import json
import logging
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse

from api.models import (
    RAGQuery,
    RAGUploadResponse,
    RAGDocument,
    RuntimeModelSettings,
)
from agent_v2.observability import CapacityExceeded
from agent_v2.run_orchestration import ChatFlowRequest
from api.chat_v2 import _build_run_orchestrator
from api.memory_v2 import resolve_request_subject, set_subject_cookie
from config import settings
from observability.langfuse_setup import reset_langfuse_session, set_langfuse_session
from memory.factory import chat_memory_status, get_chat_memory
from runtime_paths import runtime_dir

logger = logging.getLogger("ai-service.rag")
rag_router = APIRouter(prefix="/rag")

RAG_DOCS_DIR = runtime_dir("RAG_DOCS_DIR", "rag_docs")

VectorStoreRetriever = None
CloudDocsLoader = None
FeishuDefaultSyncer = None
FeishuDocLoader = None
find_feishu_links = None


def _get_vector_store_retriever_class():
    global VectorStoreRetriever
    if VectorStoreRetriever is None:
        from rag.retriever import VectorStoreRetriever as loaded

        VectorStoreRetriever = loaded
    return VectorStoreRetriever


def _get_cloud_docs_loader_class():
    global CloudDocsLoader
    if CloudDocsLoader is None:
        from rag.cloud_docs import CloudDocsLoader as loaded

        CloudDocsLoader = loaded
    return CloudDocsLoader


def _get_feishu_classes():
    global FeishuDefaultSyncer, FeishuDocLoader
    if FeishuDefaultSyncer is None:
        from rag.feishu import FeishuDefaultSyncer as loaded_syncer

        FeishuDefaultSyncer = loaded_syncer
    if FeishuDocLoader is None:
        from rag.feishu import FeishuDocLoader as loaded_loader

        FeishuDocLoader = loaded_loader
    return FeishuDefaultSyncer, FeishuDocLoader


def _find_feishu_links(text: str):
    global find_feishu_links
    if find_feishu_links is None:
        from rag.feishu import find_feishu_links as loaded

        find_feishu_links = loaded
    return find_feishu_links(text)


def _get_retriever_sqlite_store(retriever: "VectorStoreRetriever"):
    """从 retriever 取出底层 SQLiteStore（若 SQLite 后端可用）。

    返回 None 表示 retriever 未启用 SQLite 后端——摄入/同步调用方必须显式报错，
    不再存在 legacy 写入回退路径。
    13 票据：通过公共 property sqlite_store 访问，不读取 _sqlite_client 私有字段。
    在 storage_mode=="sqlite" 但 store 为 None 时记录 warning，避免静默失败。
    """
    sqlite_store = getattr(retriever, "sqlite_store", None)
    if sqlite_store is None:
        # 检查是否为 SQLite 后端但 store 丢失（静默失败风险）
        storage_mode = getattr(retriever, "storage_mode", "")
        if storage_mode == "sqlite":
            logger.warning(
                "rag ingestion: storage_mode=sqlite but sqlite_store is None; "
                "ingestion will fail with an explicit error"
            )
    return sqlite_store


def _build_embedding_provider_for_ingestion(
    runtime_settings: RuntimeModelSettings | None,
):
    """07 票据：构造 IngestionPipeline 用的 EmbeddingProvider。

    优先使用 runtime_settings 的 embedding 配置；缺失项用 settings 默认值填充。
    返回 OpenAIEmbeddingAdapter（委托 OpenAIEmbeddings.embed_documents）。
    """
    from langchain_openai import OpenAIEmbeddings
    from rag.ingestion import OpenAIEmbeddingAdapter

    kwargs = _runtime_embedding_kwargs_from_settings(runtime_settings)
    # 保守起见不传 dimensions，避免不同 langchain_openai 版本的兼容性问题
    kwargs.pop("dimensions", None)
    # 过滤 None 值并用 settings 默认值填充
    base_url = kwargs.get("base_url") or settings.embedding_base_url
    api_key = kwargs.get("api_key") or settings.embedding_api_key
    model = kwargs.get("model") or settings.embedding_model

    embeddings = OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        # 兼容非 OpenAI 原生 embedding API（如 edgefn BAAI/bge-m3）：
        # 默认 _get_len_safe_embeddings 会用 tiktoken 切 token 后发 input=[[tokens],...]
        # 嵌套数组，部分提供商返回 400 'field input must be a string, array of strings'。
        # 关闭长度切分后改为发送原始文本字符串。
        check_embedding_ctx_length=False,
    )
    return OpenAIEmbeddingAdapter(embeddings)


DIRECT_ANSWER_PATTERNS = {
    "你好",
    "您好",
    "hi",
    "hello",
    "嗨",
    "你是谁",
    "你是谁？",
    "你是谁?",
    "谢谢",
    "谢谢你",
}

FOLLOW_UP_MARKERS = {"这些", "这个", "这类", "它", "他们", "它们", "那", "上述", "上面", "刚才", "呢"}

HIKING_INTENT_TERMS = {
    "徒步",
    "登山",
    "爬山",
    "户外",
    "露营",
    "营地",
    "路线",
    "轨迹",
    "行程",
    "装备",
    "背包",
    "登山鞋",
    "冲锋衣",
    "登山杖",
    "头灯",
    "补给",
    "饮水",
    "急救",
    "导航",
    "海拔",
    "高反",
    "失温",
    "下撤",
    "穿越",
    "重装",
    "轻装",
    "雨天",
    "山洪",
}

AMBIGUOUS_ACTIVITY_TERMS = {"活动", "带我", "参加", "报名"}

RAG_UNCLEAR_MSG = "我没理解你的问题。可以换成一个具体的徒步问题，比如路线、装备、天气风险、营地选择或急救处理。"
RAG_AMBIGUOUS_ACTIVITY_MSG = "你是想问徒步活动吗？请补充目的地、天数、难度或你想了解的具体问题，我再帮你查知识库。"
RAG_OUT_OF_DOMAIN_MSG = "这个问题看起来不属于徒步知识库范围。你可以问徒步路线、装备、天气风险、营地选择、补给或急救相关问题。"


def _runtime_embedding_kwargs_from_settings(model_settings: RuntimeModelSettings | None) -> dict:
    if not model_settings or not model_settings.embedding:
        return {}

    config = model_settings.embedding
    return {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
        "dimensions": config.dimensions,
    }


def _parse_runtime_model_settings(raw: str | None) -> RuntimeModelSettings | None:
    if not raw:
        return None
    data = json.loads(raw)
    return RuntimeModelSettings(**data)


def _sse_event(event_type: str, content: str = "", metadata: dict | None = None) -> str:
    payload = {"type": event_type, "content": content}
    if metadata is not None:
        payload["metadata"] = metadata
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _document_key(doc) -> str:
    metadata = doc.metadata or {}
    for key in ("title", "file_name", "source"):
        value = metadata.get(key)
        if value:
            return f"{key}:{value}"
    return f"content:{doc.page_content[:80]}"


def _document_filename(doc) -> str:
    metadata = doc.metadata or {}
    return (
        metadata.get("title")
        or metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("id")
        or "indexed-document"
    )


def _indexed_document_refs(retriever: VectorStoreRetriever) -> list[RAGDocument]:
    grouped: dict[str, RAGDocument] = {}
    for doc in retriever.indexed_documents():
        key = _document_key(doc)
        metadata = doc.metadata or {}
        current = grouped.get(key)
        if current is None:
            grouped[key] = RAGDocument(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                filename=str(_document_filename(doc)),
                status=metadata.get("status"),
                chunk_count=1,
            )
        else:
            current.chunk_count += 1
    return list(grouped.values())


def _summarize_feishu_sync(summaries: list[dict]) -> dict:
    synced_count = sum(1 for item in summaries if not item.get("error") and item.get("chunks", 0) > 0)
    error_count = sum(1 for item in summaries if item.get("error"))
    chunk_count = sum(int(item.get("chunks") or 0) for item in summaries)
    return {
        "synced_count": synced_count,
        "error_count": error_count,
        "chunks": chunk_count,
        "documents": summaries,
    }


def _ingest_feishu_raw_docs(
    raw_docs: list[dict],
    pipeline,
) -> tuple[list[dict], list[dict]]:
    """09 票据：辅助函数：批量摄入飞书 raw_docs 到 canonical ingestion。

    单份失败 try/except 收集到 failed_docs，不中断其他文档（验收 4）。

    Args:
        raw_docs: FeishuDocLoader.fetch_raw_doc / FeishuDefaultSyncer.iter_raw_docs_* 返回的 raw dict 列表
        pipeline: IngestionPipeline 实例

    Returns:
        (summaries, failed_docs) 元组：
            summaries: 成功摄入的文档摘要（{token, title, chunks, action}）
            failed_docs: 失败的文档列表（{token, title, error}）
    """
    summaries: list[dict] = []
    failed_docs: list[dict] = []
    for raw in raw_docs:
        token = raw.get("feishu_doc_token", "")
        title = raw.get("title", token)
        # 09 票据验收 3：保留 source_url 到 summaries，让 API 层可追溯完整 URL
        # （SQLiteStore schema 暂不存 source_url 列，扩展留待 Ticket 11/12 契约统一）
        source_url = raw.get("source_url", "")
        try:
            report = pipeline.ingest_content(
                markdown=raw["markdown"],
                source=f"feishu:{token}",
                file_name=title,
                status="feishu",
            )
            summaries.append({
                "token": token,
                "title": title,
                "chunks": report.child_count,
                "action": report.action,
                "source_url": source_url,
            })
        except Exception as doc_exc:
            logger.warning("飞书文档 %s 摄入失败: %s", token, doc_exc)
            failed_docs.append({
                "token": token,
                "title": title,
                "error": str(doc_exc)[:200],
                "source_url": source_url,
            })
    return summaries, failed_docs


def _sync_feishu_links_canonical(
    question: str,
    pipeline,
    loader,
) -> tuple[list[dict], list[dict]]:
    """09 票据：通过 canonical ingestion 同步用户消息中的飞书链接。

    SQLite 后端使用：循环调用 pipeline.ingest_content，单份失败收集到 failed_docs。
    与 legacy _sync_feishu_links 不同：不返回 docs 列表（SQLite 后端通过检索获取结果，
    不绕过检索直接加入回答上下文 — 验收 6）。

    Args:
        question: 用户问题（含飞书链接）
        pipeline: IngestionPipeline 实例
        loader: FeishuDocLoader 实例

    Returns:
        (summaries, failed_docs) 元组：
            summaries: 成功摄入的文档摘要
            failed_docs: 失败的文档列表
    """
    links = _find_feishu_links(question)
    if not links:
        return [], []

    syncer_cls, _ = _get_feishu_classes()
    # canonical 路径不使用 retriever（直接走 pipeline.ingest_content）
    syncer = syncer_cls(loader)

    all_summaries: list[dict] = []
    all_failed: list[dict] = []

    for link in links:
        try:
            if link.kind == "wiki_space":
                # wiki 空间链接：批量同步
                raw_docs, space_failed = syncer.iter_raw_docs_from_space(link.token)
                all_failed.extend(space_failed)
                summaries, failed = _ingest_feishu_raw_docs(raw_docs, pipeline)
                all_summaries.extend(summaries)
                all_failed.extend(failed)
                continue

            # 单文档链接
            raw = loader.fetch_raw_doc(link)
            summaries, failed = _ingest_feishu_raw_docs([raw], pipeline)
            all_summaries.extend(summaries)
            all_failed.extend(failed)
        except Exception as exc:
            logger.warning("飞书链接同步失败: %s", exc)
            all_failed.append({
                "token": link.token,
                "title": link.token,
                "error": str(exc)[:200],
            })

    return all_summaries, all_failed


def _sync_default_feishu_knowledge_canonical(
    pipeline,
    loader,
) -> tuple[list[dict], list[dict]]:
    """09 票据：通过 canonical ingestion 同步默认飞书知识库（空库时触发）。

    SQLite 后端使用：循环调用 pipeline.ingest_content，单份失败收集到 failed_docs。

    Args:
        pipeline: IngestionPipeline 实例
        loader: FeishuDocLoader 实例

    Returns:
        (summaries, failed_docs) 元组
    """
    syncer_cls, _ = _get_feishu_classes()
    syncer = syncer_cls(loader)

    all_summaries: list[dict] = []
    all_failed: list[dict] = []

    if settings.feishu_default_space_id:
        raw_docs, space_failed = syncer.iter_raw_docs_from_space(
            settings.feishu_default_space_id
        )
        all_failed.extend(space_failed)
        summaries, failed = _ingest_feishu_raw_docs(raw_docs, pipeline)
        all_summaries.extend(summaries)
        all_failed.extend(failed)

    if settings.feishu_default_folder_token:
        raw_docs, folder_failed = syncer.iter_raw_docs_from_folder(
            settings.feishu_default_folder_token
        )
        all_failed.extend(folder_failed)
        summaries, failed = _ingest_feishu_raw_docs(raw_docs, pipeline)
        all_summaries.extend(summaries)
        all_failed.extend(failed)

    return all_summaries, all_failed


@rag_router.get("/health")
async def rag_health():
    try:
        if not RAG_DOCS_DIR.exists():
            raise RuntimeError("RAG documents directory is missing")
        if not os.access(RAG_DOCS_DIR, os.W_OK):
            raise RuntimeError("RAG documents directory is not writable")

        retriever_cls = _get_vector_store_retriever_class()
        retriever = retriever_cls()
        document_count = retriever.document_count()
        return {
            "status": "ok",
            "module": "rag",
            "service": "ai-service",
            "storage": retriever.storage_mode,
            "documents": document_count,
            "rag_docs_api_configured": bool(settings.rag_docs_api_url),
            # 09 票据验收 5：用户可见健康信息使用「外部文档」通用表述，不泄漏 legacy 品牌
            "external_doc_enabled": settings.feishu_enabled,
            "external_doc_default_configured": settings.feishu_default_configured,
            "memory": chat_memory_status(),
        }
    except Exception as e:
        logger.exception("RAG health check failed")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/upload")
async def rag_upload(
    file: UploadFile = File(...),
    status: str = Form(None),
    model_settings: str = Form(None),
):
    """Upload a document for RAG processing.

    SQLite 后端 canonical ingestion（parent/child 层级切分）是唯一摄入路径。
    扫描版 PDF 返回 415 Unsupported Media Type（不产生空索引）。
    """
    try:
        file_path = RAG_DOCS_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        runtime_settings = _parse_runtime_model_settings(model_settings)
        retriever_cls = _get_vector_store_retriever_class()
        retriever = retriever_cls(**_runtime_embedding_kwargs_from_settings(runtime_settings))

        sqlite_store = _get_retriever_sqlite_store(retriever)
        if sqlite_store is None:
            raise HTTPException(
                status_code=503,
                detail="知识库存储后端未启用 SQLite，无法摄入文档",
            )

        from rag.ingestion import IngestionPipeline

        embedding_provider = _build_embedding_provider_for_ingestion(runtime_settings)
        pipeline = IngestionPipeline(
            store=sqlite_store,
            embedding_provider=embedding_provider,
        )
        # source 用绝对路径身份标识（与 06 票据 ingest_file 默认行为一致）
        report = pipeline.ingest_file(
            file_path=file_path,
            source=str(file_path.resolve()),
            status=status or "active",
        )

        # 扫描版 PDF 或无法提取文本 → 415，不产生空索引
        if report.skipped:
            raise HTTPException(
                status_code=415,
                detail=f"无法解析文档：{report.skip_reason}",
            )

        return RAGUploadResponse(
            filename=file.filename,
            chunks=report.child_count,
            status=status or "active",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/query")
async def rag_query(req: RAGQuery, request: Request):
    """Stream a knowledge-QA run for the RAG page as legacy-frame SSE.

    Spec #13 phase 2 (T05): the query runs as a kernel run (profile=rag)
    through the shared run orchestrator; this adapter translates kernel
    event envelopes into the five legacy frame types the RAG page
    consumes (process / documents / text / done / error). The route
    path, request model, and frame contract are unchanged.

    The legacy clarify double-request fields on RAGQuery are accepted for
    wire compatibility but unused: clarification is a model text follow-up
    in the rag profile (the frontend never consumed clarification events).

    RAG 页会话标识 chat_id 直接作为 kernel 会话标识（多轮上下文由
    kernel 会话存储承担）。
    """
    runtime = getattr(request.app.state, "agent_v2_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="知识库问答服务未就绪（v2 runtime 未装配，检查 AGENT_V2_ENABLED）。",
        )

    request_id = f"rag-{uuid.uuid4()}"
    orchestrator = _build_run_orchestrator(runtime)
    try:
        leases = await orchestrator.acquire_leases(
            request_id=request_id, profile_id="rag"
        )
    except CapacityExceeded:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "run_capacity_exceeded", "retryable": True}},
        )

    _, subject = resolve_request_subject(request)
    flow_req = ChatFlowRequest(
        message=req.question,
        request_id=request_id,
        profile_id="rag",
        conversation_id=req.chat_id or None,
    )

    async def guarded_stream():
        # Langfuse 会话聚合：chat_id 优先，退回 request_id（与 chat v2 对齐）。
        session_token = set_langfuse_session(req.chat_id or request_id)
        try:
            async for envelope in orchestrator.stream_chat(
                flow_req, subject_id=getattr(request.state, "subject_id", None)
            ):
                frame = _rag_frame_from_envelope(envelope)
                if frame is not None:
                    yield frame
        finally:
            reset_langfuse_session(session_token)
            leases.release()

    response = StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (Gateway)
        },
    )
    set_subject_cookie(response, request, subject)
    return response


def _rag_frame_from_envelope(envelope) -> str | None:
    """Translate a kernel event envelope into a legacy RAG SSE frame.

    Legacy contract (frozen; frontend ragStream.ts):
      process/thought → 检索过程步骤行
      documents       → metadata.searched_count / matched_chunks / documents[]
      text            → 回答内容（前端逐字累加）
      done / error    → 终态帧

    run_started / heartbeat / memory.* / 未知信封类型丢弃；
    cancelled 映射为 done（旧契约没有 cancelled 帧）。
    """
    etype = envelope.type
    payload = envelope.payload or {}

    if etype == "progress":
        message = str(payload.get("message") or "")
        return _sse_event("process", message) if message else None

    if etype == "tool_result":
        evidence = payload.get("evidence")
        chunks = evidence.get("chunks") if isinstance(evidence, dict) else None
        if not isinstance(chunks, list) or not chunks:
            return None
        documents = [
            {
                "title": chunk.get("title") or "未命名文档",
                "source": chunk.get("source") or "unknown",
                "doc_type": "",
                "chunks": 1,
                "content": chunk.get("preview") or "",
            }
            for chunk in chunks
            if isinstance(chunk, dict)
        ]
        return _sse_event(
            "documents",
            f"已检索 {len(documents)} 篇相关文档，共 {len(documents)} 个片段",
            {
                "searched_count": len(documents),
                "matched_chunks": len(documents),
                "documents": documents,
            },
        )

    if etype == "text":
        return _sse_event("text", str(payload.get("content") or ""))
    if etype == "done":
        return _sse_event("done")
    if etype == "cancelled":
        return _sse_event("done")
    if etype == "error":
        return _sse_event("error", str(payload.get("message") or "知识库查询失败"))
    return None


@rag_router.get("/documents")
async def rag_documents():
    """List all uploaded documents."""
    retriever_cls = _get_vector_store_retriever_class()
    retriever = retriever_cls()
    docs = _indexed_document_refs(retriever)
    if not docs:
        for fpath in RAG_DOCS_DIR.iterdir():
            if fpath.is_file():
                docs.append(RAGDocument(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(fpath))),
                    filename=fpath.name,
                    chunk_count=0,
                ))
    return {"documents": docs}


@rag_router.get("/history/{chat_id}")
async def rag_history(chat_id: str):
    """Get persisted RAG chat history for a given chat_id."""
    try:
        memory = get_chat_memory(chat_id)
        return {"chat_id": chat_id, "messages": memory.get_messages()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.delete("/history/{chat_id}")
async def rag_clear_history(chat_id: str):
    """Clear persisted RAG chat history for a given chat_id."""
    try:
        memory = get_chat_memory(chat_id)
        memory.clear()
        return {"chat_id": chat_id, "status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
