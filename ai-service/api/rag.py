import asyncio
import json
import logging
import os
import re
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from api.models import (
    RAGQuery,
    RAGUploadResponse,
    RAGDocument,
    RuntimeModelSettings,
)
from config import settings
from memory.factory import chat_memory_status, get_chat_memory
from runtime_paths import runtime_dir

if TYPE_CHECKING:
    from rag.runtime import RAGRuntime

logger = logging.getLogger("ai-service.rag")
rag_router = APIRouter(prefix="/rag")

RAG_DOCS_DIR = runtime_dir("RAG_DOCS_DIR", "rag_docs")


def get_rag_runtime(request: Request) -> "RAGRuntime | None":
    """票据 01：FastAPI 依赖，返回 app.state.rag_runtime（可能为 None）。

    受控切换：当 settings.rag_use_graph_runtime=True 且 lifespan 装配成功时
    返回 RAGRuntime；否则返回 None，调用方走 legacy 路径（零回归）。
    """
    return getattr(request.app.state, "rag_runtime", None)

DocumentLoader = None
VectorStoreRetriever = None
CloudDocsLoader = None
QueryRewriter = None
ContextAugmenter = None
Reranker = None
FeishuDefaultSyncer = None
FeishuDocLoader = None
find_feishu_links = None
has_relevant_evidence = None
clean_display_text = None


def _get_document_loader_class():
    global DocumentLoader
    if DocumentLoader is None:
        from rag.loader import DocumentLoader as loaded

        DocumentLoader = loaded
    return DocumentLoader


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


def _get_query_rewriter_class():
    global QueryRewriter
    if QueryRewriter is None:
        from rag.rewriter import QueryRewriter as loaded

        QueryRewriter = loaded
    return QueryRewriter


def _get_context_augmenter_class():
    global ContextAugmenter
    if ContextAugmenter is None:
        from rag.augmenter import ContextAugmenter as loaded

        ContextAugmenter = loaded
    return ContextAugmenter


def _get_reranker_class():
    global Reranker
    if Reranker is None:
        from rag.reranker import Reranker as loaded

        Reranker = loaded
    return Reranker


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


def _has_relevant_evidence(question: str, docs: list) -> bool:
    global has_relevant_evidence
    if has_relevant_evidence is None:
        from rag.augmenter import has_relevant_evidence as loaded

        has_relevant_evidence = loaded
    return has_relevant_evidence(question, docs)


def _clean_display_text(text: str) -> str:
    global clean_display_text
    if clean_display_text is None:
        from rag.text_processing import clean_display_text as loaded

        clean_display_text = loaded
    return clean_display_text(text)


def _get_retriever_sqlite_store(retriever: "VectorStoreRetriever"):
    """07 票据：从 retriever 取出底层 SQLiteStore（若 SQLite 后端可用）。

    返回 None 表示 retriever 走 pgvector 或 memory 后端（legacy 路径）。
    13 票据：通过公共 property sqlite_store 访问，不读取 _sqlite_client 私有字段。
    在 storage_mode=="sqlite" 但 store 为 None 时记录 warning，避免静默回退。
    """
    sqlite_store = getattr(retriever, "sqlite_store", None)
    if sqlite_store is None:
        # 检查是否为 SQLite 后端但 store 丢失（静默回退风险）
        storage_mode = getattr(retriever, "storage_mode", "")
        if storage_mode == "sqlite":
            logger.warning(
                "rag_upload: storage_mode=sqlite but sqlite_store is None; "
                "falling back to legacy upload path (canonical ingestion skipped)"
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


def direct_rag_answer(question: str) -> str | None:
    normalized = question.strip().lower().rstrip("。！？!?")
    if normalized in DIRECT_ANSWER_PATTERNS:
        if normalized in {"你是谁"}:
            return "我是 AI Hiking 的 RAG 助手，可以帮你检索知识库、总结文档，也可以回答一些简单问题。"
        if normalized in {"谢谢", "谢谢你"}:
            return "不客气。需要查知识库内容时，直接把问题发给我就行。"
        return "你好，我是 AI Hiking 的 RAG 助手。你可以上传文档后向我提问，也可以先问一些简单问题。"
    return None


def _compact_question_text(question: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", question).strip().lower()


def _has_previous_user_message(history: list[dict]) -> bool:
    return any(item.get("role") == "user" and str(item.get("content") or "").strip() for item in history)


def _is_follow_up_question(question: str, history: list[dict]) -> bool:
    return _has_previous_user_message(history) and any(marker in question for marker in FOLLOW_UP_MARKERS)


def _has_hiking_intent(question: str) -> bool:
    lowered = question.lower()
    return any(term.lower() in lowered for term in HIKING_INTENT_TERMS)


def _rag_guard_answer(question: str, history: list[dict]) -> str | None:
    """Return a direct guard answer when a query should not enter retrieval."""
    if _find_feishu_links(question):
        return None

    compact = _compact_question_text(question)
    if not compact:
        return RAG_UNCLEAR_MSG

    if _is_follow_up_question(question, history):
        return None

    if _has_hiking_intent(question):
        return None

    if len(compact) < 4:
        return RAG_UNCLEAR_MSG

    if any(term in compact for term in AMBIGUOUS_ACTIVITY_TERMS):
        return RAG_AMBIGUOUS_ACTIVITY_MSG

    return RAG_OUT_OF_DOMAIN_MSG


def _should_trigger_clarification(question: str, history: list[dict]) -> tuple[bool, str]:
    """08 票据：判断首请求是否应触发澄清交互。

    触发澄清的场景（用户问题模糊到无法检索，但属于徒步领域）：
    - 问题为空或纯噪声 → 触发（澄清原因：请提供具体问题）
    - 问题过短（<4 字符）且无上下文 → 触发（澄清原因：请补充更多细节）
    - 含模糊活动词（"活动"、"带我"等）但无目的地 → 触发（澄清原因：请补充目的地/天数/难度）

    不触发澄清的场景：
    - 含飞书链接（走飞书同步流程）
    - 追问（结合历史可解析）
    - 含徒步意图词（直接检索）
    - 明确超出领域（直接返回 RAG_OUT_OF_DOMAIN_MSG，不需要澄清）

    Args:
        question: 用户原始问题
        history: 对话历史

    Returns:
        (should_trigger, clarification_reason)
    """
    if _find_feishu_links(question):
        return (False, "")

    compact = _compact_question_text(question)
    if not compact:
        return (True, "请提供具体的徒步相关问题，比如路线、装备、天气风险或急救处理。")

    if _is_follow_up_question(question, history):
        return (False, "")

    if _has_hiking_intent(question):
        return (False, "")

    if len(compact) < 4:
        return (True, "问题太简短，请补充更多细节，比如目的地、天数或想了解的方面。")

    if any(term in compact for term in AMBIGUOUS_ACTIVITY_TERMS):
        return (True, "你是想问徒步活动吗？请补充目的地、天数、难度或你想了解的具体问题。")

    # 明确超出领域，不触发澄清（直接返回 out_of_domain 消息）
    return (False, "")


def _build_clarification_questions_for_api(question: str, reason: str) -> list[str]:
    """08 票据：构造推送给用户的澄清问题列表。"""
    questions: list[str] = []
    if reason:
        questions.append(reason)
    # 默认补充一个引导性问题
    questions.append("你想了解哪条路线、哪种装备，还是其他徒步相关问题？")
    # 去重保序
    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:2]  # 至多 2 个问题


def _runtime_llm_kwargs_from_settings(model_settings: RuntimeModelSettings | None) -> dict:
    if not model_settings or not model_settings.llm:
        return {}

    config = model_settings.llm
    return {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
    }


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


def _runtime_rerank_kwargs_from_settings(model_settings: RuntimeModelSettings | None) -> dict:
    if not model_settings or not model_settings.rerank:
        return {}

    config = model_settings.rerank
    return {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
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


def _recent_rag_history(memory, limit: int = 6) -> list[dict]:
    if memory is None:
        return []
    try:
        messages = memory.get_messages()
    except Exception as e:
        logger.warning("读取 RAG 历史失败，跳过上下文改写: %s", e)
        return []

    if not isinstance(messages, list):
        return []

    normalized: list[dict] = []
    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content})
    return normalized


def _preview_text(text: str, limit: int = 260) -> str:
    normalized = _clean_display_text(text)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


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


def summarize_retrieved_documents(docs: list) -> dict:
    grouped: dict[str, dict] = {}

    for doc in docs:
        metadata = doc.metadata or {}
        key = _document_key(doc)
        current = grouped.setdefault(
            key,
            {
                "title": metadata.get("title") or metadata.get("file_name") or metadata.get("source") or "未命名文档",
                "source": metadata.get("source", "unknown"),
                "doc_type": metadata.get("doc_type") or "",
                "chunks": 0,
                "content": "",
            },
        )
        current["chunks"] += 1
        if not current["content"]:
            current["content"] = _preview_text(doc.page_content)

    documents = list(grouped.values())
    return {
        "searched_count": len(documents),
        "matched_chunks": len(docs),
        "documents": documents,
    }


def _dedupe_documents(docs: list) -> list:
    seen_keys: set[str] = set()
    deduped_docs = []
    for doc in docs:
        meta = doc.metadata or {}
        dedup_key = (
            meta.get("id")
            or meta.get("content_hash")
            or f"{meta.get('source', '')}:{doc.page_content[:100]}"
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        deduped_docs.append(doc)
    return deduped_docs


def _persist_rag_message(memory, role: str, content: str) -> None:
    if memory is None or not content:
        return
    try:
        memory.add_message(role, content)
    except Exception as exc:
        logger.warning("RAG chat persistence failed for role=%s: %s", role, exc)


async def _run_graph_runtime_sse(runtime, req: RAGQuery, memory):
    """graph runtime 路径的 SSE 流生成器（票据 02 + 票据 03）。

    流程（验收 1/2/3/4/5/6/7）：
    **首请求（clarification_answers is None）**：
    1. 读 history（M2 修复：在 persist 之前）
    2. 写入 user question（带重复检查：同 chat_id+question 已存在 clarification 时不重复写）
    3. guard 前置 1：问候 → text + done（不进入检索，验收 1）
    4. guard 前置 2：越域/不清晰 → process + text + done（不进入检索，验收 1/2）
    5. graph runtime.arun(question, chat_id, messages)
       - 清晰问题 → 完整图执行 → state 含 answer/retrieved_docs/process_messages
       - 不清晰问题 → 触发 interrupt → state 含 __interrupt__
    6. 检测 __interrupt__：emit clarification_needed + done（验收 2，干净结束）
    7. 否则：state → SSE 映射（process/documents/text + done，验收 3）

    **resume 请求（clarification_answers is not None）**：
    1. 读 history
    2. 写入 user clarification_answers（不重复写原 question，验收 4）
    3. graph runtime.arun(question, chat_id, messages, clarification_answers)
       内部用 Command(resume=answers) 恢复 → request_clarification 节点返回 →
       mark_resumed + 合并答案 + fan-out 到 retrieve → aggregate_answer
    4. state → SSE 映射（process/documents/text + done，验收 3/7）

    不支持 req.model_settings 自定义配置；不支持 req.status 过滤；不处理飞书链接。
    """
    # M2 修复：先读 history 再 persist，确保 guard 看到的 history 不含当前问题。
    history_messages = _recent_rag_history(memory)

    if req.clarification_answers is None:
        # === 首请求 ===
        # 写入 user question（按契约，首请求必写；重复首请求不重复写）
        # 重复检查：rag_enable_clarification=True 时通过 store.get(cid) 检查
        if settings.rag_enable_clarification:
            from rag.clarification import (
                clarification_id_for,
                get_clarification_store,
            )
            _cid = clarification_id_for(req.chat_id, req.question)
            _store = get_clarification_store()
            _existing = _store.get(_cid)
            if _existing is None:
                _persist_rag_message(memory, "user", req.question)
            # 已存在澄清记录：重复首请求，不重复写入 user question
        else:
            _persist_rag_message(memory, "user", req.question)

        # guard 前置 1：问候（direct_rag_answer）
        direct_answer = direct_rag_answer(req.question)
        if direct_answer is not None:
            yield _sse_event("text", direct_answer)
            _persist_rag_message(memory, "assistant", direct_answer)
            yield _sse_event("done")
            return

        # guard 前置 2：越域/不清晰（_rag_guard_answer）
        # _rag_guard_answer 内部 _is_follow_up_question 已识别有上下文支撑的追问（验收 2）
        guard_answer = _rag_guard_answer(req.question, history_messages)
        if guard_answer is not None:
            yield _sse_event(
                "process",
                "问题不清晰或超出徒步知识库范围，跳过知识库检索",
            )
            yield _sse_event("text", guard_answer)
            _persist_rag_message(memory, "assistant", guard_answer)
            yield _sse_event("done")
            return

        # graph runtime：调用 runtime.arun
        try:
            state = await runtime.arun(
                question=req.question,
                chat_id=req.chat_id or "default",
                messages=history_messages,
            )
        except Exception as exc:
            logger.warning("graph runtime arun failed: %s", exc)
            yield _sse_event(
                "error",
                f"知识库查询失败：{exc}",
                {"error": str(exc)},
            )
            yield _sse_event("done")
            return

        # 票据 03：检测 __interrupt__（澄清触发）
        from rag.graph import get_clarification_info, is_clarification_needed

        if is_clarification_needed(state):
            info = get_clarification_info(state) or {}
            yield _sse_event(
                "clarification_needed",
                "请补充更多信息以便我帮你查知识库",
                {
                    "clarification_id": info.get("clarification_id", ""),
                    "questions": info.get("questions", []),
                    "expires_at": info.get("expires_at", ""),
                },
            )
            yield _sse_event("done")
            return

        # 正常流程：state → SSE 映射
        async for event in _emit_state_sse(state, memory):
            yield event
        return

    # === resume 请求 ===
    # 写入 user clarification_answers（不重复写原 question，验收 4）
    answers = list(req.clarification_answers)
    answers_text = "；".join(a.strip() for a in answers if a and a.strip())
    if answers_text:
        _persist_rag_message(memory, "user", f"[澄清答案] {answers_text}")

    # graph runtime：用 clarification_answers 恢复执行
    try:
        state = await runtime.arun(
            question=req.question,
            chat_id=req.chat_id or "default",
            messages=history_messages,
            clarification_answers=answers,
        )
    except Exception as exc:
        logger.warning("graph runtime arun (resume) failed: %s", exc)
        yield _sse_event(
            "error",
            f"知识库查询失败：{exc}",
            {"error": str(exc)},
        )
        yield _sse_event("done")
        return

    # state → SSE 映射
    async for event in _emit_state_sse(state, memory):
        yield event


async def _emit_state_sse(state, memory):
    """state → SSE 事件映射（process/documents/text/done，票据 03 提取公共逻辑）。"""
    # 1. process_messages → process 事件（批量发送）
    for msg in state.get("process_messages") or []:
        if isinstance(msg, str) and msg:
            yield _sse_event("process", msg)

    # 2. retrieved_docs → documents 事件（复用 summarize_retrieved_documents）
    from langchain_core.documents import Document as _LCDocument

    docs = state.get("retrieved_docs") or []
    lc_docs = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        try:
            lc_docs.append(
                _LCDocument(
                    page_content=d.get("content", ""),
                    metadata={
                        "source": d.get("source", "unknown"),
                        "file_name": d.get("file_name", ""),
                        "title": d.get("title", ""),
                        "doc_type": d.get("doc_type", ""),
                    },
                )
            )
        except Exception as exc:
            logger.warning("graph runtime doc conversion failed: %s", exc)

    document_summary = summarize_retrieved_documents(lc_docs)
    yield _sse_event(
        "documents",
        f"已检索 {document_summary['searched_count']} 篇相关文档，共 {document_summary['matched_chunks']} 个片段",
        document_summary,
    )

    # 3. answer → text 事件（m1：answer 为空时 fallback，避免无 text 的 done 帧）
    answer = state.get("answer", "") or "暂未生成有效回答，请重试。"
    yield _sse_event("text", answer)
    _persist_rag_message(memory, "assistant", answer)

    # 4. done
    yield _sse_event("done")


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


def _sync_feishu_links(question: str, retriever: VectorStoreRetriever) -> tuple[list, list[dict]]:
    links = _find_feishu_links(question)
    if not links:
        return [], []

    syncer_cls, loader_cls = _get_feishu_classes()
    loader = loader_cls()
    syncer = syncer_cls(loader, retriever)
    summaries: list[dict] = []
    synced_docs = []

    for link in links:
        try:
            if link.kind == "wiki_space":
                summaries.extend(syncer.sync_from_space(link.token))
                continue

            docs = loader.load_and_split(link.raw, doc_type=link.doc_type)
            # 10 票据 legacy：pgvector/memory 后端的直接写入路径，留待 Ticket 14 cutover 移除。
            # SQLite 后端走 _sync_feishu_links_canonical，不会走到此处。
            retriever.add_documents(docs, status="feishu")
            syncer.synced_documents.extend(docs)
            synced_docs.extend(docs)
            title = docs[0].metadata.get("title", link.token) if docs else link.token
            summaries.append({"token": link.token, "title": title, "chunks": len(docs)})
        except Exception as exc:
            logger.warning("Feishu link sync failed: %s", exc)
            summaries.append({"token": link.token, "title": link.token, "chunks": 0, "error": str(exc)})

    return syncer.synced_documents or synced_docs, summaries


def _sync_default_feishu_knowledge(retriever: VectorStoreRetriever) -> tuple[list, list[dict]]:
    syncer_cls, loader_cls = _get_feishu_classes()
    loader = loader_cls()
    syncer = syncer_cls(loader, retriever)
    summaries: list[dict] = []

    if settings.feishu_default_space_id:
        summaries.extend(syncer.sync_from_space(settings.feishu_default_space_id))
    if settings.feishu_default_folder_token:
        summaries.extend(syncer.sync_from_folder(settings.feishu_default_folder_token))

    return syncer.synced_documents, summaries


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
    syncer = syncer_cls(loader, retriever=None)

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
    syncer = syncer_cls(loader, retriever=None)

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

    07 票据：SQLite 后端走 canonical ingestion（parent/child 层级切分）；
    pgvector/memory 后端走 legacy 路径（DocumentLoader + retriever.add_documents）零回归。
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

        # 07 票据：SQLite 后端 → canonical ingestion（统一父子切分 + 索引 + 状态）
        sqlite_store = _get_retriever_sqlite_store(retriever)
        if sqlite_store is not None:
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

        # legacy 路径（pgvector / memory 后端，零回归）
        loader_cls = _get_document_loader_class()
        loader = loader_cls()
        docs = loader.load_and_split(str(file_path))

        # 10 票据 legacy：pgvector/memory 后端的直接写入路径，留待 Ticket 14 cutover 移除。
        # SQLite 后端走 canonical ingestion（上方 pipeline.ingest_file 分支），不会走到此处。
        indexing_result = retriever.add_documents(docs, status)
        if (
            indexing_result is not None
            and getattr(indexing_result, "requested_count", len(docs)) > 0
            and (
                getattr(indexing_result, "indexed_count", len(docs)) < len(docs)
                or getattr(indexing_result, "fallback_used", False)
                or (
                    settings.postgres_configured
                    and not getattr(indexing_result, "durable", False)
                )
            )
        ):
            detail = "文档已接收，但未写入持久向量索引，暂时不能用于知识库检索。"
            error = getattr(indexing_result, "error", None)
            if error:
                detail = f"{detail} 原因：{error}"
            raise HTTPException(status_code=503, detail=detail)

        return RAGUploadResponse(
            filename=file.filename,
            chunks=len(docs),
            status=status or "none",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/query")
async def rag_query(
    req: RAGQuery,
    runtime: "RAGRuntime | None" = Depends(get_rag_runtime),
):
    """RAG query endpoint with SSE streaming.

    08 票据：支持澄清双请求 + SSE 契约
    - 首请求：仅 chat_id + question（+ 可选 status / model_settings）
    - resume 请求：同 chat_id + 同 question + clarification_answers
    - 取消请求：同 chat_id + cancel_clarification_id

    02 票据：graph runtime 受控切换
    - runtime 可用 + 首请求 + 无自定义 model_settings → 走 graph runtime 路径
    - 否则 → 走 legacy 路径（零回归）

    返回契约（08 票据冻结结论）：
    - 首请求触发澄清：SSE 流，事件序列 clarification_needed → done
    - 首请求不触发澄清：SSE 流，事件序列 process? → documents? → text → done（旧行为零回归）
    - resume 成功：SSE 流，事件序列 process? → documents? → text → done
    - resume 失败（取消/过期/不存在/已恢复）：JSONResponse（410/404/200）
    - 取消请求：JSONResponse（200 或 404）
    """

    # 08 票据：取消请求返回 JSONResponse（不进入 SSE 流，符合冻结契约）
    if req.cancel_clarification_id:
        from rag.clarification import get_clarification_store
        store = get_clarification_store()
        cancelled = store.cancel(req.cancel_clarification_id)
        if cancelled is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "clarification_not_found",
                    "clarification_id": req.cancel_clarification_id,
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "status": "cancelled",
                "clarification_id": cancelled.clarification_id,
            },
        )

    # 08 票据：resume 请求的状态校验（取消/过期/已恢复/不存在）返回 JSONResponse
    if req.clarification_answers is not None:
        from rag.clarification import (
            clarification_id_for,
            get_clarification_store,
        )
        store = get_clarification_store()
        cid = clarification_id_for(req.chat_id, req.question)
        status = store.get(cid)
        if status is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "clarification_not_found",
                    "clarification_id": cid,
                },
            )
        if store.is_cancelled(cid):
            return JSONResponse(
                status_code=410,
                content={
                    "error": "clarification_cancelled",
                    "clarification_id": cid,
                },
            )
        if store.is_expired(cid):
            return JSONResponse(
                status_code=410,
                content={
                    "error": "clarification_expired",
                    "clarification_id": cid,
                    "expired_at": status.expires_at,
                },
            )
        if store.is_resumed(cid):
            return JSONResponse(
                status_code=200,
                content={
                    "status": "already_resumed",
                    "clarification_id": cid,
                },
            )

    async def event_stream():
        memory = get_chat_memory(req.chat_id) if req.chat_id else None
        assistant_parts: list[str] = []

        # 票据 02/03：graph runtime 受控切换
        # 切换条件：
        # - runtime 可用（settings.rag_use_graph_runtime=True 且 lifespan 装配成功）
        # - 无自定义 model_settings（graph runtime 不支持运行时 LLM/embedding 切换）
        # - 无 status 过滤（graph runtime 的 retriever.search 不支持 status_filter）
        # - 无飞书链接（graph runtime 不执行飞书文档同步）
        # 票据 03：移除"首请求 only"和"未启用 clarification"排除项
        # - 首请求 + resume 请求都走 graph runtime（澄清 interrupt/resume 由 graph 内部处理）
        # - rag_enable_clarification=True 时走 graph runtime（澄清 interrupt 由 graph 处理）
        if (
            runtime is not None
            and req.model_settings is None
            and req.status is None
            and not _find_feishu_links(req.question)
        ):
            try:
                async for event in _run_graph_runtime_sse(runtime, req, memory):
                    yield event
            except Exception as exc:
                # 兜底：state→SSE 映射异常时不丢 done 帧（契约§3：error → done）
                logger.exception("graph runtime SSE unexpected error")
                yield _sse_event("error", f"知识库查询失败：{exc}")
                yield _sse_event("done")
            return

        try:
            # 08 票据：resume 请求（带 clarification_answers）
            # 已在上层完成状态校验（cancelled/expired/resumed/not_found）
            # 此处仅处理 active 且未过期的有效 resume 请求
            effective_question = req.question
            resume_cid: str | None = None
            if req.clarification_answers is not None:
                from rag.clarification import (
                    clarification_id_for,
                    get_clarification_store,
                )
                store = get_clarification_store()
                cid = clarification_id_for(req.chat_id, req.question)
                resume_cid = cid

                # 08 票据：mark_resumed 延迟到检索成功后调用，避免检索失败时
                # clarification 已标记 resumed 导致用户重试首请求被 is_resumed 拦截形成死锁。
                # 并发 resume 通过 mark_resumed 的 CAS 语义（仅 active 可 → resumed）保证只有一个成功。

                # 合并澄清答案与原问题为自包含检索输入
                answers = list(req.clarification_answers)
                answers_text = "；".join(
                    a.strip() for a in answers if a and a.strip()
                )
                if answers_text:
                    # 写入 user clarification_answers（不重复写原 question）
                    _persist_rag_message(memory, "user", f"[澄清答案] {answers_text}")
                    # 合并后的自包含问题用于检索
                    effective_question = f"{req.question}（补充：{answers_text}）"

            history_messages = _recent_rag_history(memory)
            # 首请求：写入 user question（按契约，即使触发澄清也写入）
            # 08 票据修复：重复首请求（同 chat_id+question）不重复写入 user question，
            # 仅在澄清启用时通过 store.get(cid) 检查幂等；关闭时走旧逻辑（零回归）
            if req.clarification_answers is None:
                if settings.rag_enable_clarification:
                    from rag.clarification import (
                        clarification_id_for,
                        get_clarification_store,
                    )
                    _cid = clarification_id_for(req.chat_id, req.question)
                    _store = get_clarification_store()
                    _existing = _store.get(_cid)
                    if _existing is None:
                        # 首次首请求：写入 user question
                        _persist_rag_message(memory, "user", req.question)
                    # 已存在澄清记录：重复首请求，不重复写入 user question
                else:
                    # 澄清未启用：走旧逻辑，始终写入 user question（零回归）
                    _persist_rag_message(memory, "user", req.question)

            direct_answer = direct_rag_answer(req.question)
            if direct_answer is not None:
                assistant_parts.append(direct_answer)
                yield _sse_event("text", direct_answer)
                _persist_rag_message(memory, "assistant", "".join(assistant_parts).strip())
                # 08 票据：resume 请求在检索/回答成功后标记 resumed（CAS 原子）
                if resume_cid is not None:
                    from rag.clarification import get_clarification_store
                    get_clarification_store().mark_resumed(resume_cid)
                yield _sse_event("done")
                return

            # 08 票据：首请求触发澄清（仅 enable_clarification=True 且首请求时检查）
            if (
                req.clarification_answers is None
                and settings.rag_enable_clarification
            ):
                from rag.clarification import (
                    clarification_id_for,
                    get_clarification_store,
                )
                _cid = clarification_id_for(req.chat_id, req.question)
                _store = get_clarification_store()
                _existing = _store.get(_cid)
                # 08 票据修复：若同 chat_id+question 的 clarification 已 resumed，
                # 说明用户重试首请求且之前 resume 已成功，直接走检索流程（避免死锁）
                if _existing is not None and _existing.status == "resumed":
                    pass  # 跳过澄清触发，走检索流程
                else:
                    should_trigger, reason = _should_trigger_clarification(
                        req.question, history_messages
                    )
                    if should_trigger:
                        clarification_questions = _build_clarification_questions_for_api(
                            req.question, reason
                        )
                        status_record = _store.create(
                            chat_id=req.chat_id,
                            question=req.question,
                            questions=clarification_questions,
                        )
                        # 08 票据修复：契约§5 - 澄清期间发新问题，旧 clarification 自动过期
                        # 但同 chat_id+question 重复首请求（幂等）不过期
                        _store.expire_active_for_chat(
                            chat_id=req.chat_id,
                            except_clarification_id=status_record.clarification_id,
                        )
                        # 推送 clarification_needed 事件 + done（按契约干净结束首请求）
                        yield _sse_event(
                            "clarification_needed",
                            "请补充更多信息以便我帮你查知识库",
                            {
                                "clarification_id": status_record.clarification_id,
                                "questions": list(status_record.questions),
                                "expires_at": status_record.expires_at,
                            },
                        )
                        yield _sse_event("done")
                        return

            # resume 请求不重复走 guard_answer（已通过澄清流程判定为需检索）
            if req.clarification_answers is None:
                guard_answer = _rag_guard_answer(req.question, history_messages)
                if guard_answer is not None:
                    yield _sse_event(
                        "process",
                        "问题不清晰或超出徒步知识库范围，跳过知识库检索",
                    )
                    assistant_parts.append(guard_answer)
                    yield _sse_event("text", guard_answer)
                    _persist_rag_message(memory, "assistant", "".join(assistant_parts).strip())
                    yield _sse_event("done")
                    return

            # 使用 effective_question（resume 时为合并后的问题）继续检索流程
            question_for_retrieval = effective_question

            embedding_kwargs = _runtime_embedding_kwargs_from_settings(req.model_settings)
            try:
                retriever_cls = _get_vector_store_retriever_class()
                retriever = retriever_cls(**embedding_kwargs)
            except Exception as e:
                logger.warning("自定义 Embedding 配置不可用，降级到默认配置: %s", e)
                retriever_cls = _get_vector_store_retriever_class()
                retriever = retriever_cls()
                yield _sse_event(
                    "process",
                    f"自定义 Embedding 配置连接失败，已降级使用默认配置",
                    {"error": str(e)[:200]},
                )

            cloud_docs: list = []
            if settings.rag_docs_api_url:
                yield _sse_event("process", "从云知识库 API 同步文档")
                await asyncio.sleep(0.2)
                try:
                    cloud_loader_cls = _get_cloud_docs_loader_class()
                    cloud_loader = cloud_loader_cls()
                    # 08 票据：SQLite 后端走 canonical ingestion（父子层级切分 + 单份失败不中断）；
                    # pgvector/memory 后端走 legacy 路径（load_and_split + add_documents）零回归
                    sqlite_store = _get_retriever_sqlite_store(retriever)
                    if sqlite_store is not None:
                        from rag.ingestion import IngestionPipeline

                        embedding_provider = _build_embedding_provider_for_ingestion(
                            req.model_settings
                        )
                        pipeline = IngestionPipeline(
                            store=sqlite_store,
                            embedding_provider=embedding_provider,
                        )
                        raw_docs = cloud_loader.load_raw_docs(settings.rag_docs_api_url)
                        failed_docs: list[dict] = []
                        successful_count = 0
                        for raw in raw_docs:
                            name = raw.get("name", "unknown")
                            content = raw.get("content", "")
                            try:
                                report = pipeline.ingest_content(
                                    markdown=content,
                                    source=f"cloud_docs:{name}",
                                    file_name=name,
                                    status="cloud_docs",
                                )
                                # 只计入实际写入的（inserted/replaced），
                                # noop（内容未变化）不计入 successful_count
                                if report.action in ("inserted", "replaced"):
                                    successful_count += 1
                                cloud_docs.append(
                                    {"name": name, "chunks": report.child_count}
                                )
                            except Exception as doc_exc:
                                # 验收 3：单份失败不中断其他文档
                                logger.warning(
                                    "云知识库文档 %s 摄入失败: %s", name, doc_exc
                                )
                                failed_docs.append(
                                    {"name": name, "error": str(doc_exc)[:200]}
                                )

                        yield _sse_event(
                            "process",
                            (
                                f"已同步云知识库文档，获得 {successful_count} 份文档"
                                if successful_count
                                else "云知识库同步未获得文档片段，请检查 API 地址及返回数据"
                            ),
                            {
                                "source": "cloud_docs_api",
                                "api_url": settings.rag_docs_api_url,
                                "chunks": successful_count,
                                "failed_docs": failed_docs,
                            },
                        )
                    else:
                        # legacy 路径（pgvector/memory 后端，零回归）
                        loaded = cloud_loader.load_and_split(settings.rag_docs_api_url)
                        if loaded:
                            # 10 票据 legacy：pgvector/memory 后端的直接写入路径，留待 Ticket 14 cutover 移除。
                            # SQLite 后端走 canonical ingestion（上方 pipeline.ingest_content 分支），不会走到此处。
                            retriever.add_documents(loaded, status="cloud_docs")
                            cloud_docs.extend(loaded)
                        yield _sse_event(
                            "process",
                            (
                                f"已同步云知识库文档，获得 {len(loaded)} 个文档片段"
                                if loaded
                                else "云知识库同步未获得文档片段，请检查 API 地址及返回数据"
                            ),
                            {
                                "source": "cloud_docs_api",
                                "api_url": settings.rag_docs_api_url,
                                "chunks": len(loaded),
                            },
                        )
                except Exception as e:
                    logger.warning("云知识库 API 同步失败: %s", e)
                    yield _sse_event(
                        "process",
                        f"云知识库同步失败：{e}。已转入普通知识库检索",
                        {"error": str(e)},
                    )
                await asyncio.sleep(0.2)

            feishu_docs: list = []  # 09 票据：SQLite 后端保持为 []（验收 6：不绕过检索）
            feishu_synced = False  # 09 票据：用于早退判断（替代 feishu_docs，跨后端统一）
            feishu_links = _find_feishu_links(question_for_retrieval)
            if feishu_links and not settings.feishu_enabled:
                yield _sse_event(
                    "process",
                    "检测到外部文档链接，但外部应用凭证未配置，已转入普通知识库检索",
                    {"link_count": len(feishu_links)},
                )
                await asyncio.sleep(0.2)
                answer = "我检测到了外部文档链接，但当前服务没有配置外部应用凭证，暂时不能读取该文档。请先配置应用凭证，或把文档导出后上传到知识库。"
                assistant_parts.append(answer)
                yield _sse_event("text", answer)
                _persist_rag_message(memory, "assistant", "".join(assistant_parts).strip())
                yield _sse_event("done")
                return
            elif feishu_links:
                yield _sse_event(
                    "process",
                    "检测到外部文档链接，正在同步外部文档",
                    {"link_count": len(feishu_links)},
                )
                await asyncio.sleep(0.2)
                # 09 票据：SQLite 后端走 canonical ingestion；
                # pgvector/memory 后端走 legacy 路径（_sync_feishu_links）零回归
                sqlite_store = _get_retriever_sqlite_store(retriever)
                if sqlite_store is not None:
                    from rag.ingestion import IngestionPipeline

                    embedding_provider = _build_embedding_provider_for_ingestion(
                        req.model_settings
                    )
                    pipeline = IngestionPipeline(
                        store=sqlite_store,
                        embedding_provider=embedding_provider,
                    )
                    _, loader_cls = _get_feishu_classes()
                    loader = loader_cls()
                    summaries, failed_docs = _sync_feishu_links_canonical(
                        question_for_retrieval, pipeline, loader
                    )
                    sync_summary = _summarize_feishu_sync(summaries + failed_docs)
                    # SQLite 后端：feishu_docs 保持 []（验收 6：通过检索获取结果）
                    feishu_synced = bool(summaries)
                else:
                    # legacy 后端（pgvector/memory，零回归）
                    feishu_docs, summaries = _sync_feishu_links(
                        question_for_retrieval, retriever
                    )
                    sync_summary = _summarize_feishu_sync(summaries)
                    feishu_synced = bool(feishu_docs)
                yield _sse_event(
                    "process",
                    (
                        f"已同步外部文档，获得 {sync_summary['chunks']} 个文档片段"
                        if sync_summary["chunks"]
                        else "外部文档同步未获得文档片段，请检查链接权限"
                    ),
                    sync_summary,
                )
                await asyncio.sleep(0.2)
                if not feishu_synced:
                    answer = "我检测到了外部文档链接，但没有同步到可检索的文档内容。请确认应用对该文档有访问权限，或先把文档导出/上传后再问。"
                    assistant_parts.append(answer)
                    yield _sse_event("text", answer)
                    _persist_rag_message(memory, "assistant", "".join(assistant_parts).strip())
                    yield _sse_event("done")
                    return
            elif (
                not cloud_docs
                and settings.feishu_enabled
                and settings.feishu_default_configured
                and retriever.document_count() == 0
            ):
                yield _sse_event("process", "知识库为空，正在同步默认外部知识库")
                await asyncio.sleep(0.2)
                # 09 票据：SQLite 后端走 canonical ingestion；
                # pgvector/memory 后端走 legacy 路径（_sync_default_feishu_knowledge）零回归
                sqlite_store = _get_retriever_sqlite_store(retriever)
                if sqlite_store is not None:
                    from rag.ingestion import IngestionPipeline

                    embedding_provider = _build_embedding_provider_for_ingestion(
                        req.model_settings
                    )
                    pipeline = IngestionPipeline(
                        store=sqlite_store,
                        embedding_provider=embedding_provider,
                    )
                    _, loader_cls = _get_feishu_classes()
                    loader = loader_cls()
                    summaries, failed_docs = _sync_default_feishu_knowledge_canonical(
                        pipeline, loader
                    )
                    sync_summary = _summarize_feishu_sync(summaries + failed_docs)
                    # SQLite 后端：feishu_docs 保持 []（验收 6：通过检索获取结果）
                    feishu_synced = bool(summaries)
                else:
                    # legacy 后端（pgvector/memory，零回归）
                    feishu_docs, summaries = _sync_default_feishu_knowledge(retriever)
                    sync_summary = _summarize_feishu_sync(summaries)
                    feishu_synced = bool(feishu_docs)
                yield _sse_event(
                    "process",
                    (
                        f"已同步默认外部知识库，获得 {sync_summary['chunks']} 个文档片段"
                        if sync_summary["chunks"]
                        else "默认外部知识库同步未获得文档片段，请检查空间/文件夹权限"
                    ),
                    sync_summary,
                )
                await asyncio.sleep(0.2)

            llm_kwargs = _runtime_llm_kwargs_from_settings(req.model_settings)
            rewriter_cls = _get_query_rewriter_class()
            rewriter = rewriter_cls(**llm_kwargs)
            try:
                augmenter_cls = _get_context_augmenter_class()
                augmenter = augmenter_cls(**llm_kwargs)
            except Exception as e:
                logger.warning("自定义 LLM 配置不可用，降级到默认配置: %s", e)
                augmenter_cls = _get_context_augmenter_class()
                augmenter = augmenter_cls()
            rerank_kwargs = _runtime_rerank_kwargs_from_settings(req.model_settings)
            try:
                reranker_cls = _get_reranker_class()
                reranker = reranker_cls(**rerank_kwargs)
            except Exception:
                reranker_cls = _get_reranker_class()
                reranker = reranker_cls()

            contextualize = getattr(rewriter, "contextualize", None)
            contextual_question = (
                contextualize(question_for_retrieval, history_messages)
                if callable(contextualize)
                else question_for_retrieval
            )
            if contextual_question != question_for_retrieval:
                yield _sse_event(
                    "process",
                    "结合历史上下文改写当前问题，用于检索",
                    {"question": contextual_question},
                )
                await asyncio.sleep(0.2)

            queries = rewriter.rewrite(contextual_question)
            yield _sse_event(
                "process",
                "调用查询改写模块生成检索查询：用户提问进入 query 改写",
                {"query_count": len(queries), "question": contextual_question},
            )
            await asyncio.sleep(0.2)

            yield _sse_event(
                "process",
                "调用 embedding 模型生成查询向量：问题向量化",
                {"query_count": len(queries)},
            )
            await asyncio.sleep(0.2)

            yield _sse_event(
                "process",
                "LangChain 混合检索组件开始召回：向量检索 + BM25",
                {"storage": getattr(retriever, "storage_mode", "unknown")},
            )
            await asyncio.sleep(0.2)

            all_docs = []
            retrieval_error = ""
            try:
                hybrid_search = getattr(retriever, "hybrid_search", None)
                if callable(hybrid_search):
                    all_docs = hybrid_search(queries, k=4, status_filter=req.status)
                else:
                    for q in queries:
                        all_docs.extend(retriever.similarity_search(q, k=2, status_filter=req.status))
                    all_docs = _dedupe_documents(all_docs)
            except Exception as e:
                retrieval_error = str(e)
                logger.warning("混合检索失败: %s", e)

            if retrieval_error and not all_docs and embedding_kwargs:
                logger.warning("自定义 Embedding 检索失败，降级到默认配置重试")
                yield _sse_event(
                    "process",
                    "自定义 Embedding 配置不可用，正在使用默认配置重试",
                    {"error": retrieval_error[:200]},
                )
                await asyncio.sleep(0.2)
                try:
                    retriever_cls = _get_vector_store_retriever_class()
                    retriever = retriever_cls()
                    hybrid_search = getattr(retriever, "hybrid_search", None)
                    if callable(hybrid_search):
                        all_docs = hybrid_search(queries, k=4, status_filter=req.status)
                    else:
                        for q in queries:
                            all_docs.extend(retriever.similarity_search(q, k=2, status_filter=req.status))
                        all_docs = _dedupe_documents(all_docs)
                    retrieval_error = ""
                except Exception as e:
                    retrieval_error = str(e)
                    logger.warning("默认 Embedding 混合检索也失败: %s", e)

            if not retrieval_error and not all_docs and embedding_kwargs:
                logger.warning("自定义 Embedding 未返回候选片段，使用默认配置重试")
                yield _sse_event(
                    "process",
                    "自定义 Embedding 未返回候选片段，正在使用默认配置重试",
                )
                await asyncio.sleep(0.2)
                try:
                    retriever_cls = _get_vector_store_retriever_class()
                    retriever = retriever_cls()
                    hybrid_search = getattr(retriever, "hybrid_search", None)
                    if callable(hybrid_search):
                        all_docs = hybrid_search(queries, k=4, status_filter=req.status)
                    else:
                        for q in queries:
                            all_docs.extend(retriever.similarity_search(q, k=2, status_filter=req.status))
                        all_docs = _dedupe_documents(all_docs)
                except Exception as e:
                    retrieval_error = str(e)
                    logger.warning("默认 Embedding 混合检索失败: %s", e)

            if retrieval_error and not all_docs:
                yield _sse_event(
                    "error",
                    f"混合检索失败：{retrieval_error[:200]}。请检查 Embedding 模型配置或网络连接。",
                    {"error": retrieval_error},
                )
                yield _sse_event("done")
                return

            yield _sse_event(
                "process",
                "使用向量在 pgvector/memory 中召回候选片段，并通过 BM25 + RRF 完成融合",
                {
                    "candidate_chunks": len(all_docs),
                    "storage": getattr(retriever, "storage_mode", "unknown"),
                },
            )
            await asyncio.sleep(0.2)

            all_docs = feishu_docs + all_docs
            if reranker.enabled and all_docs:
                yield _sse_event("process", "调用 Rerank 模型重排候选片段")
                await asyncio.sleep(0.2)
                all_docs = reranker.rerank(contextual_question, all_docs)
                yield _sse_event("process", f"Rerank 返回 {len(all_docs)} 个高相关片段")
                await asyncio.sleep(0.2)

            humanize_for_answer = getattr(rewriter, "humanize_for_answer", None)
            humanized_question = (
                humanize_for_answer(contextual_question)
                if callable(humanize_for_answer)
                else contextual_question
            )
            yield _sse_event(
                "process",
                "按 humanizer-zh 风格改写 query，用于最终回答",
                {"question": humanized_question},
            )
            await asyncio.sleep(0.2)

            if (
                all_docs
                and not feishu_docs
                and not _has_relevant_evidence(contextual_question, all_docs)
                and not _has_relevant_evidence(humanized_question, all_docs)
            ):
                yield _sse_event(
                    "process",
                    "检索片段与问题相关性不足，已按无匹配处理",
                    {"candidate_chunks": len(all_docs)},
                )
                all_docs = []

            document_summary = summarize_retrieved_documents(all_docs)
            yield _sse_event(
                "documents",
                f"已检索 {document_summary['searched_count']} 篇相关文档，共 {document_summary['matched_chunks']} 个片段",
                document_summary,
            )
            await asyncio.sleep(0.2)

            yield _sse_event("process", "构造上下文并调用 LLM 生成回答")
            try:
                has_streamed = False
                stream_fn = getattr(augmenter, "augment_stream", None)
                if callable(stream_fn):
                    async for chunk in stream_fn(humanized_question, all_docs):
                        assistant_parts.append(chunk)
                        yield _sse_event("text", chunk)
                        has_streamed = True
                if not has_streamed:
                    augmented = augmenter.augment(humanized_question, all_docs)
                    assistant_parts.append(augmented)
                    yield _sse_event("text", augmented)
            except Exception as e:
                logger.warning("LLM 生成回答失败: %s", e)
                context_preview = summarize_retrieved_documents(all_docs).get("documents", [])[:2]
                augmented = (
                    f"生成回答时出错：{e}\n\n"
                    f"但已检索到 {len(all_docs)} 个相关片段，以下是原始内容摘要：\n\n"
                    f"{chr(10).join(str(d) for d in context_preview)}\n\n"
                    f"原问题：{humanized_question}"
                )
                assistant_parts.append(augmented)
                yield _sse_event("text", augmented)

            _persist_rag_message(memory, "assistant", "".join(assistant_parts).strip())
            # 08 票据：resume 请求在检索/回答成功后标记 resumed（CAS 原子）
            if resume_cid is not None:
                from rag.clarification import get_clarification_store
                get_clarification_store().mark_resumed(resume_cid)
            yield _sse_event("done")
        except Exception as e:
            logger.exception("RAG query error")
            yield _sse_event("error", str(e))
            # 08 票据：兜底异常必须发送 done 事件，符合契约§3"终止帧顺序 error → done"
            # 防止客户端因缺少 done 帧而无限等待
            yield _sse_event("done")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


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
