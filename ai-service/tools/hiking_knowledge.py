"""RAG-backed hiking knowledge tool.

T17 (#17): the budget-controlled retrieve subgraph (retrieval planning,
parent back-tracing, compression, dedup) runs INSIDE this tool — to the
model it stays a single tool call. The subgraph compiles once per process
and is reused across calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from config import settings
from rag.retriever import VectorStoreRetriever
from rag.text_processing import clean_display_text

logger = logging.getLogger("ai-service.tools.hiking_knowledge")

# Distance numbers only (gate v1). Extracted from FULL content — the
# preview is truncated and must never be the source of gate evidence.
_DISTANCE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:千米|公里|km)", re.IGNORECASE)


def distance_values_in(text: str) -> list[float]:
    values: list[float] = []
    for match in _DISTANCE_PATTERN.finditer(text or ""):
        value = float(match.group(1))
        if value not in values:
            values.append(value)
    return values


def _preview(text: str, limit: int = 220) -> str:
    cleaned = clean_display_text(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _score_from_fields(doc: dict[str, Any]) -> float | None:
    for key in ("score_rrf", "score", "bm25_score", "score_sparse", "score_dense", "similarity"):
        value = doc.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


_retrieval_stack_lock = threading.Lock()
_retriever: VectorStoreRetriever | None = None
_subgraph: Any = None
_retrieval_stack_fingerprint: tuple[Any, ...] | None = None
# 账号级检索栈：key=owner_id，value=(retriever, subgraph, fingerprint)。
_owner_retrieval_stacks: dict[str, tuple[VectorStoreRetriever, Any, tuple[Any, ...]]] = {}


def _make_parent_retriever_fn(retriever: VectorStoreRetriever):
    """parent 回溯回调：parent_ids → 完整 parent 文档（retrieve_subgraph 契约）。"""

    def _fn(parent_ids: list[str]) -> list[dict[str, Any]]:
        store = retriever.sqlite_store
        if store is None or not parent_ids:
            return []
        from rag.hybrid_retrieval import retrieve_parent_chunks

        result = retrieve_parent_chunks(store.read_conn, parent_ids)
        return list(result.get("parents", []))

    return _fn


def _embedding_fingerprint() -> tuple[Any, ...]:
    """影响检索栈行为的配置指纹（embedding 客户端参数 + 索引文件路径）。"""
    return (
        settings.embedding_base_url,
        settings.embedding_api_key,
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.rag_sqlite_db_path,
    )


def _get_retrieval_stack() -> tuple[VectorStoreRetriever, Any]:
    """Lazily build the retriever + compiled subgraph, keyed by config fingerprint.

    惰性构建保持 import 无副作用（不打开数据库文件）；构建与替换持锁。
    配置页保存 embedding 配置后指纹变化：先构建新栈，成功后再替换旧栈
    （构建失败时保留旧栈可用）。

    旧栈只丢弃引用，不显式 close：重建可能发生在其他请求仍在用旧栈检索时，
    此时关闭它会抽走其 SQLite 连接。引用归零后由 GC 关闭，天然等待在飞
    请求释放旧栈。
    """
    from agent_v2.user_scope import current_owner_id, current_user_settings

    owner_id = current_owner_id.get()
    if owner_id:
        effective = current_user_settings.get()
        if effective is None:
            raise RuntimeError("账号检索作用域缺少配置")
        return _get_owner_retrieval_stack(owner_id, effective)

    global _retriever, _subgraph, _retrieval_stack_fingerprint
    with _retrieval_stack_lock:
        fingerprint = _embedding_fingerprint()
        if (
            _retriever is None
            or _subgraph is None
            or _retrieval_stack_fingerprint != fingerprint
        ):
            retriever, subgraph = _build_retrieval_stack(settings.rag_sqlite_db_path)
            _retriever = retriever
            _subgraph = subgraph
            _retrieval_stack_fingerprint = fingerprint
    return _retriever, _subgraph


def _build_retrieval_stack(
    db_path: str, effective_settings: Any | None = None
) -> tuple[VectorStoreRetriever, Any]:
    from rag.retrieve_subgraph import build_retrieve_subgraph

    effective = effective_settings or settings
    if effective_settings is None:
        # 旧全局作用域：保持无参构造，测试与脚本可替换 VectorStoreRetriever。
        retriever = VectorStoreRetriever()
    else:
        retriever = VectorStoreRetriever(
            base_url=effective.embedding_base_url,
            api_key=effective.embedding_api_key,
            model=effective.embedding_model,
            dimensions=effective.embedding_dimensions,
            db_path=db_path,
        )
    subgraph = build_retrieve_subgraph(
        retriever=retriever,
        parent_retriever_fn=_make_parent_retriever_fn(retriever),
    )
    return retriever, subgraph


def _get_owner_retrieval_stack(
    owner_id: str, effective_settings: Any
) -> tuple[VectorStoreRetriever, Any]:
    from auth.paths import user_rag_path

    db_path = str(user_rag_path(settings, owner_id))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    fingerprint = (
        owner_id,
        effective_settings.embedding_base_url,
        effective_settings.embedding_api_key,
        effective_settings.embedding_model,
        effective_settings.embedding_dimensions,
        db_path,
    )
    with _retrieval_stack_lock:
        entry = _owner_retrieval_stacks.get(owner_id)
        if entry is None or entry[2] != fingerprint:
            retriever, subgraph = _build_retrieval_stack(
                db_path, effective_settings
            )
            _owner_retrieval_stacks[owner_id] = (retriever, subgraph, fingerprint)
        retriever, subgraph, _ = _owner_retrieval_stacks[owner_id]
    return retriever, subgraph


def _docs_from_page_documents(page_docs: list) -> list[dict[str, Any]]:
    """hybrid_search 回退路径返回 LangChain Document → 子图同款 flat dict。"""
    docs: list[dict[str, Any]] = []
    for d in page_docs:
        metadata = d.metadata or {}
        docs.append(
            {
                "content": d.page_content,
                "source": metadata.get("source", ""),
                "file_name": metadata.get("file_name", ""),
                "title": metadata.get("title", ""),
                "score_rrf": metadata.get("rrf_score"),
            }
        )
    return docs


def _chunks_from_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.get("source") or doc.get("file_name") or "unknown"
        title = doc.get("title") or doc.get("file_name") or source
        chunks.append(
            {
                "rank": idx,
                "title": title,
                "source": source,
                "score": _score_from_fields(doc),
                "preview": _preview(doc.get("content", "")),
                "distance_values": distance_values_in(doc.get("content", "")),
            }
        )
    return chunks


@tool
async def hiking_knowledge_search(query: str, k: int = 4) -> dict[str, Any]:
    """Search the hiking RAG knowledge base and return traceable evidence chunks."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return {
            "query": normalized_query,
            "chunks": [],
            "message": "资料不足：检索问题为空。",
        }

    from agent_v2.user_scope import current_owner_id, current_user_settings

    owner_id = current_owner_id.get()
    if owner_id == "":
        return {
            "query": normalized_query,
            "chunks": [],
            "message": "资料不足：知识库未绑定当前登录账号。",
        }
    if owner_id:
        effective = current_user_settings.get()
        if effective is None or not str(
            getattr(effective, "embedding_api_key", "") or ""
        ).strip():
            return {
                "query": normalized_query,
                "chunks": [],
                "message": "资料不足：本账号尚未配置 Embedding 模型，无法检索知识库。",
            }

    try:
        retriever, subgraph = _get_retrieval_stack()
        try:
            # 预算子图为同步实现，放入线程执行避免阻塞事件循环。
            from rag.retrieve_subgraph import run_retrieve_subgraph

            state = await asyncio.to_thread(run_retrieve_subgraph, subgraph, normalized_query)
            docs = list(state.get("final_docs") or [])
        except Exception as exc:
            # 具名失败模式 retrieve_subgraph_runtime_error：退回单次混合检索。
            logger.warning(
                "retrieve subgraph failed (%s); falling back to single hybrid search", exc
            )
            docs = _docs_from_page_documents(
                retriever.hybrid_search([normalized_query], k=k)
            )
    except Exception as exc:
        logger.warning("hiking knowledge retrieval failed: %s", exc)
        return {
            "query": normalized_query,
            "chunks": [],
            "message": f"资料不足：户外知识库检索失败，原因：{exc}",
        }

    chunks = _chunks_from_docs(docs[:k])
    if not chunks:
        return {
            "query": normalized_query,
            "chunks": [],
            "message": f"资料不足：未在徒步知识库中找到与「{normalized_query}」相关的证据。",
        }

    return {
        "query": normalized_query,
        "chunks": chunks,
        "message": f"找到 {len(chunks)} 条可追溯徒步知识证据。",
    }
