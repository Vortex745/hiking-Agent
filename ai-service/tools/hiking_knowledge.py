"""RAG-backed hiking knowledge tool."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from rag.retriever import VectorStoreRetriever
from rag.text_processing import clean_display_text

logger = logging.getLogger("ai-service.tools.hiking_knowledge")


def _preview(text: str, limit: int = 220) -> str:
    cleaned = clean_display_text(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _score_from_metadata(metadata: dict[str, Any]) -> float | None:
    for key in ("score", "rrf_score", "bm25_score", "similarity"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


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

    try:
        retriever = VectorStoreRetriever()
        docs = retriever.hybrid_search([normalized_query], k=k, status_filter="hiking")
    except Exception as exc:
        logger.warning("hiking knowledge retrieval failed: %s", exc)
        return {
            "query": normalized_query,
            "chunks": [],
            "message": f"资料不足：户外知识库检索失败，原因：{exc}",
        }

    chunks = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("file_name") or "unknown"
        title = metadata.get("title") or metadata.get("file_name") or source
        chunks.append(
            {
                "rank": idx,
                "title": title,
                "source": source,
                "score": _score_from_metadata(metadata),
                "preview": _preview(doc.page_content),
            }
        )

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
