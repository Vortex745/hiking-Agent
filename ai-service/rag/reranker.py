import logging
from typing import Any

import httpx
from langchain_core.documents import Document

from config import settings


logger = logging.getLogger("ai-service.rag.reranker")


class Reranker:
    """Rerank retrieved RAG chunks through an OpenAI-compatible rerank endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
        timeout_seconds: float | None = None,
        enabled: bool | None = None,
        client: Any | None = None,
    ):
        self.base_url = (base_url if base_url is not None else settings.rerank_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.rerank_api_key
        self.model = model if model is not None else settings.rerank_model
        self.top_k = top_k if top_k is not None else settings.rerank_top_k
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.rerank_timeout_seconds
        self._enabled = settings.rerank_enabled if enabled is None else enabled
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.base_url and self.api_key and self.model)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Return documents ordered by rerank score, falling back to original order on errors."""
        if not self.enabled or len(docs) < 2:
            return docs

        try:
            payload = {
                "model": self.model,
                "query": query,
                "documents": [doc.page_content for doc in docs],
                "top_n": max(1, min(self.top_k, len(docs))),
            }
            data = self._post_json(payload)
            reranked_docs = self._documents_from_response(data, docs)
            return reranked_docs or docs
        except Exception as exc:
            logger.warning("Rerank failed, using retrieval order: %s", exc)
            return docs

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}/rerank"

        if self._client is not None:
            response = self._client.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def _documents_from_response(self, data: dict[str, Any], docs: list[Document]) -> list[Document]:
        raw_results = data.get("results") or data.get("data") or []
        if not isinstance(raw_results, list):
            return []

        scored_docs: list[Document] = []
        for rank, item in enumerate(raw_results, start=1):
            if not isinstance(item, dict):
                continue

            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(docs):
                continue

            score = item.get("relevance_score", item.get("score"))
            metadata = {
                **docs[index].metadata,
                "rerank_rank": rank,
            }
            if isinstance(score, (int, float)):
                metadata["rerank_score"] = float(score)

            scored_docs.append(Document(page_content=docs[index].page_content, metadata=metadata))

        return scored_docs
