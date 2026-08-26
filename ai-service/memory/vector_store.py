import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import settings
from memory.redis_memory import RedisCommandStore

logger = logging.getLogger("ai-service.memory.vector_store")
REDIS_VECTOR_STORE_KEY = "agent:memory:knowledge"

FAISS_AVAILABLE = False
try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    logger.info("faiss not installed, using in-memory fallback for vector search")


class VectorStore:
    """Persistent vector store for L2 knowledge items.

    Uses FAISS (L2 index) when available for efficient similarity search,
    with a pure in-memory cosine-similarity fallback for environments
    without faiss installed.
    """

    def __init__(self, store_path: str = "./memory_store"):
        self.embeddings = OpenAIEmbeddings(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._redis_store = RedisCommandStore() if settings.redis_configured else None

        # In-memory storage (always available, used by fallback)
        self._items: list[dict] = []
        self._embeddings: list[list[float]] = []

        # FAISS index (optional, for efficient search)
        self._index = None
        self._dimension = settings.embedding_dimensions
        if FAISS_AVAILABLE:
            import faiss as faiss_lib

            self._index = faiss_lib.IndexFlatL2(self._dimension)

        self._load()

    def add(self, items: list[dict]) -> None:
        """Add knowledge items to the store.

        Args:
            items: List of knowledge item dicts (from KnowledgeExtractor).
        """
        if not items:
            return

        texts = [
            f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}"
            for item in items
        ]

        try:
            embeddings = self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.warning("Embedding failed, items not stored: %s", e)
            return

        for item, emb in zip(items, embeddings):
            self._items.append(item)
            self._embeddings.append(emb)

        if self._index is not None:
            self._index.add(np.array(embeddings, dtype=np.float32))

        self._save()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for similar knowledge items by query text.

        Args:
            query: Natural language query string.
            k: Maximum number of results to return.

        Returns:
            List of knowledge item dicts ordered by relevance.
        """
        if not self._items:
            return []

        try:
            query_embedding = self.embeddings.embed_query(query)
        except Exception as e:
            logger.warning("Query embedding failed: %s", e)
            return []

        return self._search_in_memory(query_embedding, k)

    def _search_in_memory(self, query_embedding: list[float], k: int) -> list[dict]:
        """Cosine similarity search over in-memory embeddings."""
        if not self._embeddings:
            return []

        query_vec = np.array(query_embedding).reshape(1, -1)

        if self._index is not None:
            # FAISS returns L2 distances
            distances, indices = self._index.search(query_vec, min(k, len(self._items)))
            return [self._items[idx] for idx in indices[0] if idx < len(self._items)]

        # In-memory fallback: cosine similarity
        doc_array = np.array(self._embeddings)
        similarities = np.dot(doc_array, query_vec.T).flatten() / (
            np.linalg.norm(doc_array, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self._items[int(i)] for i in top_indices]

    def _save(self) -> None:
        """Persist items and embeddings to disk."""
        data = {"items": self._items, "embeddings": self._embeddings}
        if self._redis_store is not None:
            try:
                self._redis_store.set_json(REDIS_VECTOR_STORE_KEY, data)
                return
            except Exception as e:
                logger.warning("Failed to save vector store to Redis: %s", e)

        try:
            with open(self.store_path / "knowledge.pkl", "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning("Failed to save vector store: %s", e)

    def _load(self) -> None:
        """Load persisted items from disk."""
        if self._redis_store is not None:
            try:
                data = self._redis_store.get_json(REDIS_VECTOR_STORE_KEY)
                if data:
                    self._items = data.get("items", [])
                    self._embeddings = data.get("embeddings", [])
                    if self._index is not None and self._embeddings:
                        self._index.add(np.array(self._embeddings, dtype=np.float32))
                    return
            except Exception as e:
                logger.warning("Failed to load vector store from Redis: %s", e)

        pkl_path = self.store_path / "knowledge.pkl"
        if not pkl_path.exists():
            return

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self._items = data.get("items", [])
            self._embeddings = data.get("embeddings", [])

            if self._index is not None and self._embeddings:
                self._index.add(np.array(self._embeddings, dtype=np.float32))
        except Exception as e:
            logger.warning("Failed to load vector store: %s", e)
            self._items = []
            self._embeddings = []

    def clear(self) -> None:
        """Clear all stored items and reset the index."""
        self._items = []
        self._embeddings = []
        if self._index is not None:
            self._index.reset()
        self._save()

    @property
    def count(self) -> int:
        """Number of stored knowledge items."""
        return len(self._items)
