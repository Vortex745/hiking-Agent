import logging
from typing import Optional

from memory.compressor import SessionCompressor
from memory.committer import MemoryCommitter
from memory.knowledge import KnowledgeExtractor
from memory.vector_store import VectorStore

logger = logging.getLogger("ai-service.memory")


class MemoryConfig:
    """Configuration for the two-level memory system."""

    compressor_model: str = "gpt-4o-mini"
    extractor_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_api_key: str = ""
    vector_store_path: str = "./memory_store"
    top_k: int = 5

    def __init__(
        self,
        compressor_model: Optional[str] = None,
        extractor_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        if compressor_model is not None:
            self.compressor_model = compressor_model
        if extractor_model is not None:
            self.extractor_model = extractor_model
        if llm_base_url is not None:
            self.llm_base_url = llm_base_url
        if llm_api_key is not None:
            self.llm_api_key = llm_api_key
        if vector_store_path is not None:
            self.vector_store_path = vector_store_path
        if top_k is not None:
            self.top_k = top_k


class MemoryManager:
    """Orchestrates L1 session compression and L2 knowledge extraction.

    Usage::

        memory = MemoryManager()
        context = memory.process_interaction(history, user_query)
        # context["session_context"] -> compressed L1 summary
        # context["knowledge_context"] -> formatted L2 knowledge string
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.compressor = SessionCompressor(
            model=self.config.compressor_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )
        self.extractor = KnowledgeExtractor(
            model=self.config.extractor_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )
        self.vector_store = VectorStore(store_path=self.config.vector_store_path)
        self.committer = MemoryCommitter()

    def get_session_context(self, history: list[dict]) -> str:
        """Generate compressed session context from conversation history (L1)."""
        return self.compressor.compress(history)

    def get_relevant_knowledge(self, query: str) -> list[dict]:
        """Search for knowledge relevant to the current query (L2)."""
        return self.vector_store.search(query, k=self.config.top_k)

    def update_knowledge(self, history: list[dict]) -> int:
        """Extract and store knowledge from the latest exchange (L2).

        Args:
            history: Full conversation history.

        Returns:
            Number of new knowledge items stored.
        """
        if not history:
            return 0

        # Extract the latest 2 user/assistant exchanges
        recent = []
        for msg in reversed(history):
            if msg.get("role") in ("user", "assistant"):
                recent.append(msg.get("content", ""))
            if len(recent) >= 4:
                break

        if not recent:
            return 0

        conversation_text = "\n".join(reversed(recent))
        items = self.extractor.extract(conversation_text)

        if items:
            self.vector_store.add(items)

        return len(items)

    def format_knowledge_context(self, query: str) -> str:
        """Format relevant knowledge as a context string for system prompt injection."""
        knowledge = self.get_relevant_knowledge(query)
        if not knowledge:
            return ""

        lines = []
        for item in knowledge:
            subj = item.get("subject", "")
            pred = item.get("predicate", "")
            obj = item.get("object", "")
            if subj and pred:
                lines.append(f"- {subj} 的 {pred}: {obj}")
            elif subj:
                lines.append(f"- {subj}: {obj}")

        if not lines:
            return ""

        return "## 已知的用户信息\n" + "\n".join(lines)

    def process_interaction(self, history: list[dict], query: str) -> dict:
        """Full pipeline: update L2 knowledge and return both L1 and L2 context.

        Args:
            history: Full conversation history.
            query: Current user query.

        Returns:
            Dict with ``session_context`` (str) and ``knowledge_context`` (str).
        """
        self.update_knowledge(history)
        return {
            "session_context": self.get_session_context(history),
            "knowledge_context": self.format_knowledge_context(query),
        }

    def build_runtime_context(self, history: list[dict], query: str) -> dict:
        """Read-only context assembly before agent execution."""
        return {
            "session_context": self.get_session_context(history),
            "knowledge_context": self.format_knowledge_context(query),
        }

    def commit_interaction(
        self,
        history: list[dict],
        query: str,
        final_response: str = "",
        task_state: dict | None = None,
    ) -> int:
        """Commit stable profile/trip memories after final response generation."""
        candidates = self.committer.extract_candidates(
            history=history,
            query=query,
            final_response=final_response,
            task_state=task_state,
        )
        if candidates:
            self.vector_store.add(candidates)
        return len(candidates)
