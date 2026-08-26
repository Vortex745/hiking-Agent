import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger("ai-service.memory.compressor")


class SessionCompressor:
    """L1 memory: compresses conversation history into concise Chinese session summaries.

    Uses LLM for intelligent compression with a graceful fallback to
    simple truncation-based extraction when the LLM is unavailable.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model or settings.openai_model
        self.llm = ChatOpenAI(
            base_url=base_url or settings.openai_base_url,
            api_key=api_key or settings.openai_api_key,
            model=self.model,
            temperature=0.3,
        )

    def compress(self, history: list[dict]) -> str:
        """Compress a list of conversation messages into a succinct Chinese summary.

        Args:
            history: List of message dicts with 'role' and 'content' keys.

        Returns:
            Compressed summary string, or empty string if history is empty.
        """
        if not history:
            return ""

        dialogue = [m for m in history if m.get("role") in ("user", "assistant")]
        if not dialogue:
            return ""

        try:
            return self._compress_with_llm(dialogue)
        except Exception as e:
            logger.warning("LLM compression failed, using fallback: %s", e)
            return self._compress_fallback(dialogue)

    def _compress_with_llm(self, dialogue: list[dict]) -> str:
        """Use LLM to generate a concise Chinese summary."""
        conversation_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in dialogue
        )

        prompt = (
            "请将以下对话压缩为一段简洁的摘要（中文，不超过200字），"
            "保留关键信息：用户的核心需求、助手提供的关键回答或操作、以及任何重要的上下文。\n\n"
            f"对话内容：\n{conversation_text}"
        )

        messages = [
            SystemMessage(content="你是一个专业的对话摘要生成助手。"),
            HumanMessage(content=prompt),
        ]
        result = self.llm.invoke(messages)
        content = result.content.strip() if result.content else ""
        return content if content else self._compress_fallback(dialogue)

    @staticmethod
    def _compress_fallback(dialogue: list[dict]) -> str:
        """Fallback: simple truncation-based summary when LLM is unavailable."""
        if not dialogue:
            return ""

        first_user = next((m["content"] for m in dialogue if m["role"] == "user"), "")
        last_exchange = dialogue[-3:]

        summary_parts = []
        if first_user:
            summary_parts.append(f"初始提问: {first_user[:100]}")

        if len(dialogue) > 2:
            exchange_text = "; ".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:80]}"
                for m in last_exchange
            )
            summary_parts.append(f"最近对话: {exchange_text}")

        summary_parts.append(f"总消息数: {len(dialogue)}")
        return " | ".join(summary_parts)
