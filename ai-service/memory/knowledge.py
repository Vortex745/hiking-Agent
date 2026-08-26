import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger("ai-service.memory.knowledge")


class KnowledgeExtractor:
    """L2 memory: extracts structured knowledge items from conversation text.

    Parses factual statements, user preferences, and entity relationships
    from dialogue into a structured JSON format for vector storage.
    Returns empty list when LLM is unavailable.
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
            temperature=0.2,
        )

    def extract(self, text: str) -> list[dict]:
        """Extract knowledge items from conversation text.

        Args:
            text: Conversation text to analyze.

        Returns:
            List of knowledge item dicts with keys: type, subject, predicate, object, confidence.
        """
        if not text or len(text.strip()) < 20:
            return []

        try:
            return self._extract_with_llm(text)
        except Exception as e:
            logger.warning("LLM knowledge extraction failed: %s", e)
            return []

    def _extract_with_llm(self, text: str) -> list[dict]:
        """Use LLM to extract structured knowledge from text."""
        prompt = (
            "从以下对话中提取有价值的结构化知识。只提取明确陈述的事实、偏好和实体关系。\n\n"
            "输出格式为 JSON 数组，每项包含:\n"
            '- type: "entity" | "fact" | "preference"\n'
            "- subject: 主体（如人名、概念、事物）\n"
            "- predicate: 关系或属性描述\n"
            "- object: 客体或属性值\n"
            "- confidence: 0.0-1.0 置信度\n\n"
            f"对话内容：\n{text}\n\nJSON 输出："
        )

        messages = [
            SystemMessage(content="你是一个知识提取助手，严格按照 JSON 数组格式输出，不要输出其他内容。"),
            HumanMessage(content=prompt),
        ]
        result = self.llm.invoke(messages)
        content = result.content.strip() if result.content else ""
        return self._parse_result(content)

    @staticmethod
    def _parse_result(content: str) -> list[dict]:
        """Parse LLM output into knowledge item list."""
        if not content:
            return []

        # Try to extract JSON array from response (handles markdown-wrapped JSON)
        json_match = re.search(r"(\[.*?\])", content, re.DOTALL)
        raw = json_match.group(1) if json_match else content

        try:
            items = json.loads(raw)
            if isinstance(items, list):
                # Validate and normalize
                validated = []
                for item in items:
                    if isinstance(item, dict) and "subject" in item and "predicate" in item:
                        validated.append(item)
                return validated
        except json.JSONDecodeError:
            logger.debug("Failed to parse LLM knowledge output as JSON")

        return []
