import json
import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger("ai-service.rag.rewriter")

_REWRITE_SYSTEM = """你是一个查询改写专家。给定用户的原始问题，生成2-3个语义不同但相关的检索查询。
这些查询用于向量检索，目标是覆盖用户可能想表达的不同角度。

规则：
- 第一个查询保持原问题不变
- 其他查询从不同角度改写（具体化、同义替换、上下位词扩展）
- 每个查询独立可用，不依赖其他查询
- 只输出JSON数组，不要其他文字
- 示例：用户问"徒步安全" → ["徒步安全", "户外徒步注意事项和风险防范", "登山安全指南和应急处理"]"""

_HUMANIZER_SYSTEM = """你是中文问题改写编辑。把用户问题改成适合 RAG 生成回答的自然中文。
风格要求：
- 保留原意，不补充事实。
- 删除填充词、口号化表达和 AI 味套话。
- 句子短一点，像真人提问。
- 不要三段式，不要解释你的改写过程。
- 只输出改写后的问题。"""

_CONTEXTUALIZE_SYSTEM = """你是 RAG 检索问题改写器。根据对话历史，把用户当前问题改写成一个独立、清晰的检索问题。

规则：
- 只补全历史中已经出现的指代对象，不添加新事实。
- 如果当前问题已经独立，保持原问题。
- 删除客套话，保留徒步领域关键词。
- 只输出改写后的问题，不要解释。"""

_FALLBACK_TEMPLATES = [
    "{question}的具体信息和注意事项",
    "关于{question}的详细指南",
]


class QueryRewriter:
    """Rewrites user questions into multiple queries for better retrieval.

    Uses LLM semantic rewrite when available, falls back to template-based.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url if base_url is not None else settings.openai_base_url
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model if model is not None else settings.openai_model
        self.llm: Optional[ChatOpenAI] = None
        if self.api_key:
            try:
                self.llm = ChatOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning("LLM init failed, fallback to template: %s", e)
                self.llm = None

    def rewrite(self, question: str) -> list[str]:
        """Generate multiple query variations."""
        if self.llm is not None:
            try:
                return self._semantic_rewrite(question)
            except Exception as e:
                logger.warning("Semantic rewrite failed, fallback: %s", e)

        return self._template_rewrite(question)

    def contextualize(self, question: str, history: list[dict] | None = None) -> str:
        """Turn follow-up questions into standalone retrieval questions."""
        normalized = " ".join(question.strip().split())
        if not normalized or not history:
            return question

        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                history_text = self._format_history(history)
                response = self.llm.invoke([
                    SystemMessage(content=_CONTEXTUALIZE_SYSTEM),
                    HumanMessage(content=f"对话历史：\n{history_text}\n\n当前问题：{normalized}"),
                ])
                content = response.content if hasattr(response, "content") else str(response)
                content = str(content).strip().strip("\"'")
                if content:
                    return content
            except Exception as e:
                logger.warning("Contextual query rewrite failed, fallback: %s", e)

        return self._fallback_contextualize(normalized, history)

    def _semantic_rewrite(self, question: str) -> list[str]:
        """LLM-powered semantic rewrite."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=f"原始问题：{question}\n\n输出JSON数组："),
        ]
        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            queries = json.loads(content)
            if isinstance(queries, list) and len(queries) > 0:
                # Ensure original question is first
                if queries[0] != question:
                    queries.insert(0, question)
                return [str(q) for q in queries if str(q).strip()]
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse LLM rewrite output: %s", content[:200])

        return self._template_rewrite(question)

    def _template_rewrite(self, question: str) -> list[str]:
        """Fallback template-based rewrite."""
        queries = [question]
        for template in _FALLBACK_TEMPLATES:
            queries.append(template.format(question=question))
        return queries

    def _format_history(self, history: list[dict], limit: int = 6) -> str:
        lines: list[str] = []
        for item in history[-limit:]:
            role = str(item.get("role") or "").strip()
            content = " ".join(str(item.get("content") or "").split())
            if role and content:
                lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines)

    def _fallback_contextualize(self, question: str, history: list[dict]) -> str:
        if not any(marker in question for marker in ("这些", "这个", "这类", "它", "他们", "它们", "那", "上述", "上面", "刚才", "呢")):
            return question

        previous_user = ""
        for item in reversed(history):
            if item.get("role") == "user":
                previous_user = " ".join(str(item.get("content") or "").split())
                break

        if not previous_user:
            return question
        if previous_user in question:
            return question
        return f"针对{previous_user}，{question}"

    def humanize_for_answer(self, question: str) -> str:
        """Rewrite the generation question with humanizer-zh style guidance."""
        normalized = " ".join(question.strip().split())
        if not normalized:
            return question

        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                response = self.llm.invoke([
                    SystemMessage(content=_HUMANIZER_SYSTEM),
                    HumanMessage(content=f"原问题：{normalized}"),
                ])
                content = response.content if hasattr(response, "content") else str(response)
                content = str(content).strip().strip("\"'")
                if content:
                    return content
            except Exception as e:
                logger.warning("Humanized query rewrite failed, fallback: %s", e)

        return self._fallback_humanize(normalized)

    def _fallback_humanize(self, question: str) -> str:
        replacements = {
            "请问": "",
            "麻烦": "",
            "帮我": "",
            "详细": "",
            "深入": "",
            "全面": "",
            "一下": "",
        }
        text = question
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = " ".join(text.split()).strip()
        return text or question
