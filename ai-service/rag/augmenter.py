import logging
import re
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from rag.text_processing import clean_display_text, emphasize_display_terms

logger = logging.getLogger("ai-service.rag.augmenter")

_RAG_SYSTEM = """你是 AI Hiking 的中文知识助手。请按下面风格回答：
- 只回答用户问题本身，先直接给结论，再补充要点。
- 中文自然、短句、少套话，不使用"作为 AI"这类开场。
- 按 humanizer-zh 风格输出：删除填充短语，避免宣传腔、三段式和金句感。
- 只根据文档能支持的信息回答，禁止补充文档外常识、编造细节或编造来源。
- 正文不要写“根据文档”“知识库中说明”“资料显示”“检索内容提到”等元叙述。
- 每个关键事实必须有文档支撑，但不要在正文里写 [1]、[2] 这类来源编号。
- 最终回答使用自然中文，不要 Markdown 标题、引用或代码块；但对回答中的关键信息（如核心结论、重要数据、关键地名、时间节点、费用、安全警告等）使用 **加粗** 标记，每段回答加粗 2-5 处最关键的信息，不要过度加粗。
- 短答案用 1-2 句；超过 2 个事实时用普通数字分点，每行一条，格式如“1. ...”。
- 文档没说清的地方明确说不确定；如果证据不足，不要硬答。
- 面向徒步场景时，给出可执行建议，避免空泛鼓励。"""

_RAG_HUMAN = """---
检索到的相关内容，每段开头是来源编号：
{context}
---

问题：{question}

回答："""

_NO_DOCS_MSG = "我没在知识库里找到和「{question}」直接相关的文档。\n\n可以换个更具体的关键词，或先上传/同步相关资料后再查。"

_META_PHRASE_RE = re.compile(
    r"(?:根据|依据|从|结合|基于)?(?:上述|相关|检索到的)?"
    r"(?:文档|知识库|资料|检索内容|上下文|材料)"
    r"(?:内容|信息)?(?:中)?(?:可以)?(?:明确)?"
    r"(?:说明|显示|提到|指出|可知|来看)?[，,:：]?"
)
_META_SENTENCE_RE = re.compile(
    r"[^。！？\n]*(?:文档|知识库|资料|检索内容|上下文)[^。！？\n]*"
    r"(?:说明|显示|提到|指出|可知|来看)[^。！？\n]*[。！？]?"
)
_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+(?:[)、]\s*|\.(?:\s+|(?=[^\d\s])))")
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+(?:[)、]\s*|\.(?:\s+|(?=[^\d\s]))))")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
_CJK_STOP_CHARS = set("的是了在和与及或并就都而很也还把被让对中为到从个吗呢啊")
_STOP_TERMS = {
    "什么",
    "是什么",
    "什么是",
    "怎么",
    "如何",
    "需要",
    "可以",
    "是否",
    "请问",
    "相关",
    "信息",
    "内容",
    "详细",
    "指南",
    "关于",
    "问题",
    "活动",
    "带我",
}
_QUERY_EXPANSIONS = {
    "准备": {"装备", "携带", "清单"},
    "安全": {"风险", "防范", "应急", "急救"},
    "路线": {"轨迹", "行程"},
    "天气": {"气候", "降雨", "温度"},
    "徒步": {"登山", "户外"},
}
_GENERIC_QUERY_TERMS = {"徒步", "户外", "指南", "相关", "内容", "信息"}


def _build_context(docs: list[Document]) -> str:
    """Format documents into context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        source = metadata.get("title") or metadata.get("file_name") or metadata.get("source", "未知来源")
        parts.append(f"[{i}] 来源：{source}\n{doc.page_content}")
    return "\n\n".join(parts)


def _clip_display_snippet(text: str, question: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    terms = sorted(_terms_from_text(question), key=len, reverse=True)
    hit = -1
    for term in terms:
        hit = text.find(term)
        if hit >= 0:
            break

    if hit < 0:
        return f"{text[:limit].rstrip()}..."

    start = max(0, hit - limit // 3)
    end = min(len(text), start + limit)
    if end == len(text):
        start = max(0, end - limit)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _build_display_context(docs: list[Document], question: str, max_docs: int = 2, limit: int = 220) -> str:
    parts = []
    for doc in docs[:max_docs]:
        metadata = doc.metadata or {}
        source = metadata.get("title") or metadata.get("file_name") or metadata.get("source", "未知来源")
        snippet = clean_display_text(doc.page_content)
        snippet = _clip_display_snippet(snippet, question, limit)
        parts.append(f"{source}：{snippet}")
    return "；".join(parts)


def _question_focus_terms(question: str) -> set[str]:
    terms = {term for term in _terms_from_text(question) if term not in _GENERIC_QUERY_TERMS and len(term) >= 2}
    if terms:
        return terms
    return {term for term in _terms_from_text(question) if term not in _GENERIC_QUERY_TERMS}


def _build_display_snippets(docs: list[Document], question: str, max_docs: int = 3, limit: int = 180) -> list[str]:
    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    seen: set[str] = set()
    focus_terms = _question_focus_terms(question)
    for doc in docs:
        snippet = clean_display_text(doc.page_content)
        snippet = _clip_display_snippet(snippet, question, limit)
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        fallback.append(snippet)
        score = sum(1 for term in focus_terms if term and term in snippet)
        if score > 0:
            candidates.append((score, snippet))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _score, snippet in candidates[:max_docs]]

    return fallback[:max_docs]


def _strip_answer_meta(text: str) -> str:
    text = _META_SENTENCE_RE.sub("", text)
    text = _META_PHRASE_RE.sub("", text)
    text = re.sub(r"^(?:所以|因此|综上)[，,:：]?", "", text.strip())
    return text.strip()


def _split_answer_sentences(text: str) -> list[str]:
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]
    return [s if re.search(r"[。！？!?]$", s) else f"{s}。" for s in sentences]


def _has_numbered_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if _NUMBERED_LINE_RE.match(line)) >= 2


def _strip_leading_list_marker(text: str) -> str:
    return _LEADING_LIST_MARKER_RE.sub("", text).strip()


def _format_answer_text(text: str) -> str:
    cleaned = clean_display_text(
        text,
        preserve_lines=True,
        keep_list_markers=True,
        keep_markdown_emphasis=True,
    )
    cleaned = _strip_answer_meta(cleaned)
    if not cleaned:
        return ""

    if _has_numbered_lines(cleaned):
        return emphasize_display_terms(cleaned)

    normalized = " ".join(cleaned.split())
    sentences = _split_answer_sentences(normalized)
    if len(sentences) >= 3 or (len(normalized) >= 90 and len(sentences) >= 2):
        points = [_strip_leading_list_marker(sentence) for sentence in sentences]
        points = [point for point in points if point]
        return emphasize_display_terms(
            "\n".join(f"{idx}. {sentence}" for idx, sentence in enumerate(points, start=1))
        )

    return emphasize_display_terms(normalized)


def _fallback_answer(question: str, docs: list[Document]) -> str:
    snippets = _build_display_snippets(docs, question)
    if not snippets:
        return ""

    focus_terms = _question_focus_terms(question)
    question_text = clean_display_text(question)
    facts: list[str] = []
    for snippet in snippets:
        sentences = _split_answer_sentences(snippet)
        start = 0
        for idx, sentence in enumerate(sentences):
            if any(term in sentence for term in focus_terms):
                start = idx
                break
        for sentence in sentences[start:start + 3]:
            sentence = sentence.strip()
            if not sentence or sentence.startswith("...") or sentence.endswith("...。"):
                continue
            if (
                question_text
                and sentence.startswith(question_text)
                and len(sentence) > len(question_text)
                and sentence[len(question_text)].isspace()
            ):
                sentence = sentence[len(question_text):].strip(" ，,:：")
            if sentence not in facts:
                facts.append(sentence)

    if not facts:
        facts = snippets

    return emphasize_display_terms("\n".join(f"{idx}. {fact}" for idx, fact in enumerate(facts, start=1)))


def _terms_from_text(text: str) -> set[str]:
    """Extract lightweight search terms without adding a tokenizer dependency."""
    lowered = text.lower()
    terms = {m.group(0) for m in _LATIN_RE.finditer(lowered) if m.group(0) not in _STOP_TERMS}

    cjk_chars_added = False
    for match in _CJK_RE.finditer(text):
        seq = match.group(0)
        for n in range(2, min(5, len(seq) + 1)):
            for idx in range(0, len(seq) - n + 1):
                term = seq[idx:idx + n]
                if term in _STOP_TERMS:
                    continue
                if all(ch in _CJK_STOP_CHARS for ch in term):
                    continue
                terms.add(term)

        if not terms:
            for ch in seq:
                if ch not in _CJK_STOP_CHARS:
                    terms.add(ch)
                    cjk_chars_added = True

    if not terms and not cjk_chars_added:
        for match in _CJK_RE.finditer(text):
            for ch in match.group(0):
                if ch not in _CJK_STOP_CHARS:
                    terms.add(ch)

    return terms


def _expand_query_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_QUERY_EXPANSIONS.get(term, set()))
    return expanded


def _document_search_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    meta_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "file_name", "doc_type", "feishu_doc_type")
    )
    return f"{meta_text}\n{doc.page_content}"


def has_relevant_evidence(question: str, docs: list[Document]) -> bool:
    """Return True when retrieved docs have direct lexical evidence for the question."""
    if not docs:
        return False

    question_terms = _expand_query_terms(_terms_from_text(question))
    if not question_terms:
        return False

    for doc in docs:
        doc_terms = _terms_from_text(_document_search_text(doc))
        if question_terms & doc_terms:
            return True

    return False


class ContextAugmenter:
    """Augments user questions with retrieved context using LangChain ChatPromptTemplate."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url if base_url is not None else settings.openai_base_url
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = model if model is not None else settings.openai_model
        self.llm: ChatOpenAI | None = None

        if self.api_key:
            try:
                self.llm = ChatOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                    temperature=0.2,
                )
            except Exception as e:
                logger.warning("LLM init failed: %s", e)

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _RAG_SYSTEM),
            ("human", _RAG_HUMAN),
        ])

    def augment(self, question: str, docs: list[Document]) -> str:
        """Combine retrieved documents and question into an augmented prompt."""
        if not has_relevant_evidence(question, docs):
            return clean_display_text(_NO_DOCS_MSG.format(question=question))

        context = _build_context(docs)

        if self.llm is None:
            return _fallback_answer(question, docs)

        try:
            messages = self._prompt.format_messages(context=context, question=question)
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return _format_answer_text(content)
        except Exception as e:
            fallback = _fallback_answer(question, docs)
            return _format_answer_text(f"生成回答时出错：{e}。{fallback}")

    async def augment_stream(self, question: str, docs: list[Document]) -> AsyncIterator[str]:
        """Stream augmented response token by token."""
        if not has_relevant_evidence(question, docs):
            yield clean_display_text(_NO_DOCS_MSG.format(question=question))
            return

        context = _build_context(docs)

        if self.llm is None:
            yield _fallback_answer(question, docs)
            return

        try:
            messages = self._prompt.format_messages(context=context, question=question)
            chunks: list[str] = []
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    chunks.append(str(chunk.content))
            answer = _format_answer_text("".join(chunks))
            if answer:
                yield answer
        except Exception as e:
            fallback = _fallback_answer(question, docs)
            yield _format_answer_text(f"生成回答时出错：{e}。{fallback}")

    def inject_knowledge(self, question: str, knowledge_items: list[dict[str, Any]]) -> str:
        """Build a prompt enriched with extracted knowledge items from the memory system."""
        if not knowledge_items:
            return f"请回答用户的问题：{question}"

        parts: list[str] = []
        for item in knowledge_items:
            kind = item.get("type", "常识")
            confidence = item.get("confidence", 0.0)
            prefix = f"  (置信度: {confidence:.0%})" if confidence > 0 else ""
            parts.append(f"[{kind}]{prefix} {item.get('content', '')}")

        knowledge_block = "\n\n".join(parts)

        return (
            f"以下是与该问题相关的已知知识（从历史交互中提取）：\n\n"
            f"{knowledge_block}\n\n"
            f"请基于上述知识（如有）回答用户的问题，"
            f"但不要被其约束——如果知识不相关可以忽略。\n\n"
            f"问题：{question}\n\n"
            f"回答："
        )
