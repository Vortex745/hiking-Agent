"""Shared text processing helpers for the RAG pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_core.documents import Document


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
_NOISE_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)
_SPACES_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MARKDOWN_QUOTE_RE = re.compile(r"(?m)^\s*>\s?")
_MARKDOWN_LIST_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)、]\s+)")
_MARKDOWN_CITATION_RE = re.compile(r"\s*\[(?:\d+(?:\s*[,，]\s*\d+)*)\]")
_HTML_SUP_RE = re.compile(r"<sup>.*?</sup>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[^>]+>")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_DISPLAY_METRIC_RE = re.compile(
    r"(?<![A-Za-z0-9_*])"
    r"(?:海拔\s*)?\d+(?:\.\d+)?(?:\s*[-–—~到至]\s*\d+(?:\.\d+)?)?"
    r"\s*(?:°C|℃|公里|千米|km|KM|米|m|小时|h|分钟|min|元|块|升|L|l|毫升|ml|%|毫米|mm|级|天|号线)"
    r"(?![A-Za-z0-9_*])"
)
_DISPLAY_RISK_RE = re.compile(
    r"(中暑风险|失温|高温|暴雨|雷暴|大风|台风|山洪|滑坠|迷路|落石|强降雨|"
    r"风险等级|安全警告|危险|严禁|禁止|不建议|取消|改期|撤退|避开)"
)
_DISPLAY_ROUTE_PREFIX_RE = re.compile(
    r"(?m)(^|\n)(\s*(?:\d+[.)、]\s*)?)"
    r"([\u4e00-\u9fffA-Za-z0-9（）()·]{2,24})"
    r"(\s*(?:[—–]|\s-\s)\s*)"
)
_DISPLAY_PLACE_RE = re.compile(
    r"(?<![A-Za-z0-9_*])"
    r"([\u4e00-\u9fffA-Za-z0-9·]{2,24}(?:森林公园|国家公园|保护区|景区|公园|"
    r"南门|北门|东门|西门|入口|营地|山|湖|峰|岭|谷|村|镇|站|线))"
    r"(?![A-Za-z0-9_*])"
)

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
_CJK_STOP_CHARS = set("的是了在和与及或并就都而很也还把被让对中为到从个吗呢啊")


def normalize_text(text: str) -> str:
    """Unify raw document text before denoise/chunk."""
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u3000", " ")
        .replace("\x00", "")
    )


def denoise_text(text: str) -> str:
    """Remove punctuation/special symbols while preserving words and line structure."""
    cleaned_lines: list[str] = []
    for line in normalize_text(text).split("\n"):
        line = _NOISE_RE.sub(" ", line)
        line = line.replace("_", " ")
        line = _SPACES_RE.sub(" ", line).strip()
        cleaned_lines.append(line)
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(cleaned_lines)).strip()


def clean_display_text(
    text: str,
    preserve_lines: bool = False,
    keep_list_markers: bool = False,
    keep_markdown_emphasis: bool = False,
) -> str:
    """Strip markdown syntax that looks noisy in plain chat bubbles."""
    if not text:
        return ""

    cleaned = normalize_text(str(text))
    emphasis_markers: list[str] = []
    if keep_markdown_emphasis:
        def _protect_emphasis(match: re.Match[str]) -> str:
            emphasis_markers.append(match.group(0))
            return f"\x00BOLD{len(emphasis_markers) - 1}\x00"

        cleaned = _MARKDOWN_BOLD_RE.sub(_protect_emphasis, cleaned)

    cleaned = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = _HTML_SUP_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _MARKDOWN_QUOTE_RE.sub("", cleaned)
    if not keep_list_markers:
        cleaned = _MARKDOWN_LIST_RE.sub("", cleaned)

    for pattern in (
        r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)",
        r"`([^`]+)`",
    ):
        cleaned = re.sub(pattern, r"\1", cleaned)

    cleaned = _MARKDOWN_CITATION_RE.sub("", cleaned)
    cleaned = cleaned.replace("`", "")
    for idx, marker in enumerate(emphasis_markers):
        cleaned = cleaned.replace(f"\x00BOLD{idx}\x00", marker)
    cleaned = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", cleaned)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    if preserve_lines:
        cleaned = "\n".join(line for line in lines if line)
        return cleaned.strip()

    cleaned = " ".join(line for line in lines if line)
    cleaned = _SPACES_RE.sub(" ", cleaned)
    return cleaned.strip()


def _apply_outside_bold(text: str, pattern: re.Pattern[str], repl) -> str:
    parts: list[str] = []
    pos = 0
    for match in _MARKDOWN_BOLD_RE.finditer(text):
        parts.append(pattern.sub(repl, text[pos:match.start()]))
        parts.append(match.group(0))
        pos = match.end()
    parts.append(pattern.sub(repl, text[pos:]))
    return "".join(parts)


def _bold_match(match: re.Match[str]) -> str:
    return f"**{match.group(0)}**"


def _bold_route_prefix(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}**{match.group(3)}**{match.group(4)}"


def emphasize_display_terms(text: str) -> str:
    """Add lightweight bold markers around route, metric, and safety terms for chat display."""
    if not text:
        return ""

    emphasized = str(text)
    emphasized = _apply_outside_bold(emphasized, _DISPLAY_ROUTE_PREFIX_RE, _bold_route_prefix)
    emphasized = _apply_outside_bold(emphasized, _DISPLAY_METRIC_RE, _bold_match)
    emphasized = _apply_outside_bold(emphasized, _DISPLAY_RISK_RE, _bold_match)
    emphasized = _apply_outside_bold(emphasized, _DISPLAY_PLACE_RE, _bold_match)
    emphasized = re.sub(r"\*\*([^*]+)\*\*\*\*([^*]+)\*\*", r"**\1\2**", emphasized)
    return emphasized


def extract_terms(text: str) -> list[str]:
    """Extract lightweight lexical terms without adding tokenizer dependencies."""
    lowered = text.lower()
    terms = [m.group(0) for m in _LATIN_RE.finditer(lowered) if m.group(0) not in _STOP_TERMS]

    for match in _CJK_RE.finditer(text):
        seq = match.group(0)
        max_ngram = min(4, len(seq))
        for size in range(2, max_ngram + 1):
            for idx in range(0, len(seq) - size + 1):
                term = seq[idx:idx + size]
                if term in _STOP_TERMS:
                    continue
                if all(ch in _CJK_STOP_CHARS for ch in term):
                    continue
                terms.append(term)

    if not terms:
        for match in _CJK_RE.finditer(text):
            terms.extend(ch for ch in match.group(0) if ch not in _CJK_STOP_CHARS)

    return terms


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(
        metadata.get("id")
        or metadata.get("content_hash")
        or f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{doc.page_content[:80]}"
    )


def bm25_rank(query: str, docs: list[Document], k: int = 4) -> list[tuple[Document, float]]:
    """Small BM25 implementation for local hybrid retrieval."""
    if not docs:
        return []

    query_terms = extract_terms(query)
    if not query_terms:
        return []

    tokenized_docs = [extract_terms(doc.page_content) for doc in docs]
    doc_lengths = [len(tokens) or 1 for tokens in tokenized_docs]
    avg_len = sum(doc_lengths) / len(doc_lengths)
    doc_freq = Counter(term for tokens in tokenized_docs for term in set(tokens))
    query_counts = Counter(query_terms)
    total_docs = len(docs)
    k1 = 1.5
    b = 0.75

    scored: list[tuple[Document, float]] = []
    for doc, tokens, doc_len in zip(docs, tokenized_docs, doc_lengths):
        counts = Counter(tokens)
        score = 0.0
        for term, query_weight in query_counts.items():
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += query_weight * idf * (freq * (k1 + 1) / denom)
        if score > 0:
            scored.append((doc, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


def reciprocal_rank_fusion(rank_lists: list[list[Document]], k: int = 4, rrf_k: int = 60) -> list[Document]:
    """Fuse ranked result lists with reciprocal rank fusion."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    sources_by_key: dict[str, set[str]] = {}

    for source_index, docs in enumerate(rank_lists):
        source_name = "vector" if source_index % 2 == 0 else "bm25"
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
            docs_by_key.setdefault(key, doc)
            sources_by_key.setdefault(key, set()).add(source_name)

    fused_keys = sorted(scores, key=scores.get, reverse=True)[:k]
    fused_docs: list[Document] = []
    for rank, key in enumerate(fused_keys, start=1):
        doc = docs_by_key[key]
        fused_docs.append(Document(
            page_content=doc.page_content,
            metadata={
                **(doc.metadata or {}),
                "retrieval_method": "hybrid_rrf",
                "rrf_rank": rank,
                "rrf_score": scores[key],
                "retrieval_sources": sorted(sources_by_key.get(key, set())),
            },
        ))

    return fused_docs
