"""票据 03：摄入管线 — PDF/MD → 规范化 Markdown → 层级切分 → SQLiteStore。

按 03 票据 6 项验收：
1. PDF 与 Markdown 进入同一摄入管线，行为一致
2. parent 切分遵守 H1/H2/H3 + 2000-4000 字符约束（过小合并、过大拆分、相邻再平衡）
3. child 派生自 parent（500 字符、100 重叠）
4. 全部 child 引用存在的 parent；parent 元数据保留标题路径与 source
5. 内容不在配置重叠之外丢失或重复（结构属性测试）
6. 扫描版 PDF 显式拒绝/报告，不产生空索引

设计原则：
- PDF 解析与核心切分解耦：PDF → Markdown 单独函数，延迟导入 pymupdf4llm
- Embedding 抽象：Protocol + Fake 实现，测试不依赖真实 API
- 扫描版 PDF 检测：基于文本字符数与页面数比值
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.sqlite_store import (
    ChildChunk,
    IngestResult,
    SQLiteStore,
    content_hash_of,
    parent_id_for,
)
from rag.text_processing import normalize_text

logger = logging.getLogger("ai-service.rag.ingestion")

# === 默认配置（按票据要求）===

DEFAULT_PARENT_MIN_CHARS = 2000
DEFAULT_PARENT_MAX_CHARS = 4000
DEFAULT_CHILD_CHUNK_SIZE = 500
DEFAULT_CHILD_CHUNK_OVERLAP = 100
# 扫描版 PDF 检测阈值：平均每页字符数 < 此值视为扫描版
DEFAULT_SCANNED_PDF_CHARS_PER_PAGE = 100


# === Embedding 抽象 ===


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding 提供方协议。生产用 OpenAI 实现，测试用 FakeEmbedding。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """对一批文本生成 embedding。"""
        ...


class FakeEmbedding:
    """确定性 fake embedding，测试专用。

    用文本长度 hash 生成固定维度向量，相同文本永远产生相同向量。
    不依赖外部 API，速度极快。
    """

    def __init__(self, dimensions: int = 4):
        self._dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            # 用字符 ord 累加生成确定性向量
            vec = [0.0] * self._dimensions
            for i, ch in enumerate(text):
                vec[i % self._dimensions] += (ord(ch) % 100) / 100.0
            # 归一化
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            result.append(vec)
        return result

    def embed_query(self, text: str) -> list[float]:
        """单文本 embedding（对齐 langchain OpenAIEmbeddings.embed_query 接口）。

        review No.11：baseline_performance 的 monkey-patch 包装 embed_query/embed_documents
        做 embedding 调用计数，FakeEmbedding 需实现此方法才能被计数。
        """
        return self.embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文档 embedding（对齐 langchain OpenAIEmbeddings.embed_documents 接口）。"""
        return self.embed_texts(texts)


class OpenAIEmbeddingAdapter:
    """适配 langchain_openai.OpenAIEmbeddings 到 EmbeddingProvider 协议。

    07 票据：OpenAIEmbeddings 只提供 embed_documents/embed_query，而 EmbeddingProvider
    协议要求 embed_texts。本适配器委托 embed_documents 实现 embed_texts，
    使生产 OpenAIEmbeddings 可直接注入 IngestionPipeline。
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


# === 数据结构 ===


@dataclass(frozen=True)
class ParentChunk:
    """单个 parent chunk（切分后、写入前）。"""

    content: str
    headings: list[str]  # H1/H2/H3 标题路径，如 ["徒步装备", "背包", "容量"]


@dataclass(frozen=True)
class IngestionReport:
    """摄入管线报告，对运营可见。"""

    source: str
    file_name: str
    parent_count: int
    child_count: int
    action: str  # inserted / replaced / noop（透传 SQLiteStore.IngestResult.action）
    content_hash: str
    skipped: bool = False  # True 时表示扫描版 PDF 等被拒绝
    skip_reason: str = ""


# === 异常 ===


class ScannedPDFError(ValueError):
    """扫描版 PDF 被拒绝（不产生空索引）。"""


class IngestionError(ValueError):
    """摄入管线通用错误（文件不存在、格式不支持等）。"""


# === PDF → Markdown ===


def extract_markdown_from_pdf(
    file_path: Path | str,
    scanned_chars_per_page: int = DEFAULT_SCANNED_PDF_CHARS_PER_PAGE,
) -> str:
    """PDF → Markdown（用 pymupdf4llm）。

    扫描版 PDF 检测：若平均每页字符数 < scanned_chars_per_page，抛 ScannedPDFError。
    不静默索引空内容。

    Raises:
        ScannedPDFError: 扫描版 PDF 或解析出空内容
        IngestionError: pymupdf4llm 不可用或 PDF 解析失败
    """
    try:
        import pymupdf4llm  # 延迟导入，避免硬依赖
    except ImportError as exc:
        raise IngestionError(
            "pymupdf4llm not installed; cannot parse PDF. "
            "Install with: pip install pymupdf4llm"
        ) from exc

    try:
        # 用 fitz 取页数（pymupdf4llm 不直接返回页数）
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        page_count = doc.page_count
        doc.close()
    except Exception as exc:
        raise IngestionError(f"failed to open PDF: {exc}") from exc

    if page_count == 0:
        raise ScannedPDFError(f"PDF has 0 pages: {file_path}")

    try:
        markdown_text = pymupdf4llm.to_markdown(str(file_path))
    except Exception as exc:
        raise IngestionError(f"pymupdf4llm failed: {exc}") from exc

    markdown_text = normalize_text(markdown_text)

    # 扫描版 PDF 检测
    if not markdown_text.strip():
        raise ScannedPDFError(f"PDF parsed to empty content (likely scanned): {file_path}")

    chars_per_page = len(markdown_text) / page_count
    if chars_per_page < scanned_chars_per_page:
        raise ScannedPDFError(
            f"scanned PDF detected: {len(markdown_text)} chars / {page_count} pages "
            f"= {chars_per_page:.1f} chars/page (threshold: {scanned_chars_per_page})"
        )

    return markdown_text


def extract_markdown_from_md(file_path: Path | str) -> str:
    """Markdown 文件直接读取。"""
    path = Path(file_path)
    if not path.exists():
        raise IngestionError(f"file not found: {file_path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return normalize_text(text)


def extract_markdown(
    file_path: Path | str,
    scanned_chars_per_page: int = DEFAULT_SCANNED_PDF_CHARS_PER_PAGE,
) -> str:
    """根据扩展名分派到 PDF 或 MD 解析器。

    PDF 与 Markdown 进入同一管线，行为一致（票据验收 1）。
    """
    path = Path(file_path)
    if not path.exists():
        raise IngestionError(f"file not found: {file_path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_markdown_from_pdf(path, scanned_chars_per_page=scanned_chars_per_page)
    if ext in (".md", ".markdown"):
        return extract_markdown_from_md(path)
    raise IngestionError(
        f"unsupported file type: {ext} (supported: .pdf, .md, .markdown)"
    )


# === Markdown → parent chunks ===


def _build_header_splitter() -> MarkdownHeaderTextSplitter:
    """构建 H1/H2/H3 标题切分器。"""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
        ],
        strip_headers=False,  # 保留标题，便于运营调试
    )


def _merge_headers(metadata: dict[str, str | None]) -> list[str]:
    """把 MarkdownHeaderTextSplitter 的 metadata 合并为标题路径 list。"""
    headings: list[str] = []
    for key in ("H1", "H2", "H3"):
        value = metadata.get(key)
        if value:
            headings.append(str(value).strip())
    return headings


def _merge_headings_on_coalesce(prev_headings: list[str], current_headings: list[str]) -> list[str]:
    """合并两个 parent 的标题路径。

    合并后的内容以 current 为主（prev 过小才合并），标题取 current 的 headings；
    若 current 无 headings 则回退到 prev，避免丢失标题信息。
    """
    return current_headings if current_headings else prev_headings


def _coalesce_small_parents(
    parents: list[ParentChunk],
    min_chars: int,
    max_chars: int,
) -> list[ParentChunk]:
    """合并过小的相邻 parent。

    策略：若前一个 parent 内容 < min_chars，与后一个合并（不超过 max_chars）。
    合并后 headings 取 current 的层级（合并后内容以 current 为主，避免丢失子标题）。

    边界：单个 parent 即使 < min_chars 也保留（不与下一个合并超过 max_chars）。
    """
    if not parents:
        return []

    coalesced: list[ParentChunk] = [parents[0]]
    for current in parents[1:]:
        prev = coalesced[-1]
        if len(prev.content) < min_chars and len(prev.content) + len(current.content) + 2 <= max_chars:
            # 合并到 prev（用 \n\n 连接，取 current 的 headings 避免丢失子标题）
            merged_content = f"{prev.content}\n\n{current.content}"
            merged_headings = _merge_headings_on_coalesce(prev.headings, current.headings)
            coalesced[-1] = ParentChunk(content=merged_content, headings=merged_headings)
        else:
            coalesced.append(current)
    return coalesced


def _split_oversized_parent(
    parent: ParentChunk,
    max_chars: int,
) -> list[ParentChunk]:
    """拆分过大的 parent。

    用 RecursiveCharacterTextSplitter 按 max_chars 拆分，保留原 headings。
    拆分后的子 parent 共享原 headings（标题路径不变）。
    """
    if len(parent.content) <= max_chars:
        return [parent]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,  # parent 之间不重叠（重叠在 child 层处理）
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    parts = splitter.split_text(parent.content)
    return [ParentChunk(content=p, headings=parent.headings) for p in parts if p.strip()]


def split_parent_chunks(
    markdown: str,
    min_chars: int = DEFAULT_PARENT_MIN_CHARS,
    max_chars: int = DEFAULT_PARENT_MAX_CHARS,
) -> list[ParentChunk]:
    """Markdown → parent chunks。

    步骤（票据验收 2）：
    1. MarkdownHeaderTextSplitter 按 H1/H2/H3 切分
    2. 合并过小相邻 parent（< min_chars）
    3. 拆分过大 parent（> max_chars）

    Returns:
        list[ParentChunk]，每个含 content + headings
    """
    if not isinstance(markdown, str):
        raise IngestionError(f"markdown must be string, got {type(markdown)}")
    if not markdown.strip():
        raise IngestionError("markdown is empty")

    if min_chars <= 0 or max_chars <= 0:
        raise IngestionError(f"min_chars and max_chars must be positive, got {min_chars}, {max_chars}")
    if min_chars > max_chars:
        raise IngestionError(f"min_chars({min_chars}) must be <= max_chars({max_chars})")

    # 1. H1/H2/H3 切分
    splitter = _build_header_splitter()
    try:
        docs = splitter.split_text(markdown)
    except Exception as exc:
        raise IngestionError(f"MarkdownHeaderTextSplitter failed: {exc}") from exc

    if not docs:
        # 无标题的 markdown，整体作为一个 parent
        parents = [ParentChunk(content=markdown.strip(), headings=[])]
    else:
        parents = [
            ParentChunk(content=doc.page_content, headings=_merge_headers(doc.metadata))
            for doc in docs
            if doc.page_content.strip()
        ]

    if not parents:
        raise IngestionError("no parent chunks produced after header split")

    # 2. 合并过小相邻 parent
    parents = _coalesce_small_parents(parents, min_chars, max_chars)

    # 3. 拆分过大 parent
    split_parents: list[ParentChunk] = []
    for p in parents:
        split_parents.extend(_split_oversized_parent(p, max_chars))

    # 最终校验：无常规 parent 超过 max_chars（除非单段无分隔符的极端情况）
    for p in split_parents:
        if len(p.content) > max_chars:
            # 极端情况：单段超长无分隔符，记录警告但不抛错
            logger.warning(
                f"parent exceeds max_chars ({len(p.content)} > {max_chars}); "
                f"cannot split further without breaking word boundary"
            )

    return split_parents


# === parent → child chunks ===


def split_child_chunks(
    parent_content: str,
    chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> list[str]:
    """parent content → child chunks（500字符，100重叠）。

    票据验收 3：child 派生自 parent，全部 child 引用存在的 parent。

    Returns:
        list[str]，child 内容列表（child_index 即列表索引）
    """
    if not isinstance(parent_content, str):
        raise IngestionError(f"parent_content must be string, got {type(parent_content)}")
    if not parent_content.strip():
        return []
    if chunk_size <= 0:
        raise IngestionError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise IngestionError(
            f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} (chunk_size={chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(parent_content)
    return [c for c in chunks if c.strip()]


# === 摄入管线编排 ===


class IngestionPipeline:
    """摄入管线：文件 → Markdown → parent → child + embedding → SQLiteStore。

    端到端编排（票据验收 6）：摄入完成后文档立即可被检索。
    """

    def __init__(
        self,
        store: SQLiteStore,
        embedding_provider: EmbeddingProvider,
        parent_min_chars: int = DEFAULT_PARENT_MIN_CHARS,
        parent_max_chars: int = DEFAULT_PARENT_MAX_CHARS,
        child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
        child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
        scanned_chars_per_page: int = DEFAULT_SCANNED_PDF_CHARS_PER_PAGE,
    ):
        # fail-fast 校验配置参数，避免摄入时才报错
        if parent_min_chars <= 0 or parent_max_chars <= 0:
            raise IngestionError(
                f"parent_min_chars and parent_max_chars must be positive, "
                f"got {parent_min_chars}, {parent_max_chars}"
            )
        if parent_min_chars > parent_max_chars:
            raise IngestionError(
                f"parent_min_chars({parent_min_chars}) must be <= parent_max_chars({parent_max_chars})"
            )
        if child_chunk_size <= 0:
            raise IngestionError(f"child_chunk_size must be positive, got {child_chunk_size}")
        if scanned_chars_per_page <= 0:
            raise IngestionError(
                f"scanned_chars_per_page must be positive, got {scanned_chars_per_page}"
            )

        self._store = store
        self._embedding = embedding_provider
        self._parent_min = parent_min_chars
        self._parent_max = parent_max_chars
        self._child_size = child_chunk_size
        self._child_overlap = chunk_overlap_check(child_chunk_overlap, child_chunk_size)
        self._scanned_threshold = scanned_chars_per_page

    def ingest_file(
        self,
        file_path: Path | str,
        source: str | None = None,
        status: str = "active",
    ) -> IngestionReport:
        """端到端摄入一个文件。

        Args:
            file_path: 文件路径（.pdf / .md / .markdown）
            source: 文档身份标识；None 时用 file_path 绝对路径
            status: 文档状态枚举（active/archived/feishu/cloud_docs）

        Returns:
            IngestionReport：对运营可见的摄入报告

        Raises:
            ScannedPDFError: 扫描版 PDF 被拒绝
            IngestionError: 摄入失败
            ValueError: 维度不匹配等（透传 SQLiteStore）
        """
        path = Path(file_path)
        if source is None:
            source = str(path.resolve())

        # 1. 文件 → Markdown
        try:
            markdown = extract_markdown(path, scanned_chars_per_page=self._scanned_threshold)
        except ScannedPDFError as exc:
            logger.warning(f"scanned PDF rejected: {exc}")
            return IngestionReport(
                source=source,
                file_name=path.name,
                parent_count=0,
                child_count=0,
                action="skipped",
                content_hash="",
                skipped=True,
                skip_reason=str(exc),
            )

        # 2. 委托给 ingest_content（06 票据：文件输入与标准化内存内容共享同一管线）
        return self.ingest_content(
            markdown=markdown,
            source=source,
            file_name=path.name,
            status=status,
        )

    def ingest_content(
        self,
        markdown: str,
        source: str,
        file_name: str,
        status: str = "active",
    ) -> IngestionReport:
        """端到端摄入标准化内存内容（markdown 字符串）。

        06 票据：扩展 Canonical Ingestion 接收已加载内容。
        与 ingest_file 共享同一个摄入管线（split_parent_chunks + split_child_chunks +
        upsert_document）。来源 adapter（CloudDocsLoader、FeishuLoader）只负责获取内容
        与来源元数据，不自行决定分块或直接写检索后端。

        Args:
            markdown: 标准化 Markdown 文本（来源 adapter 已加载并规范化）
            source: 文档身份标识（远程稳定标识或本地路径）
            file_name: 文件名（用于检索结果展示）
            status: 文档状态枚举（active/archived/feishu/cloud_docs）

        Returns:
            IngestionReport：对运营可见的摄入报告

        Raises:
            IngestionError: 摄入失败（空内容、无 child 产出、embedding 数量不匹配等）
            ValueError: 维度不匹配等（透传 SQLiteStore）
        """
        # 1. Markdown → parent chunks
        parents = split_parent_chunks(
            markdown,
            min_chars=self._parent_min,
            max_chars=self._parent_max,
        )

        # 2. 每个 parent → child chunks + embedding
        all_child_chunks: list[ChildChunk] = []
        child_index = 0
        for parent in parents:
            child_texts = split_child_chunks(
                parent.content,
                chunk_size=self._child_size,
                chunk_overlap=self._child_overlap,
            )
            if not child_texts:
                continue
            # 批量生成 embedding（校验返回数量，防止静默截断丢数据）
            embeddings = self._embedding.embed_texts(child_texts)
            if len(embeddings) != len(child_texts):
                raise IngestionError(
                    f"embedding count mismatch: expected {len(child_texts)}, "
                    f"got {len(embeddings)} (parent headings={parent.headings})"
                )
            for text, emb in zip(child_texts, embeddings):
                all_child_chunks.append(
                    ChildChunk(content=text, embedding=emb, child_index=child_index)
                )
                child_index += 1

        if not all_child_chunks:
            raise IngestionError(
                f"no child chunks produced from source={source} (parents={len(parents)})"
            )

        # 3. 按 02 契约：parent 表存完整文档文本（一个 source = 一个 parent 行）
        # 03 的 parent chunks（2000-4000）是中间切分层，用于派生 child（500/100）
        parent_content_full = markdown

        # 4. 写入 SQLiteStore（事务原子）
        # headings 聚合所有中间 parent 的标题路径，去重保序
        headings_all: list[str] = []
        for p in parents:
            headings_all.extend(p.headings)
        seen: set[str] = set()
        headings_unique = [h for h in headings_all if not (h in seen or seen.add(h))]

        ingest_result = self._store.upsert_document(
            source=source,
            file_name=file_name,
            parent_content=parent_content_full,
            headings=headings_unique,
            child_chunks=all_child_chunks,
            status=status,  # type: ignore[arg-type]
        )

        logger.info(
            f"ingested {file_name}: parents={len(parents)} children={ingest_result.child_count} "
            f"action={ingest_result.action}"
        )

        return IngestionReport(
            source=source,
            file_name=file_name,
            parent_count=len(parents),
            child_count=ingest_result.child_count,
            action=ingest_result.action,
            content_hash=ingest_result.content_hash,
        )


def chunk_overlap_check(overlap: int, chunk_size: int) -> int:
    """校验 chunk_overlap 参数。"""
    if overlap < 0 or overlap >= chunk_size:
        raise IngestionError(
            f"chunk_overlap must be in [0, chunk_size), got {overlap} (chunk_size={chunk_size})"
        )
    return overlap


__all__ = [
    "DEFAULT_CHILD_CHUNK_OVERLAP",
    "DEFAULT_CHILD_CHUNK_SIZE",
    "DEFAULT_PARENT_MAX_CHARS",
    "DEFAULT_PARENT_MIN_CHARS",
    "DEFAULT_SCANNED_PDF_CHARS_PER_PAGE",
    "EmbeddingProvider",
    "FakeEmbedding",
    "IngestionError",
    "IngestionPipeline",
    "IngestionReport",
    "OpenAIEmbeddingAdapter",
    "ParentChunk",
    "ScannedPDFError",
    "extract_markdown",
    "extract_markdown_from_md",
    "extract_markdown_from_pdf",
    "split_child_chunks",
    "split_parent_chunks",
]
