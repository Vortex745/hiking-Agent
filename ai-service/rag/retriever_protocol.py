"""11 票据：窄检索存储 Adapter 契约。

定义面向业务行为的检索存储 adapter 边界，覆盖父子写入所需的最小存储操作、
dense/sparse 检索、文档列举、计数、过滤、健康和关闭语义。

设计原则：
- 调用方通过依赖注入消费契约，不接触连接对象、后端选择或 fallback 私有状态
- 现有 _SQLiteStoreClient / _PGVectorClient 将在 12 票据中迁移到本契约
- 本模块只定义契约，不实现具体后端逻辑（依赖方向：存储实现不依赖 service/runtime/API）

验收标准（.scratch/rag-architecture-debt-repayment/issues/11-narrow-retrieval-storage-contract.md）：
1. 契约明确搜索、列举、计数、过滤、写入结果和错误语义，避免暴露特定后端类型
2. dense、sparse 与融合结果具有稳定的文档身份、父片段回溯和排序约定
3. 后端不可用、数据为空、维度不匹配和事务失败具有可区分结果
4. runtime 通过构造/工厂注入 adapter，不在业务方法内读取后端环境或创建裸连接
5. adapter 连接生命周期由明确 owner 管理，调用方无需访问内部连接
6. 契约替身可以驱动查询与摄入测试，而无需构造真实后端私有状态
7. 依赖方向保持存储实现不依赖 service、runtime 或 API 类型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# 数据类型：dense / sparse / 融合结果（验收 2）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """检索结果条目（dense / sparse / 融合共用）。

    字段对齐 hybrid_retrieval.search_child_chunks 返回契约的 9 个字段，
    但提供稳定的 dataclass 接口（不依赖 dict 结构）。

    文档身份与父片段回溯（验收 2）：
    - child_id：子片段唯一标识（chunk 级文档身份）
    - parent_id：父片段标识（回溯到完整父文档）
    - source：来源标识（路径/远程标识，用于来源过滤）
    - status：文档状态枚举（active/upload/cloud_docs/feishu/archived）
    """

    content: str
    child_id: str
    parent_id: str
    source: str
    file_name: str
    status: str
    score_dense: float
    score_sparse: float
    score_rrf: float
    # 可选元数据：headings 用于上下文渲染，id 兼容 langchain Document.metadata["id"]
    headings: tuple[str, ...] = field(default_factory=tuple)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_metadata(self) -> dict[str, Any]:
        """转换为 LangChain Document.metadata 格式（兼容现有调用方）。"""
        meta = {
            "id": self.child_id,
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "file_name": self.file_name,
            "title": self.file_name,  # 兼容下游 title 字段读取
            "status": self.status,
            "headings": list(self.headings),
            "score_dense": self.score_dense,
            "score_sparse": self.score_sparse,
            "score_rrf": self.score_rrf,
        }
        meta.update(self.extra_metadata)
        return meta


@dataclass(frozen=True)
class DocumentSummary:
    """文档摘要（list_documents 返回）。

    每个摘要对应一个父文档（parent），不含 child chunk 内容。
    """

    source: str
    file_name: str
    title: str
    status: str
    parent_id: str
    headings: tuple[str, ...] = field(default_factory=tuple)
    content: str = ""  # parent_content（可选，列举时不一定返回）

    def to_langchain_metadata(self) -> dict[str, Any]:
        """转换为 LangChain Document.metadata 格式。"""
        return {
            "id": self.parent_id,
            "source": self.source,
            "file_name": self.file_name,
            "title": self.title,
            "status": self.status,
            "headings": list(self.headings),
        }


@dataclass(frozen=True)
class WriteResult:
    """写入结果（add_documents 返回）。

    验收 1：写入结果和错误语义明确。
    """

    requested_count: int
    indexed_count: int
    storage_mode: str  # 'sqlite' | 'pgvector' | 'memory'
    durable: bool
    action: str = "inserted"  # 'inserted' | 'replaced' | 'noop' | 'failed'
    fallback_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class HealthStatus:
    """健康状态（health 返回）。

    验收 3：后端不可用具有可区分结果（status='unavailable' + error 字段）。
    """

    status: str  # 'ok' | 'degraded' | 'unavailable'
    storage_mode: str  # 'sqlite' | 'pgvector' | 'memory' | 'unknown'
    document_count: int
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 错误类型（验收 3）
# ---------------------------------------------------------------------------


class RetrieverAdapterError(Exception):
    """检索 adapter 错误基类。

    所有 adapter 错误都继承自此基类，调用方可以捕获基类处理所有后端错误。
    """


class BackendUnavailableError(RetrieverAdapterError):
    """后端不可用（连接失败、文件损坏、权限不足等）。

    与 EmptyResultError 区分：本错误表示后端本身无法访问，
    EmptyResultError 表示后端可访问但无数据。
    """

    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"Backend '{backend}' unavailable: {reason}")


class EmptyResultError(RetrieverAdapterError):
    """后端可访问但无任何文档（数据为空）。

    验收 3：与 BackendUnavailableError 区分——本错误表示后端可访问，
    但数据库中无任何文档（has_documents 返回 False 的显式版本）。

    使用场景：
    - 调用方在 search 前 has_documents 返回 False，仍调用 search 时可抛出
    - list_documents 在后端无文档时可选抛出（常规返回空列表也合法）
    - search 返回空列表（查询无匹配）不抛此错误——空列表是合法查询结果

    与 BackendUnavailableError 的区别：
    - BackendUnavailableError: 后端不可访问（连接失败/文件损坏）
    - EmptyResultError: 后端可访问但无数据（数据库为空）
    """

    def __init__(self, reason: str = "no documents in store"):
        self.reason = reason
        super().__init__(reason)


class DimensionMismatchError(RetrieverAdapterError):
    """向量维度不匹配（query_embedding 维度与库中存储维度不一致）。

    sqlite-vec 在维度不匹配时会抛 sqlite3.OperationalError，
    pgvector 在 vector_dims != query_dimensions 时返回空结果。
    本错误统一这两种情况的错误信号。
    """

    def __init__(self, expected: int, actual: int, backend: str = ""):
        self.expected = expected
        self.actual = actual
        self.backend = backend
        msg = f"Dimension mismatch: expected {expected}, got {actual}"
        if backend:
            msg += f" (backend: {backend})"
        super().__init__(msg)


class TransactionFailedError(RetrieverAdapterError):
    """事务失败（写入时事务回滚，如 upsert_document 失败）。

    SQLiteStore.upsert_document 在事务失败时会回滚并抛 RuntimeError，
    本错误包装底层错误信号，提供稳定的错误语义。
    """

    def __init__(self, reason: str, source: str = ""):
        self.reason = reason
        self.source = source
        msg = f"Transaction failed: {reason}"
        if source:
            msg += f" (source: {source})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# RetrieverProtocol（验收 1/4/5）
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrieverProtocol(Protocol):
    """窄检索存储 Adapter 契约。

    调用方通过依赖注入消费本契约，不接触：
    - 连接对象（_conn / _store / _read_conn）
    - 后端选择（_storage_backend / _fallback_mode）
    - fallback 私有状态

    验收 1：契约明确搜索、列举、计数、过滤、写入结果和错误语义
    验收 4：runtime 通过构造/工厂注入 adapter，不在业务方法内读取后端环境
    验收 5：adapter 连接生命周期由明确 owner 管理，调用方无需访问内部连接
    """

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        query_text: str = "",
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """混合检索：dense + sparse + RRF 融合，返回 top-K 结果。

        Args:
            query_embedding: dense 查询向量
            k: 返回 top-K 条
            query_text: sparse 查询文本（空串跳过 sparse，仅 dense）
            source_filter: 仅返回该 source 的结果（None = 不过滤）
            status_filter: 仅返回该 status 的结果（None = 不过滤）

        Returns:
            按 score_rrf 降序排列的 SearchResult 列表，长度 <= k

        Raises:
            BackendUnavailableError: 后端不可用
            DimensionMismatchError: 向量维度不匹配
        """
        ...

    def list_documents(
        self,
        *,
        status_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: int = 5000,
    ) -> list[DocumentSummary]:
        """列举文档（按 status / source 过滤）。

        验收 1：契约明确过滤语义，list_documents 支持 status + source 双维度过滤。

        Args:
            status_filter: 仅返回该 status 的文档（None = 不过滤）
            source_filter: 仅返回该 source 的文档（None = 不过滤）
            limit: 返回上限

        Returns:
            DocumentSummary 列表，每个摘要对应一个父文档；
            后端无文档时返回空列表（不抛 EmptyResultError，空列表是合法结果）

        Raises:
            BackendUnavailableError: 后端不可用
        """
        ...

    def has_documents(self, *, status_filter: Optional[str] = None) -> bool:
        """检查是否有文档（按 status 过滤）。

        Args:
            status_filter: 检查该 status 是否有文档（None = 检查任意 status）

        Returns:
            True 如果有匹配文档，False 否则

        Raises:
            BackendUnavailableError: 后端不可用
        """
        ...

    def document_count(self) -> int:
        """返回文档计数（chunk 数，对齐 pgvector 语义）。

        Returns:
            文档计数

        Raises:
            BackendUnavailableError: 后端不可用
        """
        ...

    def health(self) -> HealthStatus:
        """检查后端健康状态。

        Returns:
            HealthStatus，status='ok' 表示后端可用，'degraded' 表示部分可用，
            'unavailable' 表示后端不可用

        Note:
            本方法不应抛异常，所有错误通过 HealthStatus.status='unavailable' + error 返回。
        """
        ...

    def close(self) -> None:
        """关闭 adapter 释放连接资源。

        调用方不接触内部连接，通过本方法管理生命周期。
        关闭后调用其他方法行为未定义（应抛 BackendUnavailableError）。
        """
        ...


# ---------------------------------------------------------------------------
# 写入契约（add_documents）— 11 票据不强制实现，12 票据迁移
# ---------------------------------------------------------------------------


@runtime_checkable
class WritableRetrieverProtocol(RetrieverProtocol, Protocol):
    """可写入的检索 adapter 契约（继承 RetrieverProtocol）。

    11 票据：本契约定义扁平写入语义，但现有后端适配器在 12 票据中迁移。

    职责边界（spec 行 3 "覆盖父子写入所需的最小存储操作"）：
    - **父子写入**（带层级切分）：由 IngestionPipeline.ingest_file / ingest_content
      负责（06 票据交付），执行 Markdown 解析 + H1/H2/H3 切分 + Embedding 生成 +
      事务写入。生产路径必须走 canonical ingestion（10 票据守护）。
    - **扁平写入**（不带层级切分）：由本契约的 add_documents 方法负责，
      每个 Document 作为一个 parent=一个 child（伪层级）。仅用于：
      (1) 迁移脚本批量重建索引（_SQLiteStoreClient.add_documents 直接调用）
      (2) 测试替身构造确定性状态

    生产代码不应调用本契约的 add_documents，应使用 IngestionPipeline。
    """

    def add_documents(
        self,
        docs: list,  # list[Document]
        embeddings: list[list[float]],
        *,
        status: Optional[str] = None,
    ) -> WriteResult:
        """扁平写入文档（每个 Document 作为一个 parent=一个 child）。

        11 票据：本方法为 legacy/测试专用，生产代码应使用 IngestionPipeline。
        12 票据：现有 _SQLiteStoreClient.add_documents / _PGVectorClient.add_documents
        将迁移到返回 WriteResult（当前返回 None）。

        Args:
            docs: LangChain Document 列表
            embeddings: 对应的 embedding 列表
            status: 文档状态（None = 不设置）

        Returns:
            WriteResult，包含 requested_count / indexed_count / storage_mode 等

        Raises:
            BackendUnavailableError: 后端不可用
            TransactionFailedError: 事务失败
        """
        ...


# ---------------------------------------------------------------------------
# Factory 接口（验收 4）
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrieverAdapterFactory(Protocol):
    """检索 adapter 工厂契约。

    验收 4：runtime 通过构造/工厂注入 adapter，不在业务方法内读取后端环境或创建裸连接。

    调用方注入 Factory 实例，通过 create() 获取 adapter，不接触后端选择逻辑。
    """

    def create(self) -> RetrieverProtocol:
        """创建检索 adapter 实例。

        Returns:
            RetrieverProtocol 实例

        Raises:
            BackendUnavailableError: 后端不可用（调用方应处理降级）
        """
        ...


