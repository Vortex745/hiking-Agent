import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from config import settings
from rag.backend_resolver import resolve_backend
from rag.retriever_protocol import (
    BackendUnavailableError,
    DocumentSummary,
    HealthStatus,
    SearchResult,
    TransactionFailedError,
    WritableRetrieverProtocol,
    WriteResult,
)
from rag.text_processing import bm25_rank, reciprocal_rank_fusion

logger = logging.getLogger("ai-service.rag")

_SHARED_FALLBACK_DOCS: list[Document] = []


@dataclass(frozen=True)
class AddDocumentsResult:
    requested_count: int
    indexed_count: int
    storage_mode: str
    durable: bool
    fallback_used: bool = False
    error: str | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _PGVectorClient:
    """Lightweight pgvector client using psycopg (sync), no async needed."""

    def __init__(self, db_url: str, connect_timeout: int = 2):
        self._db_url = db_url
        self._connect_timeout = connect_timeout
        self._conn = None
        self._ensure_table()

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg
            self._conn = psycopg.connect(self._db_url, connect_timeout=self._connect_timeout)
            self._conn.autocommit = True
        return self._conn

    def _ensure_table(self):
        conn = self._get_conn()
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector,
                metadata JSONB DEFAULT '{}'
            )
        """)
        logger.info("pgvector table ensured (dynamic dimensions)")

    def add_documents(
        self,
        docs: list[Document],
        embeddings: list[list[float]],
        *,
        status: Optional[str] = None,
    ) -> WriteResult:
        """扁平写入文档（12 票据：实现 WritableRetrieverProtocol）。

        每个 Document 作为一行写入 rag_documents 表（id 来自 metadata.id 或 uuid）。
        若 status 非 None，覆盖每个 doc.metadata["status"]。
        """
        conn = self._get_conn()
        indexed = 0
        for doc, emb in zip(docs, embeddings):
            doc_id = doc.metadata.get("id") or str(uuid.uuid4())
            meta = dict(doc.metadata or {})
            # Remove embedding from metadata to keep it clean
            meta.pop("embedding", None)
            if status is not None:
                meta["status"] = status
            try:
                conn.execute(
                    """
                    INSERT INTO rag_documents (id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (doc_id, doc.page_content, json.dumps(emb), json.dumps(meta, ensure_ascii=False)),
                )
                indexed += 1
            except Exception as exc:
                logger.warning("pgvector add_documents row failed (id=%s): %s", doc_id, exc)
        return WriteResult(
            requested_count=len(docs),
            indexed_count=indexed,
            storage_mode="pgvector",
            durable=True,
            action="inserted" if indexed == len(docs) else "failed",
            error=None if indexed == len(docs) else f"only {indexed}/{len(docs)} rows inserted",
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        query_text: str = "",
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """混合检索（12 票据：实现 RetrieverProtocol）。

        pgvector 后端仅支持 dense 检索（无 FTS5 sparse），query_text 参数被忽略，
        score_sparse=0.0，score_rrf=score_dense（单榜直接透传）。
        """
        rows = self.similarity_search(query_embedding, k=k)
        results: list[SearchResult] = []
        for row_id, content, metadata, similarity in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = metadata or {}
            source = metadata.get("source", "")
            file_name = metadata.get("file_name") or metadata.get("title") or source
            doc_status = metadata.get("status", "")
            if source_filter is not None and source != source_filter:
                continue
            if status_filter is not None and doc_status != status_filter:
                continue
            results.append(SearchResult(
                content=content,
                child_id=row_id,
                parent_id=row_id,  # pgvector 无父子层级，parent_id=child_id
                source=source,
                file_name=file_name,
                status=doc_status,
                score_dense=float(similarity),
                score_sparse=0.0,
                score_rrf=float(similarity),
            ))
        return results[:k]

    def similarity_search(self, query_embedding: list[float], k: int = 4) -> list[tuple[str, str, dict, float]]:
        conn = self._get_conn()
        query_dimensions = len(query_embedding)
        rows = conn.execute(
            """
            SELECT id, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag_documents
            WHERE embedding IS NOT NULL
              AND vector_dims(embedding) = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (json.dumps(query_embedding), query_dimensions, json.dumps(query_embedding), k),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def list_documents(
        self,
        *,
        status_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: int = 5000,
    ) -> list[DocumentSummary]:
        """列举文档（12 票据：实现 RetrieverProtocol，返回 DocumentSummary）。

        pgvector 无父子层级，每个 row 即一个 parent，parent_id=row_id。
        """
        conn = self._get_conn()
        sql = "SELECT id, content, metadata FROM rag_documents"
        conditions = []
        params: list[Any] = []
        if status_filter is not None:
            conditions.append("metadata->>'status' = %s")
            params.append(status_filter)
        if source_filter is not None:
            conditions.append("metadata->>'source' = %s")
            params.append(source_filter)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " LIMIT %s"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

        summaries: list[DocumentSummary] = []
        for row_id, content, metadata in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = metadata or {}
            summaries.append(DocumentSummary(
                source=metadata.get("source", ""),
                file_name=metadata.get("file_name") or metadata.get("title") or "",
                title=metadata.get("title") or metadata.get("file_name") or "",
                status=metadata.get("status", ""),
                parent_id=row_id,
                headings=tuple(metadata.get("headings", [])),
                content=content or "",
            ))
        return summaries

    def has_documents(self, *, status_filter: Optional[str] = None) -> bool:
        conn = self._get_conn()
        if status_filter:
            row = conn.execute(
                """
                SELECT 1
                FROM rag_documents
                WHERE embedding IS NOT NULL
                  AND metadata->>'status' = %s
                LIMIT 1
                """,
                (status_filter,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT 1
                FROM rag_documents
                WHERE embedding IS NOT NULL
                LIMIT 1
                """
            ).fetchone()
        return row is not None

    def document_count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM rag_documents WHERE embedding IS NOT NULL").fetchone()
        return row[0] if row else 0

    def health(self) -> HealthStatus:
        """检查后端健康状态（12 票据：实现 RetrieverProtocol）。"""
        try:
            count = self.document_count()
            return HealthStatus(
                status="ok",
                storage_mode="pgvector",
                document_count=count,
            )
        except Exception as exc:
            return HealthStatus(
                status="unavailable",
                storage_mode="pgvector",
                document_count=0,
                error=str(exc),
            )

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


class _SQLiteStoreClient:
    """10 票据：SQLite 后端适配器，接口与 _PGVectorClient 对齐。

    把 VectorStoreRetriever API 适配到 SQLiteStore + hybrid_retrieval：
    - add_documents：每个 Document 作为一个独立 source 写入（一个 parent = 一个 child）
      注：这是运行时兼容路径，真实迁移用 IngestionPipeline 重新切分保留层级
    - similarity_search：dense-only 检索（query_text="" 跳过 sparse）
    - list_documents / has_documents / document_count：直接查 parent 表
    """

    def __init__(self, db_path: str, embedding_dimensions: int = 1536):
        # 延迟导入避免循环依赖（sqlite_store import 链较长）
        from rag.sqlite_store import SQLiteStore

        self._db_path = db_path
        self._embedding_dimensions = embedding_dimensions
        self._store = SQLiteStore(db_path, embedding_dimensions=embedding_dimensions)
        # review No.10：用公开 read_conn property 替代私有 _write_conn
        self._read_conn = self._store.read_conn
        logger.info(
            "SQLiteStoreClient connected: db_path=%s dimensions=%d parents=%d",
            db_path, embedding_dimensions, self._store.count_parents(),
        )

    @property
    def store(self):
        """暴露底层 SQLiteStore，供迁移脚本/调试使用。"""
        return self._store

    @property
    def read_conn(self):
        """公开读连接（review No.10：透出 SQLiteStore.read_conn）。

        调用方（_sqlite_hybrid_search / baseline_performance）通过
        _sqlite_client.read_conn 访问，而非 _sqlite_client._read_conn。
        """
        return self._read_conn

    def add_documents(
        self,
        docs: list[Document],
        embeddings: list[list[float]],
        *,
        status: Optional[str] = None,
    ) -> WriteResult:
        """每个 Document 作为一个独立 source 写入（运行时兼容路径）。

        10 票据：该方法为 legacy/测试专用，生产代码不应直接调用。
        - 生产路径：通过 IngestionPipeline.ingest_file / ingest_content 走 canonical ingestion
          （父子层级切分 + 事务写入），VectorStoreRetriever.add_documents 在 SQLite 后端
          已被守护拦截（抛 RuntimeError）。
        - 保留场景：迁移脚本（批量重建索引）、测试替身（构造确定性 SQLite 后端状态）。
        - 行为：每个 Document 被当作一个 parent=一个 child（伪层级），不执行父子切分。

        12 票据：实现 WritableRetrieverProtocol，返回 WriteResult。
        若 status 非 None，覆盖每个 doc.metadata["status"]。

        review No.7：归一化非标准 status，避免 pgvector 旧库的 draft/deleted 等
        非标准 status 导致 upsert_document 抛 ValueError 使整批回退。
        非标准 status 映射到 "archived"（最接近"非活跃"语义）。
        """
        from rag.sqlite_store import ChildChunk, _VALID_STATUSES

        indexed = 0
        for doc, emb in zip(docs, embeddings):
            # source 优先取 metadata.source，否则用 metadata.id，最后兜底 uuid
            source = doc.metadata.get("source") or doc.metadata.get("id") or str(uuid.uuid4())
            file_name = (
                doc.metadata.get("file_name")
                or doc.metadata.get("title")
                or source
            )
            doc_status = status if status is not None else (doc.metadata.get("status") or "active")
            # review No.7：归一化非标准 status
            if doc_status not in _VALID_STATUSES:
                logger.warning(
                    "非标准 status=%r 归一化为 'archived'（source=%s）", doc_status, source,
                )
                doc_status = "archived"

            try:
                self._store.upsert_document(
                    source=source,
                    file_name=file_name,
                    parent_content=doc.page_content,
                    headings=[],
                    child_chunks=[
                        ChildChunk(content=doc.page_content, embedding=emb, child_index=0)
                    ],
                    status=doc_status,  # type: ignore[arg-type]
                )
                indexed += 1
            except Exception as exc:
                logger.warning("SQLite add_documents row failed (source=%s): %s", source, exc)
        return WriteResult(
            requested_count=len(docs),
            indexed_count=indexed,
            storage_mode="sqlite",
            durable=True,
            action="inserted" if indexed == len(docs) else "failed",
            error=None if indexed == len(docs) else f"only {indexed}/{len(docs)} rows inserted",
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        query_text: str = "",
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """混合检索（12 票据：实现 RetrieverProtocol）。

        SQLite 后端通过 search_child_chunks 实现 dense + sparse + RRF 融合：
        - query_text="" 时跳过 sparse（仅 dense）
        - query_text 非空时执行 FTS5 sparse 检索 + RRF 融合
        """
        from rag.hybrid_retrieval import search_child_chunks

        results = search_child_chunks(
            self._read_conn,
            query_embedding,
            query_text=query_text,
            k=k,
            source_filter=source_filter,
            status_filter=status_filter,
            embedding_dimensions=self._embedding_dimensions,
        )
        return [
            SearchResult(
                content=r["content"],
                child_id=r["child_id"],
                parent_id=r["parent_id"],
                source=r["source"],
                file_name=r["file_name"],
                status=r["status"],
                score_dense=r["score_dense"],
                score_sparse=r["score_sparse"],
                score_rrf=r["score_rrf"],
            )
            for r in results
        ]

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        status_filter: Optional[str] = None,
    ) -> list[tuple[str, str, dict, float]]:
        """dense-only 检索（query_text="" 跳过 sparse，对齐 pgvector 行为）。

        10 票据 review No.5：status_filter 下推到 search_child_chunks，
        避免全量 top-k*2 不含目标 status 文档时 Python 层过滤返回空。

        12 票据：保留作为内部方法供 VectorStoreRetriever.similarity_search dense 路径使用。
        新代码应使用 search() 方法（RetrieverProtocol 契约）。
        """
        from rag.hybrid_retrieval import search_child_chunks

        results = search_child_chunks(
            self._read_conn,
            query_embedding,
            query_text="",  # 空 query_text 跳过 sparse
            k=k,
            source_filter=None,
            status_filter=status_filter,
            embedding_dimensions=self._embedding_dimensions,
        )
        return [
            (
                r["child_id"],
                r["content"],
                {
                    "source": r["source"],
                    "file_name": r["file_name"],
                    "title": r["file_name"],  # 兼容下游 title 字段读取
                    "status": r["status"],
                    "id": r["child_id"],
                },
                r["score_dense"],
            )
            for r in results
        ]

    def list_documents(
        self,
        *,
        status_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: int = 5000,
    ) -> list[DocumentSummary]:
        """列举 parent 文档（12 票据：实现 RetrieverProtocol，返回 DocumentSummary）。

        10 票据 review No.2：content 返回真实 content（对齐 pgvector 行为）。
        """
        parents = self._store.list_parents(status_filter=status_filter)  # type: ignore[arg-type]
        summaries: list[DocumentSummary] = []
        # 12 票据 review：先过滤 source_filter 再应用 limit（对齐 pgvector SQL WHERE 下推语义）
        for p in parents:
            if source_filter is not None and p.get("source") != source_filter:
                continue
            if len(summaries) >= limit:
                break
            summaries.append(
                DocumentSummary(
                    source=p["source"],
                    file_name=p["file_name"],
                    title=p["file_name"],
                    status=p["status"],
                    parent_id=p["parent_id"],
                    headings=tuple(p.get("headings", [])),
                    content=p.get("content") or "",
                )
            )
        return summaries

    def has_documents(self, *, status_filter: Optional[str] = None) -> bool:
        """检查是否有文档（按 status 过滤）。"""
        if status_filter:
            row = self._read_conn.execute(
                "SELECT 1 FROM parent WHERE status = ? LIMIT 1", (status_filter,)
            ).fetchone()
        else:
            row = self._read_conn.execute("SELECT 1 FROM parent LIMIT 1").fetchone()
        return row is not None

    def document_count(self) -> int:
        """child chunk 数（对齐 pgvector document_count 的 chunk 语义）。

        10 票据 review No.1：pgvector document_count 统计 embedding IS NOT NULL
        的行数（即 chunk 数），SQLite 端改用 count_children() 对齐，避免
        retriever.document_count() 在两后端返回量级不同的值。
        """
        return self._store.count_children()

    def health(self) -> HealthStatus:
        """检查后端健康状态（12 票据：实现 RetrieverProtocol）。"""
        try:
            count = self.document_count()
            return HealthStatus(
                status="ok",
                storage_mode="sqlite",
                document_count=count,
            )
        except Exception as exc:
            return HealthStatus(
                status="unavailable",
                storage_mode="sqlite",
                document_count=0,
                error=str(exc),
            )

    def close(self):
        try:
            self._store.close()
        except Exception as exc:
            logger.warning("SQLiteStoreClient close failed: %s", exc)


class MemoryRetrieverAdapter:
    """12 票据：显式 memory 后端 adapter（实现 WritableRetrieverProtocol）。

    验收 5：memory adapter 只能通过显式配置（RAG_STORAGE_BACKEND=memory）或测试注入启用。
    不参与 pgvector/sqlite 连接失败时的静默 fallback（那是 VectorStoreRetriever 的 degraded mode）。

    内部使用 InMemoryVectorStore 做 dense 检索 + 简单 BM25 做 sparse 检索，
    与 VectorStoreRetriever 的 legacy memory 路径行为对齐。

    注意：InMemoryVectorStore 无持久化，进程退出数据丢失。仅用于：
    - 开发环境（无 pgvector/sqlite 依赖）
    - 测试注入（构造确定性状态）
    - 显式配置的 memory 模式（RAG_STORAGE_BACKEND=memory）
    """

    def __init__(
        self,
        embeddings: Any | None = None,
        fallback_docs: list[Document] | None = None,
    ):
        self._embeddings = embeddings
        self._store: InMemoryVectorStore | None = None
        self._fallback_docs: list[Document] = list(fallback_docs) if fallback_docs else []
        if embeddings is not None:
            self._store = InMemoryVectorStore(embedding=embeddings)
        self._closed = False

    def search(
        self,
        query_embedding: list[float],
        *,
        k: int = 4,
        query_text: str = "",
        source_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """混合检索（12 票据：实现 RetrieverProtocol）。

        memory 后端仅支持 dense 检索（InMemoryVectorStore.similarity_search_with_score）。
        query_text 参数被忽略，score_sparse=0.0，score_rrf=score_dense。
        若 InMemoryVectorStore 不可用（无 embeddings），回退到 fallback_docs 的简单文本匹配。
        """
        if self._closed:
            raise BackendUnavailableError("memory", "adapter closed")

        if self._store is not None:
            try:
                # 12 票据 review：用 similarity_search_by_vector 直接传入预计算的 query_embedding
                # 避免重复 embed（query_text 不再触发 InMemoryVectorStore 内部 embed）
                lc_results = self._store.similarity_search_by_vector(query_embedding, k=k)
            except Exception as exc:
                logger.warning("MemoryRetrieverAdapter InMemoryVectorStore search failed: %s", exc)
                lc_results = []
            results: list[SearchResult] = []
            for doc in lc_results:
                meta = doc.metadata or {}
                source = meta.get("source", "")
                file_name = meta.get("file_name") or meta.get("title") or source
                doc_status = meta.get("status", "")
                if source_filter is not None and source != source_filter:
                    continue
                if status_filter is not None and doc_status != status_filter:
                    continue
                doc_id = meta.get("id") or meta.get("child_id") or str(uuid.uuid4())
                results.append(SearchResult(
                    content=doc.page_content,
                    child_id=doc_id,
                    parent_id=meta.get("parent_id", doc_id),  # memory 无父子层级
                    source=source,
                    file_name=file_name,
                    status=doc_status,
                    score_dense=1.0,  # InMemoryVectorStore 不返回分数，用占位值
                    score_sparse=0.0,
                    score_rrf=1.0,
                ))
            return results[:k]

        # fallback_docs 路径：简单文本匹配（无 embedding）
        results = []
        for doc in self._fallback_docs:
            meta = doc.metadata or {}
            source = meta.get("source", "")
            file_name = meta.get("file_name") or meta.get("title") or source
            doc_status = meta.get("status", "")
            if source_filter is not None and source != source_filter:
                continue
            if status_filter is not None and doc_status != status_filter:
                continue
            if query_text and query_text not in doc.page_content:
                continue  # 简单包含匹配
            doc_id = meta.get("id") or str(uuid.uuid4())
            results.append(SearchResult(
                content=doc.page_content,
                child_id=doc_id,
                parent_id=meta.get("parent_id", doc_id),
                source=source,
                file_name=file_name,
                status=doc_status,
                score_dense=1.0,
                score_sparse=0.0,
                score_rrf=1.0,
            ))
        return results[:k]

    def list_documents(
        self,
        *,
        status_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: int = 5000,
    ) -> list[DocumentSummary]:
        """列举文档（12 票据：实现 RetrieverProtocol）。"""
        if self._closed:
            raise BackendUnavailableError("memory", "adapter closed")

        docs = self._all_docs()
        summaries: list[DocumentSummary] = []
        for doc in docs[:limit]:
            meta = doc.metadata or {}
            source = meta.get("source", "")
            doc_status = meta.get("status", "")
            if status_filter is not None and doc_status != status_filter:
                continue
            if source_filter is not None and source != source_filter:
                continue
            doc_id = meta.get("id") or str(uuid.uuid4())
            summaries.append(DocumentSummary(
                source=source,
                file_name=meta.get("file_name") or meta.get("title") or "",
                title=meta.get("title") or meta.get("file_name") or "",
                status=doc_status,
                parent_id=meta.get("parent_id", doc_id),
                headings=tuple(meta.get("headings", [])),
                content=doc.page_content,
            ))
        return summaries

    def has_documents(self, *, status_filter: Optional[str] = None) -> bool:
        """检查是否有文档（按 status 过滤）。"""
        if self._closed:
            raise BackendUnavailableError("memory", "adapter closed")
        docs = self._all_docs()
        for doc in docs:
            meta = doc.metadata or {}
            if status_filter is None or meta.get("status") == status_filter:
                return True
        return False

    def document_count(self) -> int:
        """文档计数（对齐 chunk 语义：每个 doc 视为一个 chunk）。"""
        if self._closed:
            raise BackendUnavailableError("memory", "adapter closed")
        return len(self._all_docs())

    def health(self) -> HealthStatus:
        """检查后端健康状态（12 票据：实现 RetrieverProtocol）。"""
        if self._closed:
            return HealthStatus(
                status="unavailable",
                storage_mode="memory",
                document_count=0,
                error="adapter closed",
            )
        return HealthStatus(
            status="ok",
            storage_mode="memory",
            document_count=self.document_count(),
        )

    def add_documents(
        self,
        docs: list[Document],
        embeddings: list[list[float]],
        *,
        status: Optional[str] = None,
    ) -> WriteResult:
        """扁平写入文档（12 票据：实现 WritableRetrieverProtocol）。"""
        if self._closed:
            raise BackendUnavailableError("memory", "adapter closed")

        indexed = 0
        for doc in docs:
            meta = dict(doc.metadata or {})
            if status is not None:
                meta["status"] = status
            # 写入 InMemoryVectorStore（若有 embeddings）和 fallback_docs（始终）
            if self._store is not None and embeddings:
                # 注：InMemoryVectorStore.add_documents 需要 embeddings 通过 embedding model 生成
                # 这里直接添加 Document，InMemoryVectorStore 会内部 embed
                try:
                    self._store.add_documents([Document(page_content=doc.page_content, metadata=meta)])
                    indexed += 1
                except Exception as exc:
                    logger.warning("MemoryRetrieverAdapter store add failed: %s", exc)
            else:
                self._fallback_docs.append(Document(page_content=doc.page_content, metadata=meta))
                indexed += 1
        return WriteResult(
            requested_count=len(docs),
            indexed_count=indexed,
            storage_mode="memory",
            durable=False,
            action="inserted" if indexed == len(docs) else "failed",
            error=None if indexed == len(docs) else f"only {indexed}/{len(docs)} docs inserted",
        )

    def close(self) -> None:
        """释放资源（memory 后端无外部资源，仅标记关闭状态）。"""
        self._closed = True
        self._store = None
        # fallback_docs 保留（可能被其他实例共享），不主动清空

    def _all_docs(self) -> list[Document]:
        """合并 InMemoryVectorStore 和 fallback_docs 中的所有文档。"""
        docs: list[Document] = []
        if self._store is not None:
            try:
                # InMemoryVectorStore 没有公开的 list_all 方法，用 docstore
                if hasattr(self._store, "docstore") and hasattr(self._store.docstore, "_dict"):
                    docs.extend(self._store.docstore._dict.values())
            except Exception as exc:
                logger.warning("MemoryRetrieverAdapter docstore access failed: %s", exc)
        docs.extend(self._fallback_docs)
        return docs


class VectorStoreRetriever:
    """Vector store retriever using LangChain VectorStore abstraction.

    Uses InMemoryVectorStore for local dev/fallback,
    PGVector for production when database is available.
    10 票据：支持 SQLite 后端（rag_storage_backend="sqlite"），cutover 后替代 pgvector。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        self.embedding_base_url = base_url if base_url is not None else settings.embedding_base_url
        self.embedding_api_key = api_key if api_key is not None else settings.embedding_api_key
        self.embedding_model = model if model is not None else settings.embedding_model
        self.embedding_dimensions = dimensions if dimensions is not None else settings.embedding_dimensions

        self.embeddings: OpenAIEmbeddings | None = None
        self._store: InMemoryVectorStore | None = None
        self._pg_client: _PGVectorClient | None = None
        self._sqlite_client: _SQLiteStoreClient | None = None  # 10 票据
        self._fallback_mode = True
        self._fallback_docs: list[Document] = _SHARED_FALLBACK_DOCS
        # 14 票据：runtime 后端选择通过统一 resolver 解析（marker 文件 > config env var）
        # resolver 在每次 __init__ 调用时读取 marker，保证应用重启后解析结果与切换操作一致（AC6）
        resolved = resolve_backend(
            config_backend=settings.rag_storage_backend,
            marker_path=Path(settings.rag_cutover_marker_path),
        )
        self._storage_backend = resolved.backend
        self._resolved_backend_source = resolved.source
        # 14 票据：diagnostics 已在 resolver 内部记录到 logger，此处不重复打印

        if self.embedding_api_key:
            self.embeddings = OpenAIEmbeddings(
                base_url=self.embedding_base_url,
                api_key=self.embedding_api_key,
                model=self.embedding_model,
                # 兼容非 OpenAI 原生 embedding API（详见 api/rag.py 同名参数注释）
                check_embedding_ctx_length=False,
            )

        self._try_connect_storage()

    def _try_connect_storage(self):
        """10 票据：根据 storage_backend 选择 pgvector 或 SQLite，失败回退 in-memory。

        12 票据：新增 "memory" 显式后端选项（验收 5）。
        """
        if self.embeddings is None:
            logger.info("No embeddings configured, using in-memory store")
            self._store = InMemoryVectorStore(embedding=None)
            return

        if self._storage_backend == "memory":
            self._try_connect_memory()
        elif self._storage_backend == "sqlite":
            self._try_connect_sqlite()
        else:
            self._try_connect_pgvector()

    def _try_connect_memory(self):
        """12 票据：显式 memory 后端（验收 5：memory adapter 只能通过显式配置启用）。

        使用 _SHARED_FALLBACK_DOCS 实现 cross-instance 文档持久化（与 legacy memory 路径行为对齐）。
        """
        logger.info("Using explicit memory backend (RAG_STORAGE_BACKEND=memory)")
        self._store = InMemoryVectorStore(embedding=self.embeddings)
        self._fallback_mode = False  # memory 是显式后端，不是 degraded mode
        self._memory_explicit = True

    @property
    def _memory_explicit(self) -> bool:
        """12 票据：是否启用了显式 memory 后端（vs degraded fallback）."""
        return getattr(self, "_memory_explicit_flag", False)

    @_memory_explicit.setter
    def _memory_explicit(self, value: bool):
        self._memory_explicit_flag = value

    @classmethod
    def for_testing(
        cls,
        *,
        storage_backend: str = "pgvector",
        embeddings: Any = None,
        embedding_dimensions: int = 1536,
        pg_client: Optional[WritableRetrieverProtocol] = None,
        sqlite_client: Optional[WritableRetrieverProtocol] = None,
        memory_store: Optional[InMemoryVectorStore] = None,
        fallback_mode: bool = False,
        fallback_docs: Optional[list[Document]] = None,
        memory_explicit: bool = False,
    ) -> "VectorStoreRetriever":
        """13 票据：公共测试工厂，注入确定性 adapter 状态。

        测试通过此方法构造 retriever，不绕过构造函数、不直接设置私有字段。
        adapter 注入通过 WritableRetrieverProtocol 契约，不访问私有连接。

        参数：
            storage_backend: 后端类型（"pgvector" | "sqlite" | "memory"）。
                注意：fallback_mode=True 时实际走 memory 路径，storage_backend 仅用于
                标记配置意图（与生产 _try_connect_* 失败回退行为对齐）。
            embeddings: embedding 客户端（或测试替身）
            embedding_dimensions: embedding 维度
            pg_client: pgvector adapter 替身（实现 WritableRetrieverProtocol）
            sqlite_client: SQLite adapter 替身（实现 WritableRetrieverProtocol）
            memory_store: InMemoryVectorStore 实例（memory/fallback 用）
            fallback_mode: 是否处于降级回退模式
            fallback_docs: fallback 文档列表
            memory_explicit: 是否为显式 memory 后端（vs degraded fallback）
        """
        retriever = cls.__new__(cls)
        retriever.embedding_base_url = None
        retriever.embedding_api_key = None
        retriever.embedding_model = None
        retriever.embedding_dimensions = embedding_dimensions
        retriever.embeddings = embeddings
        retriever._store = memory_store
        retriever._pg_client = pg_client
        retriever._sqlite_client = sqlite_client
        retriever._fallback_mode = fallback_mode
        retriever._fallback_docs = fallback_docs if fallback_docs is not None else []
        retriever._storage_backend = storage_backend
        retriever._memory_explicit = memory_explicit
        return retriever

    def _try_connect_pgvector(self):
        """Try to connect to pgvector, fall back to in-memory."""
        try:
            self._pg_client = _PGVectorClient(
                settings.database_url,
                connect_timeout=settings.database_connect_timeout_seconds,
            )
            count = self._pg_client.document_count()
            self._fallback_mode = False
            logger.info("Connected to pgvector, %d documents in store", count)
        except Exception as e:
            logger.warning("pgvector connection failed, using InMemoryVectorStore: %s", e)
            self._pg_client = None
            self._fallback_mode = True
            self._store = InMemoryVectorStore(embedding=self.embeddings)

    def _try_connect_sqlite(self):
        """10 票据：连接 SQLiteStore，失败回退 in-memory。"""
        try:
            self._sqlite_client = _SQLiteStoreClient(
                settings.rag_sqlite_db_path,
                embedding_dimensions=self.embedding_dimensions,
            )
            count = self._sqlite_client.document_count()
            self._fallback_mode = False
            logger.info("Connected to SQLiteStore, %d parents in store", count)
        except Exception as e:
            logger.warning("SQLiteStore connection failed, using InMemoryVectorStore: %s", e)
            self._sqlite_client = None
            self._fallback_mode = True
            self._store = InMemoryVectorStore(embedding=self.embeddings)

    @property
    def storage_mode(self) -> str:
        if self._fallback_mode and not self._memory_explicit:
            return "memory"
        return self._storage_backend  # "pgvector" | "sqlite" | "memory"

    @property
    def sqlite_store(self):
        """13 票据：公开访问底层 SQLiteStore（若 SQLite 后端可用）。

        runtime/迁移工具通过此 property 获取 SQLiteStore，不访问 _sqlite_client 私有字段。
        返回 None 表示非 SQLite 后端或 SQLite 不可用。
        """
        if self._sqlite_client is not None:
            return getattr(self._sqlite_client, "store", None)
        return None

    def health(self) -> HealthStatus:
        """检查后端健康状态（12 票据：透传 Protocol.health()）。

        验收 4：后端初始化失败不会静默切换到 memory 并报告生产成功。
        - 显式 memory 后端：status='ok'（不是 degraded）
        - pgvector/sqlite 连接失败回退 memory：status='degraded'（报告降级状态）
        - pgvector/sqlite 正常连接：status='ok'
        """
        # 显式 memory 后端
        if self._memory_explicit:
            return HealthStatus(
                status="ok",
                storage_mode="memory",
                document_count=len(self._fallback_docs),
            )
        # pgvector/sqlite degraded mode（连接失败回退 memory）
        if self._fallback_mode:
            return HealthStatus(
                status="degraded",
                storage_mode="memory",
                document_count=len(self._fallback_docs),
                error=f"configured backend '{self._storage_backend}' unavailable, using memory fallback",
            )
        # pgvector/sqlite 正常连接
        try:
            if self._sqlite_client is not None:
                return self._sqlite_client.health()
            if self._pg_client is not None:
                return self._pg_client.health()
        except Exception as exc:
            return HealthStatus(
                status="unavailable",
                storage_mode=self._storage_backend,
                document_count=0,
                error=str(exc),
            )
        return HealthStatus(
            status="degraded",
            storage_mode=self._storage_backend,
            document_count=0,
            error="no backend connected",
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the configured embedding model."""
        return self.embeddings.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)

    def add_documents(self, docs: list[Document], status: Optional[str] = None) -> AddDocumentsResult:
        """Add documents with optional status metadata.

        10 票据：该方法为 legacy 摄入入口，新代码应使用 IngestionPipeline.ingest_file /
        ingest_content 走 canonical ingestion（父子层级切分 + 事务写入）。

        收缩后的生产职责：
        - SQLite 后端（storage_mode='sqlite' 且未回退）：禁止直接写入未分层文档，
          抛 RuntimeError 引导调用方走 canonical ingestion（验收 1/2/3）。
        - pgvector/memory 后端：保留 legacy 路径零回归（Ticket 14 cutover 移除）。
        - fallback 模式（SQLite 连接失败回退 memory）：允许写入 memory，保留零回归。
        - 测试/迁移脚本：可直接调用，但应通过 _SQLiteStoreClient.add_documents 显式标识。
        """
        # 10 票据验收 1/2/3：SQLite 后端禁止通过 retriever.add_documents 直接写入
        # 未分层文档（每个 Document 被当作一个 parent=一个 child，绕过父子生命周期）
        if not self._fallback_mode and self._sqlite_client is not None:
            raise RuntimeError(
                "SQLite 后端禁止通过 VectorStoreRetriever.add_documents 直接写入未分层文档；"
                "请使用 IngestionPipeline.ingest_file / ingest_content 走 canonical ingestion。"
                "（10 票据：收缩 Runtime 直接文档写入路径）"
            )

        for doc in docs:
            if status:
                doc.metadata["status"] = status
        self._fallback_docs.extend(docs)
        requested_count = len(docs)
        # 10 票据：durable 后端可能是 pgvector 或 sqlite
        started_in_durable = (
            not self._fallback_mode
            and (self._pg_client is not None or self._sqlite_client is not None)
        )

        if requested_count == 0:
            return AddDocumentsResult(
                requested_count=0,
                indexed_count=0,
                storage_mode=self.storage_mode,
                durable=started_in_durable,
            )

        # 10 票据：dead code after guard — 守护（L449-454）已用同一条件抛 RuntimeError，
        # 此分支在任何情况下都不可达（fallback 模式下 _fallback_mode=True 使条件为 False）。
        # 保留作为防御性代码，防止守护被绕过时静默写入未分层文档。
        if not self._fallback_mode and self._sqlite_client is not None:
            try:
                texts = [doc.page_content for doc in docs]
                embs = self._embed_texts(texts)
                write_result = self._sqlite_client.add_documents(docs, embs, status=status)
                return AddDocumentsResult(
                    requested_count=write_result.requested_count,
                    indexed_count=write_result.indexed_count,
                    storage_mode="sqlite",
                    durable=True,
                )
            except Exception as e:
                logger.warning("SQLite add failed, falling back to memory: %s", e)
                self._fallback_mode = True
                if self._store is None:
                    self._store = InMemoryVectorStore(embedding=self.embeddings)
                durable_error = str(e)
        elif started_in_durable and self._pg_client is not None:
            try:
                texts = [doc.page_content for doc in docs]
                embs = self._embed_texts(texts)
                write_result = self._pg_client.add_documents(docs, embs, status=status)
                return AddDocumentsResult(
                    requested_count=write_result.requested_count,
                    indexed_count=write_result.indexed_count,
                    storage_mode="pgvector",
                    durable=True,
                )
            except Exception as e:
                logger.warning("pgvector add failed, falling back to memory: %s", e)
                self._fallback_mode = True
                if self._store is None:
                    self._store = InMemoryVectorStore(embedding=self.embeddings)
                durable_error = str(e)
        else:
            durable_error = None

        if self._store is not None:
            try:
                self._store.add_documents(docs)
                return AddDocumentsResult(
                    requested_count=requested_count,
                    indexed_count=requested_count,
                    storage_mode="memory",
                    durable=False,
                    fallback_used=started_in_durable,
                    error=durable_error,
                )
            except Exception as e:
                logger.warning("InMemoryVectorStore add failed: %s", e)
                return AddDocumentsResult(
                    requested_count=requested_count,
                    indexed_count=0,
                    storage_mode="memory",
                    durable=False,
                    fallback_used=started_in_durable,
                    error=durable_error or str(e),
                )

        return AddDocumentsResult(
            requested_count=requested_count,
            indexed_count=requested_count,
            storage_mode="memory",
            durable=False,
            fallback_used=started_in_durable,
            error=durable_error,
        )

    def similarity_search(self, query: str, k: int = 4, status_filter: Optional[str] = None) -> list[Document]:
        """Search for similar documents."""
        if self.embeddings is None:
            logger.warning("Embedding API key missing; returning no RAG matches")
            return []

        # 10 票据：SQLite 后端路径（dense-only，对齐 pgvector similarity_search 行为）
        if not self._fallback_mode and self._sqlite_client is not None:
            try:
                if status_filter and not self._sqlite_client.has_documents(status_filter=status_filter):
                    return []
                query_emb = self._embed_query(query)
                # review No.5：status_filter 下推到 search_child_chunks，避免全量 top-k*2
                # 不含目标 status 文档时 Python 层过滤返回空
                rows = self._sqlite_client.similarity_search(
                    query_emb, k=k * 2, status_filter=status_filter
                )
                docs = [
                    Document(
                        page_content=content,
                        metadata=meta,
                    )
                    for _id, content, meta, _sim in rows
                ]
                return self._apply_status_filter(docs, status_filter)[:k]
            except Exception as e:
                logger.warning("SQLite vector search failed, continuing with lexical retrieval: %s", e)
                return []

        # Try pgvector first
        if not self._fallback_mode and self._pg_client is not None:
            try:
                if status_filter and not self._pg_client.has_documents(status_filter=status_filter):
                    return []
                query_emb = self._embed_query(query)
                rows = self._pg_client.similarity_search(query_emb, k=k * 2)
                docs = [
                    Document(
                        page_content=content,
                        metadata=meta,
                    )
                    for _id, content, meta, _sim in rows
                ]
                return self._apply_status_filter(docs, status_filter)[:k]
            except Exception as e:
                logger.warning("pgvector vector search failed, continuing with lexical retrieval: %s", e)
                return []

        # InMemoryVectorStore search
        if self._store is not None:
            try:
                results = self._store.similarity_search(query, k=k * 2)
                return self._apply_status_filter(results, status_filter)[:k]
            except Exception as e:
                logger.warning("InMemoryVectorStore search failed: %s", e)

        # Fallback: simple list search
        if self._fallback_docs:
            return self._apply_status_filter(self._fallback_docs, status_filter)[:k]

        return []

    def bm25_search(self, query: str, k: int = 4, status_filter: Optional[str] = None) -> list[Document]:
        """Search documents with BM25 lexical scoring."""
        # 10 票据：SQLite 后端用 FTS5 sparse 检索（jieba 预分词，recall 79%）
        if not self._fallback_mode and self._sqlite_client is not None:
            return self._sqlite_sparse_search(query, k=k, status_filter=status_filter)

        docs = self._all_documents(status_filter=status_filter)
        ranked = bm25_rank(query, docs, k=k)
        results: list[Document] = []
        for rank, (doc, score) in enumerate(ranked, start=1):
            results.append(Document(
                page_content=doc.page_content,
                metadata={
                    **(doc.metadata or {}),
                    "bm25_rank": rank,
                    "bm25_score": score,
                    "retrieval_method": "bm25",
                },
            ))
        return results

    def _sqlite_sparse_search(
        self, query: str, k: int = 4, status_filter: Optional[str] = None
    ) -> list[Document]:
        """10 票据：SQLite 后端的 sparse 检索（FTS5 + jieba）。

        12 票据 review：该方法为 legacy BM25-only 路径，仍访问 read_conn 裸连接。
        主检索路径（hybrid_search）已通过 Protocol.search() 调用，不访问裸连接。
        Ticket 13/14 cutover 时统一移除 legacy BM25 路径或迁移到 Protocol 扩展。
        """
        from rag.hybrid_retrieval import _sparse_search

        rows = _sparse_search(
            self._sqlite_client.read_conn,  # legacy：仍访问公开 read_conn
            query,
            k=k * 2,
            source_filter=None,
            status_filter=status_filter,
        )
        results: list[Document] = []
        for rank, r in enumerate(rows, start=1):
            results.append(Document(
                page_content=r["content"],
                metadata={
                    "source": r["source"],
                    "file_name": r["file_name"],
                    "title": r["file_name"],
                    "status": r["status"],
                    "id": r["child_id"],
                    "bm25_rank": rank,
                    "bm25_score": r["score_sparse"],
                    "retrieval_method": "fts5",
                },
            ))
        return results[:k]

    def hybrid_search(self, queries: list[str], k: int = 4, status_filter: Optional[str] = None) -> list[Document]:
        """Run vector + BM25 retrieval for each query and fuse with RRF."""
        # 10 票据：SQLite 后端用 search_child_chunks（dense + sparse + RRF 一次性融合）
        if not self._fallback_mode and self._sqlite_client is not None:
            return self._sqlite_hybrid_search(queries, k=k, status_filter=status_filter)

        rank_lists: list[list[Document]] = []

        for query in queries:
            vector_docs = self.similarity_search(query, k=k, status_filter=status_filter)
            if vector_docs:
                rank_lists.append(vector_docs)

            bm25_docs = self.bm25_search(query, k=k, status_filter=status_filter)
            if bm25_docs:
                rank_lists.append(bm25_docs)

        return reciprocal_rank_fusion(rank_lists, k=k)

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """票据 02：RetrieverProtocol.search 适配方法。

        调用 hybrid_search（dense + sparse + RRF），返回 dict 列表（对齐 graph.py 的
        RetrieverProtocol.search 返回类型）。用于 graph runtime 的 retrieve 节点。

        与 hybrid_search 的差异：返回 dict 而非 LangChain Document，字段平铺到顶层
        （content/source/file_name/title/doc_type/status/child_id/parent_id/score_*）。
        """
        docs = self.hybrid_search([query], k=k)
        return [
            {
                "content": doc.page_content,
                "source": (doc.metadata or {}).get("source", "unknown"),
                "file_name": (doc.metadata or {}).get("file_name", ""),
                "title": (doc.metadata or {}).get("title", ""),
                "doc_type": (doc.metadata or {}).get("doc_type", ""),
                "status": (doc.metadata or {}).get("status", ""),
                "child_id": (doc.metadata or {}).get("id", ""),
                "parent_id": (doc.metadata or {}).get("parent_id", ""),
                "score_rrf": (doc.metadata or {}).get("rrf_score", 0.0),
                "score_dense": (doc.metadata or {}).get("score_dense", 0.0),
                "score_sparse": (doc.metadata or {}).get("score_sparse", 0.0),
                "retrieval_method": (doc.metadata or {}).get("retrieval_method", ""),
            }
            for doc in docs
        ]

    def _sqlite_hybrid_search(
        self, queries: list[str], k: int = 4, status_filter: Optional[str] = None
    ) -> list[Document]:
        """10 票据：SQLite 后端的 hybrid 检索（dense + sparse + 跨 query RRF）。

        12 票据：通过 RetrieverProtocol.search() 契约方法调用，不再访问 read_conn 私有连接。
        review No.6：移除 early break，每个 query 的结果作为独立 rank_list，
        用 reciprocal_rank_fusion 跨 query 融合（对齐 pgvector 路径语义）。
        search_child_chunks 内部已做 query 内 dense+sparse RRF，
        本方法再做跨 query RRF 二次融合。
        """
        from rag.text_processing import reciprocal_rank_fusion

        if self.embeddings is None:
            return []

        # review No.3：has_documents 检查移到循环外（对所有 query 结果相同）
        if status_filter and not self._sqlite_client.has_documents(status_filter=status_filter):  # type: ignore[union-attr]
            return []

        rank_lists: list[list[Document]] = []
        for query in queries:
            query_emb = self._embed_query(query)
            # 12 票据：通过 Protocol.search() 调用，不访问 read_conn
            results = self._sqlite_client.search(  # type: ignore[union-attr]
                query_emb,
                k=k,
                query_text=query,
                status_filter=status_filter,
            )
            if not results:
                continue
            query_docs = [
                Document(
                    page_content=r.content,
                    metadata={
                        "source": r.source,
                        "file_name": r.file_name,
                        "title": r.file_name,
                        "status": r.status,
                        "id": r.child_id,
                        "parent_id": r.parent_id,
                        "rrf_score": r.score_rrf,
                        "score_dense": r.score_dense,
                        "score_sparse": r.score_sparse,
                        "retrieval_method": "hybrid_rrf",
                    },
                )
                for r in results
            ]
            rank_lists.append(query_docs)

        if not rank_lists:
            return []

        # 跨 query RRF 融合（reciprocal_rank_fusion 内部按 _doc_key 去重）
        return reciprocal_rank_fusion(rank_lists, k=k)

    def document_count(self) -> int:
        """Count indexed documents in the active store."""
        if not self._fallback_mode and self._sqlite_client is not None:
            try:
                return self._sqlite_client.document_count()
            except Exception as e:
                logger.warning("SQLite document count failed: %s", e)
        if not self._fallback_mode and self._pg_client is not None:
            try:
                return self._pg_client.document_count()
            except Exception as e:
                logger.warning("pgvector document count failed: %s", e)
        return len(self._fallback_docs)

    def indexed_documents(self, limit: int = 5000) -> list[Document]:
        """List indexed documents from the active store."""
        if not self._fallback_mode and self._sqlite_client is not None:
            try:
                summaries = self._sqlite_client.list_documents(limit=limit)
                return [self._summary_to_document(s) for s in summaries]
            except Exception as e:
                logger.warning("SQLite document listing failed: %s", e)
        if not self._fallback_mode and self._pg_client is not None:
            try:
                summaries = self._pg_client.list_documents(limit=limit)
                return [self._summary_to_document(s) for s in summaries]
            except Exception as e:
                logger.warning("pgvector document listing failed: %s", e)
        return self._fallback_docs[:limit]

    def _all_documents(self, status_filter: Optional[str] = None) -> list[Document]:
        if not self._fallback_mode and self._sqlite_client is not None:
            try:
                summaries = self._sqlite_client.list_documents(status_filter=status_filter)
                return [self._summary_to_document(s) for s in summaries]
            except Exception as e:
                logger.warning("SQLite document listing failed: %s", e)
        if not self._fallback_mode and self._pg_client is not None:
            try:
                summaries = self._pg_client.list_documents(status_filter=status_filter)
                return [self._summary_to_document(s) for s in summaries]
            except Exception as e:
                logger.warning("pgvector document listing failed: %s", e)

        return self._apply_status_filter(self._fallback_docs, status_filter)

    @staticmethod
    def _summary_to_document(summary: DocumentSummary) -> Document:
        """将 DocumentSummary 转换为 LangChain Document（12 票据）。"""
        return Document(
            page_content=summary.content,
            metadata={
                "id": summary.parent_id,
                "parent_id": summary.parent_id,
                "source": summary.source,
                "file_name": summary.file_name,
                "title": summary.title,
                "status": summary.status,
                "headings": list(summary.headings),
            },
        )

    @staticmethod
    def _apply_status_filter(docs: list[Document], status_filter: Optional[str] = None) -> list[Document]:
        """Filter documents by status metadata."""
        if not status_filter:
            return docs
        return [doc for doc in docs if doc.metadata.get("status") == status_filter]
