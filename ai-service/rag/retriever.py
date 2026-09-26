import logging
import uuid
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import settings
from rag.retriever_protocol import (
    DocumentSummary,
    HealthStatus,
    SearchResult,
    WriteResult,
)
from rag.text_processing import reciprocal_rank_fusion

logger = logging.getLogger("ai-service.rag")

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


class VectorStoreRetriever:
    """Vector store retriever over the SQLite retrieval store.

    SQLite（data/rag.sqlite，父子层级 + FTS5 + sqlite-vec）是唯一存储后端；
    摄入统一走 IngestionPipeline（canonical 父子切分），检索走 hybrid_search。
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
        self._sqlite_client: _SQLiteStoreClient | None = None

        if self.embedding_api_key:
            self.embeddings = OpenAIEmbeddings(
                base_url=self.embedding_base_url,
                api_key=self.embedding_api_key,
                model=self.embedding_model,
                # 兼容非 OpenAI 原生 embedding API（详见 api/rag.py 同名参数注释）
                check_embedding_ctx_length=False,
            )

        self._sqlite_client = _SQLiteStoreClient(
            settings.rag_sqlite_db_path,
            embedding_dimensions=self.embedding_dimensions,
        )
        count = self._sqlite_client.document_count()
        logger.info("Connected to SQLiteStore, %d parents in store", count)

    @property
    def storage_mode(self) -> str:
        return "sqlite"

    @property
    def sqlite_store(self):
        """公开访问底层 SQLiteStore。

        runtime/迁移工具通过此 property 获取 SQLiteStore，不访问 _sqlite_client 私有字段。
        """
        return getattr(self._sqlite_client, "store", None)

    def health(self) -> HealthStatus:
        """检查后端健康状态（透传 SQLite adapter 的 Protocol.health()）。"""
        try:
            return self._sqlite_client.health()
        except Exception as exc:
            return HealthStatus(
                status="unavailable",
                storage_mode="sqlite",
                document_count=0,
                error=str(exc),
            )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the configured embedding model."""
        return self.embeddings.embed_documents(texts)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)

    def document_count(self) -> int:
        """Count indexed documents in the SQLite store."""
        try:
            return self._sqlite_client.document_count()
        except Exception as e:
            logger.warning("SQLite document count failed: %s", e)
        return 0

    def indexed_documents(self, limit: int = 5000) -> list[Document]:
        """List indexed documents from the SQLite store."""
        try:
            summaries = self._sqlite_client.list_documents(limit=limit)
            return [self._summary_to_document(s) for s in summaries]
        except Exception as e:
            logger.warning("SQLite document listing failed: %s", e)
        return []

    def similarity_search(self, query: str, k: int = 4, status_filter: Optional[str] = None) -> list[Document]:
        """Dense-only similar-document search over the SQLite store."""
        if self.embeddings is None:
            logger.warning("Embedding API key missing; returning no RAG matches")
            return []

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
            logger.warning("SQLite vector search failed, returning no matches: %s", e)
            return []

    def bm25_search(self, query: str, k: int = 4, status_filter: Optional[str] = None) -> list[Document]:
        """BM25 lexical scoring over the SQLite FTS5 index（jieba 预分词）."""
        return self._sqlite_sparse_search(query, k=k, status_filter=status_filter)

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
        """Run dense + sparse (FTS5) retrieval fused with RRF over the SQLite store."""
        return self._sqlite_hybrid_search(queries, k=k, status_filter=status_filter)

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
