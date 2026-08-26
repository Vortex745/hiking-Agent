"""票据 02：SQLite 父子分层检索存储层。

按 sqlite-storage-contract.md 冻结结论实现事务边界、幂等、级联、维度校验：
- 文档身份策略（C 复合）：同 source+同 content_hash 跳过；同 source+不同 content_hash 单事务替换
- 稳定 ID：parent_id=uuid5(NAMESPACE_URL, source)，child_id=uuid5(NAMESPACE_URL, parent_id+':'+child_index)
- vec0 维度不匹配显式报错 + 事务回滚
- ON DELETE CASCADE 级联删除（schema 层已实现）
- FTS5 应用层同步：04 票据中文评测后改为 jieba 预分词 + unicode61 tokenizer，
  写入时 jieba.cut 分词后插入 child_fts，删除时显式 DELETE（无触发器）

本模块负责文档级写入与删除，不负责检索（检索由票据 04 的 search_child_chunks 实现）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import jieba

from rag.sqlite_schema import DEFAULT_EMBEDDING_DIMENSIONS, connect, init_schema

logger = logging.getLogger("ai-service.rag.sqlite_store")

# uuid5 命名空间（按契约 §2 稳定 ID：uuid5(NAMESPACE_URL, source)）
_NAMESPACE = uuid.NAMESPACE_URL

# status 枚举（按契约 §3 source/status 语义）
Status = Literal["active", "archived", "feishu", "cloud_docs"]
_VALID_STATUSES = frozenset({"active", "archived", "feishu", "cloud_docs"})


def parent_id_for(source: str) -> str:
    """parent_id = uuid5(NAMESPACE_URL, source)。

    按契约 §2：同 source 永远产生同 parent_id（幂等重跑不变）。
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source must be non-empty string, got: {source!r}")
    return str(uuid.uuid5(_NAMESPACE, source))


def child_id_for(parent_id: str, child_index: int) -> str:
    """child_id = uuid5(NAMESPACE_URL, parent_id + ':' + child_index)。

    按契约 §2：同 parent + 同切分顺序产生同 child_id（幂等重跑不变）。
    """
    if not isinstance(parent_id, str) or not parent_id.strip():
        raise ValueError(f"parent_id must be non-empty string, got: {parent_id!r}")
    if not isinstance(child_index, int) or child_index < 0:
        raise ValueError(f"child_index must be non-negative int, got: {child_index!r}")
    return str(uuid.uuid5(_NAMESPACE, f"{parent_id}:{child_index}"))


def content_hash_of(content: str) -> str:
    """SHA256(content)，用于判定 no-op vs 替换。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def serialize_embedding(embedding: list[float]) -> bytes:
    """将 float list 序列化为 vec0 接受的二进制格式（float32 little-endian）。

    Raises:
        ValueError: embedding 非 list/tuple 或元素类型非 float/int
    """
    if not isinstance(embedding, (list, tuple)):
        raise ValueError(f"embedding must be list/tuple, got: {type(embedding)}")
    for i, v in enumerate(embedding):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(
                f"embedding[{i}] must be float/int, got {type(v).__name__}: {v!r}"
            )
    try:
        return struct.pack(f"{len(embedding)}f", *embedding)
    except struct.error as exc:
        # 兜底：理论上元素类型已校验，此处防止边界情况
        raise ValueError(f"embedding serialization failed: {exc}") from exc


def _now_iso() -> str:
    """UTC ISO 8601 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _jieba_tokenize(text: str) -> str:
    """jieba 精确分词，空格连接（04 票据中文评测结论：FTS5 unicode61 + jieba 预分词）。

    评测证据：trigram recall@5=25% vs jieba recall@5=79%
    （.scratch/rag-lifecycle-refactor/eval/fts5-chinese-eval.md）
    """
    return " ".join(t for t in jieba.cut(text, cut_all=False) if t.strip())


@dataclass(frozen=True)
class ChildChunk:
    """单个 child chunk 输入。"""

    content: str
    embedding: list[float]
    child_index: int


@dataclass(frozen=True)
class IngestResult:
    """upsert_document 的返回值，对运营可见。"""

    action: Literal["noop", "replaced", "inserted"]
    parent_id: str
    source: str
    child_count: int
    content_hash: str
    message: str = ""


class SQLiteStore:
    """SQLite 父子分层检索存储层。

    单例写：业务层应全局共享一个 SQLiteStore 实例（契约 §5：单例写连接 + 多读连接）。
    事务边界：upsert_document 内部 BEGIN/COMMIT/ROLLBACK，四层原子生效。
    """

    def __init__(
        self,
        db_path: Path | str,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ):
        self._db_path = str(db_path)
        self._embedding_dimensions = embedding_dimensions
        # 单例写连接：check_same_thread=False 允许调用方在不同线程使用（如 FastAPI async 框架），
        # 但调用方需保证不真正并发调用同一连接的写操作（SQLite 单写者模型）
        self._write_conn = connect(
            db_path, embedding_dimensions=embedding_dimensions, check_same_thread=False
        )

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_dimensions

    @property
    def read_conn(self):
        """公开读连接（WAL 模式下与写连接同一实例）。

        10 票据 review No.10：暴露公开 property 替代直接访问 _write_conn，
        避免外部调用方（_SQLiteStoreClient / baseline_performance）依赖私有属性。
        若后续改为连接池，只需修改此 property 实现即可。
        """
        return self._write_conn

    def close(self) -> None:
        """关闭写连接。"""
        try:
            self._write_conn.close()
        except Exception as exc:
            logger.warning(f"close write_conn failed: {exc}")

    # === 文档级写入 ===

    def upsert_document(
        self,
        source: str,
        file_name: str,
        parent_content: str,
        headings: list[str],
        child_chunks: list[ChildChunk],
        status: Status = "active",
    ) -> IngestResult:
        """文档级写入：parent + child + vec0 + FTS5 四层同事务提交。

        身份策略（契约 §1 C 复合）：
        - 同 source + 同 content_hash：跳过（noop），返回现有 child_count
        - 同 source + 不同 content_hash：单事务替换（DELETE 旧四层 + INSERT 新四层）
        - 不同 source：新文档（INSERT）

        Args:
            source: 文档身份标识（本地绝对路径或远程稳定标识）
            file_name: 文件名（用于检索结果展示）
            parent_content: parent 整段文本（用于 retrieve_parent_chunks）
            headings: H1/H2/H3 标题路径（JSON 序列化存 parent.headings）
            child_chunks: child chunk 列表（含 content + embedding + child_index）
            status: 文档状态枚举

        Returns:
            IngestResult：对运营可见的写入结果

        Raises:
            ValueError: status 非法、embedding 维度不匹配、参数非法
            sqlite3.OperationalError: 事务失败（已回滚）
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)}, got: {status!r}"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"source must be non-empty string, got: {source!r}")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"file_name must be non-empty string, got: {file_name!r}")
        if not isinstance(parent_content, str):
            raise ValueError(f"parent_content must be string, got: {type(parent_content)}")
        if not isinstance(headings, list):
            raise ValueError(f"headings must be list, got: {type(headings)}")
        if not isinstance(child_chunks, list):
            raise ValueError(f"child_chunks must be list, got: {type(child_chunks)}")
        if not child_chunks:
            raise ValueError("child_chunks must not be empty (parent without children is meaningless)")

        # 维度校验（写入前显式拒绝，避免事务中途失败）
        # 同时校验类型：None/str 等会被 len() 误判，元素类型错误会在 struct.pack 失败
        for chunk in child_chunks:
            if not isinstance(chunk.embedding, (list, tuple)):
                raise ValueError(
                    f"embedding must be list/tuple, got {type(chunk.embedding).__name__} "
                    f"(child_index={chunk.child_index})"
                )
            if len(chunk.embedding) != self._embedding_dimensions:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self._embedding_dimensions} "
                    f"got {len(chunk.embedding)} (child_index={chunk.child_index})"
                )

        parent_id = parent_id_for(source)
        new_hash = content_hash_of(parent_content)
        headings_json = json.dumps(headings, ensure_ascii=False)
        now = _now_iso()

        # 显式事务控制：BEGIN ... COMMIT/ROLLBACK
        # 注意：sqlite3 模块的 with conn 不会自动 BEGIN/COMMIT（只处理 commit/rollback），
        # 用显式 BEGIN 保证事务边界
        try:
            self._write_conn.execute("BEGIN IMMEDIATE")
            existing = self._write_conn.execute(
                "SELECT content_hash FROM parent WHERE parent_id = ?", (parent_id,)
            ).fetchone()

            if existing is not None:
                old_hash = existing[0]
                if old_hash == new_hash:
                    # 同 source + 同 content_hash：no-op（先查 child_count 再 ROLLBACK）
                    child_count = self._count_children(parent_id)
                    self._write_conn.execute("ROLLBACK")
                    logger.info(
                        f"ingest noop: source={source} parent_id={parent_id} children={child_count}"
                    )
                    return IngestResult(
                        action="noop",
                        parent_id=parent_id,
                        source=source,
                        child_count=child_count,
                        content_hash=new_hash,
                        message="content unchanged, skipped",
                    )
                # 同 source + 不同 content_hash：替换
                self._delete_parent_cascade(parent_id)
                action: Literal["replaced", "inserted"] = "replaced"
            else:
                action = "inserted"

            # 插入新 parent
            self._write_conn.execute(
                """
                INSERT INTO parent (parent_id, source, file_name, content, content_hash, headings, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (parent_id, source, file_name, parent_content, new_hash, headings_json, status, now, now),
            )

            # 插入 child + vec0 + FTS5（04 票据后 FTS5 由应用层 jieba 分词同步）
            for chunk in child_chunks:
                child_id = child_id_for(parent_id, chunk.child_index)
                self._write_conn.execute(
                    """
                    INSERT INTO child (child_id, parent_id, child_index, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (child_id, parent_id, chunk.child_index, chunk.content),
                )
                emb_bytes = serialize_embedding(chunk.embedding)
                try:
                    self._write_conn.execute(
                        """
                        INSERT INTO child_vec (child_id, parent_id, embedding)
                        VALUES (?, ?, ?)
                        """,
                        (child_id, parent_id, emb_bytes),
                    )
                except sqlite3.OperationalError as exc:
                    # 维度不匹配（理论上写入前已校验，此处兜底）
                    raise ValueError(
                        f"embedding dimension mismatch: expected {self._embedding_dimensions} "
                        f"(sqlite-vec rejected: {exc})"
                    ) from exc
                # FTS5 应用层同步：jieba 分词后写入（原触发器方案已移除）
                tokenized = _jieba_tokenize(chunk.content)
                self._write_conn.execute(
                    """
                    INSERT INTO child_fts (child_id, parent_id, content)
                    VALUES (?, ?, ?)
                    """,
                    (child_id, parent_id, tokenized),
                )

            self._write_conn.execute("COMMIT")
            logger.info(
                f"ingest {action}: source={source} parent_id={parent_id} "
                f"children={len(child_chunks)} hash={new_hash[:8]}"
            )
            return IngestResult(
                action=action,
                parent_id=parent_id,
                source=source,
                child_count=len(child_chunks),
                content_hash=new_hash,
            )
        except ValueError:
            # 维度校验失败，回滚事务
            try:
                self._write_conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass  # 事务可能已回滚
            raise
        except sqlite3.Error as exc:
            # 其他 SQLite 错误，回滚事务
            try:
                self._write_conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise sqlite3.OperationalError(f"ingest failed (rolled back): {exc}") from exc

    def _delete_parent_cascade(self, parent_id: str) -> None:
        """删除 parent 行，由 ON DELETE CASCADE 级联清空 child。

        vec0 + FTS5 是虚拟表，不参与 FK CASCADE，需显式删除。
        04 票据后 FTS5 触发器已移除，FTS5 同步由本方法显式 DELETE。
        """
        # 先显式删 FTS5（需在 child 行 CASCADE 删除前，通过 parent_id 定位）
        self._write_conn.execute(
            "DELETE FROM child_fts WHERE parent_id = ?", (parent_id,)
        )
        # 显式删 vec0（vec0 无 FK CASCADE）
        self._write_conn.execute(
            "DELETE FROM child_vec WHERE parent_id = ?", (parent_id,)
        )
        # 删 parent，child 由 CASCADE 级联
        self._write_conn.execute(
            "DELETE FROM parent WHERE parent_id = ?", (parent_id,)
        )

    def _count_children(self, parent_id: str) -> int:
        """统计 parent 下的 child 数量。"""
        row = self._write_conn.execute(
            "SELECT COUNT(*) FROM child WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        return row[0] if row else 0

    # === 文档级删除 ===

    def delete_document(self, source: str) -> bool:
        """按 source 删除文档（级联清空四层）。

        按 §1 级联删除：删除 parent 行时 child + vec0 + FTS5 同事务级联清空。

        Returns:
            True: 文档存在并已删除
            False: 文档不存在（无副作用）
        """
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"source must be non-empty string, got: {source!r}")

        parent_id = parent_id_for(source)
        try:
            self._write_conn.execute("BEGIN IMMEDIATE")
            existing = self._write_conn.execute(
                "SELECT 1 FROM parent WHERE parent_id = ?", (parent_id,)
            ).fetchone()
            if existing is None:
                self._write_conn.execute("ROLLBACK")
                logger.info(f"delete skip: source={source} not found")
                return False
            self._delete_parent_cascade(parent_id)
            self._write_conn.execute("COMMIT")
            logger.info(f"delete ok: source={source} parent_id={parent_id}")
            return True
        except sqlite3.Error as exc:
            try:
                self._write_conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise sqlite3.OperationalError(f"delete failed (rolled back): {exc}") from exc

    # === 查询（仅供运营/调试用，检索由票据 04 实现）===

    def get_parent(self, source: str) -> dict[str, Any] | None:
        """按 source 查询 parent 记录（运营调试用）。"""
        parent_id = parent_id_for(source)
        row = self._write_conn.execute(
            """
            SELECT parent_id, source, file_name, content_hash, headings, status, created_at, updated_at
            FROM parent WHERE parent_id = ?
            """,
            (parent_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "parent_id": row[0],
            "source": row[1],
            "file_name": row[2],
            "content_hash": row[3],
            "headings": json.loads(row[4]),
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def list_parents(self, status_filter: Status | None = None) -> list[dict[str, Any]]:
        """列出 parent 记录（运营调试用）。

        10 票据 review No.2：增加 content 列返回，避免下游 list_documents
        用 content_hash[:8] 冒充 page_content。
        """
        if status_filter is not None and status_filter not in _VALID_STATUSES:
            raise ValueError(f"status_filter must be one of {sorted(_VALID_STATUSES)} or None")
        if status_filter:
            rows = self._write_conn.execute(
                """
                SELECT parent_id, source, file_name, content_hash, content, headings, status, created_at, updated_at
                FROM parent WHERE status = ? ORDER BY created_at DESC
                """,
                (status_filter,),
            ).fetchall()
        else:
            rows = self._write_conn.execute(
                """
                SELECT parent_id, source, file_name, content_hash, content, headings, status, created_at, updated_at
                FROM parent ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "parent_id": r[0],
                "source": r[1],
                "file_name": r[2],
                "content_hash": r[3],
                "content": r[4],
                "headings": json.loads(r[5]),
                "status": r[6],
                "created_at": r[7],
                "updated_at": r[8],
            }
            for r in rows
        ]

    def count_parents(self) -> int:
        """统计 parent 数量。"""
        row = self._write_conn.execute("SELECT COUNT(*) FROM parent").fetchone()
        return row[0] if row else 0

    def count_children(self) -> int:
        """统计 child 数量。"""
        row = self._write_conn.execute("SELECT COUNT(*) FROM child").fetchone()
        return row[0] if row else 0


__all__ = [
    "ChildChunk",
    "IngestResult",
    "SQLiteStore",
    "child_id_for",
    "content_hash_of",
    "parent_id_for",
    "serialize_embedding",
]
