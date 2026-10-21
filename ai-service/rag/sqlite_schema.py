"""票据 02：SQLite 父子分层检索 schema 初始化。

按 sqlite-storage-contract.md 冻结结论实现（04 票据中文评测后 FTS5 tokenizer 更新）：
- parent/child/vec0/FTS5 四层结构
- 稳定 ID：parent_id=uuid5(NAMESPACE_URL, source)，child_id=uuid5(NAMESPACE_URL, parent_id+':'+index)
- source/status 语义：status ∈ {active, archived, feishu, cloud_docs}
- vec0 维度参数化（默认 1536，由调用方传入）
- FTS5 regular 表 + unicode61 tokenizer + jieba 预分词（04 评测结论，recall@5 79% vs 25%）
  - 原 trigram + 外部内容表 + 触发器方案对 2 字符中文查询 recall=0
  - 改为应用层同步（sqlite_store.py 负责 jieba 分词 + FTS5 insert/delete）
- ON DELETE CASCADE 级联删除（child 表）；vec0 + FTS5 由应用层显式删除
- PRAGMA journal_mode=WAL + busy_timeout=5000

本模块只负责 schema 初始化，不涉及读写逻辑。事务边界、幂等、维度校验由 sqlite_store.py 实现。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import sqlite_vec

# vec0 维度占位符：实际由调用方传入。契约示例为 1536。
DEFAULT_EMBEDDING_DIMENSIONS = 1536

# 建表 SQL（按契约 §Schema 总览实现，增加 content 列以支持 retrieve_parent_chunks）
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS parent (
    parent_id TEXT PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    headings TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS child (
    child_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    child_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES parent(parent_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_child_parent_id ON child(parent_id);
"""

# FTS5 表（04 票据中文评测后改为 unicode61 + jieba 预分词，应用层同步）
# 原 trigram + 外部内容表 + 触发器方案对 2 字符中文查询 recall=0（评测证据见
# .scratch/rag-lifecycle-refactor/eval/fts5-chinese-eval.md），改为：
# - tokenize='unicode61'（按空白分词）
# - 写入前 jieba.cut 分词，空格连接（由 sqlite_store.py 应用层完成）
# - regular FTS5 表（非外部内容表），存 child_id + parent_id + tokenized content
# - 无触发器，FTS5 同步由 sqlite_store.py upsert_document / _delete_parent_cascade 负责
_FTS5_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS child_fts USING fts5(
    child_id UNINDEXED,
    parent_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

# vec0 虚拟表维度参数化（占位符由 init_schema 替换）
_VEC0_SQL_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS child_vec USING vec0(
    child_id TEXT PRIMARY KEY,
    parent_id TEXT,
    embedding float[{dimensions}]
);
"""


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """加载 sqlite-vec 扩展。幂等：重复加载不报错。"""
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """应用 WAL + busy_timeout + foreign_keys PRAGMA。

    按 sqlite-storage-contract §5 WAL 并发契约：
    - journal_mode=WAL：多读不阻塞写
    - busy_timeout=5000：5 秒超时
    - foreign_keys=ON：启用 ON DELETE CASCADE
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")


def init_schema(conn: sqlite3.Connection, embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
    """初始化 parent/child/vec0/FTS5 四层 schema。

    幂等：所有 CREATE 语句带 IF NOT EXISTS，重复调用不抛错。
    事务原子：全部 DDL 在单事务内提交。

    Args:
        conn: SQLite 连接（已加载 sqlite-vec 扩展）
        embedding_dimensions: vec0 维度，必须与写入向量维度一致

    Raises:
        ValueError: embedding_dimensions 非正整数
        sqlite3.OperationalError: schema 创建失败
    """
    if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
        raise ValueError(
            f"embedding_dimensions must be positive int, got: {embedding_dimensions!r}"
        )

    _load_sqlite_vec(conn)
    apply_pragmas(conn)

    vec0_sql = _VEC0_SQL_TEMPLATE.format(dimensions=embedding_dimensions)

    # 全部 DDL 在单事务内（sqlite3 默认 autocommit=off，execute 后进入隐式事务）
    conn.executescript(_SCHEMA_SQL)
    conn.executescript(_FTS5_SQL)
    conn.executescript(vec0_sql)
    conn.commit()

    # 校验 vec0 表已创建（实际维度校验在写入路径由 sqlite-vec 强制）
    _verify_vec0_table_exists(conn)


def _verify_vec0_table_exists(conn: sqlite3.Connection) -> None:
    """校验 child_vec 虚拟表已创建。

    注意：sqlite-vec vec0 虚拟表不暴露维度查询接口，实际维度校验由写入时的
    OperationalError 强制（见 sqlite_store.py 的 upsert_document 维度校验）。
    本函数只做 schema 完整性检查（表存在），不验证维度。
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='child_vec'"
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("child_vec virtual table not created")


_VEC0_DIMENSIONS_RE = re.compile(r"float\[(\d+)\]")


def read_vec0_dimensions(db_path: Path | str | None) -> int | None:
    """读取现有索引里 child_vec 声明的向量维度。

    sqlite-vec 不暴露维度查询接口，只能从 sqlite_master 的建表 DDL
    （`embedding float[N]`）解析。文件/表不存在、DDL 不含维度或解析失败
    均返回 None，不抛异常、不创建数据库文件。
    """
    if not db_path:
        return None
    path = Path(db_path)
    try:
        if not path.is_file():
            return None
        conn: sqlite3.Connection | None = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1
        )
    except (OSError, ValueError, sqlite3.Error):
        return None

    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'child_vec'"
        ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass

    if not row or not row[0]:
        return None
    match = _VEC0_DIMENSIONS_RE.search(row[0])
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def connect(
    db_path: Path | str,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    *,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """打开 SQLite 连接并初始化 schema。

    工厂函数：业务层用本函数获取就绪的连接，无需手动 init_schema。
    返回的连接已应用 PRAGMA + 加载扩展 + 创建 schema。

    Args:
        db_path: SQLite 文件路径；':memory:' 用于测试
        embedding_dimensions: vec0 维度
        check_same_thread: 透传 sqlite3.connect；False 允许跨线程使用同一连接
            （调用方需自行加锁保证不并发写）

    Returns:
        就绪的 sqlite3.Connection
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    try:
        init_schema(conn, embedding_dimensions=embedding_dimensions)
    except Exception:
        conn.close()
        raise
    return conn


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "apply_pragmas",
    "connect",
    "init_schema",
    "read_vec0_dimensions",
]
