"""票据 04：混合检索工具。

按 sqlite-storage-contract.md §7 返回契约实现：
- search_child_chunks: dense (sqlite-vec) + sparse (FTS5) + RRF 融合 + source/status 过滤
- retrieve_parent_chunks: 按 parent_id 回溯完整 parent 文本，未命中返回空列表 + has_more=False

本模块只负责检索，不负责写入（写入由 sqlite_store.py 实现）。
所有检索函数接受 sqlite3.Connection（已加载 sqlite-vec + init_schema）。

FTS5 sparse 检索采用 jieba 预分词 + unicode61 tokenizer（04 票据中文评测结论，
recall@5 79% vs trigram 25%）。写入时 sqlite_store.py 对 content 做 jieba.cut 分词；
查询时本模块对 query_text 做 jieba.cut 分词后送入 FTS5 MATCH。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

import jieba

from rag.sqlite_store import serialize_embedding

logger = logging.getLogger("ai-service.rag.hybrid_retrieval")

# RRF 默认常数（与既有 rag/text_processing.py 一致）
DEFAULT_RRF_K = 60


def _jieba_query(text: str) -> str:
    """对查询文本做 jieba 分词，构建 FTS5 MATCH 表达式。

    短查询（<=4 字符）：短语匹配（所有 token 必须连续出现），精确召回。
    长查询（>4 字符）：OR 连接，任一 token 出现即命中，召回更广。
    """
    tokens = [t for t in jieba.cut(text, cut_all=False) if t.strip()]
    if not tokens:
        return ""
    if len(text) <= 4:
        # 短查询：短语匹配（FTS5 phrase query，双引号包裹）
        escaped = " ".join(tokens).replace('"', '""')
        return f'"{escaped}"'
    # 长查询：OR 连接（FTS5 默认操作符）
    # 每个 token 用双引号包裹防止 FTS5 保留字冲突
    return " OR ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens)


def search_child_chunks(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    query_text: str,
    *,
    k: int = 4,
    source_filter: str | None = None,
    status_filter: str | None = None,
    min_score: float | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    embedding_dimensions: int | None = None,
) -> list[dict[str, Any]]:
    """对 child 做 dense + sparse 混合检索，RRF 融合返回 top-K。

    按 sqlite-storage-contract.md §7 返回契约：
    每条结果含 parent_id / child_id / file_name / source / status / content /
    score_dense / score_sparse / score_rrf 九个字段。

    Args:
        conn: 已加载 sqlite-vec + init_schema 的 SQLite 连接
        query_embedding: dense 查询向量（维度必须与库中 vec0 维度一致）
        query_text: sparse 查询文本（送入 FTS5 MATCH；空串跳过 sparse）
        k: 返回 top-K 条
        source_filter: 仅返回该 source 的结果（None = 不过滤）
        status_filter: 仅返回该 status 的结果（None = 不过滤）
        min_score: RRF 分数下限，低于此值的结果被过滤（None = 不过滤）
        rrf_k: RRF 常数，默认 60
        embedding_dimensions: 预期的向量维度，仅用于错误消息（None = 从 sqlite-vec 错误推断）

    Returns:
        按 score_rrf 降序排列的 top-K 结果列表。空列表表示无命中。

    Raises:
        ValueError: query_embedding 维度与库中 vec0 维度不匹配，或元素类型非法
    """
    if k <= 0:
        return []
    started = time.perf_counter()

    # 多取用于 RRF 融合（两榜各取 k*2，避免漏掉只在单榜出现的 relevant child）
    fetch_k = k * 2

    dense_rows = _dense_search(
        conn,
        query_embedding,
        k=fetch_k,
        source_filter=source_filter,
        status_filter=status_filter,
        embedding_dimensions=embedding_dimensions,
    )
    sparse_rows = _sparse_search(
        conn,
        query_text,
        k=fetch_k,
        source_filter=source_filter,
        status_filter=status_filter,
    )

    fused = _rrf_fuse(dense_rows, sparse_rows, rrf_k=rrf_k)

    if min_score is not None:
        fused = [r for r in fused if r["score_rrf"] >= min_score]
    pre_truncate = len(fused)
    fused = fused[:k]

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "search_child_chunks: dense=%d sparse=%d fused=%d returned=%d elapsed=%.2fms",
        len(dense_rows), len(sparse_rows), pre_truncate, len(fused), elapsed_ms,
    )

    return fused


def _dense_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    *,
    k: int,
    source_filter: str | None,
    status_filter: str | None,
    embedding_dimensions: int | None,
) -> list[dict[str, Any]]:
    """sqlite-vec dense 检索，返回带 score_dense 的行列表。"""
    try:
        emb_bytes = serialize_embedding(query_embedding)
    except ValueError as exc:
        raise ValueError(f"query_embedding invalid: {exc}") from exc

    sql = """
        SELECT c.child_id, c.parent_id, c.content,
               p.file_name, p.source, p.status,
               v.distance
        FROM child_vec v
        JOIN child c ON c.child_id = v.child_id
        JOIN parent p ON p.parent_id = v.parent_id
        WHERE v.embedding MATCH ? AND v.k = ?
    """
    params: list[Any] = [emb_bytes, k]
    if source_filter is not None:
        sql += " AND p.source = ?"
        params.append(source_filter)
    if status_filter is not None:
        sql += " AND p.status = ?"
        params.append(status_filter)
    sql += " ORDER BY v.distance"

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        # sqlite-vec 维度不匹配报错关键字：dim / dimension / match
        if "dim" in msg or "match" in msg or "shape" in msg:
            expected = embedding_dimensions if embedding_dimensions is not None else "?"
            raise ValueError(
                f"query_embedding dimension mismatch: expected {expected} "
                f"got {len(query_embedding)} (sqlite-vec: {exc})"
            ) from exc
        raise

    # vec0 只允许 ORDER BY distance（不支持多列），等距时顺序未定义。
    # 在 Python 层按 (distance, child_id) 二次排序，保证 rank 分配确定性。
    rows = sorted(rows, key=lambda r: (r[6], r[0]))

    # distance → 相似度分数（单调递减，便于调试阅读；RRF 只用 rank 不用此值）
    return [
        {
            "child_id": r[0],
            "parent_id": r[1],
            "content": r[2],
            "file_name": r[3],
            "source": r[4],
            "status": r[5],
            "score_dense": 1.0 / (1.0 + float(r[6])),
            "score_sparse": 0.0,
            "score_rrf": 0.0,
        }
        for r in rows
    ]


def _sparse_search(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    k: int,
    source_filter: str | None,
    status_filter: str | None,
) -> list[dict[str, Any]]:
    """FTS5 sparse 检索（jieba 预分词 + unicode61 tokenizer），返回带 score_sparse 的行列表。

    空查询返回空列表。FTS5 语法错误降级为空榜（不中断检索）。
    """
    if not query_text or not query_text.strip():
        return []

    fts_query = _jieba_query(query_text)
    if not fts_query:
        return []

    # child_fts 是 regular FTS5 表（非外部内容表），存 child_id + parent_id + tokenized content
    # 通过 child_id 关联 child 表获取原文，通过 parent_id 关联 parent 表获取元数据
    sql = """
        SELECT c.child_id, c.parent_id, c.content,
               p.file_name, p.source, p.status,
               f.rank
        FROM child_fts f
        JOIN child c ON c.child_id = f.child_id
        JOIN parent p ON p.parent_id = f.parent_id
        WHERE child_fts MATCH ?
    """
    params: list[Any] = [fts_query]
    if source_filter is not None:
        sql += " AND p.source = ?"
        params.append(source_filter)
    if status_filter is not None:
        sql += " AND p.status = ?"
        params.append(status_filter)
    sql += " ORDER BY f.rank, c.child_id LIMIT ?"
    params.append(k)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        # FTS5 语法错误（如保留字未转义）不应中断检索，降级为空榜
        logger.warning("sparse_search FTS5 query failed: %s (query=%r)", exc, query_text)
        return []

    # FTS5 rank 是负 BM25（越小越相关），取负转为正分数（越大越好）
    return [
        {
            "child_id": r[0],
            "parent_id": r[1],
            "content": r[2],
            "file_name": r[3],
            "source": r[4],
            "status": r[5],
            "score_dense": 0.0,
            "score_sparse": -float(r[6]) if r[6] is not None else 0.0,
            "score_rrf": 0.0,
        }
        for r in rows
    ]


def _rrf_fuse(
    dense_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    *,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion：按 child_id 融合两榜。

    RRF 公式：score_rrf = sum(1 / (rrf_k + rank_i)) 跨所有榜
    融合后按 score_rrf 降序排列；同分按 child_id 升序 tie-break，保证顺序稳定可复现。

    保留每个 child 在两榜的最佳分数（score_dense / score_sparse），
    只在单榜出现的 child 另一榜分数为 0.0。
    """
    by_child: dict[str, dict[str, Any]] = {}

    for rank, row in enumerate(dense_rows, start=1):
        cid = row["child_id"]
        if cid not in by_child:
            by_child[cid] = dict(row)  # 浅拷贝，避免污染输入
        else:
            by_child[cid]["score_dense"] = row["score_dense"]
        by_child[cid]["score_rrf"] = by_child[cid].get("score_rrf", 0.0) + 1.0 / (rrf_k + rank)

    for rank, row in enumerate(sparse_rows, start=1):
        cid = row["child_id"]
        if cid not in by_child:
            by_child[cid] = dict(row)
        else:
            by_child[cid]["score_sparse"] = row["score_sparse"]
        by_child[cid]["score_rrf"] = by_child[cid].get("score_rrf", 0.0) + 1.0 / (rrf_k + rank)

    return sorted(by_child.values(), key=lambda r: (-r["score_rrf"], r["child_id"]))


def retrieve_parent_chunks(
    conn: sqlite3.Connection,
    parent_ids: list[str],
) -> dict[str, Any]:
    """按 parent_id 列表回溯完整 parent 文本。

    按 sqlite-storage-contract.md §7 返回契约：
    {"parents": [{"parent_id", "file_name", "content", "headings"}, ...], "has_more": bool}

    未命中行为（哨兵值集合）：
    - 不存在的 parent_id 跳过（不抛异常）
    - 全部未命中或 parent_ids 为空 → {"parents": [], "has_more": False}
    - has_more 始终为 False（本实现不分页，一次返回全部命中）

    Args:
        conn: SQLite 连接
        parent_ids: parent_id 列表（通常来自 search_child_chunks 的返回值）

    Returns:
        按 parent_ids 输入顺序（去重保序）返回命中的 parent 记录。
    """
    if not parent_ids:
        return {"parents": [], "has_more": False}

    started = time.perf_counter()

    # 去重保序
    seen: set[str] = set()
    unique_ids: list[str] = []
    for pid in parent_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)

    placeholders = ",".join("?" * len(unique_ids))
    sql = f"""
        SELECT parent_id, file_name, content, headings
        FROM parent
        WHERE parent_id IN ({placeholders})
    """
    rows = conn.execute(sql, unique_ids).fetchall()
    row_by_id = {r[0]: r for r in rows}

    parents: list[dict[str, Any]] = []
    for pid in unique_ids:
        r = row_by_id.get(pid)
        if r is None:
            continue  # 不存在的 id 跳过
        parents.append({
            "parent_id": r[0],
            "file_name": r[1],
            "content": r[2],
            "headings": json.loads(r[3]) if r[3] else [],
        })

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "retrieve_parent_chunks: requested=%d hit=%d elapsed=%.2fms",
        len(unique_ids), len(parents), elapsed_ms,
    )

    return {"parents": parents, "has_more": False}


__all__ = [
    "DEFAULT_RRF_K",
    "retrieve_parent_chunks",
    "search_child_chunks",
]
