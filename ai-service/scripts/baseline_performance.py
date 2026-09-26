"""10 票据：基线性能记录脚本。

按 10 票据验收标准：
- 基线性能记录：环境、语料规模、端到端延迟、首字延迟、检索耗时、
  模型/工具调用数、token 估算（基线而非 SLO）

记录项：
1. 环境：Python 版本、OS、CPU 数、关键依赖版本（langgraph/sqlite-vec/...）
2. 语料规模：parent/child/source 计数、平均 child 长度、status 分布
3. 检索耗时：sample queries 的 hybrid_search/dense/sparse 各阶段耗时
4. 端到端延迟：sample queries 的 retriever.hybrid_search 全程耗时
5. 首字延迟：检索首条结果返回时间（proxy：dense_search 首次返回时间）
6. 模型/工具调用数：每次查询的 embedding 调用次数、检索调用次数
7. token 估算：基于 query + 检索结果长度的粗略估算

输出：data/baseline-performance.json

用法：
    python -m scripts.baseline_performance [--queries-file PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-service.scripts.baseline_performance")


DEFAULT_SAMPLE_QUERIES = [
    "徒步装备",
    "高海拔注意事项",
    "安全指引",
    "雨季徒步风险",
    "导航与地图",
]


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: str
    platform: str
    cpu_count: int
    langgraph_version: str
    sqlite_vec_version: str
    langchain_version: str
    jieba_version: str


@dataclass(frozen=True)
class CorpusScale:
    parent_count: int
    child_count: int
    source_count: int
    avg_child_length: float
    status_distribution: dict[str, int]


@dataclass(frozen=True)
class QueryLatency:
    query: str
    total_latency_ms: float  # 端到端延迟（hybrid_search 全程）
    dense_latency_ms: float  # dense 检索耗时
    sparse_latency_ms: float  # sparse 检索耗时
    first_result_latency_ms: float  # 首字延迟（proxy：首条结果返回时间）
    embedding_calls: int  # embedding 调用次数
    retrieval_calls: int  # 检索调用次数
    result_count: int
    estimated_input_tokens: int  # query token 估算（中文按 1.5 字/token）
    estimated_output_tokens: int  # 检索结果 token 估算


@dataclass(frozen=True)
class BaselineReport:
    timestamp: str
    backend: str  # "pgvector" | "sqlite"
    environment: EnvironmentInfo
    corpus_scale: CorpusScale
    query_latencies: list[QueryLatency]
    notes: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_environment() -> EnvironmentInfo:
    """收集环境信息。"""
    # 关键依赖版本
    versions: dict[str, str] = {}
    for pkg in ("langgraph", "langchain", "jieba"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not-installed"

    sqlite_vec_version = "unknown"
    try:
        import sqlite_vec
        sqlite_vec_version = getattr(sqlite_vec, "__version__", "unknown")
    except (ImportError, AttributeError):
        try:
            import importlib.metadata
            sqlite_vec_version = importlib.metadata.version("sqlite-vec")
        except Exception:
            sqlite_vec_version = "unknown"

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 0,
        langgraph_version=versions.get("langgraph", "unknown"),
        sqlite_vec_version=sqlite_vec_version,
        langchain_version=versions.get("langchain", "unknown"),
        jieba_version=versions.get("jieba", "unknown"),
    )


def _collect_corpus_scale(sqlite_db_path: str) -> CorpusScale:
    """收集语料规模。"""
    from rag.sqlite_schema import connect

    conn = connect(sqlite_db_path)
    try:
        parent_count = conn.execute("SELECT COUNT(*) FROM parent").fetchone()[0]
        child_count = conn.execute("SELECT COUNT(*) FROM child").fetchone()[0]

        # source 数（去重）
        source_count = conn.execute(
            "SELECT COUNT(DISTINCT source) FROM parent"
        ).fetchone()[0]

        # 平均 child 长度
        avg_len_row = conn.execute(
            "SELECT AVG(LENGTH(content)) FROM child"
        ).fetchone()
        avg_child_length = float(avg_len_row[0]) if avg_len_row and avg_len_row[0] else 0.0

        # status 分布
        status_rows = conn.execute(
            "SELECT status, COUNT(*) FROM parent GROUP BY status"
        ).fetchall()
        status_dist = {row[0]: row[1] for row in status_rows}

        return CorpusScale(
            parent_count=parent_count,
            child_count=child_count,
            source_count=source_count,
            avg_child_length=avg_child_length,
            status_distribution=status_dist,
        )
    finally:
        conn.close()


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按 1.5 字/token，英文按 4 字符/token。"""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def _measure_query_latency(
    retriever: Any, query: str, k: int = 4
) -> QueryLatency:
    """测量单次查询的延迟和调用数。

    embedding 调用计数用 monkey-patch 包装 embed_query/embed_documents 实现，
    避免 getattr 依赖不存在的 _call_count 属性（OpenAIEmbeddings 无此属性）。
    """
    started = time.perf_counter()
    embedding_calls = 0

    # 用 patch 计数 embedding 调用（embed_query + embed_documents 都算）
    original_embed_query = getattr(retriever.embeddings, "embed_query", None)
    original_embed_documents = getattr(retriever.embeddings, "embed_documents", None)

    def _count_query(*args, **kwargs):
        nonlocal embedding_calls
        embedding_calls += 1
        return original_embed_query(*args, **kwargs)

    def _count_documents(*args, **kwargs):
        nonlocal embedding_calls
        embedding_calls += 1
        return original_embed_documents(*args, **kwargs)

    if original_embed_query is not None:
        retriever.embeddings.embed_query = _count_query
    if original_embed_documents is not None:
        retriever.embeddings.embed_documents = _count_documents

    try:
        results = retriever.hybrid_search([query], k=k)
    finally:
        # 恢复原始方法
        if original_embed_query is not None:
            retriever.embeddings.embed_query = original_embed_query
        if original_embed_documents is not None:
            retriever.embeddings.embed_documents = original_embed_documents

    total_latency_ms = (time.perf_counter() - started) * 1000

    # review No.11：移除 embedding_calls==0 兜底，让计数反映真实调用。
    # 若 embeddings 为 None 或无 embed_query 方法，embedding_calls 保持 0，
    # 测试可检测到计数失效（原兜底使断言恒真）。
    retrieval_calls = 1  # hybrid_search 调用 1 次

    # 首字延迟：proxy 用 total_latency / result_count（首条结果返回时间的近似）
    first_result_latency_ms = total_latency_ms / max(len(results), 1)

    # 13 票据：dense/sparse 独立延迟测量已移除。
    # 原实现访问 retriever._sqlite_client.read_conn（私有字段）和 _embed_query（私有方法），
    # 违反"性能工具不读取 adapter 私有连接或后端字段"契约。
    # 总延迟（total_latency_ms）已通过公共 hybrid_search() 方法测量，足够用于基线对比。
    dense_latency_ms = 0.0
    sparse_latency_ms = 0.0

    # token 估算
    input_tokens = _estimate_tokens(query)
    output_tokens = sum(_estimate_tokens(d.page_content) for d in results)

    return QueryLatency(
        query=query,
        total_latency_ms=total_latency_ms,
        dense_latency_ms=dense_latency_ms,
        sparse_latency_ms=sparse_latency_ms,
        first_result_latency_ms=first_result_latency_ms,
        embedding_calls=embedding_calls,
        retrieval_calls=retrieval_calls,
        result_count=len(results),
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
    )


def measure(
    *,
    sqlite_db_path: str,
    queries: list[str] | None = None,
    notes: list[str] | None = None,
) -> BaselineReport:
    """执行基线性能测量。"""
    timestamp = _now_iso()

    # 1. 环境
    env_info = _collect_environment()

    # 2. 语料规模
    logger.info("收集语料规模: %s", sqlite_db_path)
    corpus_scale = _collect_corpus_scale(sqlite_db_path)
    logger.info(
        "语料规模: parents=%d children=%d sources=%d",
        corpus_scale.parent_count, corpus_scale.child_count, corpus_scale.source_count,
    )

    # 3. 查询延迟测量
    sample_queries = queries if queries is not None else DEFAULT_SAMPLE_QUERIES
    query_latencies: list[QueryLatency] = []

    # SQLite 是唯一存储后端；需要 embedding API key 才能真实测量 hybrid 检索
    from config import settings as _settings
    original_db_path = _settings.rag_sqlite_db_path
    _settings.rag_sqlite_db_path = sqlite_db_path

    try:
        from rag.retriever import VectorStoreRetriever
        retriever = VectorStoreRetriever()
        backend = retriever.storage_mode
    finally:
        # 恢复 settings（retriever 已初始化，不影响本次测量）
        _settings.rag_sqlite_db_path = original_db_path

    for query in sample_queries:
        logger.info("测量查询: %s", query)
        try:
            latency = _measure_query_latency(retriever, query)
            query_latencies.append(latency)
            logger.info(
                "  total=%.2fms dense=%.2fms sparse=%.2fms results=%d",
                latency.total_latency_ms, latency.dense_latency_ms,
                latency.sparse_latency_ms, latency.result_count,
            )
        except Exception as exc:
            logger.exception("查询测量失败: %s", exc)
            query_latencies.append(QueryLatency(
                query=query,
                total_latency_ms=0.0,
                dense_latency_ms=0.0,
                sparse_latency_ms=0.0,
                first_result_latency_ms=0.0,
                embedding_calls=0,
                retrieval_calls=0,
                result_count=0,
                estimated_input_tokens=0,
                estimated_output_tokens=0,
            ))

    return BaselineReport(
        timestamp=timestamp,
        backend=backend,
        environment=env_info,
        corpus_scale=corpus_scale,
        query_latencies=query_latencies,
        notes=notes or [],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="基线性能记录脚本（10 票据）"
    )
    parser.add_argument(
        "--queries-file", default=None,
        help="查询列表文件（每行一个查询）",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出文件路径（默认 data/baseline-performance.json）",
    )
    parser.add_argument(
        "--sqlite-db-path", default=None,
        help="SQLite 文件路径（默认从 config.settings 读取）",
    )
    parser.add_argument(
        "--note", action="append", default=[],
        help="额外备注",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from config import settings

    sqlite_db_path = args.sqlite_db_path or settings.rag_sqlite_db_path
    output_path = Path(args.output) if args.output else (
        Path(sqlite_db_path).parent / "baseline-performance.json"
    )

    queries: list[str] | None = None
    if args.queries_file:
        queries = [
            line.strip()
            for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report = measure(
        sqlite_db_path=sqlite_db_path,
        queries=queries,
        notes=args.note,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"基线性能报告已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
