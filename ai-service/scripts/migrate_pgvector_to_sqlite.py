"""10 票据：pgvector → SQLite 迁移脚本。

按 10 票据验收标准：
- 迁移可重启：中途杀掉后续跑，不产生重复或遗漏
- 计数一致：迁移后 pgvector document_count 与 SQLite parent_count 比对
- 代表性查询结果可比（由 verify_migration.py 独立校验）
- cutover 为显式操作（由 cutover_storage.py 独立执行）

迁移策略：
1. 从 pgvector rag_documents 表读取所有文档（按 source 分组）
2. 每个 source 用 IngestionPipeline 重新切分（parent/child）+ 生成 embedding
3. 写入 SQLiteStore（幂等：同 source + 同 content_hash 跳过；中断重跑不重复不遗漏）
4. 进度持久化：data/migration-progress.json 记录已迁移 source 列表（断点续传）

幂等保证：
- SQLiteStore.upsert_document 的 C 复合身份策略：同 source + 同 content_hash → noop
- 同 source + 不同 content_hash → 替换（DELETE 旧四层 + INSERT 新四层）
- 中断重跑：progress.json 记录已完成 source，跳过；即使 progress.json 丢失，
  upsert_document 的 noop 机制也保证不重复写

用法：
    python -m scripts.migrate_pgvector_to_sqlite [--dry-run] [--batch-size N]
    python -m scripts.migrate_pgvector_to_sqlite --reset-progress  # 重置进度文件

注意：本脚本需要真实 embedding API key（pgvector 旧库的 chunks 不直接迁移，
而是用 IngestionPipeline 重新切分 + 重新 embedding，保留 parent-child 层级）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger("ai-service.scripts.migrate_pgvector_to_sqlite")


# === 抽象接口（供测试 mock）===


class _SourceDocument(Protocol):
    """pgvector 旧库中的一条 source 文档（聚合所有 chunks）。"""

    source: str
    file_name: str
    content: str  # 完整 markdown 文本（聚合所有 chunks）
    status: str


@dataclass(frozen=True)
class MigrationProgress:
    """迁移进度（持久化到 progress.json）。"""

    started_at: str
    last_updated_at: str
    total_sources: int  # pgvector 中的 source 总数（首次扫描时记录）
    migrated_sources: list[str]  # 已成功迁移的 source 列表
    failed_sources: list[dict[str, str]]  # 失败的 source + 原因
    skipped_sources: list[str]  # 跳过的 source（content_hash 未变，noop）


@dataclass(frozen=True)
class MigrationReport:
    """迁移报告（脚本输出）。"""

    started_at: str
    finished_at: str
    total_sources: int
    migrated_count: int  # 本次新迁移数
    skipped_count: int  # noop 跳过数
    failed_count: int
    failed_sources: list[dict[str, str]]
    pgvector_document_count: int
    sqlite_parent_count: int
    sqlite_child_count: int
    elapsed_seconds: float


# === pgvector 读取 ===


def _load_pgvector_sources(
    db_url: str,
    connect_timeout: int,
) -> tuple[list[_SourceDocument], int]:
    """从 pgvector 读取所有文档，按 source 聚合。

    Returns:
        (sources, total_document_count)
    """
    import psycopg  # 延迟导入，避免无 pgvector 环境导入失败

    conn = psycopg.connect(db_url, connect_timeout=connect_timeout)
    conn.autocommit = True
    try:
        # 读取所有文档
        rows = conn.execute(
            """
            SELECT id, content, metadata
            FROM rag_documents
            WHERE embedding IS NOT NULL
            ORDER BY metadata->>'source' NULLS LAST, id
            """
        ).fetchall()

        # 按 source 聚合（同 source 的多个 chunks 拼接为完整 content）
        grouped: dict[str, dict[str, Any]] = {}
        for row_id, content, metadata in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            metadata = metadata or {}
            source = metadata.get("source") or metadata.get("id") or row_id
            file_name = metadata.get("file_name") or metadata.get("title") or source
            status = metadata.get("status") or "active"

            if source not in grouped:
                grouped[source] = {
                    "source": source,
                    "file_name": file_name,
                    "content_parts": [],
                    "status": status,
                }
            grouped[source]["content_parts"].append(content)

        sources: list[_SourceDocument] = []
        for source_data in grouped.values():
            content = "\n\n".join(source_data["content_parts"])
            sources.append(
                _SourceDocumentImpl(
                    source=source_data["source"],
                    file_name=source_data["file_name"],
                    content=content,
                    status=source_data["status"],
                )
            )

        return sources, len(rows)
    finally:
        conn.close()


@dataclass(frozen=True)
class _SourceDocumentImpl:
    """_SourceDocument 的具体实现。"""

    source: str
    file_name: str
    content: str
    status: str


# === 进度持久化 ===


def _load_progress(progress_path: Path) -> MigrationProgress | None:
    """加载迁移进度（若存在）。"""
    if not progress_path.exists():
        return None
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return MigrationProgress(
            started_at=data["started_at"],
            last_updated_at=data["last_updated_at"],
            total_sources=data["total_sources"],
            migrated_sources=data["migrated_sources"],
            failed_sources=data["failed_sources"],
            skipped_sources=data["skipped_sources"],
        )
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("progress.json 解析失败，将重新开始: %s", exc)
        return None


def _save_progress(progress_path: Path, progress: MigrationProgress) -> None:
    """持久化迁移进度（原子写入：临时文件 + os.replace）。

    用 os.replace 保证写入原子性，避免 kill 在写入中途损坏 progress.json
    导致下次重跑从 0 开始（_load_progress 有 JSONDecodeError 兜底但会丢失断点）。
    """
    import os
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(progress), ensure_ascii=False, indent=2)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, progress_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# === 迁移核心 ===


def migrate(
    *,
    pgvector_url: str,
    pgvector_connect_timeout: int,
    sqlite_db_path: str,
    embedding_dimensions: int,
    progress_path: Path,
    batch_size: int = 50,
    dry_run: bool = False,
    embedding_provider: Any = None,  # 测试注入；None 时用 OpenAIEmbeddings
    reset_progress: bool = False,
    retry_failed: bool = False,
) -> MigrationReport:
    """执行 pgvector → SQLite 迁移。

    Args:
        pgvector_url: pgvector 数据库 URL
        pgvector_connect_timeout: 连接超时秒数
        sqlite_db_path: SQLite 文件路径
        embedding_dimensions: embedding 维度
        progress_path: 进度文件路径
        batch_size: 每批迁移的 source 数量（控制内存）
        dry_run: True 时只扫描不写入
        embedding_provider: 测试注入的 embedding 提供方
        reset_progress: True 时重置进度文件
        retry_failed: review No.9：True 时重试 progress.failed_sources 中的 source，
            不重置已完成进度。用于重试瞬时失败（如 embedding API 超时）。

    Returns:
        MigrationReport
    """
    started_at = _now_iso()
    started_perf = time.perf_counter()

    # 1. 加载/初始化进度
    if reset_progress and progress_path.exists():
        progress_path.unlink()
        logger.info("已重置进度文件: %s", progress_path)

    progress = _load_progress(progress_path)

    # 2. 从 pgvector 读取所有 source
    logger.info("从 pgvector 读取文档: %s", pgvector_url)
    sources, pgvector_doc_count = _load_pgvector_sources(
        pgvector_url, pgvector_connect_timeout
    )
    logger.info(
        "pgvector 读取完成: %d 个 source, %d 条 document",
        len(sources), pgvector_doc_count,
    )

    if progress is None:
        progress = MigrationProgress(
            started_at=started_at,
            last_updated_at=started_at,
            total_sources=len(sources),
            migrated_sources=[],
            failed_sources=[],
            skipped_sources=[],
        )

    # 3. 初始化 SQLite 写入端
    if dry_run:
        logger.info("dry-run 模式：不写入 SQLite")
        sqlite_store = None
    else:
        from rag.sqlite_store import SQLiteStore
        from rag.ingestion import IngestionPipeline, EmbeddingProvider as _EP

        sqlite_store = SQLiteStore(sqlite_db_path, embedding_dimensions=embedding_dimensions)

        # embedding provider 初始化
        if embedding_provider is None:
            from langchain_openai import OpenAIEmbeddings
            from config import settings
            embedding_provider = OpenAIEmbeddings(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
            )

        pipeline = IngestionPipeline(store=sqlite_store, embedding_provider=embedding_provider)

    # review No.4：try/finally 保证 sqlite_store 在异常时也被关闭，避免文件锁残留
    try:
        # 4. 分批迁移（断点续传：跳过已完成的 source）
        migrated_set = set(progress.migrated_sources)
        skipped_set = set(progress.skipped_sources)
        failed_set = {item["source"] for item in progress.failed_sources}

        # review No.9：retry_failed=True 时，把 failed_sources 重新加入 pending 重试
        # （不重置已完成进度，只重试失败项）
        if retry_failed and failed_set:
            retry_sources = {s for s in failed_set}
            logger.info("retry_failed 模式：重试 %d 个失败 source", len(retry_sources))
            failed_set -= retry_sources  # 从 failed_set 移除，使其重新进入 pending
            # frozen dataclass 不能修改属性，用 dataclasses.replace 创建新实例
            from dataclasses import replace as _dc_replace
            progress = _dc_replace(
                progress,
                failed_sources=[
                    item for item in progress.failed_sources
                    if item["source"] not in retry_sources
                ],
            )

        pending = [
            s for s in sources
            if s.source not in migrated_set
            and s.source not in failed_set
        ]
        logger.info(
            "待迁移 source: %d (已完成 %d, 跳过 %d, 失败 %d)",
            len(pending), len(migrated_set), len(skipped_set), len(failed_set),
        )

        new_migrated: list[str] = []
        new_skipped: list[str] = []
        new_failed: list[dict[str, str]] = []

        for i, source_doc in enumerate(pending):
            if dry_run:
                logger.info("[dry-run] %d/%d source=%s", i + 1, len(pending), source_doc.source)
                continue

            try:
                # 用 IngestionPipeline 重新切分 + embedding + 写入
                # IngestionPipeline.ingest_file 接受文件路径，但我们的 source 已经是 content
                # 所以直接调用其内部步骤
                from rag.ingestion import (
                    split_parent_chunks,
                    split_child_chunks,
                )
                from rag.sqlite_store import ChildChunk

                # parent 切分
                parents = split_parent_chunks(source_doc.content)

                # child 切分 + embedding（批量）
                all_child_chunks: list[ChildChunk] = []
                child_index = 0
                for parent in parents:
                    child_texts = split_child_chunks(parent.content)
                    if not child_texts:
                        continue
                    embeddings = embedding_provider.embed_texts(child_texts)
                    if len(embeddings) != len(child_texts):
                        raise RuntimeError(
                            f"embedding count mismatch: expected {len(child_texts)}, "
                            f"got {len(embeddings)}"
                        )
                    for text, emb in zip(child_texts, embeddings):
                        all_child_chunks.append(
                            ChildChunk(content=text, embedding=emb, child_index=child_index)
                        )
                        child_index += 1

                if not all_child_chunks:
                    raise RuntimeError(f"no child chunks produced from source={source_doc.source}")

                # 写入 SQLiteStore（幂等：同 source + 同 content_hash → noop）
                # 注：把 source 当作 file_path 调用 upsert_document（不走 ingest_file 的文件解析）
                headings_all: list[str] = []
                for p in parents:
                    headings_all.extend(p.headings)
                seen: set[str] = set()
                headings_unique = [h for h in headings_all if not (h in seen or seen.add(h))]

                # review No.7：归一化非标准 status，避免 pgvector 旧库的 draft/deleted
                # 等非标准 status 导致 upsert_document 抛 ValueError
                from rag.sqlite_store import _VALID_STATUSES
                migrate_status = source_doc.status if source_doc.status in _VALID_STATUSES else "archived"
                if migrate_status != source_doc.status:
                    logger.warning(
                        "source=%s 非标准 status=%r 归一化为 'archived'",
                        source_doc.source, source_doc.status,
                    )

                ingest_result = sqlite_store.upsert_document(
                    source=source_doc.source,
                    file_name=source_doc.file_name,
                    parent_content=source_doc.content,
                    headings=headings_unique,
                    child_chunks=all_child_chunks,
                    status=migrate_status,  # type: ignore[arg-type]
                )

                if ingest_result.action == "noop":
                    new_skipped.append(source_doc.source)
                    skipped_set.add(source_doc.source)
                    logger.info(
                        "[%d/%d] noop source=%s (content_hash 未变)",
                        i + 1, len(pending), source_doc.source,
                    )
                else:
                    new_migrated.append(source_doc.source)
                    migrated_set.add(source_doc.source)
                    logger.info(
                        "[%d/%d] %s source=%s children=%d",
                        i + 1, len(pending), ingest_result.action,
                        source_doc.source, ingest_result.child_count,
                    )

            except Exception as exc:
                logger.exception("迁移失败 source=%s", source_doc.source)
                new_failed.append({"source": source_doc.source, "reason": str(exc)})
                failed_set.add(source_doc.source)

            # 批量持久化进度（每 batch_size 或最后一条）
            if (i + 1) % batch_size == 0 or i + 1 == len(pending):
                if not dry_run:
                    progress = MigrationProgress(
                        started_at=progress.started_at,
                        last_updated_at=_now_iso(),
                        total_sources=progress.total_sources,
                        migrated_sources=sorted(migrated_set),
                        failed_sources=sorted(
                            list(new_failed) + [item for item in progress.failed_sources if item["source"] in failed_set],
                            key=lambda x: x["source"],
                        ),
                        skipped_sources=sorted(skipped_set),
                    )
                    _save_progress(progress_path, progress)
                    logger.info(
                        "进度已持久化: migrated=%d skipped=%d failed=%d",
                        len(progress.migrated_sources),
                        len(progress.skipped_sources),
                        len(progress.failed_sources),
                    )
    finally:
        # 5. 关闭 SQLite 连接，统计最终计数
        sqlite_parent_count = 0
        sqlite_child_count = 0
        if sqlite_store is not None:
            try:
                sqlite_parent_count = sqlite_store.count_parents()
                sqlite_child_count = sqlite_store.count_children()
            except Exception as exc:
                logger.warning("统计 SQLite 计数失败: %s", exc)
            finally:
                sqlite_store.close()

    finished_at = _now_iso()
    elapsed = time.perf_counter() - started_perf

    return MigrationReport(
        started_at=started_at,
        finished_at=finished_at,
        total_sources=len(sources),
        migrated_count=len(new_migrated),
        skipped_count=len(new_skipped),
        failed_count=len(new_failed),
        failed_sources=new_failed,
        pgvector_document_count=pgvector_doc_count,
        sqlite_parent_count=sqlite_parent_count,
        sqlite_child_count=sqlite_child_count,
        elapsed_seconds=elapsed,
    )


# === CLI 入口 ===


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="pgvector → SQLite 迁移脚本（10 票据）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只扫描不写入 SQLite",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="每批迁移的 source 数量（控制进度持久化频率）",
    )
    parser.add_argument(
        "--reset-progress", action="store_true",
        help="重置进度文件后重新开始",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="review No.9：重试 progress.failed_sources 中的 source（不重置已完成进度）",
    )
    parser.add_argument(
        "--pgvector-url", default=None,
        help="pgvector 数据库 URL（默认从 config.settings.database_url 读取）",
    )
    parser.add_argument(
        "--sqlite-db-path", default=None,
        help="SQLite 文件路径（默认从 config.settings.rag_sqlite_db_path 读取）",
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

    pgvector_url = args.pgvector_url or settings.database_url
    sqlite_db_path = args.sqlite_db_path or settings.rag_sqlite_db_path
    progress_path = Path(settings.rag_migration_progress_path)

    report = migrate(
        pgvector_url=pgvector_url,
        pgvector_connect_timeout=settings.database_connect_timeout_seconds,
        sqlite_db_path=sqlite_db_path,
        embedding_dimensions=settings.embedding_dimensions,
        progress_path=progress_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        reset_progress=args.reset_progress,
        retry_failed=args.retry_failed,
    )

    # 输出报告
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))

    if report.failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
