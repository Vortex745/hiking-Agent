"""10 票据：显式 cutover 脚本 — 从 pgvector 切换到 SQLite 后端。

按 10 票据验收标准：
- cutover 为显式操作并留痕
- 旧库保留至验收通过后按约定处置
- 回滚路径经测试（由 rollback_cutover.py 承接）

14 票据扩展（AC3/AC4）：
- 切换前必须消费成功的迁移验证证据（migration-progress + verify-report）
- 重复切换、强制覆盖和回滚具有确定、幂等且可审计的行为
- --force 覆盖时保留旧 marker 到审计日志（cutover-audit-log.jsonl）

cutover 留痕方式：
- 写入 data/cutover-marker.json：时间戳、计数、查询对比结果、操作者、证据引用
- 业务层（VectorStoreRetriever）通过 backend_resolver 读取 marker 决定后端（14 票据）
- 不删除旧库（pgvector），保留至验收通过后由运维处置

cutover 前置检查：
1. 迁移进度文件存在且 failed_sources 为空
2. 验证报告（verify_migration.py 输出）overall_pass=True（若提供）
3. 未存在 cutover-marker.json（避免重复 cutover，--force 可覆盖并审计旧 marker）

用法：
    python -m scripts.cutover_storage --operator <name> [--verify-report PATH]
    python -m scripts.cutover_storage --force  # 跳过前置检查（紧急切换，旧 marker 审计保留）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-service.scripts.cutover_storage")


@dataclass(frozen=True)
class CutoverMarker:
    """cutover 标记文件内容（留痕）。"""

    cutover_at: str  # ISO 8601 时间戳
    operator: str  # 操作者标识
    backend: str  # "sqlite"（cutover 后的后端）
    previous_backend: str  # "pgvector"（cutover 前的后端）
    pgvector_source_count: int
    sqlite_parent_count: int
    sqlite_child_count: int
    verify_report_path: str  # 验证报告文件路径（若提供）
    migration_progress_path: str  # 迁移进度文件路径
    notes: list[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_migration_progress(progress_path: Path) -> dict[str, Any] | None:
    """读取迁移进度文件。"""
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("migration-progress.json 解析失败: %s", exc)
        return None


def _read_verify_report(report_path: Path | None) -> dict[str, Any] | None:
    """读取验证报告文件。"""
    if report_path is None or not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("verify report 解析失败: %s", exc)
        return None


def cutover(
    *,
    operator: str,
    marker_path: Path,
    progress_path: Path,
    verify_report_path: Path | None = None,
    force: bool = False,
    notes: list[str] | None = None,
    audit_log_path: Path | None = None,
) -> CutoverMarker:
    """执行 cutover，写入标记文件。

    Args:
        operator: 操作者标识（如运维姓名/CI 用户名）
        marker_path: cutover 标记文件路径
        progress_path: 迁移进度文件路径
        verify_report_path: 验证报告文件路径（可选）
        force: True 时跳过前置检查
        notes: 额外备注
        audit_log_path: 14 票据 AC4：审计日志路径（追加写入）；None 时跳过审计
            默认由 CLI main 设置为 marker 同目录下的 cutover-audit-log.jsonl

    Returns:
        CutoverMarker

    Raises:
        RuntimeError: 前置检查失败且 force=False
    """
    cutover_at = _now_iso()
    all_notes: list[str] = list(notes or [])

    # 1. 前置检查：未存在 marker
    # 14 票据 AC4：--force 覆盖时保留旧 marker 到审计日志
    previous_marker_content: dict | None = None
    if marker_path.exists():
        if force:
            # 读取旧 marker 内容用于审计
            try:
                previous_marker_content = json.loads(
                    marker_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("读取旧 marker 失败（审计将记录 None）: %s", exc)
                previous_marker_content = None
            all_notes.append("已存在 cutover-marker.json，--force 覆盖（旧 marker 已审计）")
            logger.warning("已存在 cutover-marker.json，--force 覆盖（旧 marker 已审计）")
        else:
            raise RuntimeError(
                f"cutover-marker.json 已存在: {marker_path}；如需重新 cutover 请先删除或用 --force"
            )

    # 2. 前置检查：迁移进度
    progress_data = _read_migration_progress(progress_path)
    pgvector_source_count = 0
    sqlite_parent_count = 0
    sqlite_child_count = 0

    if progress_data is not None:
        pgvector_source_count = progress_data.get("total_sources", 0)
        failed_sources = progress_data.get("failed_sources", [])
        migrated_sources = progress_data.get("migrated_sources", [])
        all_notes.append(
            f"迁移进度: total={pgvector_source_count} migrated={len(migrated_sources)} "
            f"failed={len(failed_sources)}"
        )

        if failed_sources and not force:
            raise RuntimeError(
                f"迁移存在 {len(failed_sources)} 个失败 source，拒绝 cutover；"
                f"用 --force 强制切换"
            )
    elif not force:
        raise RuntimeError(
            f"迁移进度文件不存在: {progress_path}；用 --force 强制切换"
        )

    # 3. 前置检查：验证报告
    verify_report = _read_verify_report(verify_report_path)
    if verify_report is not None:
        overall_pass = verify_report.get("overall_pass", False)
        all_notes.append(f"验证报告 overall_pass={overall_pass}")
        if not overall_pass and not force:
            raise RuntimeError(
                "验证报告 overall_pass=False，拒绝 cutover；用 --force 强制切换"
            )
        # 从验证报告读取计数
        count_comp = verify_report.get("count_comparison", {})
        sqlite_parent_count = count_comp.get("sqlite_parent_count", 0)
        sqlite_child_count = count_comp.get("sqlite_child_count", 0)
        if pgvector_source_count == 0:
            pgvector_source_count = count_comp.get("pgvector_source_count", 0)
    elif not force:
        all_notes.append("未提供验证报告，跳过 overall_pass 检查（建议提供）")

    # 4. 写入 cutover 标记
    marker = CutoverMarker(
        cutover_at=cutover_at,
        operator=operator,
        backend="sqlite",
        previous_backend="pgvector",
        pgvector_source_count=pgvector_source_count,
        sqlite_parent_count=sqlite_parent_count,
        sqlite_child_count=sqlite_child_count,
        verify_report_path=str(verify_report_path) if verify_report_path else "",
        migration_progress_path=str(progress_path),
        notes=all_notes,
    )

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    # review No.3：原子写入（tmp + os.replace），避免 kill 中途损坏 marker 文件
    import os as _os
    content = json.dumps(asdict(marker), ensure_ascii=False, indent=2)
    tmp_path = marker_path.with_suffix(marker_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    _os.replace(tmp_path, marker_path)
    logger.info("cutover 标记已写入: %s", marker_path)

    # 14 票据 AC4：写入审计日志（追加模式），保留旧 marker 内容便于审计追溯
    if audit_log_path is not None:
        audit_entry = {
            "timestamp": cutover_at,
            "operator": operator,
            "action": "force_overwrite" if previous_marker_content is not None else "create",
            "marker_path": str(marker_path),
            "previous_marker": previous_marker_content,
            "new_marker": asdict(marker),
        }
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        logger.info("cutover 审计日志已追加: %s", audit_log_path)

    # 5. 输出环境变量提示（不自动设置，留给运维手动设置）
    print(
        f"cutover 完成。runtime 通过 backend_resolver 自动读取 marker 切换后端（14 票据）。\n"
        f"标记文件: {marker_path}"
    )

    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="显式 cutover：pgvector → SQLite（10 票据）"
    )
    parser.add_argument(
        "--operator", required=True,
        help="操作者标识（如运维姓名/CI 用户名）",
    )
    parser.add_argument(
        "--verify-report", default=None,
        help="验证报告文件路径（verify_migration.py 输出）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="跳过前置检查，强制 cutover（紧急切换）",
    )
    parser.add_argument(
        "--note", action="append", default=[],
        help="额外备注（可多次指定）",
    )
    parser.add_argument(
        "--audit-log-path", default=None,
        help="14 票据 AC4：审计日志文件路径（默认 marker 同目录下 cutover-audit-log.jsonl）",
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

    marker_path = Path(settings.rag_cutover_marker_path)
    progress_path = Path(settings.rag_migration_progress_path)
    verify_report_path = Path(args.verify_report) if args.verify_report else None
    # 14 票据 AC4：默认审计日志路径 = marker 同目录/cutover-audit-log.jsonl
    audit_log_path = Path(args.audit_log_path) if args.audit_log_path else (
        marker_path.parent / "cutover-audit-log.jsonl"
    )

    try:
        marker = cutover(
            operator=args.operator,
            marker_path=marker_path,
            progress_path=progress_path,
            verify_report_path=verify_report_path,
            force=args.force,
            notes=args.note,
            audit_log_path=audit_log_path,
        )
        print(json.dumps(asdict(marker), ensure_ascii=False, indent=2))
        return 0
    except RuntimeError as exc:
        logger.error("cutover 失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
