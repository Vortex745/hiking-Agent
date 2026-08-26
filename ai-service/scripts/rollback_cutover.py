"""10 票据：回滚 cutover — 删除 cutover 标记，业务层自动回退到 pgvector。

按 10 票据验收标准：
- 回滚路径演练通过：cutover 前旧库可读、可回退
- 旧库保留至验收通过后按约定处置（回滚不删除旧库）

14 票据扩展（AC5）：
- 回滚恢复旧 adapter 的实际可读性，而不只是修改标记文件
- 通过 verify_old_backend 回调在删除 marker 前验证旧后端可读

回滚操作：
1. 验证旧后端可读（若提供 verify_old_backend 回调且未指定 --force）
2. 删除 data/cutover-marker.json
3. 提示运维清除 RAG_STORAGE_BACKEND 环境变量（或显式设置为 pgvector）
4. 不删除 SQLite 数据文件（保留迁移结果，便于再次 cutover）
5. 不删除迁移进度文件（保留迁移记录，便于审计）

用法：
    python -m scripts.rollback_cutover --operator <name> --reason <text>
    python -m scripts.rollback_cutover --force  # 跳过验证（紧急回滚）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("ai-service.scripts.rollback_cutover")


@dataclass(frozen=True)
class RollbackRecord:
    """回滚记录（追加到 rollback log）。"""

    rollback_at: str
    operator: str
    reason: str
    marker_path: str
    marker_existed: bool
    marker_content: dict | None  # 原 marker 内容（若存在）
    old_backend_verified: bool  # 14 票据 AC5：旧后端可读性验证结果
    old_backend_diagnostic: str  # 14 票据 AC5：验证诊断信息（成功或失败原因）


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rollback(
    *,
    operator: str,
    reason: str,
    marker_path: Path,
    rollback_log_path: Path,
    force: bool = False,
    verify_old_backend: Callable[[], tuple[bool, str]] | None = None,
) -> RollbackRecord:
    """执行回滚：删除 cutover 标记，追加回滚日志。

    Args:
        operator: 操作者标识
        reason: 回滚原因
        marker_path: cutover 标记文件路径
        rollback_log_path: 回滚日志文件路径（追加写入）
        force: True 时即使 marker 不存在也记录回滚；跳过旧后端验证
        verify_old_backend: 14 票据 AC5：旧后端可读性验证回调，返回 (readable, diagnostic)。
            None 时不验证（保留旧行为，便于测试注入）。
            非 None 且 force=False 时：调用回调，若返回 False 则 raise RuntimeError
            拒绝回滚（避免回滚到不可用后端）。

    Returns:
        RollbackRecord

    Raises:
        RuntimeError: marker 不存在且 force=False；
            或旧后端验证失败（force=False）
    """
    rollback_at = _now_iso()

    # 读取 marker（用于审计 + 提取 previous_backend）
    marker_content: dict | None = None
    marker_existed = marker_path.exists()

    if marker_existed:
        try:
            marker_content = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("cutover-marker.json 解析失败: %s", exc)
            marker_content = None
    elif not force:
        raise RuntimeError(
            f"cutover-marker.json 不存在: {marker_path}；无需回滚，或用 --force 记录回滚操作"
        )

    # 14 票据 AC5：验证旧后端可读性（force=False 时强制；force=True 时跳过）
    old_backend_verified = False
    old_backend_diagnostic = "skipped (force=True or no verifier)"

    if verify_old_backend is not None and not force:
        readable, diagnostic = verify_old_backend()
        old_backend_verified = readable
        old_backend_diagnostic = diagnostic
        if not readable:
            # 拒绝回滚：旧后端不可读，删除 marker 会导致 runtime 回退到不可用后端
            raise RuntimeError(
                f"回滚前旧后端验证失败: {diagnostic}；"
                f"用 --force 强制回滚（runtime 将回退到不可用后端，谨慎操作）"
            )
        logger.info("旧后端验证通过: %s", diagnostic)

    # 删除 marker
    if marker_existed:
        marker_path.unlink()
        logger.info("已删除 cutover 标记: %s", marker_path)

    # 追加回滚日志（便于审计）
    record = RollbackRecord(
        rollback_at=rollback_at,
        operator=operator,
        reason=reason,
        marker_path=str(marker_path),
        marker_existed=marker_existed,
        marker_content=marker_content,
        old_backend_verified=old_backend_verified,
        old_backend_diagnostic=old_backend_diagnostic,
    )

    rollback_log_path.parent.mkdir(parents=True, exist_ok=True)
    # 追加模式：每次回滚追加一行 JSON
    with rollback_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    logger.info("回滚日志已追加: %s", rollback_log_path)

    # 提示运维
    print(
        f"回滚完成。runtime 通过 backend_resolver 自动回退到 config_backend（14 票据）。\n"
        f"回滚日志: {rollback_log_path}"
    )

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="回滚 cutover：删除标记文件，恢复 pgvector 后端（10 票据）"
    )
    parser.add_argument(
        "--operator", required=True,
        help="操作者标识",
    )
    parser.add_argument(
        "--reason", required=True,
        help="回滚原因（必填，便于审计）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="即使 marker 不存在也记录回滚操作；跳过旧后端验证（紧急回滚）",
    )
    parser.add_argument(
        "--skip-verify-old-backend", action="store_true",
        help="14 票据 AC5：跳过旧后端可读性验证（不推荐，仅在已知旧后端状态时使用）",
    )
    parser.add_argument(
        "--rollback-log", default=None,
        help="回滚日志文件路径（默认 data/rollback-log.jsonl）",
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
    rollback_log_path = Path(args.rollback_log) if args.rollback_log else Path(
        settings.rag_cutover_marker_path
    ).parent / "rollback-log.jsonl"

    # 14 票据 AC5：默认提供 pgvector 可读性验证器
    # --skip-verify-old-backend 或 --force 时跳过验证
    verify_old_backend = None
    if not args.skip_verify_old_backend and not args.force:
        verify_old_backend = _make_pgvector_verifier(settings)

    try:
        record = rollback(
            operator=args.operator,
            reason=args.reason,
            marker_path=marker_path,
            rollback_log_path=rollback_log_path,
            force=args.force,
            verify_old_backend=verify_old_backend,
        )
        print(json.dumps(asdict(record), ensure_ascii=False, indent=2))
        return 0
    except RuntimeError as exc:
        logger.error("回滚失败: %s", exc)
        return 1


def _make_pgvector_verifier(settings):
    """14 票据 AC5：构造 pgvector 可读性验证器。

    验证策略：
    1. 从 settings.database_url 解析 pgvector 连接配置
    2. 尝试 SELECT 1 FROM rag_documents LIMIT 1（同时验证连接 + 表存在）
    3. 返回 (readable, diagnostic)

    安全：diagnostic 不输出完整 database_url（含用户名密码），仅输出 host + dbname。
    """
    def _verify() -> tuple[bool, str]:
        try:
            import psycopg
        except ImportError:
            return False, "psycopg 未安装，无法验证 pgvector"

        # 安全地从 database_url 提取 host + dbname（不泄露用户名密码）
        safe_target = _safe_db_target(settings.database_url)

        try:
            conn = psycopg.connect(
                settings.database_url,
                connect_timeout=settings.database_connect_timeout_seconds,
            )
            try:
                # 同时验证：连接成功 + rag_documents 表存在 + 至少可查询
                conn.execute("SELECT 1 FROM rag_documents LIMIT 1")
                return True, f"pgvector readable ({safe_target})"
            finally:
                conn.close()
        except Exception as exc:
            return False, f"pgvector 不可读 ({safe_target}): {exc}"

    return _verify


def _safe_db_target(database_url: str) -> str:
    """从 database_url 提取 host + dbname（不泄露用户名密码）。

    例：
        postgresql://user:pass@localhost:5432/ai_hiking → "localhost:5432/ai_hiking"
        postgresql://user:pass@host:5432/db?sslmode=require → "host:5432/db"
        invalid_url → "unknown"
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(database_url)
        # 无 hostname 的 URL（如 "not-a-url"）无法提取有效信息
        if not parsed.hostname:
            return "unknown"
        host = parsed.hostname
        port = parsed.port
        dbname = (parsed.path or "").lstrip("/") or "unknown"
        host_part = f"{host}:{port}" if port else host
        return f"{host_part}/{dbname}"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
