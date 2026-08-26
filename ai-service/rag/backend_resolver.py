"""14 票据：统一的 runtime 后端选择 resolver。

为存储 cutover 建立单一、可审计的运行时真相源。迁移验证、显式切换、应用启动、
运行时 adapter 解析和回滚读取同一状态；配置与状态冲突时拒绝静默启动或给出明确
受控决策。

设计原则（Karpathy Guidelines）：
- Simplicity First：resolver 是纯函数，输入 (config_backend, marker_path)，
  输出 ResolvedBackend，无副作用（除读取 marker 文件）。
- Surgical Changes：不修改 cutover/rollback 脚本的核心逻辑，只新增本模块
  并在 retriever.__init__ 中调用。

解析规则（验收 1/2/6）：
1. marker 文件存在 → marker.backend 为权威后端（AC1：cutover 决定 runtime）
2. marker.backend != config_backend → 记录 diagnostic 警告，仍以 marker 为准
   （AC2：不静默运行在另一后端，而是显式记录冲突）
3. marker 文件不存在 → 使用 config_backend（默认行为，零回归）
4. marker 文件存在但解析失败 → 记录 diagnostic 错误，使用 config_backend
   （保守策略：不阻塞启动，但运维必须修复 marker）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("ai-service.rag.backend_resolver")


# 有效后端枚举（与 config.py 的 rag_storage_backend 校验对齐）
_VALID_BACKENDS = {"pgvector", "sqlite", "memory"}


def _log_diagnostics(diagnostics: list[str]) -> None:
    """将 diagnostics 列表逐条记录到 logger.warning（避免静默）。"""
    for d in diagnostics:
        logger.warning("backend_resolver: %s", d)


@dataclass(frozen=True)
class ResolvedBackend:
    """runtime 后端解析结果（不可变值对象）。

    Attributes:
        backend: 最终选定的后端（"pgvector" | "sqlite" | "memory"）。
        source: 后端来源——"cutover_marker"（marker 决定）| "config"（config 决定）|
            "default"（无 config 且无 marker，回退 pgvector）。
        marker_path: marker 文件路径（用于审计追溯；None 表示未提供）。
        marker_content: marker 文件原始内容（用于审计；None 表示文件不存在或不可解析）。
        diagnostics: 人类可读的诊断信息列表（如配置-marker 冲突、解析失败等）。
    """

    backend: str
    source: str
    marker_path: str | None
    marker_content: dict | None
    diagnostics: list[str] = field(default_factory=list)


def resolve_backend(
    *,
    config_backend: str,
    marker_path: Path,
) -> ResolvedBackend:
    """解析 runtime 后端（AC1/AC2/AC6）。

    Args:
        config_backend: 从 settings.rag_storage_backend 读取的配置后端。
        marker_path: cutover-marker.json 文件路径。

    Returns:
        ResolvedBackend：包含最终后端 + 来源 + 诊断信息。

    本函数不抛异常（除非 marker 文件存在但权限拒绝读取等系统错误）。
    所有可恢复的错误（解析失败、字段缺失、冲突）均记录到 diagnostics + logger.warning
    并返回保守结果（AC2：不静默）。
    """
    diagnostics: list[str] = []

    # 1. 校验 config_backend（防御性：config.py 已校验，但本模块独立可被外部调用）
    if config_backend not in _VALID_BACKENDS:
        diagnostics.append(
            f"config_backend={config_backend!r} 非有效后端，回退 pgvector"
        )
        config_backend = "pgvector"

    # 2. 读取 marker 文件
    marker_path_str = str(marker_path)
    marker_content: dict | None = None

    if not marker_path.exists():
        # 无 marker：使用 config_backend（默认行为，零回归）
        # 即使无 marker，diagnostics 也要记录到 logger（AC2：不静默）
        _log_diagnostics(diagnostics)
        source = "config" if config_backend != "pgvector" else "default"
        return ResolvedBackend(
            backend=config_backend,
            source=source,
            marker_path=marker_path_str,
            marker_content=None,
            diagnostics=diagnostics,
        )

    # marker 存在：读取并解析
    try:
        marker_content = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # 解析失败：保守回退 config_backend，但记录诊断（AC2：不静默）
        diagnostics.append(
            f"marker 文件解析失败 ({marker_path_str}): {exc}；"
            f"回退 config_backend={config_backend}"
        )
        _log_diagnostics(diagnostics)
        return ResolvedBackend(
            backend=config_backend,
            source="config",
            marker_path=marker_path_str,
            marker_content=None,
            diagnostics=diagnostics,
        )

    # 3. 从 marker 提取 backend
    marker_backend = marker_content.get("backend")
    if not marker_backend or marker_backend not in _VALID_BACKENDS:
        diagnostics.append(
            f"marker.backend={marker_backend!r} 缺失或无效；"
            f"回退 config_backend={config_backend}"
        )
        _log_diagnostics(diagnostics)
        return ResolvedBackend(
            backend=config_backend,
            source="config",
            marker_path=marker_path_str,
            marker_content=marker_content,
            diagnostics=diagnostics,
        )

    # 4. AC1：marker.backend 为权威后端
    # AC2：检测 config-marker 冲突，记录诊断（不静默）
    if marker_backend != config_backend:
        diagnostics.append(
            f"配置-marker 冲突：config_backend={config_backend!r} 但 "
            f"marker.backend={marker_backend!r}；以 marker 为准；"
            f"请更新 RAG_STORAGE_BACKEND={marker_backend} 或回滚 marker"
        )
        _log_diagnostics(diagnostics)

    return ResolvedBackend(
        backend=marker_backend,
        source="cutover_marker",
        marker_path=marker_path_str,
        marker_content=marker_content,
        diagnostics=diagnostics,
    )
