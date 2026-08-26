"""RAG 生命周期 checkpointer 适配层（票据 01）。

按 sse-clarification-contract 决策：thread_id = chat_id（不轮转）。
按 MEMORY.md 决策：使用 AsyncSqliteSaver + data/rag_checkpoints.sqlite。

本模块只提供 01 票据的基座：
- thread_id_for(chat_id)：chat_id → thread_id 适配函数
- create_checkpointer()：生产用 AsyncSqliteSaver 工厂
- FakeCheckpointer：测试用确定性 fake，继承 MemorySaver，可直接传给 graph.compile

主图与检索子图的编译逻辑在票据 05/06 实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from runtime_paths import runtime_dir


class FakeCheckpointer(MemorySaver):
    """确定性 fake checkpointer，单元测试专用。

    继承 langgraph 内置 MemorySaver，可直接传给 graph.compile(checkpointer=...)。
    不持久化、不跨进程；满足 01 票据"默认测试用确定性 fake，不依赖真实 Postgres"。
    """

    async def asetup(self) -> None:
        """async setup，no-op（与 AsyncSqliteSaver.asetup 接口对齐）。"""
        return None

    def setup(self) -> None:
        """sync setup，no-op。"""
        return None


def thread_id_for(chat_id: str) -> str:
    """chat_id → thread_id 适配函数。

    按 sse-clarification-contract 决策：thread_id = chat_id（澄清期间不轮转，
    resume 后继续用，不轮转）。本函数作为唯一适配点，未来若需轮转可在此修改。

    Args:
        chat_id: 用户会话标识（来自 API 请求）

    Returns:
        thread_id: LangGraph checkpointer 配置用的 thread_id

    Raises:
        ValueError: chat_id 为空或非字符串
    """
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise ValueError(f"chat_id must be non-empty string, got: {chat_id!r}")
    return chat_id


def checkpoints_dir() -> Path:
    """rag_checkpoints.sqlite 所在目录。与 rag_docs / memory_store 同级。"""
    return runtime_dir("RAG_CHECKPOINTS_DIR", "rag_checkpoints")


def create_checkpointer(db_path: Path | None = None) -> Any:
    """生产用 AsyncSqliteSaver 工厂。

    返回类型为 async context manager（含 __aenter__/__aexit__），调用方需用
    `async with`。退出时连接自动关闭，无需手动 cleanup。

    Args:
        db_path: SQLite 文件路径；None 时使用默认 data/rag_checkpoints.sqlite

    Returns:
        AsyncSqliteSaver 的 async context manager（未 enter）
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if db_path is None:
        db_path = checkpoints_dir() / "rag_checkpoints.sqlite"
    return AsyncSqliteSaver.from_conn_string(str(db_path))


__all__ = [
    "BaseCheckpointSaver",
    "FakeCheckpointer",
    "checkpoints_dir",
    "create_checkpointer",
    "thread_id_for",
]
