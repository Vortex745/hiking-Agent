"""Disk-backed UniHarness graph checkpoint store (Ticket 27).

Owns the recoverable kernel kernel-graph state ONLY — public run
lifecycle, events, sequence and API idempotency stay with
:class:`RunRepository` and :class:`EventStream`. One SQLite database
under the v2 state dir; the LangGraph ``AsyncSqliteSaver`` is created
lazily inside the running event loop and reused for every kernel
assembly so ``run_id ↔ thread_id`` resumes across process restarts.

Fail-closed rules: missing parent directory, unwritable file, corrupt
schema or failed setup raise :class:`KernelCheckpointUnavailable` with a
stable public code — never a silent memory-only fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-service.agent_v2.kernel")


class KernelCheckpointUnavailable(Exception):
    """Raised when kernel graph checkpoints cannot be made durable.

    The public failure shape mirrors ``agent_v2.kernel.KernelModelFailure``
    so the request layer maps either uniformly. This module intentionally
    avoids importing kernel.py (the assembly) to prevent cycles.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "checkpoint_unavailable"
        self.message = message


class KernelCheckpointStore:
    """Lazily-opened process-wide LangGraph checkpointer over one SQLite file.

    ``get()`` is async because the underlying connection (aiosqlite) binds
    to the event loop that created it; call it from request context, never
    at import/startup time outside a loop.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(Path(path))
        self._saver: Any | None = None

    @property
    def path(self) -> str:
        return self._path

    async def get(self) -> Any:
        """Return the AsyncSqliteSaver, opening the backing database once."""
        if self._saver is not None:
            return self._saver
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self._path, timeout=5)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
        except Exception as exc:
            logger.exception("kernel checkpoint store unavailable: %s", self._path)
            raise KernelCheckpointUnavailable(
                "运行检查点不可用，无法安全地执行或恢复该请求。"
            ) from exc
        self._saver = saver
        return saver

    @property
    def healthy(self) -> bool:
        """True once the backing store has been opened successfully."""
        return self._saver is not None

    def close(self) -> None:
        """Release the async connection after its owning loop is gone.

        Runs synchronously on shutdown paths where no loop exists. Errors
        are logged, never raised — shutdown must complete.
        """
        if self._saver is None:
            return

        async def _aclose() -> None:
            # AsyncSqliteSaver owns aiosqlite.Connection with __aexit__.
            await self._saver.conn.close()  # type: ignore[attr-defined]

        try:
            import asyncio

            try:
                asyncio.get_running_loop()
                logger.warning(
                    "kernel checkpoint store close called with a running loop; "
                    "deferring via create_task"
                )
                asyncio.get_running_loop().create_task(_aclose())
            except RuntimeError:
                asyncio.run(_aclose())
        except Exception:
            logger.warning("kernel checkpoint store close failed", exc_info=True)
        finally:
            self._saver = None
