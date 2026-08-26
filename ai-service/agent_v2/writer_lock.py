"""Ticket 06 — Cross-platform single-writer file lock.

The single-writer invariant (spec AC7) is enforced at two layers:

1. **OS-level file lock** (this module) — a process-scoped exclusive
   lock on ``writer.lock`` under the v2 state dir. Two processes on the
   same host cannot both hold the lock; the second one fails fast at
   startup.
2. **Database-level row lock** (``ControlStore.acquire_writer_lock``)
   — a single row in ``agent_control.writer_lock`` records who owns the
   lock and when it was acquired. This survives process crashes (the
   row persists) and is observable from /health.

The OS lock is the hard gate (no silent co-ownership on the same host);
the DB row is the audit trail (operators can see who owns the lock
without inspecting process state).

Platform branches:

* **POSIX** — :man:`fcntl` ``LOCK_EX | LOCK_NB`` on the file descriptor.
  Releasing on process exit is automatic (kernel closes fd).
* **Windows** — :man:`msvcrt` ``LK_NBLCK`` on a file descriptor. Same
  auto-release semantics on process exit.

Both branches raise :class:`WriterLockBusy` when the lock is held by
another live process. The startup guard turns this into a
:class:`V2StartupError` so misconfigured deploys fail loud.
"""

from __future__ import annotations

import logging
import os
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("ai-service.agent_v2.writer_lock")


class WriterLockBusy(RuntimeError):
    """Raised when another live process holds the OS-level writer lock."""


@dataclass(slots=True)
class WriterLock:
    """Active ownership of the v2 single-writer lock.

    The holder keeps the file descriptor open for the lifetime of the
    process. Closing the fd releases the OS lock; the DB row is
    released separately via :class:`ControlStore.release_writer_lock`.
    """

    writer_id: str
    lock_path: str
    _fd: int | None = None

    def release(self) -> None:
        """Close the file descriptor, releasing the OS lock.

        Safe to call multiple times — subsequent calls are no-ops. The
        DB row is NOT released here; the lifespan calls
        ``ControlStore.release_writer_lock`` explicitly so the audit
        trail is cleaned up only after the OS lock is gone.
        """
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def acquire_writer_lock(state_dir: str) -> WriterLock:
    """Acquire the OS-level exclusive writer lock.

    Raises :class:`WriterLockBusy` when another live process holds the
    lock. The returned :class:`WriterLock` keeps the fd open; the
    caller must hold the reference for the lifetime of the process.

    The ``writer_id`` (UUID4) is also written to the DB row by the
    lifespan so /health can report who owns the lock.
    """
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    lock_path = str(Path(state_dir) / "writer.lock")
    writer_id = str(uuid.uuid4())

    # O_CREAT | O_RDWR so we can both create and lock. The file is
    # intentionally empty — the lock is the gate, not the contents.
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)

    system = platform.system()
    try:
        if system == "Windows":
            _lock_windows(fd)
        else:
            _lock_posix(fd)
    except OSError as exc:
        os.close(fd)
        raise WriterLockBusy(
            f"Another process holds the v2 writer lock at {lock_path}. "
            f"Only one writer may run v2 SQLite databases at a time "
            f"(spec AC7). Detail: {exc}"
        ) from exc

    logger.info("Acquired v2 writer lock (writer_id=%s, path=%s)", writer_id, lock_path)
    return WriterLock(writer_id=writer_id, lock_path=lock_path, _fd=fd)


def _lock_posix(fd: int) -> None:
    """POSIX: fcntl.LOCK_EX | LOCK_NB. Raises OSError on contention."""
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _lock_windows(fd: int) -> None:
    """Windows: msvcrt.LK_NBLCK. Raises OSError on contention."""
    import msvcrt

    # Lock 1 byte at offset 0; LK_NBLCK = non-blocking exclusive lock.
    # msvcrt.locking returns 0 on success, raises OSError on failure.
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
