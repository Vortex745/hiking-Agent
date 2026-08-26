"""Chat v2 startup guard and runtime assembly.

This module is the single place where Chat v2 decides whether it may run.
The guard is intentionally strict: Vercel serverless cannot host the three
SQLite databases or run-scoped E2B, so ``VERCEL`` + ``agent_v2_enabled``
must fail loudly rather than silently fall back to a temp filesystem.

When v2 is disabled (the default), assembly returns ``None`` and no
checkpoint, sandbox, or MCP connection is created — v1 behavior is
completely unchanged.

Ticket 06 adds real persistence: the assembled runtime owns the
:class:`RunRepository` (control + checkpoints stores) and the
:class:`WriterLock`. The lifespan closes them on shutdown.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.memory_service import MemoryService
from agent_v2.observability import CapacityGuard, TelemetryService
from agent_v2.repositories.memory import MemoryStore
from agent_v2.runs import RunRepository
from agent_v2.writer_lock import WriterLock, WriterLockBusy, acquire_writer_lock

logger = logging.getLogger("ai-service.agent_v2")


class V2StartupError(RuntimeError):
    """Raised when Chat v2 cannot start safely.

    Carrying a dedicated exception type (rather than raising plain
    ``RuntimeError``) lets the lifespan distinguish v2 guard failures from
    unrelated errors and surface them through the health endpoint.
    """


@dataclass(slots=True)
class V2Runtime:
    """Assembled v2 runtime — replaces the Ticket 03 placeholder.

    Fields:

    * ``context`` — UniHarness AgentContext (stub for now; real fork lands
      in a later ticket).
    * ``uniharness_pin`` — immutable commit hash of the UniHarness fork.
    * ``state_dir`` — persistent volume path backing the SQLite files.
    * ``runs`` — :class:`RunRepository` facade over control + checkpoints
      stores. ``None`` when v2 is disabled.
    * ``writer_lock`` — :class:`WriterLock` proving single-writer ownership.
      ``None`` when v2 is disabled.
    * ``writer_id`` — UUID identifying this process's writer claim. Surfaced
      via /health for audit.
    """

    context: Any
    uniharness_pin: str
    state_dir: str
    runs: RunRepository | None = None
    memory: MemoryService | None = None
    telemetry: TelemetryService | None = None
    capacity: CapacityGuard | None = None
    writer_lock: WriterLock | None = None
    writer_id: str = ""
    # Ticket 23: whether bounded one-level subagents may be dispatched
    # from this runtime. Read from ``is_subagent_authorized(settings)``
    # at assembly time; defaults to False so upgrades stay root-only.
    subagent_enabled: bool = False

    # Backward compat with code that probed the Ticket 03 placeholder
    # fields. Remove once all callers migrate to ``runs`` / ``writer_lock``.
    @property
    def placeholder_context(self) -> Any:
        return self.context


def is_vercel_environment() -> bool:
    """True when running under Vercel serverless.

    Vercel sets ``VERCEL=1``. The check is a thin wrapper so tests can
    monkeypatch a single function rather than every ``os.getenv`` call.
    """
    return bool(os.getenv("VERCEL"))


def check_v2_startup_guard(settings: Any) -> None:
    """Fail fast if v2 is enabled in an environment that cannot host it.

    The rule is absolute: Vercel serverless does not own the three
    SQLite files or run-scoped E2B lifecycle, so it must never run v2.
    """
    if not getattr(settings, "agent_v2_enabled", False):
        return

    if is_vercel_environment():
        raise V2StartupError(
            "Chat v2 is enabled (AGENT_V2_ENABLED=true) but the process is "
            "running under Vercel serverless. Vercel cannot host the three "
            "v2 SQLite databases or run-scoped E2B. Disable AGENT_V2_ENABLED "
            "on Vercel deployments and route /api/v2 to the long-lived "
            "Docker Gateway origin instead."
        )


def is_cowork_authorized(settings: Any) -> bool:
    """Ticket 13 AC2 — server-side gate for the cowork profile.

    The cowork profile (Bash inside E2B) is registered in
    :mod:`agent_v2.profiles` but is NOT callable by default. Only an
    explicit operator-set policy of ``trusted`` or ``staging``
    authorizes requests with ``profile_id="cowork"``.

    The chat_v2 stream consults this function BEFORE resolving the
    profile, so a request with ``profile_id="cowork"`` against a
    ``disabled`` deployment is rejected with a terminal ``error`` event
    (HTTP 200 + SSE error) rather than reaching the profile dispatch.

    Clients cannot upgrade to cowork by sending the profile id — the
    server gate is the authority, not the request.
    """
    policy = getattr(settings, "agent_v2_cowork_policy", "disabled")
    return policy in {"trusted", "staging"}


def is_subagent_authorized(settings: Any) -> bool:
    """Ticket 23 AC7 — server-side gate for bounded one-level subagents.

    Subagents are disabled by default. Activation requires
    ``settings.agent_v2_subagent_enabled = True`` AND the post-cutover
    observation gate (Ticket 22 + 14 days of stable root-only operation)
    to be met. The observation gate is enforced out-of-band by
    operations (release coordination); this function is the in-process
    authority that the dispatcher consults at construction time.

    The chat_v2 stream and the dispatcher factory consult this function
    BEFORE constructing a :class:`SubagentDispatcher`, so a request
    against a ``disabled`` deployment gets a dispatcher whose
    ``can_dispatch`` returns ``False`` — no subagent is ever created.

    Operators can flip ``agent_v2_subagent_enabled=False`` at any time
    to immediately disable subagent dispatch on the next request
    boundary (AC7). In-flight subagents run to completion but no new
    dispatches are accepted.
    """
    return bool(getattr(settings, "agent_v2_subagent_enabled", False))


def assemble_v2_runtime(settings: Any) -> Any:
    """Assemble the v2 runtime, or return ``None`` when disabled.

    Ticket 06 steps:

    1. Resolve ``state_dir`` and ensure it exists.
    2. Acquire the OS-level writer lock (single-writer invariant, AC7).
       Failure here means another process owns the v2 databases —
       raise :class:`V2StartupError` so the lifespan fails loud.
    3. Open the two SQLite stores (control + checkpoints).
    4. Record the writer lock row in ``agent_control.writer_lock`` so
       /health can audit ownership.
    5. Compose them into :class:`RunRepository` and return the
       :class:`V2Runtime`.
    """
    if not getattr(settings, "agent_v2_enabled", False):
        logger.info("Chat v2 disabled (AGENT_V2_ENABLED=false); v1 path unchanged")
        return None

    # Guard runs again here so a misconfigured test that bypasses the
    # lifespan guard still cannot assemble v2 under Vercel.
    check_v2_startup_guard(settings)

    state_dir = getattr(settings, "agent_v2_state_dir", "./agent_state")
    pin = getattr(settings, "agent_v2_uniharness_pin", "unknown")
    logger.info(
        "Chat v2 enabled (pin=%s, state_dir=%s); assembling runtime",
        pin,
        state_dir,
    )

    # AC7: OS-level writer lock. Two processes cannot both own the v2
    # SQLite databases — the second one fails fast at startup.
    try:
        writer_lock = acquire_writer_lock(state_dir)
    except WriterLockBusy as exc:
        raise V2StartupError(str(exc)) from exc

    # Open the two SQLite stores. Paths are deterministic under state_dir.
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    control_path = str(Path(state_dir) / "agent_control.sqlite")
    checkpoints_path = str(Path(state_dir) / "chat_checkpoints.sqlite")

    try:
        control_store = ControlStore(control_path)
        checkpoints_store = CheckpointsStore(checkpoints_path)
    except Exception as exc:
        # Release the OS lock so the next startup attempt can re-acquire.
        writer_lock.release()
        raise V2StartupError(f"v2 SQLite stores failed to open: {exc}") from exc

    # AC7: record the writer row so /health can audit ownership. The
    # DB row persists across process crashes; the OS lock does not.
    # If the row already exists from a prior crash, we overwrite it
    # because we now hold the OS lock (the prior owner is dead).
    if not control_store.acquire_writer_lock(writer_lock.writer_id):
        # Same writer_id re-acquiring on the same process — fine.
        # Different writer_id means the DB row is stale from a crash;
        # we already hold the OS lock, so overwrite the row.
        logger.warning(
            "Stale writer_lock row found; overwriting with current writer_id=%s",
            writer_lock.writer_id,
        )
        # Force overwrite by direct update (acquire_writer_lock only inserts)
        control_store._conn.execute(
            "UPDATE writer_lock SET writer_id = ?, acquired_at = ? WHERE lock_id = 1",
            (writer_lock.writer_id, _now_iso()),
        )

    runs = RunRepository(control=control_store, checkpoints=checkpoints_store)
    try:
        memory = MemoryService(
            MemoryStore(str(Path(state_dir) / "agent_memory.sqlite"))
        )
    except Exception as exc:
        runs.close()
        writer_lock.release()
        raise V2StartupError(f"v2 memory store failed to open: {exc}") from exc

    # Import locally so disabled deployments never pay the import cost.
    from .uniharness import create_agent

    try:
        context = create_agent()
        return V2Runtime(
            context=context,
            uniharness_pin=pin,
            state_dir=state_dir,
            runs=runs,
            memory=memory,
            telemetry=TelemetryService(control_store),
            capacity=CapacityGuard(
                max_runs=20,
                max_sandboxes=10,
                queue_timeout_seconds=60.0,
            ),
            writer_lock=writer_lock,
            writer_id=writer_lock.writer_id,
            # Ticket 23: inject the server-side subagent gate so the
            # chat_v2 stream can construct a SubagentDispatcher with
            # the correct enabled state without re-reading settings.
            subagent_enabled=is_subagent_authorized(settings),
        )
    except Exception as exc:
        try:
            control_store.release_writer_lock(writer_lock.writer_id)
        except Exception:
            logger.warning("writer_lock DB cleanup failed during startup")
        runs.close()
        memory.close()
        writer_lock.release()
        raise V2StartupError(f"v2 agent failed to initialize: {exc}") from exc


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def release_v2_runtime(runtime: Any) -> None:
    """Release all resources held by an assembled v2 runtime.

    Called by the lifespan on shutdown. Safe to call with ``None`` or a
    Ticket 03 placeholder (no-op).
    """
    if runtime is None:
        return
    if not isinstance(runtime, V2Runtime):
        # Ticket 03 placeholder — nothing to release.
        return

    if runtime.runs is not None:
        try:
            # Release the DB writer row first so /health immediately
            # reports no owner; the OS lock follows.
            if runtime.writer_id:
                runtime.runs.control.release_writer_lock(runtime.writer_id)
        except Exception as exc:
            logger.warning("writer_lock DB release failed: %s", exc)
        try:
            runtime.runs.close()
        except Exception as exc:
            logger.warning("RunRepository close failed: %s", exc)

    if runtime.memory is not None:
        try:
            runtime.memory.close()
        except Exception as exc:
            logger.warning("MemoryService close failed: %s", exc)

    if runtime.writer_lock is not None:
        runtime.writer_lock.release()
