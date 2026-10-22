"""Ticket 06 — Chat v2 SQLite persistence layer.

Two separate SQLite databases own the run lifecycle:

* ``chat_checkpoints.sqlite`` — conversation history, user/assistant
  messages, and the persisted v2 event stream. This is the source of
  truth for ``GET /snapshot`` and ``GET /events?from_seq=N`` reconnect.
* ``agent_control.sqlite`` — run metadata, status transitions, cancel
  requests, and the single-writer lock. This is the source of truth
  for "is this run cancelled / terminal / continuing".

Both databases use WAL + foreign keys + 5-second busy timeout +
explicit transactions, per spec acceptance criterion 1. They live
under ``settings.agent_v2_state_dir`` so a persistent volume can back
them independently of the application image.

The stores are synchronous (``sqlite3``) on purpose: the v2 endpoint
is request-scoped, the writes are small, and avoiding ``aiosqlite``
keeps the stub-stage dependency surface minimal. Real LLM integration
may revisit this when throughput pressure appears.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("ai-service.agent_v2.persistence")


def _now_iso() -> str:
    """UTC ISO 8601 timestamp with timezone suffix."""
    return datetime.now(timezone.utc).isoformat()


def _connect(path: str) -> sqlite3.Connection:
    """Open a connection with v2-required PRAGMAs.

    * ``journal_mode=WAL`` — concurrent readers + single writer without
      blocking reads; required by spec AC1.
    * ``foreign_keys=ON`` — ON DELETE CASCADE works; required by spec AC1.
    * ``busy_timeout=5000`` — 5-second wait on lock contention before
      raising ``SQLITE_BUSY``; required by spec AC1.

    ``check_same_thread=False`` is safe because the single-writer lock
    (:mod:`agent_v2.writer_lock`) guarantees only one process writes,
    and within that process the v2 endpoint is request-scoped with
    explicit transactions. FastAPI's TestClient runs the ASGI app in a
    separate thread, so the connection must be transferable.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path, isolation_level=None, check_same_thread=False
    )  # explicit tx
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ---- Run status + terminal types -----------------------------------------


RunStatus = Literal[
    "admission_pending",  # durable run exists, admission fence not yet complete
    "admitted",           # admission fence complete; execution may start
    "running",            # actively streaming
    "done",               # natural terminal (done event emitted)
    "error",              # failure terminal (error event emitted)
    "cancelled",          # cancel requested + accepted
    "paused",             # Ticket 11: awaiting user approval; NOT terminal
    "admission_failed",   # fence failed; repairable by admission repair only
]

TerminalType = Literal["done", "error", "cancelled", "admission_failed"]

_VALID_STATUSES: frozenset[str] = frozenset(
    {
        "admission_pending",
        "admitted",
        "running",
        "done",
        "error",
        "cancelled",
        "paused",
        "admission_failed",
    }
)
# ``paused`` is NOT terminal — the run is awaiting a user decision and
# will resume on ``POST /approvals/{approval_id}/decision``. The
# RunAlreadyTerminal guard therefore allows ``running`` → ``paused`` and
# ``paused`` → ``running`` transitions.
#
# ``admission_failed`` is terminal for execution: it has no outgoing
# lifecycle edge. BCDEF ticket 02 adds one dedicated repair transition
# (``admission_failed`` → ``admitted``) used only by admission repair; it
# is not a resume and can never start a second model call for a run that
# already executed.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"done", "error", "cancelled", "admission_failed"}
)


# ---- Control store: runs + cancel + writer lock --------------------------


_CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    terminal_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_at TEXT,
    -- BCDEF ticket 03: set when the run is terminal but its terminal
    -- event could not be durably written. Snapshot/replay surface it as
    -- ``terminal_persist_failed``; the row keeps its terminal status so
    -- the run can never be resumed or re-executed.
    terminal_persist_failed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_conversation ON runs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

-- BCDEF ticket 08: the execution policy frozen at admission. One row per
-- run (run_id is UNIQUE/PRIMARY KEY). ``policy_json`` is the canonical
-- policy content; ``policy_hash`` is its SHA-256 from
-- ``execution_contract.policy_hash``. Snapshot insertion shares the
-- admission commit gate with the run row (same transaction).
CREATE TABLE IF NOT EXISTS policy_snapshots (
    run_id TEXT PRIMARY KEY,
    policy_hash TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    source_version TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    compatibility_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_policy_snapshots_hash
ON policy_snapshots(policy_hash);

CREATE TABLE IF NOT EXISTS cancel_requests (
    cancel_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    requested_at TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    reason_text TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cancel_run ON cancel_requests(run_id);
CREATE INDEX IF NOT EXISTS idx_cancel_idem ON cancel_requests(idempotency_key);

CREATE TABLE IF NOT EXISTS writer_lock (
    lock_id INTEGER PRIMARY KEY CHECK (lock_id = 1),
    writer_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS sandbox_leases (
    lease_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    sandbox_type TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    destroyed_at TEXT,
    resource_usage_json TEXT,
    failure_reason TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sandbox_run ON sandbox_leases(run_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_status ON sandbox_leases(status);

-- Ticket 10: persisted artifacts after host-side manifest verification.
-- ``storage_path`` is host-side and never exposed to clients; the wire
-- only carries ``artifact_id`` + ``public_filename``.
-- ``status`` transitions: ``available`` → ``expired`` (TTL elapsed) or
-- ``failed`` (export validation failed but evidence retained).
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    public_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_conv ON artifacts(conversation_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status);

-- Ticket 11: persisted approvals for high-risk tool calls. The approval
-- id is derived deterministically from (run_id, tool_name, canonical_args)
-- in :mod:`agent_v2.approvals`; the UNIQUE(run_id, tool_name, args_canonical)
-- constraint enforces idempotency at the DB layer so duplicate inserts
-- raise IntegrityError rather than silently creating a second row (spec AC5).
-- ``status`` transitions: ``pending`` → ``approved``/``rejected`` (user
-- decision) or ``expired`` (TTL elapsed). ``decision_request_id`` is the
-- idempotency key for the decision endpoint — duplicate decisions return
-- the original outcome (spec AC5).
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_canonical TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    impact TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    resolved_at TEXT,
    decision_request_id TEXT,
    UNIQUE (run_id, tool_name, args_canonical),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_conv ON approvals(conversation_id);
CREATE INDEX IF NOT EXISTS idx_approvals_decision_idem ON approvals(decision_request_id);

-- Ticket 12: effect ledger for crash-safe side-effect tracking. Every
-- side-effecting tool call writes a row here BEFORE execution so a
-- crash can be detected and recovered safely (spec AC1). The effect
-- key is deterministic (computed in :mod:`agent_v2.effects` from
-- run_id + tool_name + canonical_args); the UNIQUE(run_id, tool_name,
-- args_canonical) constraint enforces one row per logical tool call
-- (spec AC5: parameter tampering is a new effect, not a replay).
-- ``status`` transitions:
--   authorized → executing → committed (success, result_json set)
--                         | → outcome_unknown (crash, spec AC3)
--                         | → failed (definitive error)
-- The ``authorized → executing`` transition uses a CAS
-- (UPDATE ... WHERE status='authorized') so only one executor wins
-- under concurrent decisions or service restart (spec AC7).
-- ``external_idempotency_key`` is the key sent to external APIs that
-- support idempotency (spec AC2); for local tools it is stored for
-- audit but the local FS does not deduplicate.
CREATE TABLE IF NOT EXISTS effects (
    effect_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_canonical TEXT NOT NULL,
    status TEXT NOT NULL,
    external_idempotency_key TEXT NOT NULL,
    result_json TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT,
    UNIQUE (run_id, tool_name, args_canonical),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_effects_run ON effects(run_id);
CREATE INDEX IF NOT EXISTS idx_effects_status ON effects(status);
CREATE INDEX IF NOT EXISTS idx_effects_tool ON effects(tool_name);

-- Ticket 16: redacted, metadata-only run telemetry. This table has no
-- columns for prompts, messages, memory, file contents, or tool output.
CREATE TABLE IF NOT EXISTS run_telemetry (
    run_id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    tool_names_json TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_cost_usd REAL NOT NULL,
    e2b_cost_usd REAL NOT NULL,
    approval_statuses_json TEXT NOT NULL,
    effect_statuses_json TEXT NOT NULL,
    resource_usage_json TEXT NOT NULL,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_telemetry_status ON run_telemetry(status);
CREATE INDEX IF NOT EXISTS idx_run_telemetry_profile ON run_telemetry(profile_name);

-- Raw sampling is an audited switch only. Raw payloads are deliberately
-- not stored in agent_control.sqlite.
CREATE TABLE IF NOT EXISTS raw_sampling_audit (
    sample_id TEXT PRIMARY KEY,
    operator TEXT NOT NULL,
    reason TEXT NOT NULL,
    enabled_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    disabled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_raw_sampling_expiry
ON raw_sampling_audit(disabled_at, expires_at);
"""


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Snapshot of a run row."""

    run_id: str
    request_id: str
    conversation_id: str
    profile_id: str
    status: str
    terminal_type: str | None
    created_at: str
    updated_at: str
    terminal_at: str | None
    terminal_persist_failed: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class PolicySnapshotWrite:
    """Primitive values of a snapshot insert (BCDEF ticket 08).

    Kept storage-local so :mod:`agent_v2.persistence` does not import the
    policy builder; hash/serialization happen in :mod:`agent_v2.runs`.
    """

    policy_json: str
    policy_hash: str
    schema_version: int
    source_version: str
    compatibility_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class PolicySnapshotRecord:
    """Snapshot of a policy_snapshots row."""

    run_id: str
    policy_hash: str
    schema_version: int
    source_version: str
    policy_json: str
    compatibility_json: str
    created_at: str


# ---- Sandbox lease statuses -----------------------------------------------


SandboxType = Literal["local_native", "wsl_localvm", "remote_e2b"]
SandboxStatus = Literal["active", "destroyed", "failed"]

_VALID_SANDBOX_TYPES: frozenset[str] = frozenset(
    {"local_native", "wsl_localvm", "remote_e2b"}
)
_VALID_SANDBOX_STATUSES: frozenset[str] = frozenset(
    {"active", "destroyed", "failed"}
)


@dataclass(frozen=True, slots=True)
class SandboxLeaseRecord:
    """Snapshot of a sandbox_leases row (Ticket 09).

    A lease binds a run to exactly one sandbox. ``status`` transitions:
    ``active`` → ``destroyed`` (clean cleanup) or ``failed`` (cleanup
    error or creation failure). ``resource_usage_json`` records cpu/memory
    /duration metrics gathered at cleanup time so operators can audit cost.
    """

    lease_id: str
    run_id: str
    sandbox_type: str
    status: str
    workspace_path: str
    created_at: str
    destroyed_at: str | None
    resource_usage: dict[str, Any] | None
    failure_reason: str | None


# ---- Artifact statuses (Ticket 10) ---------------------------------------


ArtifactStatus = Literal["available", "expired", "failed"]

_VALID_ARTIFACT_STATUSES: frozenset[str] = frozenset(
    {"available", "expired", "failed"}
)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Snapshot of an artifacts row (Ticket 10).

    ``storage_path`` is host-side and NEVER exposed to clients — the
    download endpoint only returns ``artifact_id`` + ``public_filename``.
    ``status`` transitions: ``available`` → ``expired`` (TTL elapsed) or
    ``failed`` (export validation failed but evidence row retained so
    operators can diagnose).
    """

    artifact_id: str
    run_id: str
    conversation_id: str
    public_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    storage_path: str
    status: str
    failure_reason: str | None
    created_at: str
    expires_at: str


# ---- Approval statuses (Ticket 11) ---------------------------------------


ApprovalStatus = Literal["pending", "approved", "rejected", "expired"]

_VALID_APPROVAL_STATUSES: frozenset[str] = frozenset(
    {"pending", "approved", "rejected", "expired"}
)

_TERMINAL_APPROVAL_STATUSES: frozenset[str] = frozenset(
    {"approved", "rejected", "expired"}
)


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """Snapshot of an approvals row (Ticket 11).

    The approval id is deterministic (computed in
    :mod:`agent_v2.approvals` from run_id + tool_name + canonical args).
    ``args_canonical`` is the JSON-serialized canonical form; the
    decision endpoint re-derives it from the request body to detect
    parameter tampering (spec AC5). ``impact`` is a short user-facing
    description of the tool's effect, surfaced in the
    ``approval.required`` event payload (spec AC2).

    ``decision_request_id`` is the idempotency key for the decision
    endpoint — duplicate decisions with the same key return the
    original outcome (spec AC5).
    """

    approval_id: str
    run_id: str
    conversation_id: str
    tool_name: str
    args_canonical: str
    risk_level: str
    impact: str
    status: str
    created_at: str
    expires_at: str
    resolved_at: str | None
    decision_request_id: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_APPROVAL_STATUSES


# ---- Effect ledger statuses (Ticket 12) -----------------------------------

EffectStatus = Literal[
    "authorized",       # written before execution; safe to retry
    "executing",        # tool in flight; crash → outcome_unknown
    "committed",        # success; result_json holds the durable result
    "outcome_unknown",  # crash left result indeterminate; needs human review
    "failed",           # definitive failure; terminal
]

_VALID_EFFECT_STATUSES: frozenset[str] = frozenset(
    {"authorized", "executing", "committed", "outcome_unknown", "failed"}
)
_TERMINAL_EFFECT_STATUSES: frozenset[str] = frozenset(
    {"committed", "outcome_unknown", "failed"}
)


@dataclass(frozen=True, slots=True)
class EffectRecord:
    """Snapshot of an effects row (Ticket 12).

    The effect ledger records the execution state of every side-effecting
    tool call. The ``effect_key`` is deterministic (computed in
    :mod:`agent_v2.effects`) so a retry finds the same row. The
    ``authorized → executing`` transition uses a CAS
    (:meth:`ControlStore.transition_effect_to_executing`) so only one
    executor wins under concurrent decisions or service restart
    (spec AC7).

    ``result_json`` holds the durable tool result (serialized via
    :func:`agent_v2.effects.serialize_result`). On retry, if the effect
    is ``committed``, the stored result is returned without re-executing
    (spec AC2). ``external_idempotency_key`` is the key sent to external
    APIs that support idempotency (spec AC2).
    """

    effect_key: str
    run_id: str
    conversation_id: str
    tool_name: str
    args_canonical: str
    status: str
    external_idempotency_key: str
    result_json: str | None
    failure_reason: str | None
    created_at: str
    updated_at: str
    committed_at: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_EFFECT_STATUSES


class ControlStore:
    """agent_control.sqlite — run metadata + cancel + writer lock + sandbox leases + artifacts."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = _connect(db_path)
        self._conn.executescript(_CONTROL_SCHEMA)
        self._migrate_terminal_persist_failed()
        logger.info("ControlStore ready at %s", db_path)

    def _migrate_terminal_persist_failed(self) -> None:
        """Add the ticket 03 marker column to pre-existing runs tables."""
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(runs)")
        }
        if "terminal_persist_failed" not in columns:
            self._conn.execute(
                "ALTER TABLE runs ADD COLUMN terminal_persist_failed "
                "INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self._conn.close()

    # -- runs -----------------------------------------------------------

    def create_run_idempotent(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
    ) -> tuple[str, bool]:
        """Insert a run row, or return the existing run_id for the request_id.

        Returns ``(run_id, created)`` where ``created`` is True iff a new
        row was inserted. The unique constraint on ``request_id`` makes
        repeated calls return the same run_id without re-emitting user
        messages or re-invoking tools (spec AC2).

        The new run starts in ``running`` status. ``update_run_status``
        transitions it to a terminal state on done/error/cancelled.
        """
        existing = self._conn.execute(
            "SELECT run_id FROM runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if existing is not None:
            return existing["run_id"], False

        run_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO runs "
                "(run_id, request_id, conversation_id, profile_id, "
                " status, terminal_type, created_at, updated_at, terminal_at) "
                "VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, NULL)",
                (run_id, request_id, conversation_id, profile_id, now, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return run_id, True

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            request_id=row["request_id"],
            conversation_id=row["conversation_id"],
            profile_id=row["profile_id"],
            status=row["status"],
            terminal_type=row["terminal_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            terminal_at=row["terminal_at"],
            terminal_persist_failed=bool(row["terminal_persist_failed"]),
        )

    # -- terminal fence (BCDEF ticket 03) -------------------------------

    def finalize_run_status(
        self,
        *,
        run_id: str,
        terminal_type: str,
    ) -> tuple[str, bool]:
        """CAS a run to its terminal status after the durable event commit.

        Returns ``(status, transitioned)``. Re-applying the SAME terminal
        is an idempotent no-op (``transitioned=False``) so duplicate
        finalize calls cannot create a second terminal or resurrect a
        finished run. A DIFFERENT terminal on an already-terminal run
        raises :class:`RunAlreadyTerminal` — the first terminal wins.
        """
        if terminal_type not in ("done", "error", "cancelled"):
            raise ValueError(f"invalid terminal type: {terminal_type!r}")
        current = self.get_run(run_id)
        if current is None:
            raise RunNotFound(f"run_id={run_id}")
        if current.is_terminal:
            if current.status == terminal_type or current.terminal_type == terminal_type:
                self._conn.execute(
                    "UPDATE runs SET terminal_persist_failed = 0 WHERE run_id = ?",
                    (run_id,),
                )
                return current.status, False
            raise RunAlreadyTerminal(
                f"run {run_id} already terminal "
                f"(status={current.status}, type={current.terminal_type})"
            )

        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, terminal_type = ?, updated_at = ?, "
                "terminal_at = ?, terminal_persist_failed = 0 "
                "WHERE run_id = ? AND status NOT IN "
                "('done', 'error', 'cancelled', 'admission_failed')",
                (terminal_type, terminal_type, now, now, run_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        if cursor.rowcount == 0:
            # A concurrent writer won the CAS. Re-read: the same terminal
            # is idempotent, a different one is a duplicate-terminal error.
            refreshed = self.get_run(run_id)
            if refreshed is None:
                raise RunNotFound(f"run_id={run_id}")
            if (
                refreshed.status == terminal_type
                or refreshed.terminal_type == terminal_type
            ):
                return refreshed.status, False
            raise RunAlreadyTerminal(
                f"run {run_id} already terminal "
                f"(status={refreshed.status}, type={refreshed.terminal_type})"
            )
        return terminal_type, True

    def mark_terminal_persist_failed(self, run_id: str) -> str:
        """Record that the run's terminal event could not be persisted.

        The run is moved to ``error`` (unless it already reached a
        terminal status, which is preserved) and flagged so snapshot and
        replay can expose a stable ``terminal_persist_failed`` state.
        Idempotent.
        """
        current = self.get_run(run_id)
        if current is None:
            raise RunNotFound(f"run_id={run_id}")
        if current.is_terminal:
            self._conn.execute(
                "UPDATE runs SET terminal_persist_failed = 1 WHERE run_id = ?",
                (run_id,),
            )
            return current.status
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE runs SET status = 'error', terminal_type = 'error', "
                "updated_at = ?, terminal_at = ?, terminal_persist_failed = 1 "
                "WHERE run_id = ? AND status NOT IN "
                "('done', 'error', 'cancelled', 'admission_failed')",
                (now, now, run_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        refreshed = self.get_run(run_id)
        return refreshed.status if refreshed is not None else "error"

    # -- admission fence (BCDEF ticket 02) ------------------------------

    def admit_run(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
        policy: PolicySnapshotWrite | None = None,
    ) -> tuple[RunRecord, bool]:
        """Insert a run in ``admission_pending`` or return the existing row.

        Unlike :meth:`create_run_idempotent` (legacy callers that start in
        ``running``), admission runs start pending: the orchestrator may
        not start a model turn until the repository marks the fence
        complete. The request_id UNIQUE constraint makes concurrent
        admissions converge on one durable run.

        When ``policy`` is supplied, the run row and its policy snapshot
        are inserted in ONE transaction: a run without a snapshot can
        never be observed, so execution cannot start without frozen policy.
        """
        existing = self.get_run_by_request(request_id)
        if existing is not None:
            return existing, False

        run_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO runs "
                "(run_id, request_id, conversation_id, profile_id, "
                " status, terminal_type, created_at, updated_at, terminal_at) "
                "VALUES (?, ?, ?, ?, 'admission_pending', NULL, ?, ?, NULL)",
                (run_id, request_id, conversation_id, profile_id, now, now),
            )
            if policy is not None:
                self._conn.execute(
                    "INSERT INTO policy_snapshots "
                    "(run_id, policy_hash, schema_version, source_version, "
                    " policy_json, compatibility_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        policy.policy_hash,
                        policy.schema_version,
                        policy.source_version,
                        policy.policy_json,
                        policy.compatibility_json,
                        policy.created_at,
                    ),
                )
            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            self._conn.execute("ROLLBACK")
            existing = self.get_run_by_request(request_id)
            if existing is not None:
                return existing, False
            raise
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        record = self.get_run(run_id)
        if record is None:  # pragma: no cover - defensive
            raise RuntimeError(f"admitted run {run_id} disappeared")
        return record, True

    # -- policy snapshots (BCDEF ticket 08) -----------------------------

    @staticmethod
    def _row_to_policy_snapshot(row: sqlite3.Row) -> PolicySnapshotRecord:
        return PolicySnapshotRecord(
            run_id=row["run_id"],
            policy_hash=row["policy_hash"],
            schema_version=row["schema_version"],
            source_version=row["source_version"],
            policy_json=row["policy_json"],
            compatibility_json=row["compatibility_json"],
            created_at=row["created_at"],
        )

    def get_policy_snapshot(self, run_id: str) -> PolicySnapshotRecord | None:
        row = self._conn.execute(
            "SELECT * FROM policy_snapshots WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return self._row_to_policy_snapshot(row) if row is not None else None

    def insert_policy_snapshot(
        self,
        *,
        run_id: str,
        policy: PolicySnapshotWrite,
    ) -> tuple[PolicySnapshotRecord, bool]:
        """Idempotently insert the run's snapshot; duplicate returns existing.

        The first write wins: an existing snapshot is never overwritten,
        so a config change between attempts cannot rewrite an admitted
        run's policy. Returns ``(record, created)``.
        """
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO policy_snapshots "
            "(run_id, policy_hash, schema_version, source_version, "
            " policy_json, compatibility_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                policy.policy_hash,
                policy.schema_version,
                policy.source_version,
                policy.policy_json,
                policy.compatibility_json,
                policy.created_at,
            ),
        )
        record = self.get_policy_snapshot(run_id)
        if record is None:  # pragma: no cover - defensive
            raise RuntimeError(f"policy snapshot for run {run_id} disappeared")
        return record, cursor.rowcount == 1

    def mark_run_admitted(self, *, run_id: str, repair: bool = False) -> bool:
        """CAS the admission fence to ``admitted``; True iff this caller won.

        Exactly one concurrent admission may win the CAS, so at most one
        orchestration continues into execution. ``repair=True`` is the
        dedicated admission repair edge (``admission_failed`` → ``admitted``);
        it is not a lifecycle resume and never follows an executed model turn.
        """
        allowed = ("admission_failed",) if repair else ("admission_pending",)
        placeholders = ", ".join("?" for _ in allowed)
        now = _now_iso()
        cursor = self._conn.execute(
            f"UPDATE runs SET status = 'admitted', terminal_type = NULL, "
            f"updated_at = ?, terminal_at = NULL "
            f"WHERE run_id = ? AND status IN ({placeholders})",
            (now, run_id, *allowed),
        )
        return cursor.rowcount == 1

    def mark_admission_failed(self, run_id: str) -> bool:
        """Record a repairable admission failure (never from an executed run)."""
        now = _now_iso()
        cursor = self._conn.execute(
            "UPDATE runs SET status = 'admission_failed', "
            "terminal_type = 'admission_failed', updated_at = ?, terminal_at = ? "
            "WHERE run_id = ? AND status IN ('admission_pending', 'admission_failed')",
            (now, now, run_id),
        )
        return cursor.rowcount == 1

    def mark_run_running(self, run_id: str) -> bool:
        """Idempotently move an admitted run into execution."""
        now = _now_iso()
        cursor = self._conn.execute(
            "UPDATE runs SET status = 'running', updated_at = ? "
            "WHERE run_id = ? AND status IN ('admitted', 'running')",
            (now, run_id),
        )
        return cursor.rowcount == 1

    def get_run_by_request(self, request_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_paused_runs_updated_before(self, updated_before: str) -> list[RunRecord]:
        """List paused runs whose ``updated_at`` predates ``updated_before``.

        Used by the paused-run janitor: an approval-paused run holds a
        CapacityGuard sandbox slot until the user decides (or the resume
        stream releases it). Abandoned pauses must be reclaimed or the
        in-process slot leaks until restart.
        """
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE status = 'paused' AND updated_at < ?",
            (updated_before,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_completed_run_ids(self, conversation_id: str) -> set[str]:
        """Ticket 05 (T07): run_ids of ``done`` runs in a conversation.

        Only naturally completed runs count as conversation history —
        running, failed (``error``), cancelled and paused runs are
        unfinished turns and never enter the bounded recent history.
        """
        rows = self._conn.execute(
            "SELECT run_id FROM runs WHERE conversation_id = ? AND status = 'done'",
            (conversation_id,),
        ).fetchall()
        return {row["run_id"] for row in rows}

    def update_run_status(
        self,
        *,
        run_id: str,
        status: str,
        terminal_type: str | None = None,
    ) -> None:
        """Transition a run's status.

        Once terminal (done/error/cancelled), further transitions are
        rejected — spec AC5 requires a unique cancelled terminal and
        rejects continuation. Calling this on a terminal run raises
        ``RunAlreadyTerminal``.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")

        current = self.get_run(run_id)
        if current is None:
            raise RunNotFound(f"run_id={run_id}")
        if current.is_terminal:
            raise RunAlreadyTerminal(
                f"run {run_id} already terminal "
                f"(status={current.status}, type={current.terminal_type})"
            )

        now = _now_iso()
        terminal_at = now if status in _TERMINAL_STATUSES else None
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE runs SET status = ?, terminal_type = ?, "
                "updated_at = ?, terminal_at = ? WHERE run_id = ?",
                (status, terminal_type, now, terminal_at, run_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            request_id=row["request_id"],
            conversation_id=row["conversation_id"],
            profile_id=row["profile_id"],
            status=row["status"],
            terminal_type=row["terminal_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            terminal_at=row["terminal_at"],
            terminal_persist_failed=bool(row["terminal_persist_failed"]),
        )

    # -- cancel ---------------------------------------------------------

    def request_cancel(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> tuple[bool, str]:
        """Idempotent cancel request.

        Returns ``(accepted, reason)``:

        * ``accepted=True, reason="cancelled"`` — first cancel on a
          running run; status transitions to ``cancelled``.
        * ``accepted=False, reason="already_terminal:<type>"`` — the run
          already terminated (done/error/cancelled); cancel is rejected
          because the run is not live. Repeat cancels with a NEW
          idempotency_key on an already-cancelled run also return this
          (the run is no longer live, so the cancel cannot be accepted).
        * ``accepted=False, reason="unknown_run"`` — run_id not found.

        Duplicate cancel requests with the SAME idempotency_key always
        return the original outcome (idempotent at the HTTP layer too).
        """
        run = self.get_run(run_id)
        if run is None:
            return False, "unknown_run"

        # Idempotency: same key → same outcome
        prior = self._conn.execute(
            "SELECT accepted, reason_text FROM cancel_requests "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if prior is not None:
            accepted = bool(prior["accepted"])
            # reason_text stored at first request; reuse it verbatim
            return accepted, prior["reason_text"]

        if run.is_terminal:
            # Store the reject so a replay doesn't accidentally accept
            self._insert_cancel_request(
                run_id=run_id,
                idempotency_key=idempotency_key,
                accepted=False,
                reason_text=f"already_terminal:{run.terminal_type or run.status}",
            )
            return False, f"already_terminal:{run.terminal_type or run.status}"

        # First cancel on a live run — transition to cancelled.
        cancel_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO cancel_requests "
                "(cancel_id, run_id, idempotency_key, requested_at, "
                " accepted, reason_text) VALUES (?, ?, ?, ?, 1, 'cancelled')",
                (cancel_id, run_id, idempotency_key, now),
            )
            self._conn.execute(
                "UPDATE runs SET status = 'cancelled', terminal_type = 'cancelled', "
                "updated_at = ?, terminal_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return True, "cancelled"

    def _insert_cancel_request(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        accepted: bool,
        reason_text: str,
    ) -> None:
        cancel_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO cancel_requests "
                "(cancel_id, run_id, idempotency_key, requested_at, "
                " accepted, reason_text) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cancel_id,
                    run_id,
                    idempotency_key,
                    now,
                    1 if accepted else 0,
                    reason_text,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_cancel_requests(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM cancel_requests WHERE run_id = ? "
            "ORDER BY requested_at ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- writer lock ----------------------------------------------------

    def acquire_writer_lock(self, writer_id: str) -> bool:
        """Try to acquire the single-writer lock.

        Returns True if this process now owns the lock, False if another
        live writer holds it. ``expires_at`` is NULL — the lock is
        released explicitly via :meth:`release_writer_lock` on shutdown.
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            existing = self._conn.execute(
                "SELECT writer_id FROM writer_lock WHERE lock_id = 1"
            ).fetchone()
            if existing is not None:
                self._conn.execute("ROLLBACK")
                return existing["writer_id"] == writer_id
            self._conn.execute(
                "INSERT INTO writer_lock (lock_id, writer_id, acquired_at, expires_at) "
                "VALUES (1, ?, ?, NULL)",
                (writer_id, now),
            )
            self._conn.execute("COMMIT")
            return True
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def release_writer_lock(self, writer_id: str) -> None:
        """Release the lock on shutdown if this process owns it."""
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "DELETE FROM writer_lock WHERE lock_id = 1 AND writer_id = ?",
                (writer_id,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_writer_lock(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM writer_lock WHERE lock_id = 1"
        ).fetchone()
        return dict(row) if row is not None else None

    # -- sandbox leases (Ticket 09) -----------------------------------

    def create_sandbox_lease(
        self,
        *,
        run_id: str,
        sandbox_type: str,
        workspace_path: str,
    ) -> str:
        """Insert a new sandbox lease for ``run_id``.

        A run may have at most one lease (UNIQUE on run_id). Re-calling
        for an existing lease raises ``sqlite3.IntegrityError`` — callers
        that want idempotent semantics check ``get_sandbox_lease`` first.

        ``sandbox_type`` must be one of :data:`_VALID_SANDBOX_TYPES`.
        """
        if sandbox_type not in _VALID_SANDBOX_TYPES:
            raise ValueError(f"invalid sandbox_type: {sandbox_type!r}")
        lease_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO sandbox_leases "
                "(lease_id, run_id, sandbox_type, status, workspace_path, "
                " created_at, destroyed_at, resource_usage_json, failure_reason) "
                "VALUES (?, ?, ?, 'active', ?, ?, NULL, NULL, NULL)",
                (lease_id, run_id, sandbox_type, workspace_path, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return lease_id

    def get_sandbox_lease(self, run_id: str) -> SandboxLeaseRecord | None:
        row = self._conn.execute(
            "SELECT * FROM sandbox_leases WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return self._row_to_lease(row) if row is not None else None

    def destroy_sandbox_lease(
        self,
        *,
        run_id: str,
        resource_usage: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> SandboxLeaseRecord | None:
        """Mark a lease destroyed (clean) or failed (cleanup error).

        Returns the updated record, or ``None`` if no lease exists for
        ``run_id`` (e.g. creation failed before the row was inserted).
        """
        lease = self.get_sandbox_lease(run_id)
        if lease is None:
            return None
        now = _now_iso()
        status = "failed" if failure_reason else "destroyed"
        usage_json = (
            json.dumps(resource_usage, ensure_ascii=False)
            if resource_usage is not None
            else None
        )
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE sandbox_leases SET status = ?, destroyed_at = ?, "
                "resource_usage_json = ?, failure_reason = ? WHERE run_id = ?",
                (status, now, usage_json, failure_reason, run_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_sandbox_lease(run_id)

    def _row_to_lease(self, row: sqlite3.Row) -> SandboxLeaseRecord:
        usage_raw = row["resource_usage_json"]
        return SandboxLeaseRecord(
            lease_id=row["lease_id"],
            run_id=row["run_id"],
            sandbox_type=row["sandbox_type"],
            status=row["status"],
            workspace_path=row["workspace_path"],
            created_at=row["created_at"],
            destroyed_at=row["destroyed_at"],
            resource_usage=json.loads(usage_raw) if usage_raw else None,
            failure_reason=row["failure_reason"],
        )

    # -- artifacts (Ticket 10) -----------------------------------------

    def insert_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        conversation_id: str,
        public_filename: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        storage_path: str,
        status: str,
        failure_reason: str | None,
        created_at: str,
        expires_at: str,
    ) -> ArtifactRecord:
        """Insert a new artifact row.

        Spec AC3: ``artifact_id`` is opaque (caller generates uuid4).
        ``storage_path`` is host-side and never returned to clients.
        ``status='failed'`` is allowed so export failures still persist
        an evidence row (spec AC4: do not destroy sandbox before
        evidence is saved).
        """
        if status not in _VALID_ARTIFACT_STATUSES:
            raise ValueError(f"invalid artifact status: {status!r}")
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO artifacts "
                "(artifact_id, run_id, conversation_id, public_filename, "
                " mime_type, size_bytes, sha256, storage_path, status, "
                " failure_reason, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    run_id,
                    conversation_id,
                    public_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    storage_path,
                    status,
                    failure_reason,
                    created_at,
                    expires_at,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self._row_to_artifact(
            self._conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        return self._row_to_artifact(row) if row is not None else None

    def list_artifacts_for_run(self, run_id: str) -> list[ArtifactRecord]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def mark_artifact_expired(self, artifact_id: str) -> ArtifactRecord | None:
        """Transition an artifact to ``expired`` (TTL elapsed).

        Returns the updated record, or ``None`` if no row exists. The
        transition is one-way: a ``failed`` row is not mutated (the
        failure evidence is preserved for diagnosis).
        """
        existing = self.get_artifact(artifact_id)
        if existing is None or existing.status == "failed":
            return existing
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE artifacts SET status = 'expired' WHERE artifact_id = ?",
                (artifact_id,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_artifact(artifact_id)

    def _row_to_artifact(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            conversation_id=row["conversation_id"],
            public_filename=row["public_filename"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            storage_path=row["storage_path"],
            status=row["status"],
            failure_reason=row["failure_reason"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    # -- approvals (Ticket 11) -----------------------------------------

    def insert_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        conversation_id: str,
        tool_name: str,
        args_canonical: str,
        risk_level: str,
        impact: str,
        created_at: str,
        expires_at: str,
    ) -> ApprovalRecord:
        """Insert a new approval row in ``pending`` status.

        Spec AC2: ``approval_id`` is the deterministic id computed in
        :mod:`agent_v2.approvals`. The UNIQUE(run_id, tool_name,
        args_canonical) constraint catches duplicate inserts at the DB
        layer — callers that want idempotent semantics check
        :meth:`get_approval` first (spec AC5: repeat approval requests
        for the same logical tool call must not create a second row).
        """
        if risk_level not in {"low", "medium", "high"}:
            raise ValueError(f"invalid risk_level: {risk_level!r}")
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO approvals "
                "(approval_id, run_id, conversation_id, tool_name, "
                " args_canonical, risk_level, impact, status, "
                " created_at, expires_at, resolved_at, decision_request_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)",
                (
                    approval_id,
                    run_id,
                    conversation_id,
                    tool_name,
                    args_canonical,
                    risk_level,
                    impact,
                    created_at,
                    expires_at,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self._row_to_approval(
            self._conn.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        )

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        row = self._conn.execute(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        return self._row_to_approval(row) if row is not None else None

    def list_pending_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        """Return all pending approvals for ``run_id`` (oldest first)."""
        rows = self._conn.execute(
            "SELECT * FROM approvals WHERE run_id = ? AND status = 'pending' "
            "ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_approval(r) for r in rows]

    def list_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        """Return all approvals for ``run_id`` (oldest first)."""
        rows = self._conn.execute(
            "SELECT * FROM approvals WHERE run_id = ? "
            "ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_approval(r) for r in rows]

    def resolve_approval(
        self,
        *,
        approval_id: str,
        status: str,
        decision_request_id: str,
    ) -> ApprovalRecord | None:
        """Transition an approval to ``approved`` or ``rejected``.

        Spec AC5 idempotency: if ``decision_request_id`` matches an
        existing row's ``decision_request_id``, the call is a duplicate
        decision (e.g. user double-clicked) and the existing row is
        returned unchanged. This prevents a second tool execution.

        Returns the updated record, or ``None`` if no row exists. The
        caller (decision endpoint) maps ``None`` to 404 and idempotent
        duplicates to the original outcome.

        Raises ``ValueError`` if ``status`` is not approved/rejected or
        if the row is already terminal with a different request_id.
        """
        if status not in {"approved", "rejected"}:
            raise ValueError(
                f"resolve_approval: status must be approved/rejected, got {status!r}"
            )

        existing = self.get_approval(approval_id)
        if existing is None:
            return None

        # Idempotency: same decision_request_id → return existing outcome.
        if (
            existing.decision_request_id is not None
            and existing.decision_request_id == decision_request_id
            and existing.is_terminal
        ):
            return existing

        # Reject transition on already-terminal approvals — the first
        # decision wins. This is the spec AC5 invariant: "已解决... 不会
        # 再次执行工具". A second decision with a NEW request_id is
        # rejected so the tool does not execute twice.
        if existing.is_terminal:
            raise RunAlreadyTerminal(
                f"approval {approval_id} already terminal "
                f"(status={existing.status})"
            )

        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE approvals SET status = ?, resolved_at = ?, "
                "decision_request_id = ? WHERE approval_id = ?",
                (status, now, decision_request_id, approval_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_approval(approval_id)

    def mark_approval_expired(self, approval_id: str) -> ApprovalRecord | None:
        """Transition a pending approval to ``expired`` (TTL elapsed).

        Returns the updated record, or ``None`` if no row exists. A
        terminal row (approved/rejected) is NOT mutated — the user's
        decision stands even if the TTL elapsed before the resume flow
        observed it.
        """
        existing = self.get_approval(approval_id)
        if existing is None or existing.is_terminal:
            return existing
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE approvals SET status = 'expired', resolved_at = ? "
                "WHERE approval_id = ?",
                (now, approval_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_approval(approval_id)

    def _row_to_approval(self, row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row["approval_id"],
            run_id=row["run_id"],
            conversation_id=row["conversation_id"],
            tool_name=row["tool_name"],
            args_canonical=row["args_canonical"],
            risk_level=row["risk_level"],
            impact=row["impact"],
            status=row["status"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            resolved_at=row["resolved_at"],
            decision_request_id=row["decision_request_id"],
        )

    # -- effects (Ticket 12) --------------------------------------------

    def insert_effect(
        self,
        *,
        effect_key: str,
        run_id: str,
        conversation_id: str,
        tool_name: str,
        args_canonical: str,
        external_idempotency_key: str,
    ) -> EffectRecord:
        """Insert a new effect row in ``authorized`` status (spec AC1).

        The row is written BEFORE any side effect so a crash here is
        safe to retry. The UNIQUE(run_id, tool_name, args_canonical)
        constraint catches duplicate inserts at the DB layer — callers
        that want idempotent semantics check :meth:`get_effect` first
        (spec AC5).
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO effects "
                "(effect_key, run_id, conversation_id, tool_name, "
                " args_canonical, status, external_idempotency_key, "
                " result_json, failure_reason, created_at, updated_at, "
                " committed_at) "
                "VALUES (?, ?, ?, ?, ?, 'authorized', ?, NULL, NULL, ?, ?, NULL)",
                (
                    effect_key,
                    run_id,
                    conversation_id,
                    tool_name,
                    args_canonical,
                    external_idempotency_key,
                    now,
                    now,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self._row_to_effect(
            self._conn.execute(
                "SELECT * FROM effects WHERE effect_key = ?",
                (effect_key,),
            ).fetchone()
        )

    def get_effect(self, effect_key: str) -> EffectRecord | None:
        row = self._conn.execute(
            "SELECT * FROM effects WHERE effect_key = ?",
            (effect_key,),
        ).fetchone()
        return self._row_to_effect(row) if row is not None else None

    def get_effect_by_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        args_canonical: str,
    ) -> EffectRecord | None:
        """Look up an effect by its logical tool call identity."""
        row = self._conn.execute(
            "SELECT * FROM effects "
            "WHERE run_id = ? AND tool_name = ? AND args_canonical = ?",
            (run_id, tool_name, args_canonical),
        ).fetchone()
        return self._row_to_effect(row) if row is not None else None

    def list_effects_for_run(self, run_id: str) -> list[EffectRecord]:
        rows = self._conn.execute(
            "SELECT * FROM effects WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        return [self._row_to_effect(r) for r in rows]

    def transition_effect_to_executing(
        self,
        effect_key: str,
    ) -> tuple[EffectRecord | None, bool]:
        """CAS transition ``authorized`` → ``executing`` (spec AC7).

        Returns ``(record, won)`` where ``won`` is True iff THIS caller
        performed the transition. Under concurrent decisions or service
        restart, only one caller sees ``won=True``; others see the
        already-``executing`` row with ``won=False`` and must NOT
        re-execute (spec AC7: "只有一个执行者获得执行权").

        The CAS is implemented as ``UPDATE ... WHERE status='authorized'``
        followed by a ``changes()`` check. SQLite's default isolation
        makes this atomic within a single transaction; the writer_lock
        serializes cross-process contention.
        """
        existing = self.get_effect(effect_key)
        if existing is None:
            return None, False
        if existing.status != "authorized":
            # Already past authorized — return current state, did not win.
            return existing, False
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            cur = self._conn.execute(
                "UPDATE effects SET status = 'executing', updated_at = ? "
                "WHERE effect_key = ? AND status = 'authorized'",
                (now, effect_key),
            )
            won = cur.rowcount == 1
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_effect(effect_key), won

    def commit_effect(
        self,
        effect_key: str,
        result_json: str,
    ) -> EffectRecord | None:
        """Transition ``executing`` → ``committed`` with the durable result.

        Spec AC2: the stored result is returned on retry without
        re-executing. ``result_json`` is the canonical serialization
        from :func:`agent_v2.effects.serialize_result`.

        CAS guard (Ticket 12 review H2): the UPDATE only matches
        ``status = 'executing'``. If a concurrent caller already moved
        the effect to ``outcome_unknown`` or ``failed``, this UPDATE
        affects 0 rows and we return the current record (not None) so
        the caller can detect the state mismatch. This prevents illegal
        transitions like ``outcome_unknown`` → ``committed``.
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE effects SET status = 'committed', result_json = ?, "
                "committed_at = ?, updated_at = ? "
                "WHERE effect_key = ? AND status = 'executing'",
                (result_json, now, now, effect_key),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_effect(effect_key)

    def mark_effect_outcome_unknown(
        self,
        effect_key: str,
        reason: str = "",
    ) -> EffectRecord | None:
        """Transition an ``executing`` effect to ``outcome_unknown`` (spec AC3).

        Called on crash recovery when the effect was left in
        ``executing`` — the side effect may or may not have completed,
        so the run must NOT auto-retry. The row is terminal; the run
        terminates with ``effect.outcome_unknown`` and requires human
        review (spec AC5).

        CAS guard (Ticket 12 review H2): the UPDATE only matches
        ``status = 'executing'``. If a concurrent caller already
        committed or failed the effect, this UPDATE affects 0 rows and
        we return the current record so the caller can detect the
        mismatch. This prevents illegal transitions like
        ``committed`` → ``outcome_unknown``.
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE effects SET status = 'outcome_unknown', "
                "failure_reason = ?, updated_at = ? "
                "WHERE effect_key = ? AND status = 'executing'",
                (reason, now, effect_key),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_effect(effect_key)

    def mark_effect_failed(
        self,
        effect_key: str,
        reason: str = "",
    ) -> EffectRecord | None:
        """Transition an effect to ``failed`` (definitive error, terminal).

        Called when the tool returned a clear failure (e.g. sandbox
        timeout). Unlike ``outcome_unknown``, ``failed`` means we KNOW
        the side effect did not complete. The run terminates with
        ``error``.

        CAS guard (Ticket 12 review H2): the UPDATE only matches
        ``status = 'executing'``. If a concurrent caller already
        committed or marked outcome_unknown, this UPDATE affects 0
        rows and we return the current record so the caller can detect
        the mismatch. This prevents illegal transitions like
        ``committed`` → ``failed``.
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE effects SET status = 'failed', "
                "failure_reason = ?, updated_at = ? "
                "WHERE effect_key = ? AND status = 'executing'",
                (reason, now, effect_key),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return self.get_effect(effect_key)

    def _row_to_effect(self, row: sqlite3.Row) -> EffectRecord:
        return EffectRecord(
            effect_key=row["effect_key"],
            run_id=row["run_id"],
            conversation_id=row["conversation_id"],
            tool_name=row["tool_name"],
            args_canonical=row["args_canonical"],
            status=row["status"],
            external_idempotency_key=row["external_idempotency_key"],
            result_json=row["result_json"],
            failure_reason=row["failure_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            committed_at=row["committed_at"],
        )

    # -- redacted observability (Ticket 16) ---------------------------

    def upsert_run_telemetry(
        self,
        *,
        run_id: str,
        profile_name: str,
        tool_names: list[str],
        status: str,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        model_cost_usd: float,
        e2b_cost_usd: float,
        approval_statuses: list[str],
        effect_statuses: list[str],
        resource_usage: dict[str, Any],
        error_code: str | None,
    ) -> None:
        """Persist only the fixed, metadata-only telemetry contract."""
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO run_telemetry "
            "(run_id, profile_name, tool_names_json, status, duration_ms, "
            " input_tokens, output_tokens, model_name, model_cost_usd, "
            " e2b_cost_usd, approval_statuses_json, effect_statuses_json, "
            " resource_usage_json, error_code, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "profile_name=excluded.profile_name, "
            "tool_names_json=excluded.tool_names_json, status=excluded.status, "
            "duration_ms=excluded.duration_ms, input_tokens=excluded.input_tokens, "
            "output_tokens=excluded.output_tokens, model_name=excluded.model_name, "
            "model_cost_usd=excluded.model_cost_usd, "
            "e2b_cost_usd=excluded.e2b_cost_usd, "
            "approval_statuses_json=excluded.approval_statuses_json, "
            "effect_statuses_json=excluded.effect_statuses_json, "
            "resource_usage_json=excluded.resource_usage_json, "
            "error_code=excluded.error_code, updated_at=excluded.updated_at",
            (
                run_id,
                profile_name,
                json.dumps(tool_names, separators=(",", ":")),
                status,
                duration_ms,
                input_tokens,
                output_tokens,
                model_name,
                model_cost_usd,
                e2b_cost_usd,
                json.dumps(approval_statuses, separators=(",", ":")),
                json.dumps(effect_statuses, separators=(",", ":")),
                json.dumps(resource_usage, separators=(",", ":")),
                error_code,
                now,
                now,
            ),
        )

    def get_run_telemetry(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM run_telemetry WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._telemetry_row(row) if row is not None else None

    def list_run_telemetry(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM run_telemetry ORDER BY created_at"
        ).fetchall()
        return [self._telemetry_row(row) for row in rows]

    @staticmethod
    def _telemetry_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "profile_name": row["profile_name"],
            "tool_names": json.loads(row["tool_names_json"]),
            "status": row["status"],
            "duration_ms": row["duration_ms"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "model_name": row["model_name"],
            "model_cost_usd": row["model_cost_usd"],
            "e2b_cost_usd": row["e2b_cost_usd"],
            "approval_statuses": json.loads(row["approval_statuses_json"]),
            "effect_statuses": json.loads(row["effect_statuses_json"]),
            "resource_usage": json.loads(row["resource_usage_json"]),
            "error_code": row["error_code"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def insert_raw_sampling_audit(
        self,
        *,
        sample_id: str,
        operator: str,
        reason: str,
        enabled_at: str,
        expires_at: str,
    ) -> None:
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE raw_sampling_audit SET disabled_at = ? "
                "WHERE disabled_at IS NULL",
                (enabled_at,),
            )
            self._conn.execute(
                "INSERT INTO raw_sampling_audit "
                "(sample_id, operator, reason, enabled_at, expires_at, disabled_at) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (sample_id, operator, reason, enabled_at, expires_at),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def disable_expired_raw_sampling(self, now: str) -> None:
        self._conn.execute(
            "UPDATE raw_sampling_audit SET disabled_at = ? "
            "WHERE disabled_at IS NULL AND expires_at <= ?",
            (now, now),
        )

    def get_active_raw_sampling(self, now: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM raw_sampling_audit "
            "WHERE disabled_at IS NULL AND expires_at > ? "
            "ORDER BY enabled_at DESC LIMIT 1",
            (now,),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_raw_sampling_audit(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM raw_sampling_audit ORDER BY enabled_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


# ---- Checkpoints store: conversations + messages + events ----------------


_CHECKPOINTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id)
        REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id);
CREATE INDEX IF NOT EXISTS idx_messages_seq ON messages(run_id, sequence);
-- BCDEF ticket 02: one request_id may produce at most one user message.
-- The admission fence relies on this UNIQUE constraint for concurrent
-- duplicates; a racing loser re-reads the winner's row instead of
-- appending a second copy.
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_run_user
    ON messages(run_id) WHERE role = 'user';

CREATE TABLE IF NOT EXISTS run_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    request_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON run_events(run_id, sequence);
"""


@dataclass(frozen=True, slots=True)
class MessageRecord:
    message_id: str
    conversation_id: str
    run_id: str
    role: str
    content: str
    sequence: int
    created_at: str


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """Map one ``messages`` row to its record — the single construction
    site shared by the oldest-first snapshot read and the newest-first
    history page."""
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        run_id=row["run_id"],
        role=row["role"],
        content=row["content"],
        sequence=row["sequence"],
        created_at=row["created_at"],
    )


@dataclass(frozen=True, slots=True)
class StoredEvent:
    event_id: str
    run_id: str
    sequence: int
    event_type: str
    payload_json: str
    request_id: str
    conversation_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class FinalizeRecord:
    """Result of the fused assistant + terminal write (ticket 03).

    ``terminal`` is the authoritative persisted terminal — either the one
    just written or the pre-existing one returned idempotently.
    ``assistant_content`` is the content actually stored (an existing row
    wins so replay never contradicts the first write).
    """

    terminal: StoredEvent
    terminal_created: bool
    assistant_message_id: str | None
    assistant_sequence: int | None
    assistant_content: str


class AssistantMessagePersistenceError(RuntimeError):
    """Raised when the assistant message of a finalize transaction fails."""


class TerminalEventPersistenceError(RuntimeError):
    """Raised when the terminal event of a finalize transaction fails."""


class CheckpointsStore:
    """chat_checkpoints.sqlite — conversation + message + event history."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = _connect(db_path)
        self._conn.executescript(_CHECKPOINTS_SCHEMA)
        logger.info("CheckpointsStore ready at %s", db_path)

    def close(self) -> None:
        self._conn.close()

    # -- conversations --------------------------------------------------

    def ensure_conversation(
        self,
        *,
        conversation_id: str,
        profile_id: str,
    ) -> bool:
        """Insert the conversation row if missing; touch last_active_at.

        Returns True iff a new conversation row was created. Admission
        repair uses the flag to report what it backfilled.
        """
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            existing = self._conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            created = existing is None
            if created:
                self._conn.execute(
                    "INSERT INTO conversations "
                    "(conversation_id, profile_id, created_at, last_active_at) "
                    "VALUES (?, ?, ?, ?)",
                    (conversation_id, profile_id, now, now),
                )
            else:
                self._conn.execute(
                    "UPDATE conversations SET last_active_at = ? "
                    "WHERE conversation_id = ?",
                    (now, conversation_id),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return created

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    # -- messages -------------------------------------------------------

    def append_message(
        self,
        *,
        conversation_id: str,
        run_id: str,
        role: str,
        content: str,
        sequence: int,
    ) -> str:
        """Append a user or assistant message to the conversation.

        ``sequence`` is the v2 event sequence at which the message was
        emitted; this lets the snapshot endpoint order messages by
        emission order without a separate counter.
        """
        message_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO messages "
                "(message_id, conversation_id, run_id, role, content, "
                " sequence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, run_id, role, content, sequence, now),
            )
            self._conn.execute(
                "UPDATE conversations SET last_active_at = ? "
                "WHERE conversation_id = ?",
                (now, conversation_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return message_id

    def ensure_user_message(
        self,
        *,
        conversation_id: str,
        run_id: str,
        content: str,
        sequence: int | None = None,
    ) -> tuple[str, bool]:
        """Idempotently persist the ONE user message of ``run_id``.

        Returns ``(message_id, created)``. Concurrent admissions race on
        the partial UNIQUE(run_id) WHERE role='user' index; the loser
        re-reads the winner's row, so a request_id can never accumulate a
        second user message.

        ``sequence`` is derived from the conversation's existing messages
        when omitted (ticket 03): a fresh conversation starts at 0, and a
        follow-up run continues after the previous assistant message
        instead of reusing 0.
        """
        row = self._conn.execute(
            "SELECT message_id FROM messages WHERE run_id = ? AND role = 'user' "
            "ORDER BY rowid ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is not None:
            return row["message_id"], False
        if sequence is None:
            sequence = self.get_next_message_sequence(conversation_id)
        try:
            message_id = self.append_message(
                conversation_id=conversation_id,
                run_id=run_id,
                role="user",
                content=content,
                sequence=sequence,
            )
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT message_id FROM messages WHERE run_id = ? AND role = 'user' "
                "ORDER BY rowid ASC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                raise
            return row["message_id"], False
        return message_id, True

    def get_next_message_sequence(self, conversation_id: str) -> int:
        """Next free per-conversation message sequence (0 for a new one).

        Message sequences order the user/assistant turns of ONE
        conversation; every event sequence is per-run. Deriving instead
        of hardcoding keeps a second run's user/assistant rows after the
        first run's assistant row.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sequence), -1) AS max_seq FROM messages "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row["max_seq"]) + 1

    def get_assistant_message(self, run_id: str) -> MessageRecord | None:
        """The ONE assistant message persisted for ``run_id``, if any."""
        row = self._conn.execute(
            "SELECT * FROM messages WHERE run_id = ? AND role = 'assistant' "
            "ORDER BY rowid ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        return _row_to_message(row) if row is not None else None

    def get_terminal_event(self, run_id: str) -> StoredEvent | None:
        """The persisted run-lifecycle terminal of ``run_id``, if any.

        ``run.paused`` is a stream terminal, not a run terminal, so it is
        deliberately excluded.
        """
        row = self._conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? "
            "AND event_type IN ('done', 'error', 'cancelled') "
            "ORDER BY sequence ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        return self._row_to_stored_event(row) if row is not None else None

    def finalize_terminal(
        self,
        *,
        run_id: str,
        conversation_id: str,
        request_id: str,
        terminal_type: str,
        payload: dict[str, Any],
        sequence: int,
        event_id: str | None = None,
        timestamp: str | None = None,
        assistant_content: str | None = None,
    ) -> FinalizeRecord:
        """Persist the assistant reply and the terminal event together.

        Ticket 03 fence: the assistant message (when any) is written
        exactly once with a derived conversation sequence, in the SAME
        transaction as the terminal event. A failure in either half rolls
        both back, so a terminal can never become durable before the
        assistant it reports.

        Idempotent: if the run already has a run-lifecycle terminal, that
        row is returned unchanged (``terminal_created=False``) and only a
        missing assistant row is backfilled. Duplicate finalizes therefore
        cannot create a second terminal.
        """
        if terminal_type not in ("done", "error", "cancelled"):
            raise ValueError(f"invalid terminal type: {terminal_type!r}")
        eid = event_id or str(uuid.uuid4())
        now = timestamp or _now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False)

        self._conn.execute("BEGIN")
        try:
            existing = self.get_terminal_event(run_id)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        assistant_id: str | None = None
        assistant_sequence: int | None = None
        assistant_text = ""
        try:
            if assistant_content:
                existing_assistant = self.get_assistant_message(run_id)
                if existing_assistant is not None:
                    assistant_id = existing_assistant.message_id
                    assistant_sequence = existing_assistant.sequence
                    assistant_text = existing_assistant.content
                else:
                    assistant_sequence = self.get_next_message_sequence(
                        conversation_id
                    )
                    assistant_id = str(uuid.uuid4())
                    self._conn.execute(
                        "INSERT INTO messages "
                        "(message_id, conversation_id, run_id, role, content, "
                        " sequence, created_at) "
                        "VALUES (?, ?, ?, 'assistant', ?, ?, ?)",
                        (
                            assistant_id,
                            conversation_id,
                            run_id,
                            assistant_content,
                            assistant_sequence,
                            now,
                        ),
                    )
                    self._conn.execute(
                        "UPDATE conversations SET last_active_at = ? "
                        "WHERE conversation_id = ?",
                        (now, conversation_id),
                    )
                    assistant_text = assistant_content
        except Exception as exc:
            self._conn.execute("ROLLBACK")
            raise AssistantMessagePersistenceError(
                f"assistant message write failed for run_id={run_id}"
            ) from exc

        if existing is not None:
            try:
                self._conn.execute("COMMIT")
            except Exception as exc:
                self._conn.execute("ROLLBACK")
                raise TerminalEventPersistenceError(
                    f"could not commit finalize for run_id={run_id}"
                ) from exc
            return FinalizeRecord(
                terminal=existing,
                terminal_created=False,
                assistant_message_id=assistant_id,
                assistant_sequence=assistant_sequence,
                assistant_content=assistant_text,
            )

        try:
            inserted = self._insert_terminal_event(
                event_id=eid,
                run_id=run_id,
                sequence=sequence,
                event_type=terminal_type,
                payload_json=payload_json,
                request_id=request_id,
                conversation_id=conversation_id,
                created_at=now,
            )
            self._conn.execute("COMMIT")
        except Exception as exc:
            self._conn.execute("ROLLBACK")
            raise TerminalEventPersistenceError(
                f"terminal event write failed for run_id={run_id}"
            ) from exc
        return FinalizeRecord(
            terminal=inserted,
            terminal_created=True,
            assistant_message_id=assistant_id,
            assistant_sequence=assistant_sequence,
            assistant_content=assistant_text,
        )

    def _insert_terminal_event(
        self,
        *,
        event_id: str,
        run_id: str,
        sequence: int,
        event_type: str,
        payload_json: str,
        request_id: str,
        conversation_id: str,
        created_at: str,
    ) -> StoredEvent:
        """Raw terminal insert inside the caller's transaction."""
        self._conn.execute(
            "INSERT INTO run_events "
            "(event_id, run_id, sequence, event_type, payload_json, "
            " request_id, conversation_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                run_id,
                sequence,
                event_type,
                payload_json,
                request_id,
                conversation_id,
                created_at,
            ),
        )
        return StoredEvent(
            event_id=event_id,
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            payload_json=payload_json,
            request_id=request_id,
            conversation_id=conversation_id,
            created_at=created_at,
        )

    def clear_conversation(self, conversation_id: str) -> int:
        """Delete all messages of a conversation; return the removed count.

        Spec #13 phase 2 (T07): the RAG page's history-clear endpoint is
        backed by the kernel conversation store. The conversation row
        itself is kept — ensure_conversation re-creates it idempotently
        on the next run.
        """
        self._conn.execute("BEGIN")
        try:
            cursor = self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return cursor.rowcount

    def get_messages(
        self,
        *,
        conversation_id: str,
        limit: int = 100,
    ) -> list[MessageRecord]:
        # rowid order = insertion order = true conversation chronology;
        # the per-run `sequence` alone cannot order across runs
        # (every run restarts at 0).
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY rowid ASC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    def get_messages_newest_first(
        self,
        *,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MessageRecord]:
        """Ticket 05 (T07): newest-first page of model-visible messages.

        This is the read seam behind the bounded recent-history query. It
        deliberately differs from :meth:`get_messages` (which keeps its
        OLDEST-first snapshot semantics for the UI): pages are ordered by
        ``rowid DESC`` — the stable true insertion order — and only
        ``user`` / ``assistant`` roles are returned, so internal tool /
        approval rows can never enter a model turn.
        """
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "AND role IN ('user', 'assistant') "
            "ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (conversation_id, limit, offset),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    # -- events ---------------------------------------------------------

    def append_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        payload: dict[str, Any],
        request_id: str,
        conversation_id: str,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Persist a single v2 SSE event for replay on reconnect.

        The UNIQUE(run_id, sequence) constraint guarantees monotonic
        sequence per run — a duplicate insert raises IntegrityError,
        which is exactly the spec AC4 invariant.

        ``event_id`` and ``timestamp`` are optional. When provided (from
        the wire envelope), they are stored directly so the DB row and
        the SSE frame share the same values (spec decision #26). When
        omitted, the store generates them — this keeps backward
        compatibility with tests that call ``persist_event`` directly.
        """
        import json

        eid = event_id or str(uuid.uuid4())
        now = timestamp or _now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False)
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO run_events "
                "(event_id, run_id, sequence, event_type, payload_json, "
                " request_id, conversation_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eid,
                    run_id,
                    sequence,
                    event_type,
                    payload_json,
                    request_id,
                    conversation_id,
                    now,
                ),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return eid

    def get_events_since(
        self,
        *,
        run_id: str,
        from_seq: int,
        limit: int = 1000,
    ) -> list[StoredEvent]:
        """Return events with sequence > from_seq, in ascending order.

        ``from_seq=0`` returns the full stream. The endpoint uses this
        to serve ``GET /events?from_seq=N`` reconnects (spec AC4).
        """
        rows = self._conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? AND sequence > ? "
            "ORDER BY sequence ASC LIMIT ?",
            (run_id, from_seq, limit),
        ).fetchall()
        return [
            StoredEvent(
                event_id=r["event_id"],
                run_id=r["run_id"],
                sequence=r["sequence"],
                event_type=r["event_type"],
                payload_json=r["payload_json"],
                request_id=r["request_id"],
                conversation_id=r["conversation_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_last_sequence(self, run_id: str) -> int:
        """Return the highest sequence emitted for the run, or 0."""
        row = self._conn.execute(
            "SELECT MAX(sequence) AS max_seq FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None or row["max_seq"] is None:
            return 0
        return int(row["max_seq"])

    # -- admission event (BCDEF ticket 02) ------------------------------

    @staticmethod
    def _row_to_stored_event(r: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            event_id=r["event_id"],
            run_id=r["run_id"],
            sequence=r["sequence"],
            event_type=r["event_type"],
            payload_json=r["payload_json"],
            request_id=r["request_id"],
            conversation_id=r["conversation_id"],
            created_at=r["created_at"],
        )

    def get_admission_event(self, run_id: str) -> StoredEvent | None:
        """Return the durable ``run_started`` admission event of ``run_id``."""
        row = self._conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? "
            "AND event_type = 'run_started' ORDER BY sequence ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        return self._row_to_stored_event(row) if row is not None else None

    def ensure_admission_event(
        self,
        *,
        run_id: str,
        conversation_id: str,
        request_id: str,
        payload: dict[str, Any],
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> tuple[StoredEvent, bool]:
        """Idempotently persist the ONE admission event of ``run_id``.

        Returns ``(event, created)``. A duplicate admission refreshes the
        payload (admitted/repaired/replayed outcome) without adding a
        second ``run_started`` row, so replay always has exactly one
        admission fact. Concurrent inserts converge via the
        UNIQUE(run_id, sequence) constraint.
        """
        eid = event_id or str(uuid.uuid4())
        now = timestamp or _now_iso()
        payload_json = json.dumps(payload, ensure_ascii=False)
        for _attempt in range(3):
            existing = self.get_admission_event(run_id)
            if existing is not None:
                if existing.payload_json != payload_json:
                    self._conn.execute(
                        "UPDATE run_events SET payload_json = ? WHERE event_id = ?",
                        (payload_json, existing.event_id),
                    )
                refreshed = self._conn.execute(
                    "SELECT * FROM run_events WHERE event_id = ?",
                    (existing.event_id,),
                ).fetchone()
                return self._row_to_stored_event(refreshed), False
            sequence = self.get_last_sequence(run_id) + 1
            try:
                self.append_event(
                    run_id=run_id,
                    sequence=sequence,
                    event_type="run_started",
                    payload=payload,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    event_id=eid,
                    timestamp=now,
                )
            except sqlite3.IntegrityError:
                continue
            row = self._conn.execute(
                "SELECT * FROM run_events WHERE event_id = ?",
                (eid,),
            ).fetchone()
            if row is not None:
                return self._row_to_stored_event(row), True
        raise RuntimeError(
            f"could not persist admission event for run_id={run_id}"
        )

    def clear_admission_failure_events(self, run_id: str) -> int:
        """Delete stale ``admission_failed`` error terminals of ``run_id``.

        A failed admission attempt may have persisted an ``error`` terminal
        before the repair retry succeeded. That frame belonged to the
        failed attempt, not to the run: keeping it would give the repaired
        run two terminals and put an error before its admission event.
        The durable admission event carries the repaired outcome, so the
        stale terminal is removed in the same repair operation.
        """
        rows = self._conn.execute(
            "SELECT event_id, payload_json FROM run_events "
            "WHERE run_id = ? AND event_type = 'error'",
            (run_id,),
        ).fetchall()
        removed = 0
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError):
                continue
            if payload.get("code") != "admission_failed":
                continue
            self._conn.execute(
                "DELETE FROM run_events WHERE event_id = ?",
                (row["event_id"],),
            )
            removed += 1
        return removed

    def get_pending_actions(self, run_id: str) -> list[dict[str, Any]]:
        """Return tool_call events that have no matching tool_result yet.

        A tool_call without a tool_result means the tool was started but
        did not finish before the stream closed (network drop, process
        restart). The snapshot endpoint exposes these as "pending
        actions" so the user knows which steps did not complete.
        """
        import json

        rows = self._conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? "
            "AND event_type = 'tool_call' "
            "ORDER BY sequence ASC",
            (run_id,),
        ).fetchall()
        pending: list[dict[str, Any]] = []
        for r in rows:
            payload = json.loads(r["payload_json"])
            tool = payload.get("tool", "unknown")
            # Look for a later tool_result with the same tool name
            result_row = self._conn.execute(
                "SELECT sequence FROM run_events "
                "WHERE run_id = ? AND event_type = 'tool_result' "
                "AND sequence > ? "
                "AND json_extract(payload_json, '$.tool') = ? "
                "ORDER BY sequence ASC LIMIT 1",
                (run_id, r["sequence"], tool),
            ).fetchone()
            if result_row is None:
                pending.append(
                    {
                        "tool": tool,
                        "server_id": payload.get("server_id", ""),
                        "sequence": r["sequence"],
                        "args": payload.get("args", {}),
                    }
                )
        return pending


# ---- Exceptions ----------------------------------------------------------


class RunNotFound(LookupError):
    """Raised when a run_id does not exist in the control store."""


class RunAlreadyTerminal(RuntimeError):
    """Raised when an operation tries to mutate a terminal run.

    Spec AC5: cancelled runs produce a unique cancelled terminal and
    subsequent continuation is rejected. Same applies to done/error.
    """
