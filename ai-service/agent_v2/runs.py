"""Ticket 06 — Run lifecycle facade.

Composes :class:`ControlStore` + :class:`CheckpointsStore` into a
single API the endpoint layer consumes. The facade hides the two-database
split so ``api/chat_v2.py`` deals with one object: ``RunRepository``.

Responsibilities:

* Idempotent run creation — same ``request_id`` returns the same run
  without re-emitting user messages or re-invoking tools (spec AC2).
* Run status transitions — ``running`` → ``done``/``error``/``cancelled``.
  Terminal runs reject further transitions (spec AC5).
* Snapshot assembly — conversation + status + sequence cursor + pending
  actions + user-recoverable content, no raw framework events (spec AC3).
* Event replay — return events with sequence > ``from_seq`` for
  reconnect (spec AC4).
* Idempotent cancel — repeat cancels return the original outcome
  (spec AC5).
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from agent_v2.execution_contract import (
    EXECUTION_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    # BCDEF review S3: the terminal run-status vocabulary has ONE source.
    TERMINAL_STATUSES,
    BoundaryError,
    ExecutionErrorCode,
    ProgressGap,
    compute_gap,
    policy_hash,
)
from agent_v2.effect_repair import effect_known_facts, effect_unknown_part
from agent_v2.effects import (
    EffectStatus,
    repair_age_seconds,
    repair_next_action,
)
from agent_v2.observability import (
    observe_boundary_metric,
    observe_boundary_repair_age,
)
from agent_v2.execution_policy import (
    ResolvedPolicySnapshot,
    profile_from_policy,
    serialize_policy,
)
from agent_v2.policy_adapters import (
    adapt_policy_snapshot,
    describe_policy_record,
    missing_snapshot_failure,
)
from agent_v2.persistence import (
    ApprovalRecord,
    ArtifactRecord,
    AssistantMessagePersistenceError,
    CheckpointsStore,
    ControlStore,
    EffectRecord,
    FinalizeRecord,
    JanitorTaskRecord,
    PolicySnapshotRecord,
    PolicySnapshotWrite,
    RunAlreadyTerminal,
    RunNotFound,
    RunRecord,
    SandboxLeaseRecord,
    TerminalEventPersistenceError,
)
from agent_v2.profiles import CapabilityProfile

logger = logging.getLogger("ai-service.agent_v2.runs")

# BCDEF review S3: the run terminal vocabulary in this module is derived
# from :data:`agent_v2.execution_contract.TERMINAL_STATUSES`, never
# re-spelled. Every set below (replay, admission, snapshot) is a view of
# that one source, so ``outcome_unknown`` — terminal and non-continuable —
# cannot be forgotten by a single guard again.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    status.value for status in TERMINAL_STATUSES
)

# Ticket 05 (T07): the bounded window for the model-visible conversation
# history. The limits apply to the HISTORY only — the current user message
# is appended by the kernel afterwards and never counts against them.
HISTORY_MAX_MESSAGES = 20
HISTORY_MAX_CHARS = 8000

# Newest-first page size for the history scan. The scan stops as soon as
# the next older completed turn cannot fit the window, so a window-sized
# conversation is read with one or two pages.
_HISTORY_PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Spec AC3 snapshot payload.

    Fields are deliberately product-facing — no raw framework events,
    no internal store paths. ``pending_actions`` exposes tool_calls
    that started but did not complete before the stream closed.
    """

    run_id: str
    request_id: str
    conversation_id: str
    profile_id: str
    status: str
    terminal_type: str | None
    created_at: str
    updated_at: str
    terminal_at: str | None
    sequence_cursor: int
    messages: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]
    # BCDEF ticket 08: the frozen execution policy identity, so snapshot
    # readers can explain which policy a run executed under.
    policy_hash: str = ""
    policy_schema_version: int = 0
    # BCDEF ticket 03: terminal persistence / replay explanation. All
    # additive — the existing envelope and snapshot path are unchanged.
    control_status: str = ""
    terminal_event_type: str | None = None
    terminal_event_id: str = ""
    terminal_event_sequence: int = 0
    terminal_persisted: bool = False
    terminal_persist_failed: bool = False
    terminal_error_code: str | None = None
    error_code: str | None = None
    recoverable: bool = False
    state_reconciled: bool = False
    # BCDEF ticket 04: replay-visible effect metadata so status/result
    # are reconstructible from the snapshot as well as the event log.
    effects: list[dict[str, Any]] = field(default_factory=list)
    # BCDEF ticket 06: lease status + fencing token for replay/audit.
    sandbox_lease: dict[str, Any] | None = None
    # BCDEF ticket 09: adapter provenance (source/target version, adapter
    # id, timestamp) and, when the stored policy cannot be adapted, the
    # stable failure reason + next action. Both are additive.
    policy_provenance: dict[str, Any] | None = None
    policy_failure: dict[str, Any] | None = None
    # BCDEF ticket 11: pending approval rows projected into the
    # ``approval.required`` payload shape, so a refresh while paused can
    # restore the approval prompt even when the event replay starts
    # after the approval frame. Additive — old clients ignore it.
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    # BCDEF ticket 10: additive projection summary of the frozen policy —
    # policy hash, capability ids/count, MCP services, risk/approval
    # policy. None when the run has no projectable frozen snapshot.
    capability_projection: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CancelOutcome:
    """Spec AC5 cancel result."""

    accepted: bool
    reason: str
    run_id: str


class AssistantPersistFailure(BoundaryError):
    """The assistant reply could not be made durable before the terminal."""

    def __init__(self, message: str = "") -> None:
        super().__init__(
            ExecutionErrorCode.ASSISTANT_PERSIST_FAILED,
            message,
            recoverable=True,
        )


class TerminalPersistFailure(BoundaryError):
    """The terminal event could not be made durable; the stream must close."""

    def __init__(self, message: str = "") -> None:
        super().__init__(
            ExecutionErrorCode.TERMINAL_PERSIST_FAILED,
            message,
            recoverable=True,
        )


@dataclass(frozen=True, slots=True)
class FinalizeResult:
    """Outcome of the assistant + terminal finalize fence.

    ``terminal`` is the authoritative persisted envelope (a dict in the v2
    wire shape), ``created`` is False when an existing terminal was
    returned idempotently, and ``reconciled`` marks a control-DB CAS that
    failed after the durable event commit (snapshot/replay reconcile the
    status from the event).
    """

    terminal: dict[str, Any]
    created: bool
    status: str
    assistant_persisted: bool
    assistant_content: str
    reconciled: bool = False


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Outcome of the admission fence (BCDEF ticket 02).

    ``execute`` is True for exactly one caller per request_id: the CAS
    winner of ``admission_pending``/``admission_failed`` → ``admitted``.
    Duplicates and terminal replays therefore never start a second model
    turn. ``admission_event`` is the persisted ``run_started`` envelope
    data so orchestration can emit the exact durable frame (same
    event_id/sequence/payload as replay).
    """

    run_id: str
    conversation_id: str
    status: str
    created: bool = False
    replayed: bool = False
    repaired: bool = False
    execute: bool = False
    error_code: str | None = None
    recoverable: bool = False
    admission_event: dict[str, Any] | None = None
    policy_hash: str = ""
    policy_schema_version: int = 0
    policy: Mapping[str, Any] | None = None
    profile: CapabilityProfile | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None


# Statuses that a retry must replay, never repair or execute again.
# BCDEF review S3: derived from the shared contract, so a new terminal
# status cannot silently become repairable. ``admission_failed`` is the one
# documented exception — ticket 02's admission repair backfills the missing
# admission records of the SAME request; it never continues an execution.
_NON_REPAIRABLE_STATUSES = _TERMINAL_STATUSES - {"admission_failed"}

# Run-lifecycle terminal event types (``run.paused`` is a stream terminal
# only and deliberately excluded).
_RUN_TERMINAL_TYPES = frozenset({"done", "error", "cancelled"})

# Terminal control statuses whose replay may need a synthetic terminal
# when the durable event write never happened. ``outcome_unknown`` (BCDEF
# ticket 11 surface for ticket 05's reconciliation statuses) is terminal
# and non-continuable, so its replay carries a stable synthetic terminal
# instead of reading as an endless running stream.
_REPLAY_TERMINAL_STATUSES = _TERMINAL_STATUSES


def _admission_payload(
    *,
    status: str,
    created: bool,
    replayed: bool,
    repaired: bool,
    policy_hash_value: str = "",
    schema_version: int = 0,
) -> dict[str, Any]:
    admission: dict[str, Any] = {
        "status": status,
        "created": created,
        "replayed": replayed,
        "repaired": repaired,
    }
    if policy_hash_value:
        admission["policy_hash"] = policy_hash_value
        admission["schema_version"] = schema_version
    return {"admission": admission}


def _observe_admission_outcome(admit):
    """BCDEF ticket 12: count every failed admission at the one fence.

    :meth:`RunRepository.admit` is the only production entry that creates a
    durable run (chat and the memory export both go through it), so this
    single wrapper covers every caller without touching the failure paths
    themselves. The counter is observability only: it can never change the
    returned :class:`AdmissionResult`.
    """

    @functools.wraps(admit)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        result = admit(self, *args, **kwargs)
        if (
            isinstance(result, AdmissionResult)
            and result.status == "admission_failed"
        ):
            observe_boundary_metric(ExecutionErrorCode.ADMISSION_FAILED)
        return result

    return wrapper


class RunRepository:
    """Facade over ControlStore + CheckpointsStore.

    The two stores live in separate SQLite files (spec decision #40)
    but are composed here so the endpoint layer doesn't need to know
    the split. Operations that span both stores (e.g. create_run
    inserts into control.runs + checkpoints.conversations) are wrapped
    in a single method — partial failure leaves the system recoverable
    because the run_id is the join key and missing conversation rows
    are recreated on the next message.
    """

    def __init__(self, control: ControlStore, checkpoints: CheckpointsStore) -> None:
        self.control = control
        self.checkpoints = checkpoints

    def close(self) -> None:
        self.control.close()
        self.checkpoints.close()

    # -- create + status ------------------------------------------------

    def create_run_idempotent(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
        user_message: str,
    ) -> tuple[str, bool]:
        """Create a run + persist the user message, idempotent on request_id.

        Returns ``(run_id, created)``. When ``created=False`` the run
        already exists from a prior call with the same request_id — the
        user message is NOT re-appended and no model/tool call happens
        (spec AC2).

        The conversation row is upserted so a returning user (same
        conversation_id) extends the same history across runs.
        """
        run_id, created = self.control.create_run_idempotent(
            request_id=request_id,
            conversation_id=conversation_id,
            profile_id=profile_id,
        )
        if created:
            # Only persist user message for a fresh run — replays skip.
            self.checkpoints.ensure_conversation(
                conversation_id=conversation_id,
                profile_id=profile_id,
            )
            # Sequence 0 = user input emitted before the first v2 event.
            # Real v2 events start at sequence 1 (per EventStream).
            self.checkpoints.append_message(
                conversation_id=conversation_id,
                run_id=run_id,
                role="user",
                content=user_message,
                sequence=0,
            )
        return run_id, created

    # -- admission fence (BCDEF ticket 02) ------------------------------

    @_observe_admission_outcome
    def admit(
        self,
        *,
        request_id: str,
        conversation_id: str | None,
        profile_id: str,
        user_message: str,
        policy: Mapping[str, Any] | None = None,
    ) -> AdmissionResult:
        """Durably admit one request before any model or sandbox work.

        The fence is: run row + frozen policy snapshot → canonical
        conversation → user message → admission (``run_started``) event →
        ``admitted`` CAS. Any required write failure returns
        ``admission_failed`` (recoverable) without a model turn. A
        same-request retry keeps the original run_id and only backfills
        the missing records; at most one durable run, one user message and
        one admission event exist per request_id.

        ``policy`` is the complete execution policy built at THIS
        admission from current config. It is hashed and stored atomically
        with the run, and the returned ``profile`` is reconstructed from
        the stored snapshot so execution never re-reads config.
        """
        supplied = (conversation_id or "").strip()
        canonical = supplied or f"conv-{request_id}"

        policy_write: PolicySnapshotWrite | None = None
        expected_hash = ""
        expected_schema = 0
        if policy is not None:
            try:
                expected_schema = int(
                    policy.get("schema_version", EXECUTION_SCHEMA_VERSION)
                )
                if expected_schema not in SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(
                        f"unsupported policy schema {expected_schema}"
                    )
                expected_hash = policy_hash(policy)
                policy_write = PolicySnapshotWrite(
                    policy_json=serialize_policy(policy),
                    policy_hash=expected_hash,
                    schema_version=expected_schema,
                    source_version=str(policy.get("source_version", "")),
                    compatibility_json=json.dumps(
                        dict(policy.get("compatibility", {})),
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception:
                logger.exception(
                    "admission: policy snapshot build failed for request_id=%s",
                    request_id,
                )
                return AdmissionResult(
                    run_id="",
                    conversation_id=canonical,
                    status="admission_failed",
                    error_code=ExecutionErrorCode.POLICY_INCOMPATIBLE.value,
                    recoverable=False,
                )

        try:
            run, created = self.control.admit_run(
                request_id=request_id,
                conversation_id=canonical,
                profile_id=profile_id,
                policy=policy_write,
            )
        except Exception:
            logger.exception(
                "admission: run write failed for request_id=%s", request_id
            )
            return AdmissionResult(
                run_id="",
                conversation_id=canonical,
                status="admission_failed",
                error_code="admission_failed",
                recoverable=True,
            )

        run_id = run.run_id
        canonical = run.conversation_id
        prior_status = run.status
        replayed = not created

        if prior_status in _NON_REPAIRABLE_STATUSES:
            stored = self.control.get_policy_snapshot(run_id)
            return AdmissionResult(
                run_id=run_id,
                conversation_id=canonical,
                status=prior_status,
                created=created,
                replayed=True,
                execute=False,
                admission_event=self._admission_event_dict(run_id),
                policy_hash=stored.policy_hash if stored is not None else "",
                policy_schema_version=(
                    stored.schema_version if stored is not None else 0
                ),
            )

        # Ticket 08: a run without a frozen snapshot may not execute. A
        # legacy/pre-policy run gets its snapshot on the first admission
        # attempt; failure here keeps the run repairable and refuses
        # execution.
        if policy_write is not None:
            try:
                if self.control.get_policy_snapshot(run_id) is None:
                    self.control.insert_policy_snapshot(
                        run_id=run_id, policy=policy_write
                    )
            except Exception:
                logger.exception(
                    "admission: policy snapshot write failed for run_id=%s",
                    run_id,
                )
                if prior_status in ("admission_pending", "admission_failed"):
                    self._mark_admission_failed(run_id)
                return AdmissionResult(
                    run_id=run_id,
                    conversation_id=canonical,
                    status="admission_failed",
                    created=created,
                    replayed=replayed,
                    execute=False,
                    error_code=ExecutionErrorCode.POLICY_INCOMPATIBLE.value,
                    recoverable=False,
                    admission_event=self._admission_event_dict(run_id),
                )

        repaired = False
        try:
            conversation_created = self.checkpoints.ensure_conversation(
                conversation_id=canonical,
                profile_id=profile_id,
            )
            _message_id, message_created = self.checkpoints.ensure_user_message(
                conversation_id=canonical,
                run_id=run_id,
                content=user_message,
            )
            repaired = replayed and (conversation_created or message_created)
            if prior_status == "admission_failed":
                repaired = True
            outcome = "repaired" if repaired else "admitted"
            _event, event_created = self.checkpoints.ensure_admission_event(
                run_id=run_id,
                conversation_id=canonical,
                request_id=request_id,
                payload=_admission_payload(
                    status=outcome,
                    created=created,
                    replayed=replayed,
                    repaired=repaired,
                    policy_hash_value=expected_hash,
                    schema_version=expected_schema,
                ),
            )
            if replayed and event_created:
                repaired = True
                self.checkpoints.ensure_admission_event(
                    run_id=run_id,
                    conversation_id=canonical,
                    request_id=request_id,
                    payload=_admission_payload(
                        status="repaired",
                        created=created,
                        replayed=True,
                        repaired=True,
                        policy_hash_value=expected_hash,
                        schema_version=expected_schema,
                    ),
                )
        except Exception:
            logger.exception(
                "admission: checkpoint fence failed for request_id=%s run_id=%s",
                request_id,
                run_id,
            )
            self._mark_admission_failed(run_id)
            return AdmissionResult(
                run_id=run_id,
                conversation_id=canonical,
                status="admission_failed",
                created=created,
                replayed=replayed,
                repaired=repaired,
                execute=False,
                error_code="admission_failed",
                recoverable=True,
                admission_event=self._admission_event_dict(run_id),
                policy_hash=expected_hash,
                policy_schema_version=expected_schema,
            )

        # Validate the stored snapshot BEFORE claiming the run for
        # execution: a missing/corrupt snapshot fails the admission fence
        # instead of reaching the model.
        resolved: ResolvedPolicySnapshot | None = None
        frozen_profile: CapabilityProfile | None = None
        if policy_write is not None:
            try:
                resolved = self.resolve_policy_snapshot(run_id)
                frozen_profile = resolved.profile()
            except BoundaryError as exc:
                logger.warning(
                    "admission: policy snapshot unusable for run_id=%s: %s",
                    run_id,
                    exc,
                )
                if prior_status in ("admission_pending", "admission_failed"):
                    self._mark_admission_failed(run_id)
                return AdmissionResult(
                    run_id=run_id,
                    conversation_id=canonical,
                    status="admission_failed",
                    created=created,
                    replayed=replayed,
                    repaired=repaired,
                    execute=False,
                    error_code=exc.code.value,
                    recoverable=exc.recoverable,
                    admission_event=self._admission_event_dict(run_id),
                    policy_hash=expected_hash,
                    policy_schema_version=expected_schema,
                )

        try:
            repair_edge = prior_status == "admission_failed"
            won = self.control.mark_run_admitted(run_id=run_id, repair=repair_edge)
            if not won:
                current = self.control.get_run(run_id)
                if current is not None and current.status == "admission_failed":
                    won = self.control.mark_run_admitted(run_id=run_id, repair=True)
                    if won:
                        repair_edge = True
                        repaired = True
                elif current is not None:
                    return AdmissionResult(
                        run_id=run_id,
                        conversation_id=canonical,
                        status=current.status,
                        created=created,
                        replayed=True,
                        repaired=repaired,
                        execute=False,
                        admission_event=self._admission_event_dict(run_id),
                        policy_hash=expected_hash,
                        policy_schema_version=expected_schema,
                    )
            if won and repair_edge:
                # Supersede the failed attempt: drop its stale error
                # terminal and record the repaired admission outcome.
                self.checkpoints.clear_admission_failure_events(run_id)
                self.checkpoints.ensure_admission_event(
                    run_id=run_id,
                    conversation_id=canonical,
                    request_id=request_id,
                    payload=_admission_payload(
                        status="repaired",
                        created=created,
                        replayed=True,
                        repaired=True,
                        policy_hash_value=expected_hash,
                        schema_version=expected_schema,
                    ),
                )
                repaired = True
        except Exception:
            logger.exception(
                "admission: admitted transition failed for request_id=%s run_id=%s",
                request_id,
                run_id,
            )
            self._mark_admission_failed(run_id)
            return AdmissionResult(
                run_id=run_id,
                conversation_id=canonical,
                status="admission_failed",
                created=created,
                replayed=replayed,
                repaired=repaired,
                execute=False,
                error_code="admission_failed",
                recoverable=True,
                admission_event=self._admission_event_dict(run_id),
                policy_hash=expected_hash,
                policy_schema_version=expected_schema,
            )

        if not won:
            return AdmissionResult(
                run_id=run_id,
                conversation_id=canonical,
                status="admitted",
                created=created,
                replayed=True,
                repaired=repaired,
                execute=False,
                admission_event=self._admission_event_dict(run_id),
                policy_hash=expected_hash,
                policy_schema_version=expected_schema,
            )

        return AdmissionResult(
            run_id=run_id,
            conversation_id=canonical,
            status="admitted",
            created=created,
            replayed=replayed,
            repaired=repaired,
            execute=True,
            admission_event=self._admission_event_dict(run_id),
            policy_hash=expected_hash,
            policy_schema_version=expected_schema,
            policy=(dict(resolved.policy) if resolved is not None else None),
            profile=frozen_profile,
        )

    def resolve_policy_snapshot(self, run_id: str) -> ResolvedPolicySnapshot:
        """Return the frozen policy of ``run_id`` through the adapter chain.

        Fails closed with ``policy_snapshot_missing`` when no snapshot was
        written, and with ``policy_incompatible`` when the stored content
        cannot be adapted safely (unknown schema version, hash mismatch,
        missing critical field, adapter failure). Ticket 09: a legacy
        snapshot (version 0) is lifted to the native version by the
        registered, pure adapter; an incompatible one NEVER falls back to
        current defaults.
        """
        record = self.control.get_policy_snapshot(run_id)
        if record is None:
            raise BoundaryError(
                ExecutionErrorCode.POLICY_SNAPSHOT_MISSING,
                f"run {run_id} has no frozen execution policy snapshot",
            )
        return adapt_policy_snapshot(record)

    def get_policy_snapshot(self, run_id: str) -> PolicySnapshotRecord | None:
        """Raw persisted snapshot row (no validation), or None."""
        return self.control.get_policy_snapshot(run_id)

    def _admission_event_dict(self, run_id: str) -> dict[str, Any] | None:
        event = self.checkpoints.get_admission_event(run_id)
        if event is None:
            return None
        return {
            "version": "2",
            "event_id": event.event_id,
            "sequence": event.sequence,
            "run_id": event.run_id,
            "conversation_id": event.conversation_id,
            "timestamp": event.created_at,
            "type": event.event_type,
            "request_id": event.request_id,
            "payload": json.loads(event.payload_json),
        }

    def _mark_admission_failed(self, run_id: str) -> None:
        try:
            self.control.mark_admission_failed(run_id)
        except Exception:
            logger.exception(
                "admission: could not persist admission_failed for run_id=%s",
                run_id,
            )

    def mark_run_running(self, run_id: str) -> bool:
        """Idempotently enter execution after the admission fence."""
        return self.control.mark_run_running(run_id)

    def get_admission_event(self, run_id: str) -> dict[str, Any] | None:
        """The persisted admission envelope data, or None."""
        return self._admission_event_dict(run_id)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.control.get_run(run_id)

    def list_paused_runs_updated_before(self, updated_before: str) -> list[RunRecord]:
        """List abandoned paused runs — paused-run janitor support.

        See :meth:`ControlStore.list_paused_runs_updated_before`.
        """
        return self.control.list_paused_runs_updated_before(updated_before)

    def resolve_run_id(self, identifier: str) -> str | None:
        """Resolve an identifier to a real run_id.

        Tries ``run_id`` first, then falls back to ``request_id``. This
        supports early cancel (spec AC4): when the user cancels before
        the first v2 event has delivered ``run_id`` to the client, the
        frontend only knows ``request_id`` and sends it as the path
        parameter. Without this fallback the server returns
        ``unknown_run`` and the run keeps running on the server.
        """
        run = self.control.get_run(identifier)
        if run is not None:
            return run.run_id
        run = self.control.get_run_by_request(identifier)
        if run is not None:
            return run.run_id
        return None

    # -- sandbox leases (Ticket 09 + BCDEF ticket 06) ------------------

    def create_sandbox_lease(
        self,
        *,
        run_id: str,
        sandbox_type: str,
        workspace_path: str,
        owner_id: str = "",
        sandbox_identity: str = "",
        ttl_seconds: float | None = None,
    ) -> str:
        """Persist a sandbox lease for ``run_id`` (spec AC1).

        Delegates to :class:`ControlStore`. The caller (stream endpoint)
        creates the workspace via :class:`Computer` first, then records
        the lease. A run may have at most one lease.
        """
        return self.control.create_sandbox_lease(
            run_id=run_id,
            sandbox_type=sandbox_type,
            workspace_path=workspace_path,
            owner_id=owner_id,
            sandbox_identity=sandbox_identity,
            ttl_seconds=ttl_seconds,
        )

    def acquire_sandbox_lease(
        self,
        *,
        run_id: str,
        owner_id: str,
        sandbox_type: str,
        workspace_path: str,
        sandbox_identity: str = "",
        ttl_seconds: float | None = None,
    ) -> tuple[SandboxLeaseRecord | None, bool]:
        """Acquire the run's active lease (one generation, ticket 06)."""
        return self.control.acquire_sandbox_lease(
            run_id=run_id,
            owner_id=owner_id,
            sandbox_type=sandbox_type,
            workspace_path=workspace_path,
            sandbox_identity=sandbox_identity,
            ttl_seconds=ttl_seconds,
        )

    def get_sandbox_lease(self, run_id: str) -> SandboxLeaseRecord | None:
        """Return the sandbox lease bound to ``run_id``, if any."""
        return self.control.get_sandbox_lease(run_id)

    def renew_sandbox_lease(
        self,
        *,
        run_id: str,
        fencing_token: int,
        owner_id: str = "",
        ttl_seconds: float | None = None,
        now: Any | None = None,
    ) -> tuple[SandboxLeaseRecord | None, bool]:
        """Heartbeat the lease under its fencing token (ticket 06)."""
        return self.control.renew_sandbox_lease(
            run_id=run_id,
            fencing_token=fencing_token,
            owner_id=owner_id,
            ttl_seconds=ttl_seconds,
            now=now,
        )

    def expire_sandbox_lease(
        self, run_id: str, *, now: Any | None = None
    ) -> SandboxLeaseRecord | None:
        """Mark an overdue active lease expired (idempotent CAS).

        BCDEF ticket 12: the ``lease_expired`` metric counts real
        transitions (active → expired). Re-expiring an already expired row
        is not a new incident, so it is not counted again.
        """
        before = self.control.get_sandbox_lease(run_id)
        record = self.control.expire_sandbox_lease(run_id, now=now)
        if (
            record is not None
            and record.status == "expired"
            and (before is None or before.status != "expired")
        ):
            observe_boundary_metric(ExecutionErrorCode.LEASE_EXPIRED)
        return record

    def release_sandbox_lease(
        self,
        *,
        run_id: str,
        fencing_token: int | None = None,
        resource_usage: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> tuple[SandboxLeaseRecord | None, bool]:
        """Idempotent fenced release of the active lease (ticket 06)."""
        return self.control.release_sandbox_lease(
            run_id=run_id,
            fencing_token=fencing_token,
            resource_usage=resource_usage,
            failure_reason=failure_reason,
        )

    def destroy_sandbox_lease(
        self,
        *,
        run_id: str,
        resource_usage: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> SandboxLeaseRecord | None:
        """Mark the lease destroyed (clean) or failed (cleanup error).

        Spec AC6: run completion or cancellation triggers sandbox cleanup.
        The lease records success/failure + resource usage so operators
        can audit cost and investigate stale sandboxes. Idempotent —
        repeated cleanup keeps the first release's evidence.
        """
        return self.control.destroy_sandbox_lease(
            run_id=run_id,
            resource_usage=resource_usage,
            failure_reason=failure_reason,
        )

    def record_sandbox_cleanup(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        janitor_task_id: str | None = None,
    ) -> SandboxLeaseRecord | None:
        """Record the terminal cleanup outcome (BCDEF ticket 07)."""
        return self.control.record_sandbox_cleanup(
            run_id,
            status=status,
            error=error,
            janitor_task_id=janitor_task_id,
        )

    # -- janitor work items (BCDEF ticket 07) --------------------------

    def enqueue_janitor_task(
        self,
        *,
        kind: str,
        evidence: dict[str, Any] | None = None,
        run_id: str = "",
        lease_id: str = "",
        effect_id: str = "",
        dedupe_key: str | None = None,
        task_id: str | None = None,
    ) -> tuple[JanitorTaskRecord, bool]:
        task, created = self.control.enqueue_janitor_task(
            kind=kind,
            evidence=evidence,
            run_id=run_id,
            lease_id=lease_id,
            effect_id=effect_id,
            dedupe_key=dedupe_key,
            task_id=task_id,
        )
        # BCDEF ticket 12: ``cleanup_pending`` counts sandbox cleanups that
        # did not finish inline and became durable janitor work. Only a
        # NEW item counts — the dedupe key means a retry re-observes the
        # same outstanding backlog item, not a new one.
        if created and kind == "cleanup":
            observe_boundary_metric(ExecutionErrorCode.CLEANUP_PENDING)
        return task, created

    def get_janitor_task(self, task_id: str) -> JanitorTaskRecord | None:
        return self.control.get_janitor_task(task_id)

    def list_janitor_tasks(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 100,
        claimable: bool = False,
    ) -> list[JanitorTaskRecord]:
        return self.control.list_janitor_tasks(
            status=status, kind=kind, limit=limit, claimable=claimable
        )

    def claim_janitor_task(
        self, task_id: str, *, owner: str
    ) -> tuple[JanitorTaskRecord | None, bool]:
        """CAS claim; the returned record carries the generation + deadline.

        BCDEF review S4: the caller MUST pass ``claimed.attempts`` back as
        ``expected_attempt`` when it settles, otherwise the store rejects
        the write.
        """
        return self.control.claim_janitor_task(task_id, owner=owner)

    def complete_janitor_task(
        self,
        task_id: str,
        *,
        owner: str,
        expected_attempt: int,
        result: dict[str, Any] | None = None,
    ) -> tuple[JanitorTaskRecord | None, bool]:
        return self.control.complete_janitor_task(
            task_id,
            owner=owner,
            expected_attempt=expected_attempt,
            result=result,
        )

    def fail_janitor_task(
        self,
        task_id: str,
        error: str,
        *,
        owner: str,
        expected_attempt: int,
    ) -> tuple[JanitorTaskRecord | None, bool]:
        return self.control.fail_janitor_task(
            task_id, error, owner=owner, expected_attempt=expected_attempt
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
        """Persist an exported artifact (Ticket 10).

        Spec AC3: ``artifact_id`` is opaque; ``storage_path`` never
        leaves the host. Spec AC4: a ``status='failed'`` row is allowed
        so evidence is retained before the sandbox is destroyed.
        """
        return self.control.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            public_filename=public_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            storage_path=storage_path,
            status=status,
            failure_reason=failure_reason,
            created_at=created_at,
            expires_at=expires_at,
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        return self.control.get_artifact(artifact_id)

    def list_artifacts_for_run(self, run_id: str) -> list[ArtifactRecord]:
        return self.control.list_artifacts_for_run(run_id)

    def mark_artifact_expired(self, artifact_id: str) -> ArtifactRecord | None:
        return self.control.mark_artifact_expired(artifact_id)

    def mark_run_done(self, run_id: str) -> None:
        """Transition a running run to ``done`` terminal.

        Raises ``RunAlreadyTerminal`` if the run already terminated
        (e.g. cancel beat the stream to the transition). Callers that
        want idempotent semantics wrap this in ``_mark_terminal_safe``
        — the repository surfaces the constraint so direct callers
        know continuation was rejected (spec AC5).
        """
        self.control.update_run_status(
            run_id=run_id,
            status="done",
            terminal_type="done",
        )

    def mark_run_error(self, run_id: str) -> None:
        """Transition a running run to ``error`` terminal.

        Raises ``RunAlreadyTerminal`` if the run already terminated;
        see :meth:`mark_run_done` for the rationale.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="error",
            terminal_type="error",
        )

    # -- paused/resume (Ticket 11) ------------------------------------

    def mark_run_paused(self, run_id: str) -> None:
        """Transition a running run to ``paused`` (spec AC3).

        ``paused`` is NOT terminal — the run is awaiting user approval
        and will resume via :meth:`resume_run`. The RunAlreadyTerminal
        guard allows this transition because ``paused`` is not in
        :data:`_TERMINAL_STATUSES`.

        Raises ``RunAlreadyTerminal`` if the run already reached a
        terminal state (done/error/cancelled) — a terminal run cannot
        pause because its lifecycle is over.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="paused",
            terminal_type=None,
        )

    def resume_run(self, run_id: str) -> None:
        """Transition a paused run back to ``running`` (spec AC4).

        Called by the decision endpoint after the user approves or
        rejects the pending approval. The stream that follows reads the
        run's current state and continues from where it paused.

        Raises ``RunAlreadyTerminal`` if the run reached a terminal
        state while paused (e.g. cancel API was called, or an operator
        force-terminated the run). This is the spec AC5 invariant: a
        terminal run cannot be resumed.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="running",
            terminal_type=None,
        )

    def is_paused(self, run_id: str) -> bool:
        """Check whether a run is paused awaiting approval."""
        run = self.control.get_run(run_id)
        if run is None:
            return False
        return run.status == "paused"

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
        prepared_content: str | None = None,
    ) -> ApprovalRecord:
        """Persist a new approval row in ``pending`` status (spec AC2).

        Delegates to :class:`ControlStore`. The approval id is computed
        deterministically in :mod:`agent_v2.approvals`; the DB UNIQUE
        constraint catches duplicate inserts at the layer of truth.

        BCDEF ticket 07: ``prepared_content`` is the tool input drafted
        BEFORE the pause (the plan text for ``generate_pdf``). Resume
        reads it instead of re-running the model.
        """
        return self.control.insert_approval(
            approval_id=approval_id,
            run_id=run_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
            risk_level=risk_level,
            impact=impact,
            created_at=created_at,
            expires_at=expires_at,
            prepared_content=prepared_content,
        )

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        return self.control.get_approval(approval_id)

    def list_pending_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        return self.control.list_pending_approvals_for_run(run_id)

    def list_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        return self.control.list_approvals_for_run(run_id)

    def resolve_approval(
        self,
        *,
        approval_id: str,
        status: str,
        decision_request_id: str,
    ) -> ApprovalRecord | None:
        """Transition an approval to ``approved`` or ``rejected`` (spec AC4/AC5).

        Delegates to :class:`ControlStore`. The store enforces
        idempotency on ``decision_request_id`` and rejects transitions
        on already-terminal approvals — the first decision wins, so the
        tool executes at most once.
        """
        return self.control.resolve_approval(
            approval_id=approval_id,
            status=status,
            decision_request_id=decision_request_id,
        )

    def mark_approval_expired(self, approval_id: str) -> ApprovalRecord | None:
        return self.control.mark_approval_expired(approval_id)

    # -- effects (Ticket 12) -------------------------------------------

    def insert_effect(
        self,
        *,
        effect_key: str,
        run_id: str,
        conversation_id: str,
        tool_name: str,
        args_canonical: str,
        external_idempotency_key: str,
        effect_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> EffectRecord:
        """Persist a new effect row in ``authorized`` status (spec AC1).

        Delegates to :class:`ControlStore`. The UNIQUE(run_id, tool_name,
        args_canonical) constraint catches duplicate inserts at the DB
        layer; callers that want idempotent semantics call
        :meth:`get_effect_by_tool_call` first (spec AC5).
        """
        return self.control.insert_effect(
            effect_key=effect_key,
            run_id=run_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
            external_idempotency_key=external_idempotency_key,
            effect_id=effect_id,
            idempotency_key=idempotency_key,
        )

    def get_effect(self, effect_key: str) -> EffectRecord | None:
        return self.control.get_effect(effect_key)

    def get_effect_by_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        args_canonical: str,
    ) -> EffectRecord | None:
        return self.control.get_effect_by_tool_call(
            run_id=run_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
        )

    def list_effects_for_run(self, run_id: str) -> list[EffectRecord]:
        return self.control.list_effects_for_run(run_id)

    def transition_effect_to_executing(
        self,
        effect_key: str,
    ) -> tuple[EffectRecord | None, bool]:
        """CAS transition ``authorized`` → ``executing`` (spec AC7).

        Returns ``(record, won)``. Only one concurrent caller sees
        ``won=True``; others see the already-``executing`` row with
        ``won=False`` and must NOT re-execute.
        """
        return self.control.transition_effect_to_executing(effect_key)

    def claim_effect(
        self,
        effect_key: str,
        *,
        owner: str,
        claim_token: str,
        ttl_seconds: float,
        lease_id: str = "",
        fencing_token: int = 0,
    ) -> tuple[EffectRecord | None, bool]:
        """CAS claim with a fenced claim token (BCDEF ticket 04).

        Returns ``(record, won)``; the winner owns ``claim_token`` and is
        the only caller allowed to execute the side effect and settle it.
        """
        return self.control.claim_effect(
            effect_key,
            owner=owner,
            claim_token=claim_token,
            ttl_seconds=ttl_seconds,
            lease_id=lease_id,
            fencing_token=fencing_token,
        )

    def commit_effect(
        self,
        effect_key: str,
        result_json: str,
        *,
        claim_token: str | None = None,
        fencing_token: int | None = None,
    ) -> EffectRecord | None:
        return self.control.commit_effect(
            effect_key,
            result_json,
            claim_token=claim_token,
            fencing_token=fencing_token,
        )

    def mark_effect_outcome_unknown(
        self,
        effect_key: str,
        reason: str = "",
        *,
        claim_token: str | None = None,
    ) -> EffectRecord | None:
        record = self.control.mark_effect_outcome_unknown(
            effect_key, reason, claim_token=claim_token
        )
        # BCDEF ticket 12: the ledger settling an effect as unknown is the
        # durable ``outcome_unknown`` observation (spec 07). A CAS miss
        # returning an already-unknown row is still one more real
        # observation of an unresolved external effect.
        if record is not None and record.status == EffectStatus.OUTCOME_UNKNOWN.value:
            observe_boundary_metric(ExecutionErrorCode.OUTCOME_UNKNOWN)
        return record

    def mark_effect_failed(
        self,
        effect_key: str,
        reason: str = "",
        *,
        claim_token: str | None = None,
    ) -> EffectRecord | None:
        return self.control.mark_effect_failed(
            effect_key, reason, claim_token=claim_token
        )

    def mark_effect_publication_failed(
        self,
        effect_key: str,
        reason: str = "",
        *,
        claim_token: str | None = None,
    ) -> EffectRecord | None:
        """Fence a committed effect whose success event could not persist."""
        record = self.control.mark_effect_publication_failed(
            effect_key, reason, claim_token=claim_token
        )
        # Same metric as the crash path: from the outside the external
        # effect ran and its outcome is not durable (ticket 05 semantics).
        if record is not None and record.status == EffectStatus.OUTCOME_UNKNOWN.value:
            observe_boundary_metric(ExecutionErrorCode.OUTCOME_UNKNOWN)
        return record

    # -- effect repair (BCDEF ticket 05) -------------------------------

    def get_effect_by_effect_id(self, effect_id: str) -> EffectRecord | None:
        return self.control.get_effect_by_effect_id(effect_id)

    def claim_effect_repair(
        self,
        effect_key: str,
        *,
        owner: str,
    ) -> tuple[EffectRecord | None, bool]:
        """CAS-claim the reconciliation of an ``outcome_unknown`` effect."""
        return self.control.claim_effect_repair(effect_key, owner=owner)

    def complete_effect_repair(
        self,
        effect_key: str,
        result_json: str,
        *,
        request_id: str = "",
        actor: str = "",
    ) -> tuple[EffectRecord | None, bool]:
        """CAS-commit the operator-confirmed result on the original row."""
        return self.control.complete_effect_repair(
            effect_key,
            result_json,
            request_id=request_id,
            actor=actor,
        )

    def abandon_effect_repair(
        self,
        effect_key: str,
        *,
        request_id: str = "",
        actor: str = "",
        reason: str = "",
    ) -> tuple[EffectRecord | None, bool]:
        """Fail-closed repair decision; the effect stays ``outcome_unknown``."""
        return self.control.abandon_effect_repair(
            effect_key,
            request_id=request_id,
            actor=actor,
            reason=reason,
        )

    def fail_effect_repair(self, effect_key: str, error: str) -> EffectRecord | None:
        return self.control.fail_effect_repair(effect_key, error)

    def revert_effect_repair(
        self,
        effect_key: str,
        error: str,
        *,
        request_id: str = "",
    ) -> EffectRecord | None:
        return self.control.revert_effect_repair(
            effect_key, error, request_id=request_id
        )

    # -- events ---------------------------------------------------------

    def persist_event(
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
    ) -> None:
        """Append a v2 SSE event for replay.

        Spec AC4: sequence is monotonic per run. The UNIQUE(run_id,
        sequence) constraint enforces this at the DB layer — duplicate
        inserts raise IntegrityError, signaling a programming bug.

        ``event_id`` and ``timestamp`` are passed through from the wire
        envelope so the persisted row matches what the client saw (spec
        decision #26). When omitted, the store generates them.
        """
        self.checkpoints.append_event(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            request_id=request_id,
            conversation_id=conversation_id,
            event_id=event_id,
            timestamp=timestamp,
        )

    @staticmethod
    def _stored_event_dict(stored: Any) -> dict[str, Any]:
        return {
            "version": "2",
            "event_id": stored.event_id,
            "sequence": stored.sequence,
            "run_id": stored.run_id,
            "conversation_id": stored.conversation_id,
            "timestamp": stored.created_at,
            "type": stored.event_type,
            "request_id": stored.request_id,
            "payload": json.loads(stored.payload_json),
        }

    def get_events_since(
        self,
        *,
        run_id: str,
        from_seq: int,
    ) -> list[dict[str, Any]]:
        """Return events with sequence > from_seq as envelope dicts.

        Each dict matches the v2 wire format: ``version``, ``event_id``,
        ``sequence``, ``run_id``, ``conversation_id``, ``timestamp``,
        ``type``, ``request_id``, ``payload`` (spec decision #26).
        """
        stored = self.checkpoints.get_events_since(
            run_id=run_id,
            from_seq=from_seq,
        )
        return [self._stored_event_dict(s) for s in stored]

    # -- terminal finalize fence (BCDEF ticket 03) ----------------------

    def finalize_run(
        self,
        *,
        run_id: str,
        request_id: str,
        terminal_type: str,
        payload: dict[str, Any],
        sequence: int,
        event_id: str | None = None,
        timestamp: str | None = None,
        assistant_content: str | None = None,
    ) -> FinalizeResult:
        """Durably finalize one run: assistant → terminal event → status CAS.

        The assistant message (exactly once, derived per-conversation
        sequence) and the terminal event commit in ONE checkpoints
        transaction. Only after that commit does the control DB CAS the
        run status. The canonical conversation comes from the run row, so
        callers never need to pass ``conversation_id``.

        Failure semantics:

        * assistant write fails → :class:`AssistantPersistFailure`; no
          terminal was written, so the orchestration can safely publish a
          stable ``assistant_persist_failed`` error terminal instead;
        * terminal event write fails → :class:`TerminalPersistFailure` and
          the control row is flagged ``terminal_persist_failed``;
        * the status CAS fails after the event commit → the persisted
          terminal is still returned (``reconciled=True``); snapshot and
          replay reconcile the status from the event, and no second
          terminal is ever created.
        """
        if terminal_type not in _RUN_TERMINAL_TYPES:
            raise ValueError(f"invalid terminal type: {terminal_type!r}")
        run = self.control.get_run(run_id)
        if run is None:
            raise RunNotFound(f"run_id={run_id}")
        if run.status in ("admission_pending", "admission_failed"):
            raise BoundaryError(
                ExecutionErrorCode.ADMISSION_FAILED,
                f"run {run_id} is {run.status}; finalize is not an admission repair",
            )

        effective_type = terminal_type
        effective_payload = dict(payload)
        existing = self.checkpoints.get_terminal_event(run_id)
        if existing is not None:
            # A durable terminal already exists; return it exactly.
            effective_type = existing.event_type
        elif run.status in _RUN_TERMINAL_TYPES and run.status != terminal_type:
            # Cancelled (or otherwise terminated) while the stream was
            # finishing: the control fact wins. Persist that terminal
            # instead of publishing a contradicting one.
            effective_type = run.status
            effective_payload = {
                "code": run.status,
                "message": "运行已结束。",
                "unknown_dropped": payload.get("unknown_dropped", 0),
            }

        try:
            record: FinalizeRecord = self.checkpoints.finalize_terminal(
                run_id=run_id,
                conversation_id=run.conversation_id,
                request_id=request_id,
                terminal_type=effective_type,
                payload=effective_payload,
                sequence=sequence,
                event_id=event_id,
                timestamp=timestamp,
                assistant_content=assistant_content,
            )
        except AssistantMessagePersistenceError as exc:
            raise AssistantPersistFailure(str(exc)) from exc
        except TerminalEventPersistenceError as exc:
            self.record_terminal_persist_failure(run_id)
            raise TerminalPersistFailure(str(exc)) from exc

        status = record.terminal.event_type
        reconciled = False
        try:
            status, _transitioned = self.control.finalize_run_status(
                run_id=run_id,
                terminal_type=record.terminal.event_type,
            )
        except RunAlreadyTerminal:
            current = self.control.get_run(run_id)
            status = current.status if current is not None else status
            reconciled = True
        except Exception:
            logger.exception(
                "finalize_run: status CAS failed after terminal commit for "
                "run_id=%s; snapshot/replay will reconcile from the event",
                run_id,
            )
            reconciled = True

        assistant_persisted = record.assistant_message_id is not None or not (
            assistant_content or ""
        )
        return FinalizeResult(
            terminal=self._stored_event_dict(record.terminal),
            created=record.terminal_created,
            status=status,
            assistant_persisted=assistant_persisted,
            assistant_content=record.assistant_content,
            reconciled=reconciled,
        )

    def record_terminal_persist_failure(self, run_id: str) -> None:
        """Mark a run whose terminal event could not be made durable.

        Best effort: the stream is already closing, so a failure here is
        logged and the snapshot falls back to "terminal status without a
        persisted terminal event" = ``terminal_persist_failed``.

        BCDEF ticket 12: this is the single durable site for the
        ``terminal_persist_failed`` metric — both the finalize fence here
        and the four stream-fence call sites in
        :mod:`agent_v2.run_orchestration` route through it.
        """
        observe_boundary_metric(ExecutionErrorCode.TERMINAL_PERSIST_FAILED)
        try:
            self.control.mark_terminal_persist_failed(run_id)
        except Exception:
            logger.exception(
                "finalize_run: could not mark terminal_persist_failed for "
                "run_id=%s",
                run_id,
            )

    def get_replay_events(
        self,
        *,
        run_id: str,
        from_seq: int = 0,
    ) -> list[dict[str, Any]]:
        """Replay events, surfaced progress gaps, and a stable synthetic terminal.

        A terminal run whose terminal event was never persisted (write
        failure, hard crash) must not replay as an endless running stream.
        This returns a DETERMINISTIC synthetic terminal (stable event_id
        and sequence derived from the run row) with a stable
        ``terminal_persist_failed`` (or ``cancelled``/``admission_failed``)
        explanation. Nothing is written, so repeated reads agree.

        BCDEF ticket 11: ordinary progress is droppable, but a dropped
        progress must never be a silent hole. When the durable event rows
        reveal a gap in the per-run sequence, replay synthesizes a
        ``progress.gap`` envelope (via
        :func:`agent_v2.execution_contract.compute_gap`, payload
        ``from_sequence``/``to_sequence``/``dropped``) positioned exactly
        where the hole occurred. The synthesis is deterministic per
        ``(run, from_seq)`` — repeated reads agree — never reorders or
        duplicates the terminal, and never emits a gap after it.
        """
        events = self.get_events_since(run_id=run_id, from_seq=from_seq)
        run = self.control.get_run(run_id)
        full_events = (
            events
            if from_seq == 0
            else self.get_events_since(run_id=run_id, from_seq=0)
        )
        events = _replay_with_progress_gaps(run_id, run, events, full_events, from_seq)
        if run is None or run.status not in _REPLAY_TERMINAL_STATUSES:
            return events
        has_terminal = any(
            event["type"] in _RUN_TERMINAL_TYPES for event in full_events
        )
        if has_terminal:
            return events
        synthetic = self._synthetic_terminal_event(run)
        if synthetic["sequence"] > from_seq:
            events.append(synthetic)
        return events

    def _synthetic_terminal_event(self, run: RunRecord) -> dict[str, Any]:
        """Stable replay terminal for a terminal run with no stored one."""
        if run.status == "cancelled":
            event_type = "cancelled"
            marker = "cancelled"
            payload: dict[str, Any] = {
                "code": "cancelled",
                "message": "运行已取消。",
                "recoverable": False,
            }
        elif run.status == "outcome_unknown":
            # BCDEF ticket 11: an outcome-unknown run is terminal and
            # NON-continuable. The stable code is the same one the live
            # effect driver puts on its error terminal, so clients map
            # both paths onto their first-class ``outcome_unknown``
            # status — rendered apart from a generic error, never
            # auto-replayed (spec: repairing it is reconciliation, not a
            # model re-run).
            event_type = "error"
            marker = "outcome-unknown"
            payload = {
                "code": "effect_outcome_unknown",
                "run_status": "outcome_unknown",
                "message": "外部操作结果未知，等待后台对账修复；不会自动重放。",
                "recoverable": False,
            }
        elif run.status == "admission_failed":
            event_type = "error"
            marker = "admission-failed"
            payload = {
                "code": "admission_failed",
                "message": "请求未被接纳，未执行模型调用。",
                "recoverable": True,
            }
        else:
            event_type = "error"
            marker = "terminal-persist-failed"
            payload = {
                "code": ExecutionErrorCode.TERMINAL_PERSIST_FAILED.value,
                "message": (
                    "运行已结束，但终态事件未能持久化；请以快照状态为准。"
                ),
                "recoverable": True,
                "terminal_persisted": False,
            }
        payload["unknown_dropped"] = 0
        return {
            "version": "2",
            "event_id": f"{run.run_id}:{marker}",
            "sequence": self.checkpoints.get_last_sequence(run.run_id) + 1,
            "run_id": run.run_id,
            "conversation_id": run.conversation_id,
            "timestamp": run.updated_at,
            "type": event_type,
            "request_id": run.request_id,
            "payload": payload,
        }

    # -- bounded conversation history for the model (Ticket 05 / T07) ---

    def get_recent_completed_history(
        self,
        *,
        conversation_id: str,
        current_run_id: str | None = None,
        max_messages: int = HISTORY_MAX_MESSAGES,
        max_chars: int = HISTORY_MAX_CHARS,
    ) -> list[dict[str, str]]:
        """Recent completed conversation turns for the NEXT model turn.

        Server-side read: the caller passes the CURRENT run's
        conversation_id (and the run itself so it is excluded). Only
        ``user`` / ``assistant`` messages of naturally completed
        (``done``) runs qualify; running, failed, cancelled, paused and
        internal (tool/approval) turns never enter history. Messages are
        returned oldest-first in stable true insertion order (``rowid``);
        a turn is ONE whole run, ordered by the run's first inserted
        message, so runs whose messages interleave in storage (two
        concurrent completed runs of one conversation) still never split
        into pseudo-turns.

        The window keeps at most ``max_messages`` messages AND at most
        ``max_chars`` body characters, trimmed from the OLDEST end by
        COMPLETE turns: a turn is either fully present or fully absent —
        never half a fact. When a newer turn does not fit, older turns
        are omitted with it, so stale constraints can never outrank the
        newest ones. No completed history at all returns ``[]`` (fresh
        conversation semantics).

        Storage failures propagate to the caller, which maps them to the
        explicit ``context_unavailable`` terminal — a broken store must
        never be silently treated as a fresh conversation.
        """
        if not conversation_id:
            return []
        completed_run_ids = self.control.list_completed_run_ids(conversation_id)
        if not completed_run_ids:
            return []

        def _wanted(message: Any) -> bool:
            if message.role not in ("user", "assistant"):
                return False
            if message.run_id not in completed_run_ids:
                return False
            return not (current_run_id is not None and message.run_id == current_run_id)

        def _select(
            turns: list[list[Any]],
            *,
            last_resolved: bool,
        ) -> tuple[list[list[Any]], bool]:
            """Newest turns that fit; ``final`` = no older turn can change it.

            Stop at the FIRST turn that would overflow: an unresolved
            (page-boundary) turn can only grow, so it can never fit later
            either; and once a turn is dropped, every OLDER turn is
            dropped with it — stale constraints must never outrank the
            newest ones.
            """
            selected: list[list[Any]] = []
            used_messages = 0
            used_chars = 0
            for turn in turns:
                if not _wanted(turn[0]):
                    # Excluded by run status — skipped whole, independent
                    # of the window.
                    continue
                turn_chars = sum(len(message.content) for message in turn)
                if (
                    used_messages + len(turn) > max_messages
                    or used_chars + turn_chars > max_chars
                ):
                    return selected, True
                selected.append(turn)
                used_messages += len(turn)
                used_chars += turn_chars
            # Keep reading only while the oldest turn may still grow.
            return selected, not turns or last_resolved

        # Newest-first scan. EVERY message of one run belongs to ONE block
        # keyed by ``run_id`` — a run's messages are never split by other
        # runs interleaved between them, or trimming would keep half a
        # pair (an assistant answer without its user question). Blocks are
        # ordered (newest first) by the run's FIRST inserted message — the
        # run's start — so a later-starting turn stays last and is
        # trimmed first; a run whose OLDER message just arrived moves
        # back with it. The scan stops as soon as the next older block
        # cannot fit the window, so reading cost stays bounded with the
        # window.
        blocks: dict[str, list[Any]] = {}
        block_order: list[str] = []
        selected: list[list[Any]] = []
        offset = 0
        exhausted = False
        while True:
            page = self.checkpoints.get_messages_newest_first(
                conversation_id=conversation_id,
                limit=_HISTORY_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                exhausted = True
                break
            offset += len(page)
            for message in page:
                block = blocks.get(message.run_id)
                if block is None:
                    blocks[message.run_id] = [message]
                    block_order.append(message.run_id)
                else:
                    block.append(message)
                    # The scan walks newest → oldest, so this message sits
                    # before every message already seen for the run: the
                    # run started earlier than the scan thought, and its
                    # block belongs after the newer-starting ones.
                    block_order.remove(message.run_id)
                    block_order.append(message.run_id)
            selected, final = _select(
                [blocks[run_id] for run_id in block_order],
                last_resolved=False,
            )
            if final:
                break
        if exhausted:
            # The oldest fetched turn is complete — every message exists.
            selected, _ = _select(
                [blocks[run_id] for run_id in block_order],
                last_resolved=True,
            )

        ordered: list[Any] = []
        for turn in reversed(selected):
            ordered.extend(reversed(turn))
        return [
            {"role": message.role, "content": message.content} for message in ordered
        ]

    # -- snapshot -------------------------------------------------------

    def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        """Assemble the spec AC3 snapshot.

        Returns None when the run_id is unknown. The snapshot includes:

        * ``status`` / ``terminal_type`` — run lifecycle state
        * ``sequence_cursor`` — last persisted event sequence
        * ``messages`` — conversation history (user + assistant)
        * ``pending_actions`` — tool_calls without matching tool_result
        """
        run = self.control.get_run(run_id)
        if run is None:
            return None

        messages = self.checkpoints.get_messages(
            conversation_id=run.conversation_id
        )
        sequence_cursor = self.checkpoints.get_last_sequence(run_id)
        pending = self.checkpoints.get_pending_actions(run_id)
        policy_snapshot = self.control.get_policy_snapshot(run_id)
        stored_terminal = self.checkpoints.get_terminal_event(run_id)

        # Ticket 03: the persisted terminal event is authoritative for the
        # user-visible status. A successful event commit whose control CAS
        # failed must not read back as "still running"; a terminal control
        # row without an event is explained as terminal_persist_failed.
        control_status = run.status
        status = control_status
        terminal_type = run.terminal_type
        state_reconciled = False
        if (
            stored_terminal is not None
            and control_status not in _RUN_TERMINAL_TYPES
            and control_status != "admission_failed"
        ):
            status = terminal_type = stored_terminal.event_type
            state_reconciled = True
        elif (
            stored_terminal is not None
            and control_status in _RUN_TERMINAL_TYPES
            and control_status != stored_terminal.event_type
        ):
            status = terminal_type = stored_terminal.event_type
            state_reconciled = True

        terminal_payload: dict[str, Any] = {}
        if stored_terminal is not None:
            try:
                decoded = json.loads(stored_terminal.payload_json)
                if isinstance(decoded, dict):
                    terminal_payload = decoded
            except (TypeError, ValueError):
                terminal_payload = {}
        terminal_error_code = (
            terminal_payload.get("code")
            if stored_terminal is not None
            and stored_terminal.event_type == "error"
            else None
        )
        terminal_persist_failed = bool(run.terminal_persist_failed) or (
            control_status in _RUN_TERMINAL_TYPES
            and stored_terminal is None
            and control_status != "cancelled"
        )

        error_code: str | None = None
        if terminal_persist_failed:
            error_code = ExecutionErrorCode.TERMINAL_PERSIST_FAILED.value
        elif control_status == "admission_failed" and stored_terminal is None:
            error_code = "admission_failed"
        elif control_status == "outcome_unknown" and stored_terminal is None:
            # BCDEF review S3: replay of this run synthesizes the stable
            # ``effect_outcome_unknown`` terminal (ticket 11), so the
            # snapshot names the SAME failure instead of leaving an
            # unexplained terminal status. ``outcome_unknown`` is not a
            # terminal-persist failure — the terminal was never supposed to
            # be a plain done/error/cancelled event.
            error_code = "effect_outcome_unknown"
        elif terminal_error_code:
            error_code = str(terminal_error_code)
        recoverable = bool(
            terminal_persist_failed
            or (
                stored_terminal is not None
                and stored_terminal.event_type == "error"
                and terminal_payload.get("recoverable") is True
            )
        )
        effects = self.control.list_effects_for_run(run_id)
        lease = self.control.get_sandbox_lease(run_id)
        # BCDEF ticket 11: pending approvals ride the snapshot so a
        # refresh while paused can restore the approval prompt even when
        # the replay cursor already advanced past the ``approval.required``
        # frame. Projected into the ``approval.required`` payload shape —
        # additive; old clients ignore the field.
        pending_approvals = [
            _approval_snapshot_payload(approval)
            for approval in self.control.list_pending_approvals_for_run(run_id)
        ]
        if policy_snapshot is None:
            policy_provenance = None
            policy_failure = missing_snapshot_failure(run_id)
        else:
            description = describe_policy_record(policy_snapshot)
            policy_provenance = description["policy_provenance"]
            policy_failure = description["policy_failure"]

        # BCDEF ticket 10: expose the frozen policy's projection summary
        # additively for snapshot/replay consumers. A read-only snapshot
        # must never fail because of a projection problem — the execution
        # paths fail closed separately; here the summary is simply omitted.
        capability_projection: dict[str, Any] | None = None
        if policy_snapshot is not None and policy_failure is None:
            try:
                from agent_v2.capability_projection import (
                    build_capability_projection,
                )

                resolved = self.resolve_policy_snapshot(run_id)
                capability_projection = build_capability_projection(
                    resolved.policy, run_id=run_id
                ).summary()
            except Exception:
                capability_projection = None

        snapshot = RunSnapshot(
            run_id=run.run_id,
            request_id=run.request_id,
            conversation_id=run.conversation_id,
            profile_id=run.profile_id,
            status=status,
            terminal_type=terminal_type,
            created_at=run.created_at,
            updated_at=run.updated_at,
            terminal_at=run.terminal_at,
            sequence_cursor=sequence_cursor,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                    "sequence": m.sequence,
                    "run_id": m.run_id,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
            pending_actions=pending,
            policy_hash=(
                policy_snapshot.policy_hash if policy_snapshot is not None else ""
            ),
            policy_schema_version=(
                policy_snapshot.schema_version
                if policy_snapshot is not None
                else 0
            ),
            control_status=control_status,
            terminal_event_type=(
                stored_terminal.event_type
                if stored_terminal is not None
                else None
            ),
            terminal_event_id=(
                stored_terminal.event_id if stored_terminal is not None else ""
            ),
            terminal_event_sequence=(
                stored_terminal.sequence if stored_terminal is not None else 0
            ),
            terminal_persisted=stored_terminal is not None,
            terminal_persist_failed=terminal_persist_failed,
            terminal_error_code=(
                str(terminal_error_code) if terminal_error_code else None
            ),
            error_code=error_code,
            recoverable=recoverable,
            state_reconciled=state_reconciled,
            effects=[
                {
                    "effect_id": record.resolved_effect_id,
                    "effect_key": record.effect_key,
                    "tool": record.tool_name,
                    "status": record.phase,
                    "stored_status": record.status,
                    "idempotency_key": record.resolved_idempotency_key,
                    "claim_owner": record.claim_owner,
                    "lease_id": record.lease_id,
                    "fencing_token": record.fencing_token,
                    "result": _effect_result_payload(record.result_json),
                    "failure_reason": record.failure_reason,
                    "committed_at": record.committed_at,
                    # BCDEF ticket 05: unknown-outcome reconciliation is
                    # replay-visible — identity, known facts, the unknown
                    # part, repair state/attempts, last error, age and the
                    # next operator/janitor action. All additive.
                    "repair_status": record.repair_status or "none",
                    "repair_attempts": record.repair_attempts,
                    "repair_last_error": (
                        record.repair_last_error or record.failure_reason
                    ),
                    "repair_request_id": record.repair_request_id,
                    "repair_claimed_by": record.repair_claimed_by,
                    "repair_updated_at": record.repair_updated_at,
                    "repair_note": record.repair_note,
                    "repair_age_seconds": repair_age_seconds(
                        record.repair_updated_at or record.updated_at
                    ),
                    "repair_next_action": repair_next_action(
                        record.status,
                        record.repair_status,
                        result_known=bool(record.result_json),
                    ),
                    "known_facts": effect_known_facts(record),
                    "unknown_part": effect_unknown_part(record),
                }
                for record in effects
            ],
            sandbox_lease=(
                {
                    "lease_id": lease.lease_id,
                    "status": lease.status,
                    "owner_id": lease.owner_id,
                    "fencing_token": lease.fencing_token,
                    "sandbox_identity": lease.sandbox_identity,
                    "expires_at": lease.expires_at,
                    "heartbeat_at": lease.heartbeat_at,
                    "destroyed_at": lease.destroyed_at,
                    "failure_reason": lease.failure_reason,
                    # BCDEF ticket 07: terminal cleanup outcome.
                    "cleanup_status": lease.cleanup_status,
                    "cleanup_error": lease.cleanup_error,
                    "janitor_task_id": lease.janitor_task_id,
                }
                if lease is not None
                else None
            ),
            policy_provenance=policy_provenance,
            policy_failure=policy_failure,
            pending_approvals=pending_approvals,
            capability_projection=capability_projection,
        )

        # BCDEF ticket 12: ``repair_age`` becomes observable. An
        # ``outcome_unknown`` effect whose reconciliation is still open is
        # exactly the backlog the rollout gate watches, and the number is
        # the SAME age the snapshot just published to the client — no
        # second source of truth (only the opaque run id travels, per spec
        # Further Notes on redaction).
        for effect in snapshot.effects:
            if effect.get("repair_next_action") not in (None, "", "none"):
                observe_boundary_repair_age(
                    float(effect.get("repair_age_seconds") or 0.0),
                    run_id=snapshot.run_id,
                )

        return snapshot

    # -- cancel ---------------------------------------------------------

    def request_cancel(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> CancelOutcome:
        """Idempotent cancel — spec AC5.

        The idempotency_key is required so duplicate HTTP cancel
        requests (e.g. user double-clicks) return the original outcome
        rather than re-attempting the state transition.
        """
        accepted, reason = self.control.request_cancel(
            run_id=run_id,
            idempotency_key=idempotency_key,
        )
        return CancelOutcome(
            accepted=accepted,
            reason=reason,
            run_id=run_id,
        )

    def is_cancelled(self, run_id: str) -> bool:
        """Check whether a run has been cancelled.

        The streaming endpoint polls this between yields so a cancelled
        run stops emitting new events promptly.
        """
        run = self.control.get_run(run_id)
        if run is None:
            return False
        return run.status == "cancelled"


def _approval_snapshot_payload(approval: ApprovalRecord) -> dict[str, Any]:
    """Project a pending approval row into the ``approval.required`` shape."""
    try:
        args = json.loads(approval.args_canonical) if approval.args_canonical else {}
    except (TypeError, ValueError):
        args = {}
    return {
        "approval_id": approval.approval_id,
        "run_id": approval.run_id,
        "tool": approval.tool_name,
        "args": args if isinstance(args, dict) else {},
        "risk_level": approval.risk_level,
        "impact": approval.impact,
        "expires_at": approval.expires_at,
        "status": approval.status,
        "message": "需要您的批准才能继续。",
    }


def _replay_with_progress_gaps(
    run_id: str,
    run: RunRecord | None,
    replay_events: list[dict[str, Any]],
    full_events: list[dict[str, Any]],
    from_seq: int,
) -> list[dict[str, Any]]:
    """Insert deterministic ``progress.gap`` envelopes at durable-sequence holes.

    BCDEF ticket 11, spec 07: droppable progress may leave the durable
    event log non-contiguous; replay must surface that hole as a visible
    ``progress.gap`` instead of letting clients read a silent skip as
    stalled progress or completion. Rules:

    * one gap per observed hole, computed with
      :func:`agent_v2.execution_contract.compute_gap` — payload is
      exactly ``from_sequence`` / ``to_sequence`` / ``dropped``;
    * the gap envelope is inserted immediately BEFORE the first persisted
      event after the hole, so ascending order is kept and terminals are
      never reordered, duplicated, or followed by a gap;
    * the synthesized sequence is the first MISSING sequence of the hole —
      never a sequence a real event already occupies;
    * ``event_id`` is deterministic from the hole boundaries, so repeated
      reads return byte-identical replay lists.
    """
    gaps: list[tuple[ProgressGap, dict[str, Any]]] = []
    previous: dict[str, Any] | None = None
    for event in full_events:
        if previous is not None:
            last_sequence = previous["sequence"]
            next_sequence = event["sequence"]
            if next_sequence > last_sequence + 1 and next_sequence > from_seq:
                gap = compute_gap(run_id, max(last_sequence, from_seq), next_sequence)
                if gap is not None:
                    gaps.append((gap, event))
        previous = event
    if not gaps:
        return replay_events

    # BCDEF ticket 12: ``replay_gap`` counts what a client actually saw.
    # Ordinary progress is droppable (spec out-of-scope: no delivery
    # guarantee), but every dropped stretch must become explainable — and
    # an operator needs to know how often readers hit a hole. This is a
    # read-side observation, so a reconnect that re-reads the same hole
    # re-observes it; nothing is written here.
    for gap, _anchor in gaps:
        observe_boundary_metric(ExecutionErrorCode.REPLAY_GAP)

    conversation_id = run.conversation_id if run is not None else ""
    request_id = run.request_id if run is not None else ""
    out: list[dict[str, Any]] = []
    pending = list(gaps)
    for event in replay_events:
        while pending and pending[0][0].to_sequence <= event["sequence"]:
            gap, anchor = pending.pop(0)
            out.append(_progress_gap_envelope(gap, anchor, conversation_id, request_id))
        out.append(event)
    # A gap whose anchor event sits at/below from_seq cannot happen here
    # (to_sequence > from_seq by construction), but keep the tail flush for
    # defensive completeness.
    for gap, anchor in pending:
        out.append(_progress_gap_envelope(gap, anchor, conversation_id, request_id))
    return out


def _progress_gap_envelope(
    gap: ProgressGap,
    anchor: dict[str, Any],
    conversation_id: str,
    request_id: str,
) -> dict[str, Any]:
    """Build the replay-facing ``progress.gap`` envelope for one hole."""
    return {
        "version": "2",
        "event_id": f"{gap.run_id}:gap:{gap.from_sequence}:{gap.to_sequence}",
        "sequence": gap.from_sequence + 1,
        "run_id": gap.run_id,
        "conversation_id": conversation_id or anchor.get("conversation_id", ""),
        "timestamp": anchor.get("timestamp", ""),
        "type": "progress.gap",
        "request_id": request_id or anchor.get("request_id", ""),
        "payload": gap.to_payload(),
    }


def _effect_result_payload(result_json: str | None) -> dict[str, Any]:
    """Parse a stored effect result for the snapshot.

    A corrupt/absent result is surfaced as an empty dict — the status
    (committed vs outcome_unknown) remains authoritative.
    """
    if not result_json:
        return {}
    try:
        parsed = json.loads(result_json)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "ApprovalRecord",
    "ArtifactRecord",
    "AssistantPersistFailure",
    "CancelOutcome",
    "FinalizeResult",
    "RunAlreadyTerminal",
    "RunNotFound",
    "RunRepository",
    "RunSnapshot",
    "TerminalPersistFailure",
]
