"""Ticket 01 — BCDEF shared execution contract.

This module is the single, dependency-light description of the execution
boundary that every later ticket imports. It deliberately contains no I/O,
no SQLite, no orchestration imports, and no mutable module state beyond the
legacy-adapter registry.

The contract covers:

* **Lifecycle** — :class:`RunStatus`, the allowed transition table, terminal
  uniqueness, and resumable vs non-continuable rules.
* **Durability** — which event types must survive before a success or
  terminal may be published, and the progress-gap projection for the events
  that may be dropped.
* **Identity** — the request/run/conversation/message/event/effect/lease/
  policy relations and their uniqueness rules.
* **Commit gate** — :class:`TerminalFence`, which refuses to publish a
  terminal before its assistant and effect writes are durable.
* **Versioning** — :data:`EXECUTION_SCHEMA_VERSION`, canonical
  :func:`policy_hash`, :class:`PolicySnapshotRef`, and the versioned
  :class:`LegacyStateAdapter` registry.

Storage-neutral seam: :class:`RunRepositoryContract` describes what an
orchestrator may call. Orchestrators must not mutate persisted state
directly; every status, message, event, effect, and lease write goes
through a repository that implements the contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# Bumped whenever the persisted execution representation gains a field or
# changes meaning. Adapters translate older snapshots up to this version.
EXECUTION_SCHEMA_VERSION: int = 1
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


# ---------------------------------------------------------------------------
# Violations and stable reason codes
# ---------------------------------------------------------------------------


class ReasonCode(str, Enum):
    """Stable machine-readable reasons for contract violations.

    Callers may compare against either the enum member or its string value;
    the value is the stable wire/log form.
    """

    INVALID_TRANSITION = "invalid_transition"
    UNKNOWN_STATUS = "unknown_status"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    SEQUENCE_NOT_MONOTONIC = "sequence_not_monotonic"
    INVALID_GAP = "invalid_gap"
    EMPTY_ID = "empty_id"
    INVALID_RELATION = "invalid_relation"
    INVALID_SEQUENCE = "invalid_sequence"
    MISSING_MESSAGE_SEQUENCE = "missing_message_sequence"
    REQUEST_RUN_CONFLICT = "request_run_conflict"
    RUN_CONVERSATION_CONFLICT = "run_conversation_conflict"
    MESSAGE_SEQUENCE_CONFLICT = "message_sequence_conflict"
    DUPLICATE_MESSAGE_ID = "duplicate_message_id"
    DUPLICATE_EVENT_ID = "duplicate_event_id"
    EFFECT_RUN_CONFLICT = "effect_run_conflict"
    LEASE_RUN_CONFLICT = "lease_run_conflict"
    POLICY_RUN_CONFLICT = "policy_run_conflict"
    POLICY_NOT_CANONICAL = "policy_not_canonical"
    POLICY_INCOMPATIBLE = "policy_incompatible"
    INVALID_TERMINAL_EVENT = "invalid_terminal_event"
    TERMINAL_FENCE_NOT_TERMINAL = "terminal_fence_not_terminal"
    ASSISTANT_NOT_PERSISTED = "assistant_not_persisted"
    EFFECT_NOT_COMMITTED = "effect_not_committed"
    UNKNOWN_LEGACY_STATUS = "unknown_legacy_status"
    ADAPTER_CONFLICT = "adapter_conflict"
    INVALID_ADAPTER = "invalid_adapter"


class ContractViolation(RuntimeError):
    """Raised when a contract invariant would be broken.

    ``code`` is a :class:`ReasonCode` (a ``str`` enum, so equality against
    the value also works). ``message`` is diagnostic only and must never be
    parsed; the code is the stable classification.
    """

    def __init__(self, code: ReasonCode | str, message: str = "") -> None:
        self.code = code
        self.message = message or str(code)
        super().__init__(f"{self.message} [{getattr(code, 'value', code)}]")


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@unique
class RunStatus(str, Enum):
    """Authoritative run lifecycle states.

    The five values that already exist in ``agent_control.sqlite.runs``
    (``running``, ``paused``, ``done``, ``error``, ``cancelled``) keep their
    exact stored strings. The admission states and ``outcome_unknown`` are
    added by the boundary contract; persisting them is a later ticket.
    """

    ADMISSION_PENDING = "admission_pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"
    ADMISSION_FAILED = "admission_failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


# Statuses whose execution is over. A terminal run is never continued:
# resume of a terminal run must not create a new execution.
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.DONE,
        RunStatus.ERROR,
        RunStatus.CANCELLED,
        RunStatus.ADMISSION_FAILED,
        RunStatus.OUTCOME_UNKNOWN,
    }
)

# Only a paused run may be resumed. ``outcome_unknown`` looks resumable to a
# user but is not: repairing it is a reconciliation/janitor action, not a
# model re-run.
RESUMABLE_STATUSES: frozenset[RunStatus] = frozenset({RunStatus.PAUSED})

# Explicit name for the "do not continue" rule; kept equal to the terminal
# set so a new terminal status cannot silently become continuable.
NON_CONTINUABLE_STATUSES: frozenset[RunStatus] = TERMINAL_STATUSES


ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.ADMISSION_PENDING: frozenset(
        {RunStatus.ADMITTED, RunStatus.ADMISSION_FAILED}
    ),
    RunStatus.ADMITTED: frozenset({RunStatus.RUNNING, RunStatus.ADMISSION_FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.PAUSED,
            RunStatus.DONE,
            RunStatus.ERROR,
            RunStatus.CANCELLED,
            RunStatus.OUTCOME_UNKNOWN,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.ERROR,
            RunStatus.CANCELLED,
            RunStatus.OUTCOME_UNKNOWN,
        }
    ),
    RunStatus.DONE: frozenset(),
    RunStatus.ERROR: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.ADMISSION_FAILED: frozenset(),
    RunStatus.OUTCOME_UNKNOWN: frozenset(),
}


# Admission repair edge (BCDEF ticket 02). Kept OUT of
# ``ALLOWED_TRANSITIONS`` on purpose: ``admission_failed`` is execution
# terminal (no lifecycle continuation), and this edge exists only so a
# same-request retry can backfill the missing admission records and start
# the FIRST execution. It is never a resume: a run that already executed
# can never enter this transition.
ADMISSION_REPAIR_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.ADMISSION_FAILED: frozenset({RunStatus.ADMITTED}),
}


def assert_admission_repair(current: RunStatus | str, target: RunStatus | str) -> None:
    """Raise unless ``current -> target`` is the admission repair edge.

    Only ``admission_failed -> admitted`` qualifies. Callers must not use
    this for execution resume or for any other terminal status.
    """
    current_status = coerce_run_status(current)
    target_status = coerce_run_status(target)
    if target_status not in ADMISSION_REPAIR_TRANSITIONS.get(
        current_status, frozenset()
    ):
        raise ContractViolation(
            ReasonCode.INVALID_TRANSITION,
            f"admission repair does not allow {current_status.value!r} -> "
            f"{target_status.value!r}",
        )


def coerce_run_status(value: RunStatus | str) -> RunStatus:
    """Convert a stored/string status to :class:`RunStatus`.

    Unknown values raise :class:`ContractViolation` — callers must never
    default an unrecognized status to ``running``.
    """
    if isinstance(value, RunStatus):
        return value
    try:
        return RunStatus(value)
    except ValueError as exc:
        raise ContractViolation(
            ReasonCode.UNKNOWN_STATUS, f"unknown run status {value!r}"
        ) from exc


def assert_transition(current: RunStatus | str, target: RunStatus | str) -> None:
    """Raise unless ``current -> target`` is an allowed lifecycle transition.

    Self transitions are not allowed: the repository records transitions,
    not heartbeats. Terminal statuses have no outgoing edges, so a resume
    attempt on a finished run fails here instead of starting new execution.
    """
    current_status = coerce_run_status(current)
    target_status = coerce_run_status(target)
    if target_status not in ALLOWED_TRANSITIONS[current_status]:
        raise ContractViolation(
            ReasonCode.INVALID_TRANSITION,
            f"run cannot transition {current_status.value!r} -> "
            f"{target_status.value!r}",
        )


def assert_terminal_unique(
    prior_terminal_event_id: str | None, terminal_event_id: str
) -> None:
    """Raise if a second, different terminal would be recorded for a run.

    Re-observing the same terminal event id is an idempotent replay and is
    allowed; a different terminal event always violates the one-terminal
    rule. An empty new id is rejected because terminals must be identifiable
    for replay and audit.
    """
    if not terminal_event_id:
        raise ContractViolation(
            ReasonCode.DUPLICATE_TERMINAL, "terminal event id must be non-empty"
        )
    if prior_terminal_event_id and prior_terminal_event_id != terminal_event_id:
        raise ContractViolation(
            ReasonCode.DUPLICATE_TERMINAL,
            f"run already has terminal event {prior_terminal_event_id!r}; "
            f"refusing second terminal {terminal_event_id!r}",
        )


# ---------------------------------------------------------------------------
# Event durability
# ---------------------------------------------------------------------------


@unique
class Durability(str, Enum):
    """How hard an event's durable write binds the boundary.

    ``ADMISSION`` — run intake facts; a run cannot be acknowledged without
    them. ``EFFECT`` — tool/effect and gate events that authorize or record
    side effects; a success claim requires them. ``ASSISTANT`` — model
    output shown to the user; it must be persisted before a terminal.
    ``TERMINAL`` — the run outcome; exactly one, always replayable.
    ``PROGRESS`` — droppable telemetry; dropping it must leave a visible
    :class:`ProgressGap`, never a silent hole.
    """

    ADMISSION = "admission"
    EFFECT = "effect"
    ASSISTANT = "assistant"
    TERMINAL = "terminal"
    PROGRESS = "progress"


# Every member of ``agent_v2.events.EventType`` plus ``progress.gap``. The
# keys are strings so this module has no runtime dependency on the envelope
# module; ``test_execution_contract`` asserts the two sets stay identical.
# Gate events (approval, memory consent) are durable with the effect floor:
# they authorize or interrupt a tool effect and must be replayable.
EVENT_DURABILITY: dict[str, Durability] = {
    "run_started": Durability.ADMISSION,
    "progress": Durability.PROGRESS,
    "progress.gap": Durability.PROGRESS,
    "text": Durability.ASSISTANT,
    "thought": Durability.ASSISTANT,
    "tool_call": Durability.EFFECT,
    "tool_result": Durability.EFFECT,
    "artifact_candidate": Durability.EFFECT,
    "artifact.available": Durability.EFFECT,
    "approval.required": Durability.EFFECT,
    "approval.resolved": Durability.EFFECT,
    "memory.consent_required": Durability.EFFECT,
    "run.paused": Durability.TERMINAL,
    "effect.outcome_unknown": Durability.EFFECT,
    "error": Durability.TERMINAL,
    "done": Durability.TERMINAL,
    "cancelled": Durability.TERMINAL,
    "heartbeat": Durability.PROGRESS,
}

# The replayable floor: admission + tool/effect + assistant + terminal.
DURABLE_EVENT_TYPES: frozenset[str] = frozenset(
    event_type
    for event_type, durability in EVENT_DURABILITY.items()
    if durability is not Durability.PROGRESS
)


def durability_of(event_type: str) -> Durability | None:
    """Return the durability class of ``event_type``, or ``None`` if unknown."""
    return EVENT_DURABILITY.get(event_type)


def is_durable_event(event_type: str) -> bool:
    """True unless ``event_type`` is classified as droppable progress.

    Unknown event types are treated as durable (fail closed): a new adapter
    event is kept until it is explicitly classified, never silently dropped.
    """
    durability = EVENT_DURABILITY.get(event_type)
    return durability is None or durability is not Durability.PROGRESS


@dataclass(frozen=True, slots=True)
class ProgressGap:
    """Projection of progress events that were not persisted.

    ``from_sequence`` is the last sequence the client is known to have,
    ``to_sequence`` is the next persisted sequence, and ``dropped`` is the
    number of missing sequences in between. The payload shape is stable and
    can be emitted as a normal ``progress.gap`` envelope.
    """

    run_id: str
    from_sequence: int
    to_sequence: int
    dropped: int

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ContractViolation(
                ReasonCode.INVALID_GAP, "progress gap requires a run_id"
            )
        if self.to_sequence <= self.from_sequence:
            raise ContractViolation(
                ReasonCode.INVALID_GAP,
                f"gap must advance: {self.from_sequence} -> {self.to_sequence}",
            )
        expected = self.to_sequence - self.from_sequence - 1
        if self.dropped != expected:
            raise ContractViolation(
                ReasonCode.INVALID_GAP,
                f"dropped={self.dropped} does not match skipped range "
                f"({expected})",
            )

    def to_payload(self) -> dict[str, int]:
        """Return the replayable ``progress.gap`` payload (JSON-safe ints)."""
        return {
            "from_sequence": self.from_sequence,
            "to_sequence": self.to_sequence,
            "dropped": self.dropped,
        }


def compute_gap(
    run_id: str, last_sequence: int, next_sequence: int
) -> ProgressGap | None:
    """Return the gap between two observed sequences, or ``None`` if contiguous.

    ``next_sequence == last_sequence + 1`` is contiguous. A forward skip
    produces a :class:`ProgressGap` whose payload can be replayed through the
    existing v2 envelope. Equal or backward sequences are a contract
    violation (duplicate/reordered events are not gaps).
    """
    if next_sequence <= last_sequence:
        raise ContractViolation(
            ReasonCode.SEQUENCE_NOT_MONOTONIC,
            f"next sequence {next_sequence} must be greater than "
            f"last sequence {last_sequence}",
        )
    if next_sequence == last_sequence + 1:
        return None
    return ProgressGap(
        run_id=run_id,
        from_sequence=last_sequence,
        to_sequence=next_sequence,
        dropped=next_sequence - last_sequence - 1,
    )


# ---------------------------------------------------------------------------
# Identity relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdRelation:
    """One boundary identity relation to validate.

    ``request_id``, ``run_id`` and ``conversation_id`` are always required.
    ``message_id``, ``event_id``, ``effect_id``, ``lease_id`` and
    ``policy_snapshot_id`` are optional, but when present they must be
    non-empty and obey their binding rule. ``sequence`` is the message
    sequence (0 = user input) and is required when ``message_id`` is set.

    ``policy_snapshot_id`` must be the run-scoped snapshot identity (see
    :meth:`PolicySnapshotRef.snapshot_id`), not a bare policy hash: a shared
    hash is fine, a shared snapshot id across runs is not.
    """

    run_id: str
    request_id: str
    conversation_id: str
    message_id: str = ""
    event_id: str = ""
    effect_id: str = ""
    lease_id: str = ""
    policy_snapshot_id: str = ""
    sequence: int | None = None


_REQUIRED_ID_FIELDS = ("request_id", "run_id", "conversation_id")
_OPTIONAL_ID_FIELDS = (
    "message_id",
    "event_id",
    "effect_id",
    "lease_id",
    "policy_snapshot_id",
)


def _require_id(value: object, field: str, *, required: bool) -> str:
    if value is None or (required and value == ""):
        raise ContractViolation(ReasonCode.EMPTY_ID, f"{field} must be non-empty")
    if not isinstance(value, str):
        raise ContractViolation(ReasonCode.EMPTY_ID, f"{field} must be a string")
    if value == "":
        return ""
    if not value.strip():
        raise ContractViolation(ReasonCode.EMPTY_ID, f"{field} must be non-empty")
    return value


def validate_identifiers(
    relations: IdRelation | Iterable[IdRelation],
) -> None:
    """Validate identity uniqueness and binding for one or more relations.

    Pure and storage-neutral: the caller supplies the relations it is about
    to persist (a new run, a message, an event, an effect claim, a lease, a
    policy snapshot) and this function rejects contradictions before any
    write happens.

    Rules enforced:

    * required ids are non-empty; optional ids, when present, are non-empty;
    * a ``request_id`` maps to at most one ``run_id``;
    * a ``run_id`` maps to exactly one ``conversation_id``;
    * ``(conversation_id, sequence)`` maps to at most one ``message_id``;
    * a ``message_id`` identifies one message;
    * an ``event_id`` is globally unique;
    * effect, lease and policy snapshot ids are bound to a single run.

    Raises :class:`ContractViolation` with a stable :class:`ReasonCode`.
    """
    if isinstance(relations, IdRelation):
        items: Iterable[IdRelation] = (relations,)
    else:
        items = relations

    request_runs: dict[str, str] = {}
    run_conversations: dict[str, str] = {}
    conversation_sequences: dict[tuple[str, int], str] = {}
    message_locations: dict[str, tuple[str, int]] = {}
    event_ids: set[str] = set()
    effect_runs: dict[str, str] = {}
    lease_runs: dict[str, str] = {}
    policy_runs: dict[str, str] = {}

    for relation in items:
        if not isinstance(relation, IdRelation):
            raise ContractViolation(
                ReasonCode.INVALID_RELATION,
                "identifier relation must be an IdRelation",
            )
        for field_name in _REQUIRED_ID_FIELDS:
            _require_id(getattr(relation, field_name), field_name, required=True)
        for field_name in _OPTIONAL_ID_FIELDS:
            _require_id(getattr(relation, field_name), field_name, required=False)

        sequence = relation.sequence
        if relation.message_id:
            if sequence is None:
                raise ContractViolation(
                    ReasonCode.MISSING_MESSAGE_SEQUENCE,
                    f"message {relation.message_id!r} requires a sequence",
                )
            if not isinstance(sequence, int) or sequence < 0:
                raise ContractViolation(
                    ReasonCode.INVALID_SEQUENCE,
                    f"message sequence {sequence!r} must be a non-negative int",
                )

        prior_run = request_runs.setdefault(relation.request_id, relation.run_id)
        if prior_run != relation.run_id:
            raise ContractViolation(
                ReasonCode.REQUEST_RUN_CONFLICT,
                f"request {relation.request_id!r} already maps to run "
                f"{prior_run!r}, not {relation.run_id!r}",
            )

        prior_conversation = run_conversations.setdefault(
            relation.run_id, relation.conversation_id
        )
        if prior_conversation != relation.conversation_id:
            raise ContractViolation(
                ReasonCode.RUN_CONVERSATION_CONFLICT,
                f"run {relation.run_id!r} already maps to conversation "
                f"{prior_conversation!r}, not {relation.conversation_id!r}",
            )

        if relation.message_id:
            key = (relation.conversation_id, sequence)  # type: ignore[arg-type]
            prior_message = conversation_sequences.setdefault(key, relation.message_id)
            if prior_message != relation.message_id:
                raise ContractViolation(
                    ReasonCode.MESSAGE_SEQUENCE_CONFLICT,
                    f"conversation {relation.conversation_id!r} sequence "
                    f"{sequence} already maps to message {prior_message!r}",
                )
            location = (relation.conversation_id, sequence)
            prior_location = message_locations.setdefault(
                relation.message_id, location
            )
            if prior_location != location:
                raise ContractViolation(
                    ReasonCode.DUPLICATE_MESSAGE_ID,
                    f"message {relation.message_id!r} appears at two locations",
                )

        if relation.event_id:
            if relation.event_id in event_ids:
                raise ContractViolation(
                    ReasonCode.DUPLICATE_EVENT_ID,
                    f"event {relation.event_id!r} appears more than once",
                )
            event_ids.add(relation.event_id)

        for identifier, owner_map, conflict_code, label in (
            (
                relation.effect_id,
                effect_runs,
                ReasonCode.EFFECT_RUN_CONFLICT,
                "effect",
            ),
            (relation.lease_id, lease_runs, ReasonCode.LEASE_RUN_CONFLICT, "lease"),
            (
                relation.policy_snapshot_id,
                policy_runs,
                ReasonCode.POLICY_RUN_CONFLICT,
                "policy snapshot",
            ),
        ):
            if not identifier:
                continue
            owner = owner_map.setdefault(identifier, relation.run_id)
            if owner != relation.run_id:
                raise ContractViolation(
                    conflict_code,
                    f"{label} {identifier!r} is bound to run {owner!r}, "
                    f"not {relation.run_id!r}",
                )


# ---------------------------------------------------------------------------
# Frozen execution policy
# ---------------------------------------------------------------------------


def _normalize_policy_value(value: Any) -> Any:
    """Normalize policy values to their canonical JSON representation.

    ``bytes``/``bytearray`` decode as UTF-8 (so ``b"x"`` and ``"x"`` hash the
    same), mappings become plain dicts, and tuples become lists. Anything
    that is not JSON-representable raises :class:`ContractViolation`.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractViolation(
                ReasonCode.POLICY_NOT_CANONICAL, "policy bytes must be valid UTF-8"
            ) from exc
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, (bytes, bytearray)):
                try:
                    key = bytes(key).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ContractViolation(
                        ReasonCode.POLICY_NOT_CANONICAL,
                        "policy keys must be valid UTF-8",
                    ) from exc
            if not isinstance(key, str):
                raise ContractViolation(
                    ReasonCode.POLICY_NOT_CANONICAL,
                    f"policy keys must be strings, got {type(key).__name__}",
                )
            normalized[key] = _normalize_policy_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_policy_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ContractViolation(
        ReasonCode.POLICY_NOT_CANONICAL,
        f"policy value of type {type(value).__name__} is not JSON-representable",
    )


def policy_hash(mapping: Mapping[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of a frozen execution policy.

    Canonical form: UTF-8, sorted keys, ``(",", ":")`` separators,
    ``ensure_ascii=False``, no NaN/Infinity. Key order therefore never
    changes the hash; content does. ``bytes`` values are UTF-8 decoded first
    so the same policy hashes identically whether it was built from JSON or
    from persisted bytes.
    """
    if not isinstance(mapping, Mapping):
        raise ContractViolation(
            ReasonCode.POLICY_NOT_CANONICAL, "policy must be a mapping"
        )
    normalized = _normalize_policy_value(mapping)
    try:
        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolation(
            ReasonCode.POLICY_NOT_CANONICAL, "policy is not canonicalizable"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicySnapshotRef:
    """Run-scoped reference to the policy frozen at admission.

    ``snapshot_id`` is ``"{run_id}:{policy_hash}"`` so two runs with the same
    policy share the hash but never the snapshot identity. ``created_at`` is
    informational (UTC ISO 8601) and not part of the hash.
    """

    run_id: str
    policy_hash: str
    schema_version: int = EXECUTION_SCHEMA_VERSION
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ContractViolation(ReasonCode.EMPTY_ID, "run_id must be non-empty")
        if not self.policy_hash or not self.policy_hash.strip():
            raise ContractViolation(
                ReasonCode.EMPTY_ID, "policy_hash must be non-empty"
            )

    @property
    def snapshot_id(self) -> str:
        return f"{self.run_id}:{self.policy_hash}"

    def assert_compatible(self) -> None:
        """Raise if this snapshot cannot be adapted to the current runtime.

        Version incompatibility is a terminal policy decision; callers must
        move the run to paused/error instead of guessing defaults.
        """
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractViolation(
                ReasonCode.POLICY_INCOMPATIBLE,
                f"policy schema {self.schema_version!r} is not supported "
                f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})",
            )
        digest = self.policy_hash.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractViolation(
                ReasonCode.POLICY_INCOMPATIBLE,
                "policy_hash must be a SHA-256 hex digest",
            )


# ---------------------------------------------------------------------------
# Terminal fence (shared commit gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TerminalFence:
    """Preconditions for publishing a terminal event.

    A terminal may only be published after the assistant content (for
    ``done``) and every effect it claims (for all terminal statuses) are
    durably settled. ``pending_effect_ids`` lists effects that are still
    authorized/executing without a recorded outcome; a non-empty list rejects
    the terminal. A recorded ``outcome_unknown`` effect is a durable outcome,
    not a pending one: it is valid evidence for an ``outcome_unknown``
    terminal. Error and cancellation terminals do not require assistant
    content, but they still may not publish over dangling effects.
    """

    run_id: str
    status: RunStatus
    terminal_event_id: str
    terminal_sequence: int
    assistant_persisted: bool = False
    pending_effect_ids: tuple[str, ...] = ()

    def assert_complete(self) -> None:
        status = coerce_run_status(self.status)
        if status not in TERMINAL_STATUSES:
            raise ContractViolation(
                ReasonCode.TERMINAL_FENCE_NOT_TERMINAL,
                f"status {status.value!r} is not terminal; refusing to fence",
            )
        if not self.run_id or not self.run_id.strip():
            raise ContractViolation(ReasonCode.EMPTY_ID, "run_id must be non-empty")
        if not self.terminal_event_id or self.terminal_sequence < 1:
            raise ContractViolation(
                ReasonCode.INVALID_TERMINAL_EVENT,
                "terminal requires a non-empty event id and sequence >= 1",
            )
        if status is RunStatus.DONE and not self.assistant_persisted:
            raise ContractViolation(
                ReasonCode.ASSISTANT_NOT_PERSISTED,
                "assistant message must be persisted before a done terminal",
            )
        if self.pending_effect_ids:
            raise ContractViolation(
                ReasonCode.EFFECT_NOT_COMMITTED,
                "effects still pending: " + ", ".join(self.pending_effect_ids),
            )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@unique
class ExecutionErrorCode(str, Enum):
    """Stable codes for boundary failures (metrics and client mapping)."""

    ADMISSION_FAILED = "admission_failed"
    TERMINAL_PERSIST_FAILED = "terminal_persist_failed"
    ASSISTANT_PERSIST_FAILED = "assistant_persist_failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    LEASE_EXPIRED = "lease_expired"
    CLEANUP_PENDING = "cleanup_pending"
    REPLAY_GAP = "replay_gap"
    POLICY_INCOMPATIBLE = "policy_incompatible"
    POLICY_SNAPSHOT_MISSING = "policy_snapshot_missing"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    REPAIR_PENDING = "repair_pending"


# Whether a failure can be repaired without re-running the model or
# re-executing an effect. ``False`` means the run must not be continued and
# a new run (with new admission) is required.
ERROR_RECOVERABLE: dict[ExecutionErrorCode, bool] = {
    ExecutionErrorCode.ADMISSION_FAILED: True,
    ExecutionErrorCode.TERMINAL_PERSIST_FAILED: True,
    ExecutionErrorCode.ASSISTANT_PERSIST_FAILED: True,
    ExecutionErrorCode.OUTCOME_UNKNOWN: True,
    ExecutionErrorCode.LEASE_EXPIRED: True,
    ExecutionErrorCode.CLEANUP_PENDING: True,
    ExecutionErrorCode.REPLAY_GAP: True,
    ExecutionErrorCode.POLICY_INCOMPATIBLE: False,
    ExecutionErrorCode.POLICY_SNAPSHOT_MISSING: False,
    ExecutionErrorCode.DUPLICATE_TERMINAL: False,
    ExecutionErrorCode.REPAIR_PENDING: True,
}


class BoundaryError(RuntimeError):
    """Classified boundary failure carrying a stable code and recovery hint."""

    def __init__(
        self,
        code: ExecutionErrorCode | str,
        message: str = "",
        *,
        recoverable: bool | None = None,
    ) -> None:
        self.code = (
            code if isinstance(code, ExecutionErrorCode) else ExecutionErrorCode(code)
        )
        self.recoverable = (
            ERROR_RECOVERABLE[self.code] if recoverable is None else recoverable
        )
        self.message = message
        detail = f": {message}" if message else ""
        super().__init__(f"{self.code.value}{detail}")


# ---------------------------------------------------------------------------
# Legacy state adaptation
# ---------------------------------------------------------------------------


@runtime_checkable
class LegacyStateAdapter(Protocol):
    """Versioned translation of a persisted execution snapshot.

    ``adapt`` receives the decoded payload of a snapshot written by
    ``from_version`` and returns a payload in ``to_version`` shape. Adapters
    are pure: they must not read current configuration or run side effects.
    """

    from_version: int
    to_version: int

    def adapt(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...


_LEGACY_ADAPTERS: dict[tuple[int, int], LegacyStateAdapter] = {}


def register_adapter(adapter: LegacyStateAdapter) -> None:
    """Register ``adapter`` for its exact ``(from_version, to_version)`` pair.

    Registers are explicit and non-overwriting: a duplicate pair raises
    :class:`ContractViolation` so two code paths cannot silently disagree
    about how a legacy snapshot is interpreted.
    """
    if not isinstance(adapter, LegacyStateAdapter):
        raise ContractViolation(
            ReasonCode.INVALID_ADAPTER, "adapter must expose adapt()"
        )
    from_version = getattr(adapter, "from_version", None)
    to_version = getattr(adapter, "to_version", None)
    if (
        isinstance(from_version, bool)
        or isinstance(to_version, bool)
        or not isinstance(from_version, int)
        or not isinstance(to_version, int)
    ):
        raise ContractViolation(
            ReasonCode.INVALID_ADAPTER, "adapter versions must be integers"
        )
    if to_version <= from_version:
        raise ContractViolation(
            ReasonCode.INVALID_ADAPTER,
            f"adapter must upgrade versions: {from_version} -> {to_version}",
        )
    key = (from_version, to_version)
    if key in _LEGACY_ADAPTERS:
        raise ContractViolation(
            ReasonCode.ADAPTER_CONFLICT,
            f"adapter {from_version} -> {to_version} is already registered",
        )
    _LEGACY_ADAPTERS[key] = adapter


def get_adapter(
    from_version: int, to_version: int
) -> LegacyStateAdapter | None:
    """Return the adapter registered for exactly this version pair, if any."""
    return _LEGACY_ADAPTERS.get((from_version, to_version))


def adapt_legacy_run_status(value: str) -> RunStatus:
    """Map a stored run status to :class:`RunStatus`.

    Unknown or malformed values raise with ``unknown_legacy_status``. There
    is deliberately no fallback: defaulting an unrecognized row to
    ``running`` would resurrect a run whose lifecycle is not understood.
    """
    if not isinstance(value, str):
        raise ContractViolation(
            ReasonCode.UNKNOWN_LEGACY_STATUS,
            f"stored run status must be a string, got {type(value).__name__}",
        )
    try:
        return RunStatus(value)
    except ValueError as exc:
        raise ContractViolation(
            ReasonCode.UNKNOWN_LEGACY_STATUS, f"unknown stored run status {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Persistence seam
# ---------------------------------------------------------------------------


@runtime_checkable
class RunRepositoryContract(Protocol):
    """The only way an orchestrator may touch persisted execution state.

    Implementations own the durable writes (today two SQLite databases; later
    any store). Orchestrators decide *what* should happen and call these
    methods; they must never mutate run rows, message rows, event rows,
    effect rows, or leases directly. Effect and lease stores may be separate,
    but they are bound to the same ``run_id`` and share this lifecycle and
    :class:`TerminalFence` commit gate.

    A terminal transition must receive a :class:`TerminalFence`; the
    repository validates it before persisting the terminal. ``read_snapshot``
    and ``read_events`` are the authoritative replay path after a stream
    closes. The payload shapes are mappings rather than store records so this
    protocol does not bind SQLite or any dataclass layout.
    """

    async def create_or_get_run(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
        user_message: str,
    ) -> tuple[str, bool]:
        """Admit a request; return ``(run_id, created)``, idempotent on request_id."""
        ...

    async def transition(
        self,
        *,
        run_id: str,
        target: RunStatus,
        fence: TerminalFence | None = None,
    ) -> RunStatus:
        """Apply one lifecycle transition; terminal targets require ``fence``."""
        ...

    async def append_message(
        self,
        *,
        run_id: str,
        conversation_id: str,
        role: str,
        content: str,
        sequence: int,
        message_id: str = "",
    ) -> str:
        """Persist a message and return its ``message_id``."""
        ...

    async def append_event(
        self,
        *,
        run_id: str,
        conversation_id: str,
        event_id: str,
        sequence: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Persist one v2 event row; ``event_id`` is globally unique."""
        ...

    async def read_snapshot(self, *, run_id: str) -> Mapping[str, Any] | None:
        """Return the authoritative snapshot mapping, or ``None`` if unknown."""
        ...

    async def read_events(
        self, *, run_id: str, from_sequence: int = 0
    ) -> Sequence[Mapping[str, Any]]:
        """Replay events with ``sequence > from_sequence`` in ascending order."""
        ...
