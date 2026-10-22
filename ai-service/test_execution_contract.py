"""Ticket 01 — BCDEF shared execution contract.

Pure contract coverage. No SQLite, network, model, or orchestration imports:
the module under test must stay usable by every boundary component.

Covered invariants:

* run lifecycle status strings, allowed transitions, terminal uniqueness,
  resumable vs non-continuable rules;
* per-event durability classification and the hard-droppable progress class;
* progress gap computation and replay through the existing v2 envelope;
* identifier relations (request/run/conversation/message/event/effect/lease/
  policy snapshot) and their uniqueness constraints;
* canonical policy hash, policy snapshot refs, and the terminal fence;
* legacy adapter registry + legacy run status adaptation;
* v2 envelope wire-key compatibility;
* the persistence seam protocol (fake implementation, no SQLite).
"""

from __future__ import annotations

import hashlib
import json
from typing import get_args

import pytest

from agent_v2 import execution_contract as contract
from agent_v2.events import (
    TERMINAL_EVENT_TYPES,
    EventEnvelope,
    EventStream,
    EventType,
)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_status_strings_match_persisted_values() -> None:
    assert contract.RunStatus.ADMISSION_PENDING == "admission_pending"
    assert contract.RunStatus.ADMITTED == "admitted"
    assert contract.RunStatus.RUNNING == "running"
    assert contract.RunStatus.PAUSED == "paused"
    assert contract.RunStatus.DONE == "done"
    assert contract.RunStatus.ERROR == "error"
    assert contract.RunStatus.CANCELLED == "cancelled"
    assert contract.RunStatus.ADMISSION_FAILED == "admission_failed"
    assert contract.RunStatus.OUTCOME_UNKNOWN == "outcome_unknown"


def test_terminal_and_resumable_sets() -> None:
    assert contract.TERMINAL_STATUSES == frozenset(
        {
            contract.RunStatus.DONE,
            contract.RunStatus.ERROR,
            contract.RunStatus.CANCELLED,
            contract.RunStatus.ADMISSION_FAILED,
            contract.RunStatus.OUTCOME_UNKNOWN,
        }
    )
    assert contract.RESUMABLE_STATUSES == frozenset({contract.RunStatus.PAUSED})
    assert contract.NON_CONTINUABLE_STATUSES == contract.TERMINAL_STATUSES
    # Admission failure is terminal: a request that failed admission cannot
    # later start executing.
    assert contract.RunStatus.ADMISSION_FAILED in contract.TERMINAL_STATUSES


def test_transition_table_is_exhaustive() -> None:
    assert set(contract.ALLOWED_TRANSITIONS) == set(contract.RunStatus)
    for status in contract.ALLOWED_TRANSITIONS:
        assert isinstance(contract.ALLOWED_TRANSITIONS[status], frozenset)
        for target in contract.ALLOWED_TRANSITIONS[status]:
            assert isinstance(target, contract.RunStatus)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (contract.RunStatus.ADMISSION_PENDING, contract.RunStatus.ADMITTED),
        (contract.RunStatus.ADMISSION_PENDING, contract.RunStatus.ADMISSION_FAILED),
        (contract.RunStatus.ADMITTED, contract.RunStatus.RUNNING),
        (contract.RunStatus.ADMITTED, contract.RunStatus.ADMISSION_FAILED),
        (contract.RunStatus.RUNNING, contract.RunStatus.PAUSED),
        (contract.RunStatus.RUNNING, contract.RunStatus.DONE),
        (contract.RunStatus.RUNNING, contract.RunStatus.ERROR),
        (contract.RunStatus.RUNNING, contract.RunStatus.CANCELLED),
        (contract.RunStatus.RUNNING, contract.RunStatus.OUTCOME_UNKNOWN),
        (contract.RunStatus.PAUSED, contract.RunStatus.RUNNING),
        (contract.RunStatus.PAUSED, contract.RunStatus.ERROR),
        (contract.RunStatus.PAUSED, contract.RunStatus.CANCELLED),
        (contract.RunStatus.PAUSED, contract.RunStatus.OUTCOME_UNKNOWN),
    ],
)
def test_allowed_transition_does_not_raise(
    current: contract.RunStatus, target: contract.RunStatus
) -> None:
    contract.assert_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        # Admission cannot skip straight to a terminal success.
        (contract.RunStatus.ADMISSION_PENDING, contract.RunStatus.RUNNING),
        (contract.RunStatus.ADMISSION_PENDING, contract.RunStatus.DONE),
        # admitted is not directly terminable; the run must start first.
        (contract.RunStatus.ADMITTED, contract.RunStatus.DONE),
        (contract.RunStatus.ADMITTED, contract.RunStatus.CANCELLED),
        # Terminal states never transition (resume creates no new execution).
        (contract.RunStatus.DONE, contract.RunStatus.RUNNING),
        (contract.RunStatus.ERROR, contract.RunStatus.PAUSED),
        (contract.RunStatus.CANCELLED, contract.RunStatus.RUNNING),
        (contract.RunStatus.ADMISSION_FAILED, contract.RunStatus.DONE),
        (contract.RunStatus.OUTCOME_UNKNOWN, contract.RunStatus.RUNNING),
        # paused cannot become done without running again.
        (contract.RunStatus.PAUSED, contract.RunStatus.DONE),
        # self transitions are not no-ops.
        (contract.RunStatus.RUNNING, contract.RunStatus.RUNNING),
        (contract.RunStatus.PAUSED, contract.RunStatus.PAUSED),
    ],
)
def test_invalid_transition_raises_stable_code(
    current: contract.RunStatus, target: contract.RunStatus
) -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.assert_transition(current, target)
    assert excinfo.value.code == contract.ReasonCode.INVALID_TRANSITION


def test_assert_transition_accepts_stored_strings_and_rejects_unknown() -> None:
    contract.assert_transition("running", "done")
    contract.assert_transition("paused", "running")
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.assert_transition("running", "sleeping")
    assert excinfo.value.code == contract.ReasonCode.UNKNOWN_STATUS


def test_every_status_pair_matches_the_transition_table() -> None:
    for current in contract.RunStatus:
        for target in contract.RunStatus:
            if target in contract.ALLOWED_TRANSITIONS[current]:
                contract.assert_transition(current, target)
            else:
                with pytest.raises(contract.ContractViolation) as excinfo:
                    contract.assert_transition(current, target)
                assert excinfo.value.code == contract.ReasonCode.INVALID_TRANSITION


def test_only_terminal_statuses_have_no_outgoing_edges() -> None:
    closed = {
        status
        for status, targets in contract.ALLOWED_TRANSITIONS.items()
        if not targets
    }
    assert closed == contract.TERMINAL_STATUSES
    assert contract.RESUMABLE_STATUSES.isdisjoint(contract.TERMINAL_STATUSES)
    for status in contract.RunStatus:
        if status is not contract.RunStatus.PAUSED:
            assert status not in contract.RESUMABLE_STATUSES


def test_admission_repair_edge_is_separate_from_lifecycle_resume() -> None:
    assert contract.ADMISSION_REPAIR_TRANSITIONS == {
        contract.RunStatus.ADMISSION_FAILED: frozenset(
            {contract.RunStatus.ADMITTED}
        )
    }
    contract.assert_admission_repair("admission_failed", "admitted")
    # Every other pair — including the same edge through the lifecycle
    # table — stays invalid: repair is not a resume.
    for current in contract.RunStatus:
        for target in contract.RunStatus:
            if current is contract.RunStatus.ADMISSION_FAILED and (
                target is contract.RunStatus.ADMITTED
            ):
                continue
            with pytest.raises(contract.ContractViolation) as excinfo:
                contract.assert_admission_repair(current, target)
            assert excinfo.value.code == contract.ReasonCode.INVALID_TRANSITION
    with pytest.raises(contract.ContractViolation):
        contract.assert_transition("admission_failed", "admitted")


def test_terminal_uniqueness_is_idempotent_for_same_event() -> None:
    contract.assert_terminal_unique(None, "evt-1")
    contract.assert_terminal_unique("evt-1", "evt-1")
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.assert_terminal_unique("evt-1", "evt-2")
    assert excinfo.value.code == contract.ReasonCode.DUPLICATE_TERMINAL


# ---------------------------------------------------------------------------
# Event durability + progress gaps
# ---------------------------------------------------------------------------

EXPECTED_DURABILITY = {
    "run_started": contract.Durability.ADMISSION,
    "progress": contract.Durability.PROGRESS,
    "progress.gap": contract.Durability.PROGRESS,
    "text": contract.Durability.ASSISTANT,
    "thought": contract.Durability.ASSISTANT,
    "tool_call": contract.Durability.EFFECT,
    "tool_result": contract.Durability.EFFECT,
    "artifact_candidate": contract.Durability.EFFECT,
    "artifact.available": contract.Durability.EFFECT,
    "approval.required": contract.Durability.EFFECT,
    "approval.resolved": contract.Durability.EFFECT,
    "memory.consent_required": contract.Durability.EFFECT,
    "run.paused": contract.Durability.TERMINAL,
    "effect.outcome_unknown": contract.Durability.EFFECT,
    "error": contract.Durability.TERMINAL,
    "done": contract.Durability.TERMINAL,
    "cancelled": contract.Durability.TERMINAL,
    "heartbeat": contract.Durability.PROGRESS,
}


def test_every_event_type_has_exactly_one_durability_class() -> None:
    assert set(contract.EVENT_DURABILITY) == set(get_args(EventType))
    assert contract.EVENT_DURABILITY == EXPECTED_DURABILITY


def test_durability_enum_is_the_five_contract_levels() -> None:
    assert {member.name for member in contract.Durability} == {
        "ADMISSION",
        "EFFECT",
        "ASSISTANT",
        "TERMINAL",
        "PROGRESS",
    }


def test_durable_set_matches_spec_and_drops_only_progress() -> None:
    assert "progress.gap" in contract.EVENT_DURABILITY
    assert contract.DURABLE_EVENT_TYPES == frozenset(
        event_type
        for event_type, durability in EXPECTED_DURABILITY.items()
        if durability is not contract.Durability.PROGRESS
    )
    # Admission, tool/effect, assistant and terminal classes are durable.
    for event_type, durability in EXPECTED_DURABILITY.items():
        assert contract.is_durable_event(event_type) is (
            durability is not contract.Durability.PROGRESS
        )
    assert not contract.is_durable_event("progress")
    assert not contract.is_durable_event("heartbeat")
    assert contract.is_durable_event("text")
    assert contract.is_durable_event("done")
    # Unknown types fail closed (kept, never silently dropped).
    assert contract.is_durable_event("future.event") is True


def test_progress_gap_is_none_when_contiguous() -> None:
    assert contract.compute_gap("run-1", 4, 5) is None
    assert contract.compute_gap("run-1", 0, 1) is None


def test_progress_gap_reports_skipped_sequences() -> None:
    gap = contract.compute_gap("run-1", 4, 8)
    assert gap == contract.ProgressGap(
        run_id="run-1", from_sequence=4, to_sequence=8, dropped=3
    )
    assert gap.to_payload() == {
        "from_sequence": 4,
        "to_sequence": 8,
        "dropped": 3,
    }


def test_progress_gap_rejects_non_monotonic_sequences() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.compute_gap("run-1", 5, 5)
    assert excinfo.value.code == contract.ReasonCode.SEQUENCE_NOT_MONOTONIC
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.compute_gap("run-1", 5, 3)
    assert excinfo.value.code == contract.ReasonCode.SEQUENCE_NOT_MONOTONIC


def test_gap_payload_replays_through_existing_envelope() -> None:
    stream = EventStream(request_id="req-1", conversation_id="conv-1", run_id="run-1")
    stream.build("text", {"delta": "hello"})
    gap = contract.compute_gap("run-1", 1, 4)
    assert gap is not None
    envelope = stream.build("progress.gap", gap.to_payload())

    wire = json.loads(envelope.to_json_line())
    assert wire["version"] == "2"
    assert wire["type"] == "progress.gap"
    assert wire["payload"] == {
        "from_sequence": 1,
        "to_sequence": 4,
        "dropped": 2,
    }
    # A gap is a progress event, not a stream terminal.
    assert envelope.type not in TERMINAL_EVENT_TYPES
    done = stream.build("done", {})
    assert done.sequence == envelope.sequence + 1


def test_v2_envelope_wire_keys_are_unchanged() -> None:
    envelope = EventStream(request_id="req-1").build("progress", {"pct": 10})
    wire = json.loads(envelope.to_json_line())
    assert set(wire) == {
        "version",
        "event_id",
        "sequence",
        "run_id",
        "conversation_id",
        "timestamp",
        "type",
        "request_id",
        "payload",
    }
    assert wire["version"] == "2"
    assert isinstance(envelope, EventEnvelope)
    # Adding progress.gap must not change the stream-terminal contract.
    assert TERMINAL_EVENT_TYPES == frozenset(
        {"error", "done", "cancelled", "run.paused"}
    )


# ---------------------------------------------------------------------------
# Identifier relations
# ---------------------------------------------------------------------------


def _relation(**overrides: object) -> contract.IdRelation:
    values: dict[str, object] = {
        "run_id": "run-1",
        "request_id": "req-1",
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "event_id": "evt-1",
        "effect_id": "eff-1",
        "lease_id": "lease-1",
        "policy_snapshot_id": "run-1:hash",
        "sequence": 0,
    }
    values.update(overrides)
    return contract.IdRelation(**values)  # type: ignore[arg-type]


def test_identifier_validation_accepts_consistent_relations() -> None:
    # A single full relation and a second event in the same run/conversation
    # (same request -> run) are both valid.
    contract.validate_identifiers(_relation())
    contract.validate_identifiers(
        [
            _relation(),
            _relation(message_id="", event_id="evt-2", sequence=1),
        ]
    )
    # Optional ids may be omitted entirely.
    contract.validate_identifiers(
        contract.IdRelation(
            run_id="run-2", request_id="req-2", conversation_id="conv-2"
        )
    )


def test_identifier_validation_rejects_non_relation_values() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers([object()])  # type: ignore[list-item]
    assert excinfo.value.code == contract.ReasonCode.INVALID_RELATION


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": ""},
        {"run_id": ""},
        {"conversation_id": "   "},
        {"effect_id": "  "},
        {"lease_id": "\t"},
        {"policy_snapshot_id": " "},
    ],
)
def test_identifier_validation_rejects_empty_required_ids(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(_relation(**overrides))
    assert excinfo.value.code == contract.ReasonCode.EMPTY_ID


def test_identifier_validation_rejects_message_without_sequence() -> None:
    relation = _relation(sequence=None)
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(relation)
    assert excinfo.value.code == contract.ReasonCode.MISSING_MESSAGE_SEQUENCE


def test_one_request_maps_to_at_most_one_run() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(
                    run_id="run-2",
                    conversation_id="conv-2",
                    message_id="msg-2",
                    event_id="evt-2",
                    effect_id="eff-2",
                    lease_id="lease-2",
                    policy_snapshot_id="run-2:hash",
                ),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.REQUEST_RUN_CONFLICT


def test_one_run_maps_to_one_conversation() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(conversation_id="conv-2", message_id="msg-2", event_id="evt-2"),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.RUN_CONVERSATION_CONFLICT


def test_conversation_and_sequence_map_to_one_message() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(message_id="msg-2", event_id="evt-2", sequence=0),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.MESSAGE_SEQUENCE_CONFLICT


def test_event_id_is_globally_unique() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(run_id="run-2", request_id="req-2", conversation_id="conv-2",
                          message_id="msg-2", event_id="evt-1", sequence=0,
                          effect_id="eff-2", lease_id="lease-2",
                          policy_snapshot_id="run-2:hash"),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.DUPLICATE_EVENT_ID


def test_effect_and_lease_ids_cannot_cross_runs() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(
                    run_id="run-2",
                    request_id="req-2",
                    conversation_id="conv-2",
                    message_id="msg-2",
                    event_id="evt-2",
                    effect_id="eff-1",
                    lease_id="lease-2",
                    policy_snapshot_id="run-2:hash",
                ),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.EFFECT_RUN_CONFLICT

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(
                    run_id="run-2",
                    request_id="req-2",
                    conversation_id="conv-2",
                    message_id="msg-2",
                    event_id="evt-2",
                    effect_id="eff-2",
                    lease_id="lease-1",
                    policy_snapshot_id="run-2:hash",
                ),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.LEASE_RUN_CONFLICT


def test_policy_snapshot_id_is_bound_to_its_run() -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.validate_identifiers(
            [
                _relation(),
                _relation(
                    run_id="run-2",
                    request_id="req-2",
                    conversation_id="conv-2",
                    message_id="msg-2",
                    event_id="evt-2",
                    effect_id="eff-2",
                    lease_id="lease-2",
                    policy_snapshot_id="run-1:hash",
                ),
            ]
        )
    assert excinfo.value.code == contract.ReasonCode.POLICY_RUN_CONFLICT


# ---------------------------------------------------------------------------
# Policy hash, snapshot refs, terminal fence, error classification
# ---------------------------------------------------------------------------


def test_policy_hash_is_deterministic_and_key_order_independent() -> None:
    first = contract.policy_hash({"model": "m1", "budget": {"tokens": 10, "steps": 2}})
    reordered = contract.policy_hash(
        {"budget": {"steps": 2, "tokens": 10}, "model": "m1"}
    )
    assert first == reordered
    assert len(first) == 64
    int(first, 16)  # hex digest
    assert first == contract.policy_hash({"model": "m1", "budget": {"tokens": 10, "steps": 2}})


def test_policy_hash_uses_documented_canonical_json() -> None:
    canonical = '{"a":1,"b":"中文"}'.encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    assert contract.policy_hash({"b": "中文", "a": 1}) == expected


def test_policy_hash_is_change_sensitive() -> None:
    base = contract.policy_hash({"profile": "chat", "tools": ["read", "write"]})
    assert base != contract.policy_hash({"profile": "chat", "tools": ["read"]})
    assert base != contract.policy_hash({"profile": "chat", "tools": ["read", "write", ""]})
    assert base != contract.policy_hash(
        {"profile": "chat", "tools": ["read", "write"], "schema_version": 2}
    )


def test_policy_hash_normalizes_bytes_and_rejects_non_json() -> None:
    assert contract.policy_hash({"blob": b"abc", "n": [b"x"]}) == contract.policy_hash(
        {"blob": "abc", "n": ["x"]}
    )
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.policy_hash({"blob": b"\xff\xfe"})
    assert excinfo.value.code == contract.ReasonCode.POLICY_NOT_CANONICAL
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.policy_hash({"bad": object()})
    assert excinfo.value.code == contract.ReasonCode.POLICY_NOT_CANONICAL


def test_policy_snapshot_ref_is_run_scoped_and_version_checkable() -> None:
    digest = contract.policy_hash({"profile": "chat"})
    ref = contract.PolicySnapshotRef(run_id="run-1", policy_hash=digest)
    assert ref.snapshot_id == f"run-1:{digest}"
    ref.assert_compatible()

    stale = contract.PolicySnapshotRef(
        run_id="run-1", policy_hash=digest, schema_version=999
    )
    with pytest.raises(contract.ContractViolation) as excinfo:
        stale.assert_compatible()
    assert excinfo.value.code == contract.ReasonCode.POLICY_INCOMPATIBLE

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.PolicySnapshotRef(run_id="", policy_hash=digest)
    assert excinfo.value.code == contract.ReasonCode.EMPTY_ID


def test_terminal_fence_requires_durable_assistant_and_effects() -> None:
    fence = contract.TerminalFence(
        run_id="run-1",
        status=contract.RunStatus.DONE,
        terminal_event_id="evt-done",
        terminal_sequence=9,
        assistant_persisted=True,
    )
    fence.assert_complete()

    missing_assistant = contract.TerminalFence(
        run_id="run-1",
        status=contract.RunStatus.DONE,
        terminal_event_id="evt-done",
        terminal_sequence=9,
        assistant_persisted=False,
    )
    with pytest.raises(contract.ContractViolation) as excinfo:
        missing_assistant.assert_complete()
    assert excinfo.value.code == contract.ReasonCode.ASSISTANT_NOT_PERSISTED

    pending_effect = contract.TerminalFence(
        run_id="run-1",
        status=contract.RunStatus.DONE,
        terminal_event_id="evt-done",
        terminal_sequence=9,
        assistant_persisted=True,
        pending_effect_ids=("eff-1",),
    )
    with pytest.raises(contract.ContractViolation) as excinfo:
        pending_effect.assert_complete()
    assert excinfo.value.code == contract.ReasonCode.EFFECT_NOT_COMMITTED


def test_terminal_fence_rules_per_status() -> None:
    # An error need not have a persisted assistant, but may not leave effects
    # dangling; cancellation must fence effects the same way.
    contract.TerminalFence(
        run_id="run-1",
        status=contract.RunStatus.ERROR,
        terminal_event_id="evt-error",
        terminal_sequence=5,
    ).assert_complete()

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.TerminalFence(
            run_id="run-1",
            status=contract.RunStatus.CANCELLED,
            terminal_event_id="evt-cancel",
            terminal_sequence=5,
            pending_effect_ids=("eff-1",),
        ).assert_complete()
    assert excinfo.value.code == contract.ReasonCode.EFFECT_NOT_COMMITTED

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.TerminalFence(
            run_id="run-1",
            status=contract.RunStatus.RUNNING,
            terminal_event_id="evt-x",
            terminal_sequence=5,
            assistant_persisted=True,
        ).assert_complete()
    assert excinfo.value.code == contract.ReasonCode.TERMINAL_FENCE_NOT_TERMINAL

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.TerminalFence(
            run_id="run-1",
            status=contract.RunStatus.DONE,
            terminal_event_id="",
            terminal_sequence=5,
            assistant_persisted=True,
        ).assert_complete()
    assert excinfo.value.code == contract.ReasonCode.INVALID_TERMINAL_EVENT

    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.TerminalFence(
            run_id="run-1",
            status=contract.RunStatus.DONE,
            terminal_event_id="evt-done",
            terminal_sequence=0,
            assistant_persisted=True,
        ).assert_complete()
    assert excinfo.value.code == contract.ReasonCode.INVALID_TERMINAL_EVENT


def test_execution_error_codes_are_stable_and_classified() -> None:
    assert {member.value for member in contract.ExecutionErrorCode} >= {
        "admission_failed",
        "terminal_persist_failed",
        "assistant_persist_failed",
        "outcome_unknown",
        "lease_expired",
        "cleanup_pending",
        "replay_gap",
        "policy_incompatible",
        "duplicate_terminal",
        "repair_pending",
    }
    unknown = contract.BoundaryError(
        contract.ExecutionErrorCode.OUTCOME_UNKNOWN, "ledger result indeterminate"
    )
    assert unknown.code is contract.ExecutionErrorCode.OUTCOME_UNKNOWN
    assert unknown.recoverable is True
    assert "outcome_unknown" in str(unknown)

    terminal = contract.BoundaryError(
        contract.ExecutionErrorCode.DUPLICATE_TERMINAL, "second terminal"
    )
    assert terminal.recoverable is False
    incompatible = contract.BoundaryError(
        contract.ExecutionErrorCode.POLICY_INCOMPATIBLE, "old policy"
    )
    assert incompatible.recoverable is False
    explicit = contract.BoundaryError(
        contract.ExecutionErrorCode.REPAIR_PENDING, "janitor", recoverable=False
    )
    assert explicit.recoverable is False


# ---------------------------------------------------------------------------
# Legacy state adapters
# ---------------------------------------------------------------------------


class _V10ToV11Adapter:
    from_version = 10
    to_version = 11

    def adapt(self, payload: dict[str, object]) -> dict[str, object]:
        adapted = dict(payload)
        adapted["schema_version"] = 11
        adapted["status"] = contract.adapt_legacy_run_status(
            str(payload.get("status", ""))
        ).value
        return adapted


class _V12ToV13Adapter:
    from_version = 12
    to_version = 13

    def adapt(self, payload: dict[str, object]) -> dict[str, object]:
        return dict(payload)


def test_adapter_registry_is_exact_version_paired() -> None:
    adapter = _V10ToV11Adapter()
    contract.register_adapter(adapter)
    assert contract.get_adapter(10, 11) is adapter
    assert contract.get_adapter(11, 10) is None
    assert contract.get_adapter(10, 12) is None
    assert contract.get_adapter(0, 1) is None

    adapted = contract.get_adapter(10, 11)
    assert adapted is not None
    payload = adapted.adapt({"status": "paused"})
    assert payload["schema_version"] == 11
    assert payload["status"] == "paused"


def test_adapter_registry_rejects_duplicates_and_invalid_adapters() -> None:
    contract.register_adapter(_V12ToV13Adapter())
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.register_adapter(_V12ToV13Adapter())
    assert excinfo.value.code == contract.ReasonCode.ADAPTER_CONFLICT

    class _NoAdapt:
        from_version = 20
        to_version = 21

    class _Backwards:
        from_version = 22
        to_version = 21

        def adapt(self, payload: dict[str, object]) -> dict[str, object]:
            return dict(payload)

    class _SameVersion:
        from_version = 23
        to_version = 23

        def adapt(self, payload: dict[str, object]) -> dict[str, object]:
            return dict(payload)

    for bad in (_NoAdapt(), _Backwards(), _SameVersion()):
        with pytest.raises(contract.ContractViolation) as excinfo:
            contract.register_adapter(bad)  # type: ignore[arg-type]
        assert excinfo.value.code == contract.ReasonCode.INVALID_ADAPTER


def test_legacy_run_status_adapter_maps_every_stored_status() -> None:
    assert contract.adapt_legacy_run_status("running") is contract.RunStatus.RUNNING
    assert contract.adapt_legacy_run_status("paused") is contract.RunStatus.PAUSED
    assert contract.adapt_legacy_run_status("done") is contract.RunStatus.DONE
    assert contract.adapt_legacy_run_status("error") is contract.RunStatus.ERROR
    assert contract.adapt_legacy_run_status("cancelled") is contract.RunStatus.CANCELLED


@pytest.mark.parametrize("value", ["", "RUNNING", "queued", "unknown", "done "])
def test_legacy_run_status_adapter_rejects_unknown_without_default(
    value: str,
) -> None:
    with pytest.raises(contract.ContractViolation) as excinfo:
        contract.adapt_legacy_run_status(value)
    assert excinfo.value.code == contract.ReasonCode.UNKNOWN_LEGACY_STATUS


# ---------------------------------------------------------------------------
# Persistence seam (fabricated repository, no SQLite)
# ---------------------------------------------------------------------------


class _HarnessRunRepository:
    """In-memory implementation of :class:`RunRepositoryContract`.

    The harness proves the seam is storage-neutral and that the lifecycle,
    identity and terminal-fence rules can be exercised without SQLite,
    orchestration, a model, or the network.
    """

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, object]] = {}
        self.by_request: dict[str, str] = {}
        self.messages: dict[tuple[str, int], dict[str, object]] = {}
        self.events: dict[str, list[dict[str, object]]] = {}
        self.event_ids: set[str] = set()
        self._counter = 0

    async def create_or_get_run(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
        user_message: str,
    ) -> tuple[str, bool]:
        if not request_id or not conversation_id:
            raise contract.ContractViolation(
                contract.ReasonCode.EMPTY_ID, "request and conversation required"
            )
        if request_id in self.by_request:
            return self.by_request[request_id], False
        self._counter += 1
        run_id = f"run-{self._counter}"
        contract.validate_identifiers(
            contract.IdRelation(
                run_id=run_id,
                request_id=request_id,
                conversation_id=conversation_id,
                message_id=f"msg-{self._counter}",
                sequence=0,
            )
        )
        self.by_request[request_id] = run_id
        self.runs[run_id] = {
            "run_id": run_id,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "profile_id": profile_id,
            "status": contract.RunStatus.ADMISSION_PENDING.value,
            "sequence_cursor": 0,
            "messages": [],
        }
        await self.append_message(
            run_id=run_id,
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            sequence=0,
            message_id=f"msg-{self._counter}",
        )
        return run_id, True

    async def transition(
        self,
        *,
        run_id: str,
        target: contract.RunStatus,
        fence: contract.TerminalFence | None = None,
    ) -> contract.RunStatus:
        run = self.runs.get(run_id)
        if run is None:
            raise contract.ContractViolation(
                contract.ReasonCode.UNKNOWN_STATUS, f"unknown run {run_id!r}"
            )
        status = contract.coerce_run_status(str(run["status"]))
        contract.assert_transition(status, target)
        target_status = contract.coerce_run_status(target)
        if target_status in contract.TERMINAL_STATUSES:
            if fence is None:
                raise contract.ContractViolation(
                    contract.ReasonCode.TERMINAL_FENCE_NOT_TERMINAL,
                    "terminal transition requires a fence",
                )
            if fence.run_id != run_id or fence.status != target_status:
                raise contract.ContractViolation(
                    contract.ReasonCode.INVALID_TERMINAL_EVENT,
                    "fence does not match run/target",
                )
            fence.assert_complete()
        run["status"] = target_status.value
        return target_status

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
        message_id = message_id or f"msg-auto-{len(self.messages) + 1}"
        key = (conversation_id, sequence)
        if key in self.messages:
            existing = self.messages[key]
            if existing["message_id"] != message_id:
                raise contract.ContractViolation(
                    contract.ReasonCode.MESSAGE_SEQUENCE_CONFLICT,
                    "sequence already used by another message",
                )
            return str(existing["message_id"])
        self.messages[key] = {
            "message_id": message_id,
            "run_id": run_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sequence": sequence,
        }
        run = self.runs[run_id]
        run["messages"].append(self.messages[key])  # type: ignore[union-attr]
        return message_id

    async def append_event(
        self,
        *,
        run_id: str,
        conversation_id: str,
        event_id: str,
        sequence: int,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        if event_id in self.event_ids:
            raise contract.ContractViolation(
                contract.ReasonCode.DUPLICATE_EVENT_ID, "event already stored"
            )
        events = self.events.setdefault(run_id, [])
        last = int(events[-1]["sequence"]) if events else 0  # type: ignore[arg-type]
        contract.compute_gap(run_id, last, sequence)  # reject non-monotonic
        self.event_ids.add(event_id)
        events.append(
            {
                "event_id": event_id,
                "sequence": sequence,
                "type": event_type,
                "payload": dict(payload),
                "conversation_id": conversation_id,
            }
        )
        run = self.runs[run_id]
        run["sequence_cursor"] = sequence

    async def read_snapshot(self, *, run_id: str) -> dict[str, object] | None:
        run = self.runs.get(run_id)
        return dict(run) if run is not None else None

    async def read_events(
        self, *, run_id: str, from_sequence: int = 0
    ) -> list[dict[str, object]]:
        return [
            dict(event)
            for event in self.events.get(run_id, [])
            if int(event["sequence"]) > from_sequence  # type: ignore[arg-type]
        ]


def test_harness_repository_satisfies_the_seam() -> None:
    repo = _HarnessRunRepository()
    assert isinstance(repo, contract.RunRepositoryContract)


def test_harness_runs_full_lifecycle_and_replays_gap() -> None:
    import asyncio

    async def scenario() -> None:
        repo = _HarnessRunRepository()
        run_id, created = await repo.create_or_get_run(
            request_id="req-harness",
            conversation_id="conv-harness",
            profile_id="chat",
            user_message="hello",
        )
        assert created is True
        again, created_again = await repo.create_or_get_run(
            request_id="req-harness",
            conversation_id="conv-harness",
            profile_id="chat",
            user_message="hello",
        )
        assert again == run_id
        assert created_again is False

        await repo.transition(run_id=run_id, target=contract.RunStatus.ADMITTED)
        await repo.transition(run_id=run_id, target=contract.RunStatus.RUNNING)
        await repo.append_event(
            run_id=run_id,
            conversation_id="conv-harness",
            event_id="evt-1",
            sequence=1,
            event_type="text",
            payload={"delta": "hi"},
        )
        await repo.append_event(
            run_id=run_id,
            conversation_id="conv-harness",
            event_id="evt-4",
            sequence=4,
            event_type="text",
            payload={"delta": "there"},
        )
        await repo.append_message(
            run_id=run_id,
            conversation_id="conv-harness",
            role="assistant",
            content="hi there",
            sequence=1,
            message_id="msg-assistant",
        )
        await repo.transition(
            run_id=run_id,
            target=contract.RunStatus.DONE,
            fence=contract.TerminalFence(
                run_id=run_id,
                status=contract.RunStatus.DONE,
                terminal_event_id="evt-done",
                terminal_sequence=5,
                assistant_persisted=True,
            ),
        )

        snapshot = await repo.read_snapshot(run_id=run_id)
        assert snapshot is not None
        assert snapshot["status"] == "done"

        events = await repo.read_events(run_id=run_id)
        assert [(e["sequence"], e["type"]) for e in events] == [
            (1, "text"),
            (4, "text"),
        ]
        replay_gap = contract.compute_gap(
            run_id, int(events[0]["sequence"]), int(events[1]["sequence"])  # type: ignore[arg-type]
        )
        assert replay_gap is not None
        assert replay_gap.to_payload() == {
            "from_sequence": 1,
            "to_sequence": 4,
            "dropped": 2,
        }

        # A terminal run cannot be continued through the seam.
        with pytest.raises(contract.ContractViolation) as excinfo:
            await repo.transition(run_id=run_id, target=contract.RunStatus.RUNNING)
        assert excinfo.value.code == contract.ReasonCode.INVALID_TRANSITION

    asyncio.run(scenario())




