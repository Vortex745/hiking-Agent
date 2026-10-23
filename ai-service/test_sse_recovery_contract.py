"""BCDEF ticket 11 — SSE recovery contract: replay gaps, terminals, snapshot.

Behavior asserted at the replay/snapshot boundary (no real LLM, tmp SQLite):

* admission / tool-effect / assistant / terminal durable events replay
  through ``get_replay_events`` (the ``/events`` + resume read) in
  ascending sequence order;
* a dropped progress leaves a VISIBLE ``progress.gap`` in replay — a
  deterministic envelope built via ``execution_contract.compute_gap`` with
  the stable ``from_sequence``/``to_sequence``/``dropped`` payload — that
  never reorders, delays or duplicates the terminal;
* at most one terminal per run is replayed; duplicate finalize calls
  return the one persisted terminal;
* a cancelled run whose terminal event never persisted still replays
  exactly one ``cancelled`` terminal (cancel is replayable);
* ``outcome_unknown`` replay terminals are stable, non-recoverable error
  frames the client maps onto its first-class ``outcome_unknown`` status;
* the snapshot keeps the terminal persistence fields consistent so a
  client can never render ``done`` on a terminal that failed to persist;
* the envelope field set is unchanged (old clients keep parsing every
  replayed frame; the synthesized gap is an additive known type).

Run (ai-service working directory):
    C:/Users/14253/Desktop/hiking-ai-main/ai-service/.venv/Scripts/python.exe \
        -m pytest -q test_sse_recovery_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.execution_contract import DURABLE_EVENT_TYPES, compute_gap
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository

RUN_TERMINALS = {"done", "error", "cancelled"}
ENVELOPE_KEYS = {
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


# ---- helpers -----------------------------------------------------------------


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _admit(repo: RunRepository, *, request_id: str) -> str:
    result = repo.admit(
        request_id=request_id,
        conversation_id=None,
        profile_id="chat",
        user_message="徒步建议",
    )
    assert result.ok and result.execute, result
    repo.mark_run_running(result.run_id)
    # ``admit`` already persisted the durable admission (``run_started``)
    # event at sequence 1 — the tests below never re-persist it.
    assert result.admission_event is not None
    assert result.admission_event["type"] == "run_started"
    return result.run_id


def _persist(
    repo: RunRepository,
    run_id: str,
    *,
    sequence: int,
    event_type: str,
    payload: dict[str, Any] | None = None,
    request_id: str = "req-gap",
    conversation_id: str = "conv-gap",
) -> None:
    repo.persist_event(
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload or {},
        request_id=request_id,
        conversation_id=conversation_id,
    )


def _sequences(events: list[dict[str, Any]]) -> list[int]:
    return [event["sequence"] for event in events]


# ---- replay: progress gaps are visible, terminals keep their order -----------


def test_replay_surfaces_progress_gap_for_dropped_progress(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # sequences 3 and 4 ("dropped progress") have no durable row.
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="progress")
    _persist(repo, run_id, sequence=5, event_type="text", payload={"content": "答"})
    repo.finalize_run(
        run_id=run_id,
        request_id="req-gap",
        terminal_type="done",
        payload={"code": "done"},
        sequence=6,
        assistant_content="答",
    )

    events = repo.get_replay_events(run_id=run_id, from_seq=0)

    gap_events = [e for e in events if e["type"] == "progress.gap"]
    assert len(gap_events) == 1
    assert gap_events[0]["payload"] == {
        "from_sequence": 2,
        "to_sequence": 5,
        "dropped": 2,
    }
    # The gap envelope payload is exactly what compute_gap defines.
    assert gap_events[0]["payload"] == compute_gap(run_id, 2, 5).to_payload()

    # Order: gap sits inside the hole (before seq 5), the terminal stays
    # LAST and exactly one terminal is replayed.
    assert _sequences(events) == sorted(_sequences(events))
    assert [e["type"] for e in events] == [
        "run_started",
        "progress",
        "progress.gap",
        "text",
        "done",
    ]
    terminals = [e for e in events if e["type"] in RUN_TERMINALS]
    assert len(terminals) == 1 and terminals[0]["sequence"] == 6

    # Every replayed frame — stored or synthesized — keeps the v2
    # envelope field set (AC1: old clients still parse the stream).
    for event in events:
        assert set(event.keys()) == ENVELOPE_KEYS
        json.dumps(event)  # wire-serializable


def test_replay_gap_is_clamped_to_client_cursor_and_deterministic(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="progress")
    _persist(repo, run_id, sequence=7, event_type="text", payload={"content": "答"})

    from_cursor3 = repo.get_replay_events(run_id=run_id, from_seq=3)
    assert [(e["sequence"], e["type"]) for e in from_cursor3] == [
        (4, "progress.gap"),
        (7, "text"),
    ]
    # The visible hole is measured from the client's own cursor.
    assert from_cursor3[0]["payload"] == {
        "from_sequence": 3,
        "to_sequence": 7,
        "dropped": 3,
    }
    # Repeated reads agree byte-for-byte (nothing is written).
    assert repo.get_replay_events(run_id=run_id, from_seq=3) == from_cursor3
    # A cursor past the hole sees no gap at all.
    assert [e["sequence"] for e in repo.get_replay_events(run_id=run_id, from_seq=6)] == [7]


def test_gap_never_reorders_or_duplicates_the_terminal(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # The hole is directly before the terminal: the gap must NOT take the
    # terminal's sequence nor land after it.
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="progress")
    repo.finalize_run(
        run_id=run_id,
        request_id="req-gap",
        terminal_type="done",
        payload={"code": "done"},
        sequence=5,
        assistant_content="答",
    )
    # A second finalize returns the ONE persisted terminal unchanged.
    again = repo.finalize_run(
        run_id=run_id,
        request_id="req-gap",
        terminal_type="done",
        payload={"code": "done"},
        sequence=9,
        assistant_content="答",
    )
    assert again.created is False

    events = repo.get_replay_events(run_id=run_id, from_seq=0)
    assert [e["type"] for e in events] == [
        "run_started",
        "progress",
        "progress.gap",
        "done",
    ]
    assert events[-1]["sequence"] == 5
    assert len([e for e in events if e["type"] in RUN_TERMINALS]) == 1
    # The gap envelope never occupies the terminal's sequence.
    assert events[2]["sequence"] == 3


def test_terminal_without_persisted_event_replays_one_synthetic_terminal_after_gaps(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="progress")
    _persist(repo, run_id, sequence=5, event_type="text", payload={"content": "答"})
    # Terminal event write failed but the control row reached ``done``:
    # replay still ends with exactly one stable synthetic terminal.
    repo.mark_run_done(run_id)

    events = repo.get_replay_events(run_id=run_id, from_seq=0)

    assert [e["type"] for e in events] == [
        "run_started",
        "progress",
        "progress.gap",
        "text",
        "error",
    ]
    terminal = events[-1]
    assert terminal["payload"]["code"] == "terminal_persist_failed"
    assert terminal["payload"]["terminal_persisted"] is False
    assert _sequences(events) == sorted(_sequences(events))
    # Repeated replay reads are identical (the terminal is not re-invented).
    assert repo.get_replay_events(run_id=run_id, from_seq=0) == events


# ---- durable classes replay ---------------------------------------------------


def test_durable_event_classes_all_replay_in_order(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    durable = [
        ("approval.required", 3),
        ("approval.resolved", 4),
        ("tool_call", 5),
        ("tool_result", 6),
        ("effect.outcome_unknown", 7),
        ("text", 8),
    ]
    for event_type, sequence in durable:
        _persist(repo, run_id, sequence=sequence, event_type=event_type)
    repo.finalize_run(
        run_id=run_id,
        request_id="req-gap",
        terminal_type="error",
        payload={"code": "effect_outcome_unknown"},
        sequence=9,
        assistant_content="答",
    )

    events = repo.get_replay_events(run_id=run_id, from_seq=0)
    replayed_types = {e["type"] for e in events}
    # Every durable class — admission (run_started), tool/effect gates,
    # assistant text and the terminal — is present in the replay.
    assert replayed_types >= {t for t, _ in durable} | {"run_started", "error"}
    for event_type, _ in durable:
        assert event_type in DURABLE_EVENT_TYPES
    assert _sequences(events) == sorted(_sequences(events))
    # Exactly one run terminal, last.
    terminals = [e for e in events if e["type"] in RUN_TERMINALS]
    assert len(terminals) == 1 and terminals[0]["sequence"] == 9


def test_paused_run_replays_paused_frame_without_a_run_terminal(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="approval.required")
    _persist(repo, run_id, sequence=3, event_type="run.paused")
    repo.mark_run_paused(run_id)

    events = repo.get_replay_events(run_id=run_id, from_seq=0)

    # ``run.paused`` is a STREAM terminal, not a run terminal: it replays
    # verbatim and no synthetic done/error/cancelled is appended.
    assert [e["type"] for e in events] == ["run_started", "approval.required", "run.paused"]
    assert not any(e["type"] in RUN_TERMINALS for e in events)


def test_cancelled_without_terminal_event_is_a_replayable_terminal(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    _persist(repo, run_id, sequence=2, event_type="progress")

    outcome = repo.request_cancel(run_id=run_id, idempotency_key="cancel-1")
    assert outcome.accepted is True

    events = repo.get_replay_events(run_id=run_id, from_seq=0)
    terminals = [e for e in events if e["type"] in RUN_TERMINALS]
    assert len(terminals) == 1
    assert terminals[0]["type"] == "cancelled"
    assert terminals[0]["payload"]["code"] == "cancelled"

    snapshot = repo.get_snapshot(run_id)
    assert snapshot.status == "cancelled"
    # Cancel without a stored terminal event is NOT a persist failure —
    # it is the designed replayable terminal path.
    assert snapshot.terminal_persist_failed is False


def test_outcome_unknown_replays_a_stable_non_recoverable_terminal(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    # Ticket 05 persists the run-level ``outcome_unknown`` status; here the
    # control read is patched so the replay contract is pinned without
    # pre-empting that ticket's storage work.
    fake = SimpleNamespace(
        run_id=run_id,
        status="outcome_unknown",
        conversation_id="conv-gap",
        request_id="req-gap",
        updated_at="2026-10-23T00:00:00+00:00",
        terminal_persist_failed=False,
    )
    repo.control.get_run = lambda rid: fake  # type: ignore[method-assign]

    events = repo.get_replay_events(run_id=run_id, from_seq=0)
    terminals = [e for e in events if e["type"] in RUN_TERMINALS]
    assert len(terminals) == 1
    payload = terminals[0]["payload"]
    assert payload["code"] == "effect_outcome_unknown"
    assert payload["run_status"] == "outcome_unknown"
    assert payload["recoverable"] is False
    # Deterministic across reads.
    assert repo.get_replay_events(run_id=run_id, from_seq=0) == events


# ---- snapshot terminal-field consistency -------------------------------------


def test_snapshot_after_successful_finalize_is_consistent(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    repo.finalize_run(
        run_id=run_id,
        request_id="req-gap",
        terminal_type="done",
        payload={"code": "done"},
        sequence=2,
        assistant_content="安全第一。",
    )

    snapshot = repo.get_snapshot(run_id)
    assert snapshot.status == "done"
    assert snapshot.terminal_persisted is True
    assert snapshot.terminal_persist_failed is False
    assert snapshot.terminal_event_type == "done"
    assert snapshot.error_code is None
    assert snapshot.sequence_cursor == 2
    # AC3: messages are part of the authoritative recovery source.
    assert [(m["role"], m["content"]) for m in snapshot.messages] == [
        ("user", "徒步建议"),
        ("assistant", "安全第一。"),
    ]


def test_snapshot_terminal_persist_failed_never_reads_as_clean_done(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    # ``run_started`` at sequence 1 is the durable admission event (see ``_admit``).
    # Control row reached ``done`` but the terminal event row is missing
    # (write lost): the snapshot must explain it as terminal_persist_failed
    # with a stable error code, so no client can render a clean done.
    repo.mark_run_done(run_id)

    snapshot = repo.get_snapshot(run_id)
    assert snapshot.terminal_persisted is False
    assert snapshot.terminal_persist_failed is True
    assert snapshot.error_code == "terminal_persist_failed"
    assert snapshot.recoverable is True


def test_snapshot_surfaces_pending_approvals_for_paused_runs(tmp_path: Path):
    repo = _repo(tmp_path)
    run_id = _admit(repo, request_id="req-gap")
    repo.insert_approval(
        approval_id="apr-1",
        run_id=run_id,
        conversation_id="conv-gap",
        tool_name="generate_pdf",
        args_canonical=json.dumps({"filename": "hiking-plan.pdf"}),
        risk_level="high",
        impact="生成徒步计划 PDF 并导出到宿主存储",
        created_at="2026-10-23T00:00:00+00:00",
        expires_at="2026-10-23T00:30:00+00:00",
    )
    repo.mark_run_paused(run_id)

    snapshot = repo.get_snapshot(run_id)
    assert snapshot.status == "paused"
    assert len(snapshot.pending_approvals) == 1
    approval = snapshot.pending_approvals[0]
    # Projected into the ``approval.required`` payload shape so the same
    # client parser rehydrates the prompt after a refresh.
    assert approval["approval_id"] == "apr-1"
    assert approval["tool"] == "generate_pdf"
    assert approval["args"] == {"filename": "hiking-plan.pdf"}
    assert approval["risk_level"] == "high"
    assert approval["expires_at"] == "2026-10-23T00:30:00+00:00"
