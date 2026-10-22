"""BCDEF ticket 04 — effect ledger claim/commit, fencing and replay.

Seam: the public effect-ledger API (:mod:`agent_v2.effects` +
``RunRepository``) and the orchestration driver :func:`drive_effect`.
All persistence is real SQLite under a temp directory; the external
executor is a counting fake. No network, no sandbox.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.effects import (
    EffectClaim,
    EffectNotClaimedError,
    EffectStatus,
    canonical_effect_status,
    compute_effect_identity,
    execute_claimed,
)
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import (
    EFFECT_COPY_GENERATE_PDF,
    EffectDriveResult,
    drive_effect,
)
from agent_v2.runs import RunRepository


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _make_run(repo: RunRepository, request_id: str = "req-effect") -> str:
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="m",
    )
    return run_id


def _insert_effect(repo: RunRepository, run_id: str, args: dict[str, Any]):
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args=args
    )
    repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id="conv-effect",
        tool_name="generate_pdf",
        args_canonical='{"filename": "p.pdf"}',
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    return identity


# ---- stable identity --------------------------------------------------------


def test_effect_identity_is_deterministic_across_retries(tmp_path):
    same = compute_effect_identity(
        run_id="run-1", tool_name="sandbox_bash", args={"b": 2, "a": 1}
    )
    reordered = compute_effect_identity(
        run_id="run-1", tool_name="sandbox_bash", args={"a": 1, "b": 2}
    )
    other_run = compute_effect_identity(
        run_id="run-2", tool_name="sandbox_bash", args={"a": 1, "b": 2}
    )
    assert same.effect_id == reordered.effect_id
    assert same.idempotency_key == reordered.idempotency_key
    assert same.effect_id != other_run.effect_id
    assert same.audit_summary()["effect_id"] == same.effect_id
    assert same.audit_summary()["idempotency_key"] == same.idempotency_key


def test_ledger_statuses_are_pending_claimed_then_committed(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.phase == EffectStatus.PENDING.value
        assert row.resolved_effect_id == identity.effect_id
        assert row.resolved_idempotency_key == identity.idempotency_key

        _row, won = repo.claim_effect(
            identity.effect_id,
            owner="w1",
            claim_token="token-1",
            ttl_seconds=60,
        )
        assert won is True
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.phase == EffectStatus.CLAIMED.value
        # Legacy spelling stays readable (old rows keep working).
        assert row.status in ("executing", "claimed")

        repo.commit_effect(identity.effect_id, '{"ok": true}', claim_token="token-1")
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.phase == EffectStatus.COMMITTED.value
        assert canonical_effect_status(row.status) == "committed"
    finally:
        repo.close()


# ---- claim CAS + concurrency ------------------------------------------------


def test_concurrent_claim_has_exactly_one_winner(tmp_path):
    shared = tmp_path / "control.sqlite"
    seeder = RunRepository(
        control=ControlStore(str(shared)),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    run_id = _make_run(seeder)
    identity = _insert_effect(seeder, run_id, {"filename": "p.pdf"})
    seeder.close()

    store_a = ControlStore(str(shared))
    store_b = ControlStore(str(shared))
    results: list[tuple[str, bool, str]] = []
    barrier = threading.Barrier(2)
    executions: list[str] = []

    def _claim(store: ControlStore, token: str) -> None:
        barrier.wait()
        record, won = store.claim_effect(
            identity.effect_id,
            owner=f"worker-{token}",
            claim_token=token,
            ttl_seconds=60,
        )
        results.append((token, won, record.status if record else "missing"))
        if won:
            executions.append(token)
            store.commit_effect(
                identity.effect_id, '{"path": "p.pdf"}', claim_token=token
            )

    threads = [
        threading.Thread(target=_claim, args=(store_a, "token-a")),
        threading.Thread(target=_claim, args=(store_b, "token-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    winners = [token for token, won, _status in results if won]
    assert len(winners) == 1
    assert executions == winners  # exactly one external execution path
    conn = sqlite3.connect(str(shared))
    try:
        row = conn.execute(
            "SELECT status, claim_token, result_json FROM effects "
            "WHERE effect_key = ?",
            (identity.effect_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "committed"
    assert row[1] == winners[0]
    assert row[2] == '{"path": "p.pdf"}'

    # A retry sees the committed result instead of claiming again.
    retry_winner = [token for token, won, _ in results if not won]
    assert retry_winner
    store_a.close()
    store_b.close()


def test_execution_requires_a_winning_claim(tmp_path):
    calls: list[str] = []

    def executor() -> str:
        calls.append("ran")
        return "side-effect"

    with pytest.raises(EffectNotClaimedError) as excinfo:
        execute_claimed(None, executor)
    assert excinfo.value.code.value == "effect_not_claimed"
    assert calls == []

    claim = EffectClaim(effect_id="e-1", claim_token="t-1", owner="w")
    assert execute_claimed(claim, executor) == "side-effect"
    assert calls == ["ran"]


def test_claim_is_fenced_by_token_and_lease_metadata(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        row, won = repo.claim_effect(
            identity.effect_id,
            owner="sandbox-worker",
            claim_token="claim-1",
            ttl_seconds=60,
            lease_id="lease-9",
            fencing_token=7,
        )
        assert won and row is not None
        assert row.claim_owner == "sandbox-worker"
        assert row.claim_token == "claim-1"
        assert row.lease_id == "lease-9"
        assert row.fencing_token == 7

        # A stale/wrong token cannot settle the claim.
        repo.commit_effect(
            identity.effect_id, '{"x": 1}', claim_token="stale-token"
        )
        still_claimed = repo.get_effect(identity.effect_id)
        assert still_claimed is not None
        assert still_claimed.phase == EffectStatus.CLAIMED.value

        # The owner token settles it exactly once.
        repo.commit_effect(
            identity.effect_id, '{"x": 1}', claim_token="claim-1"
        )
        committed = repo.get_effect(identity.effect_id)
        assert committed is not None and committed.phase == "committed"

        # Second claim (duplicate) never executes again.
        _again, won_again = repo.claim_effect(
            identity.effect_id,
            owner="other",
            claim_token="claim-2",
            ttl_seconds=60,
        )
        assert won_again is False
    finally:
        repo.close()


# ---- drive_effect publication gate -----------------------------------------


def _drive(
    repo: RunRepository,
    run_id: str,
    identity,
    *,
    executor: Any,
    emit_required: Any | None = None,
) -> tuple[EffectDriveResult, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event_type: str, payload: dict[str, Any]) -> Any:
        events.append((event_type, payload))
        return SimpleNamespace(type=event_type, payload=payload)

    outcome = EffectDriveResult()

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=repo,
            run_id=run_id,
            conversation_id="conv-effect",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=identity.effect_id,
            external_idempotency_key=identity.idempotency_key,
            args_canonical='{"filename": "p.pdf"}',
            tool_args={"filename": "p.pdf"},
            all_events=[],
            has_tool_call=False,
            has_tool_result_ok=False,
            execute=executor,
            emit=emit,
            outcome=outcome,
            emit_required=emit_required,
        ):
            pass

    asyncio.run(run())
    return outcome, events


def _ok_executor(calls: list[int], result: dict[str, Any] | None = None):
    def execute() -> dict[str, Any]:
        calls.append(1)
        return result if result is not None else {"path": "p.pdf"}

    return execute


def test_success_result_published_only_after_commit_and_durable_event(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        calls: list[int] = []
        observations: list[tuple[str, str | None]] = []

        def emitting(event_type: str, payload: dict[str, Any]) -> Any:
            row = repo.get_effect(identity.effect_id)
            observations.append(
                (event_type, row.status if row is not None else None)
            )
            return SimpleNamespace(type=event_type, payload=payload)

        outcome = EffectDriveResult()

        async def run() -> None:
            async for _ in drive_effect(
                runs_repo=repo,
                run_id=run_id,
                conversation_id="conv-effect",
                copy=EFFECT_COPY_GENERATE_PDF,
                effect_key=identity.effect_id,
                external_idempotency_key=identity.idempotency_key,
                args_canonical='{"filename": "p.pdf"}',
                tool_args={"filename": "p.pdf"},
                all_events=[],
                has_tool_call=False,
                has_tool_result_ok=False,
                execute=_ok_executor(calls),
                emit=emitting,
                outcome=outcome,
                emit_required=emitting,
            ):
                pass

        asyncio.run(run())

        assert calls == [1]
        assert outcome.status == "executed"
        # tool_call is emitted while the ledger is claimed; tool_result is
        # emitted only AFTER the ledger committed.
        assert observations == [
            ("tool_call", "executing"),
            ("tool_result", "committed"),
        ]
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == "committed"
    finally:
        repo.close()


@pytest.mark.parametrize(
    "failure_kind",
    ["commit_raises", "serialization", "event_persist"],
)
def test_partial_success_paths_become_outcome_unknown(
    tmp_path, monkeypatch, failure_kind
):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        calls: list[int] = []

        result: dict[str, Any] = {"path": "p.pdf"}
        if failure_kind == "serialization":
            result = {"path": object()}  # not JSON serializable
        if failure_kind == "commit_raises":
            def boom(*_a, **_k):
                raise sqlite3.OperationalError("ledger write failed")

            monkeypatch.setattr(repo, "commit_effect", boom)

        strict_calls: list[int] = []

        def strict_emit(event_type: str, payload: dict[str, Any]) -> None:
            strict_calls.append(1)
            raise sqlite3.OperationalError("event write failed")

        outcome, events = _drive(
            repo,
            run_id,
            identity,
            executor=_ok_executor(calls, result),
            emit_required=(
                strict_emit if failure_kind == "event_persist" else None
            ),
        )

        types = [event_type for event_type, _payload in events]
        success_results = [
            payload
            for event_type, payload in events
            if event_type == "tool_result" and payload.get("status") == "ok"
        ]
        assert calls == [1]  # external side effect ran exactly once
        assert success_results == []
        assert outcome.status == "outcome_unknown"
        assert "effect.outcome_unknown" in types
        assert types[-1] == "error"
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == "outcome_unknown"
        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"

        # No model/effect rerun on a retry.
        calls.clear()
        outcome2, events2 = _drive(
            repo, run_id, identity, executor=_ok_executor(calls)
        )
        assert calls == []
        assert outcome2.status == "outcome_unknown"
        assert not [
            payload
            for event_type, payload in events2
            if event_type == "tool_result" and payload.get("status") == "ok"
        ]
    finally:
        repo.close()


def test_committed_result_is_returned_on_retry_without_execution(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        calls: list[int] = []
        outcome, events = _drive(
            repo, run_id, identity, executor=_ok_executor(calls)
        )
        assert outcome.status == "executed" and calls == [1]

        retry_calls: list[int] = []
        outcome2, events2 = _drive(
            repo, run_id, identity, executor=_ok_executor(retry_calls)
        )
        assert retry_calls == []
        assert outcome2.status == "in_flight" or outcome2.status == "recovered"
        # The ledger is the unique source of truth for the result.
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == "committed"
        assert row.result_json == '{"path":"p.pdf"}'
    finally:
        repo.close()


# ---- replay / snapshot reconstruction --------------------------------------


def test_snapshot_and_events_reconstruct_effect_status_and_result(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        persisted: list[tuple[str, dict[str, Any]]] = []

        def emit(event_type: str, payload: dict[str, Any]) -> Any:
            repo.persist_event(
                run_id=run_id,
                sequence=10 + len(persisted),
                event_type=event_type,
                payload=payload,
                request_id="req-effect",
                conversation_id="conv-effect",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            persisted.append((event_type, payload))
            return SimpleNamespace(type=event_type, payload=payload)

        outcome = EffectDriveResult()

        async def run() -> None:
            async for _ in drive_effect(
                runs_repo=repo,
                run_id=run_id,
                conversation_id="conv-effect",
                copy=EFFECT_COPY_GENERATE_PDF,
                effect_key=identity.effect_id,
                external_idempotency_key=identity.idempotency_key,
                args_canonical='{"filename": "p.pdf"}',
                tool_args={"filename": "p.pdf"},
                all_events=[],
                has_tool_call=False,
                has_tool_result_ok=False,
                execute=_ok_executor([], {"path": "p.pdf", "content": "ok"}),
                emit=emit,
                outcome=outcome,
                emit_required=emit,
            ):
                pass

        asyncio.run(run())
        assert outcome.status == "executed"

        # Snapshot reconstructs status + result.
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert len(snapshot.effects) == 1
        effect = snapshot.effects[0]
        assert effect["effect_id"] == identity.effect_id
        assert effect["status"] == "committed"
        assert effect["result"] == {"path": "p.pdf", "content": "ok"}

        # Replay reconstructs the same identity + result.
        stored = repo.get_events_since(run_id=run_id, from_seq=0)
        tool_call = next(e for e in stored if e["type"] == "tool_call")
        tool_result = next(e for e in stored if e["type"] == "tool_result")
        assert tool_call["payload"]["effect_id"] == identity.effect_id
        assert (
            tool_call["payload"]["idempotency_key"]
            == identity.idempotency_key
        )
        assert tool_result["payload"]["effect_id"] == identity.effect_id
    finally:
        repo.close()


def test_legacy_status_rows_stay_readable(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo)
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        # Simulate a pre-ticket-04 row.
        repo.control._conn.execute(
            "UPDATE effects SET status = 'authorized', effect_id = NULL, "
            "idempotency_key = NULL WHERE effect_key = ?",
            (identity.effect_id,),
        )
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == "authorized"
        assert row.phase == "pending"
        assert row.resolved_effect_id == identity.effect_id
        assert row.resolved_idempotency_key == identity.idempotency_key
    finally:
        repo.close()
