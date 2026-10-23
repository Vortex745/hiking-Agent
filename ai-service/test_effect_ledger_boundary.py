"""BCDEF ticket 04 — effect ledger claim/commit, fencing and replay.

Seam: the public effect-ledger API (:mod:`agent_v2.effects` +
``RunRepository``) and the orchestration driver :func:`drive_effect`.
All persistence is real SQLite under a temp directory; the external
executor is a counting fake. No network, no sandbox.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
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


def test_zero_token_effect_claim_cannot_commit_over_a_tokened_row(tmp_path):
    """BCDEF review B1: ``fencing_token = 0`` is a generation, not "unset".

    The effect CAS must reject a token mismatch instead of skipping the
    check, so a claim that carries no lease generation can never settle a
    row owned by a fenced lease generation.
    """
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, request_id="req-fence-commit")
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        row, won = repo.claim_effect(
            identity.effect_id,
            owner="lease-worker",
            claim_token="claim-1",
            ttl_seconds=60,
            lease_id="lease-1",
            fencing_token=1,
        )
        assert won and row is not None
        assert row.fencing_token == 1

        # A zero-generation claim (an unleased/stale worker) cannot commit
        # over the tokened row: the CAS matches nothing.
        stale = repo.commit_effect(
            identity.effect_id,
            '{"x": 1}',
            claim_token="claim-1",
            fencing_token=0,
        )
        assert stale is not None
        assert stale.phase == EffectStatus.CLAIMED.value

        # A different, non-zero generation is refused too.
        other = repo.commit_effect(
            identity.effect_id,
            '{"x": 1}',
            claim_token="claim-1",
            fencing_token=2,
        )
        assert other is not None and other.phase == EffectStatus.CLAIMED.value

        # Only the owning generation settles the ledger.
        settled = repo.commit_effect(
            identity.effect_id,
            '{"x": 1}',
            claim_token="claim-1",
            fencing_token=1,
        )
        assert settled is not None and settled.phase == "committed"
    finally:
        repo.close()


def test_lease_bound_effect_claim_requires_a_generation(tmp_path):
    """BCDEF review B1: a claim bound to a lease without a generation is
    a contradiction, so construction fails instead of defaulting to 0."""
    with pytest.raises(ValueError):
        EffectClaim(
            effect_id="e-1", claim_token="t-1", lease_id="lease-1", fencing_token=0
        )
    # An unleased claim keeps its (generation-less) shape.
    claim = EffectClaim(effect_id="e-1", claim_token="t-1")
    assert claim.fencing_token == 0


# ---- drive_effect publication gate -----------------------------------------


def _drive(
    repo: RunRepository,
    run_id: str,
    identity,
    *,
    executor: Any,
    emit_required: Any | None = None,
    emit_frame: Any | None = None,
) -> tuple[EffectDriveResult, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    def emit(event_type: str, payload: dict[str, Any]) -> Any:
        events.append((event_type, payload))
        return SimpleNamespace(type=event_type, payload=payload)

    outcome = EffectDriveResult()
    # Production requires a durable publisher (BCDEF review S6). This seam's
    # default emitter is the test's own recorder; failure-injection tests
    # override it explicitly.
    required = emit_required if emit_required is not None else emit
    frame = emit_frame if emit_frame is not None else emit

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
            emit=frame,
            outcome=outcome,
            emit_required=required,
        ):
            pass

    asyncio.run(run())
    return outcome, events


def _ok_executor(calls: list[int], result: dict[str, Any] | None = None):
    def execute() -> dict[str, Any]:
        calls.append(1)
        return result if result is not None else {"path": "p.pdf"}

    return execute


def _durable_emit(repo: RunRepository, run_id: str):
    """A ``emit_required`` stand-in that really persists the event."""
    sequence = {"n": 0}

    def emit(event_type: str, payload: dict[str, Any]) -> SimpleNamespace:
        sequence["n"] += 1
        repo.persist_event(
            run_id=run_id,
            sequence=sequence["n"],
            event_type=event_type,
            payload=payload,
            request_id=run_id,
            conversation_id="conv-effect",
        )
        return SimpleNamespace(type=event_type, payload=payload)

    return emit


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
        first_row = repo.get_effect(identity.effect_id)
        assert first_row is not None

        retry_calls: list[int] = []
        outcome2, _events2 = _drive(
            repo,
            run_id,
            identity,
            executor=_ok_executor(retry_calls),
            emit_required=_durable_emit(repo, run_id),
        )
        # AC6 (review S1): the committed result is reused, never re-executed.
        assert retry_calls == []
        assert outcome2.status == "recovered"
        assert outcome2.stored_result == {"path": "p.pdf"}
        # The ledger is the unique source of truth for the result.
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == "committed"
        assert row.result_json == '{"path":"p.pdf"}'
        # Reuse did not touch the ledger: one row, same identity, and the
        # original claim stamp is untouched (no new execution right was taken).
        assert len(repo.list_effects_for_run(run_id)) == 1
        assert row.effect_key == first_row.effect_key
        assert row.resolved_effect_id == first_row.resolved_effect_id
        assert row.resolved_idempotency_key == first_row.resolved_idempotency_key
        assert row.created_at == first_row.created_at
        assert row.claim_token == first_row.claim_token
        assert row.committed_at == first_row.committed_at
        # The retry replays the stored success through the required durable
        # publisher instead of executing again.
        stored = repo.get_events_since(run_id=run_id, from_seq=0)
        assert [event["type"] for event in stored] == ["tool_result"]
        published = stored[0]["payload"]
        assert published["status"] == "ok"
        assert published["path"] == "p.pdf"
        assert published["effect_id"] == identity.effect_id
    finally:
        repo.close()


# ---- durable ledger write failure (review S1) --------------------------------


def test_effect_row_write_failure_is_a_stable_persist_failure_not_in_flight(
    tmp_path, monkeypatch, caplog
):
    """A lost ``authorized`` row is a persist failure, and nothing executes.

    Spec AC1's durable write is what grants execution rights. When it fails
    there is no row to claim, so the old bare ``except`` walked into
    ``claim_effect`` on a missing row and reported ``effect_in_flight`` —
    a lie that also left the external side effect about to run.
    """
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-insert-failure")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )
        calls: list[int] = []

        def boom(**_kwargs: Any) -> None:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(repo, "insert_effect", boom)

        with caplog.at_level(
            logging.ERROR, logger="ai-service.run_orchestration"
        ):
            outcome, events = _drive(
                repo,
                run_id,
                identity,
                executor=_ok_executor(calls),
                emit_required=_durable_emit(repo, run_id),
            )

        assert calls == []  # the external effect never ran
        assert outcome.status == "persist_failed"
        errors = [payload for event_type, payload in events if event_type == "error"]
        assert errors, events
        assert errors[-1]["code"] == "effect_persist_failed"
        assert errors[-1]["boundary_code"] == "effect_persist_failed"
        assert "effect_in_flight" not in str(events)
        # No phantom row and no tool_call either.
        assert repo.get_effect(identity.effect_id) is None
        assert [event_type for event_type, _payload in events] == ["error"]
        # The run stops terminally.
        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"

        # The durable write failure is in the log WITH its traceback.
        records = [
            record
            for record in caplog.records
            if record.levelno >= logging.ERROR and record.exc_info
        ]
        assert records, caplog.text
    finally:
        repo.close()


def test_duplicate_effect_row_insert_still_falls_back_to_the_existing_row(
    tmp_path, monkeypatch, caplog
):
    """The one benign insert failure keeps its old (non-fatal) behaviour.

    A concurrent resume may have written the SAME row first; that is a
    duplicate-key conflict with a readable row behind it, not a persist
    failure, and the CAS claim decides the single executor.

    BCDEF ticket 03 evidence note: the concurrent row is really written by a
    SECOND, independent connection at the instant the driver performs its own
    ledger write, so the driver's first read still sees nothing and the
    duplicate-key ``IntegrityError`` is raised by SQLite inside the driver's
    own ``INSERT``. (An earlier revision of this case raised a synthetic
    IntegrityError while the row was already readable, so the read-after-write
    branch it documented was never reached.)
    """
    repo = _repo(tmp_path)
    concurrent = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        run_id = _make_run(repo, "req-insert-duplicate")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )
        assert repo.get_effect(identity.effect_id) is None
        assert _effect_row_count(repo) == 0
        real_insert = repo.insert_effect
        original_created_at: dict[str, str] = {}
        calls: list[int] = []

        def raced(**kwargs: Any) -> Any:
            # The concurrent resume commits the same identity right now...
            written = concurrent.insert_effect(**kwargs)
            original_created_at["value"] = written.created_at
            # ...so the driver's own real write is the one SQLite rejects.
            return real_insert(**kwargs)

        monkeypatch.setattr(repo, "insert_effect", raced)

        with caplog.at_level(logging.ERROR, logger="ai-service.run_orchestration"):
            outcome, events = _drive(
                repo,
                run_id,
                identity,
                executor=_ok_executor(calls),
                emit_required=_durable_emit(repo, run_id),
            )

        # The existing authorized row is claimed by this caller, exactly as
        # before the S1 change: a duplicate insert is not a persist failure.
        assert outcome.status == "executed"
        assert calls == [1]
        assert "effect_persist_failed" not in str(events)
        assert "effect_in_flight" not in str(events)

        # The conflict really happened at the driver's own ledger write.
        conflicts = [
            record
            for record in caplog.records
            if "effect ledger insert" in record.getMessage()
        ]
        assert conflicts, caplog.text
        assert isinstance(conflicts[-1].exc_info[1], sqlite3.IntegrityError)

        # AC4 (review S1): the ORIGINAL durable identity is reused, not
        # re-created. Same row, same identity columns, the concurrent
        # writer's creation stamp; only the claim CAS fields and the result
        # changed.
        rows = repo.list_effects_for_run(run_id)
        assert len(rows) == 1
        reused = rows[0]
        assert reused.effect_key == identity.effect_id
        assert reused.resolved_effect_id == identity.effect_id
        assert reused.resolved_idempotency_key == identity.idempotency_key
        assert reused.created_at == original_created_at["value"]
        assert reused.status == "committed"
        assert reused.result_json == '{"path":"p.pdf"}'
        # Execution rights came from the CAS claim, which stamped THIS
        # caller's token/owner on that same row (see
        # test_concurrent_claim_has_exactly_one_winner for the two-connection
        # race that proves at most one executor wins).
        assert reused.claim_token and reused.claim_owner
        # The concurrent connection agrees on the same durable fact.
        via_writer = concurrent.get_effect(identity.effect_id)
        assert via_writer is not None and via_writer.status == "committed"
    finally:
        concurrent.close()
        repo.close()


# ---- durable write failure injected at the REAL storage write (review S1) ---
#
# Review S1 asks for the failure to be produced where storage actually
# writes, so the repository write path (``RunRepository.insert_effect``,
# its transaction and its rollback) and :func:`drive_effect` both run
# unpatched. Both faults below are raised by SQLite itself from INSIDE the
# real ``INSERT INTO effects`` statement:
#
# * ``io_error`` — a storage object the write depends on is gone, so the
#   engine aborts with ``sqlite3.OperationalError`` and no row exists.
# * ``duplicate_key_without_row`` — the engine aborts with
#   ``sqlite3.IntegrityError`` on a duplicate-key conflict while NO row is
#   readable at the requested effect identity.
_STORAGE_FAULT_SQL = {
    "io_error": (
        "CREATE TRIGGER s1_effects_write_failure BEFORE INSERT ON effects BEGIN "
        "SELECT * FROM s1_missing_storage_object; END"
    ),
    "duplicate_key_without_row": (
        "CREATE TRIGGER s1_effects_write_failure BEFORE INSERT ON effects BEGIN "
        "SELECT RAISE(ABORT, 'UNIQUE constraint failed: effects.effect_key'); END"
    ),
}
# Which SQLite exception the real write actually raises.
_STORAGE_FAULT_ERRORS = {
    "io_error": sqlite3.OperationalError,
    "duplicate_key_without_row": sqlite3.IntegrityError,
}


def _install_effects_write_fault(conn, fault: str) -> None:
    conn.execute(_STORAGE_FAULT_SQL[fault])


def _remove_effects_write_fault(conn) -> None:
    conn.execute("DROP TRIGGER IF EXISTS s1_effects_write_failure")


def _effect_row_count(repo: RunRepository) -> int:
    return int(
        repo.control._conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0]
    )


def _durable_publisher(
    repo: RunRepository,
    run_id: str,
    sink: list[tuple[str, dict[str, Any]]],
    *,
    seq_start: int = 0,
) -> Any:
    """A frame emitter that REALLY persists every event and records it."""
    counter = {"n": seq_start}

    def emit(event_type: str, payload: dict[str, Any]) -> SimpleNamespace:
        counter["n"] += 1
        repo.persist_event(
            run_id=run_id,
            sequence=counter["n"],
            event_type=event_type,
            payload=payload,
            request_id=run_id,
            conversation_id="conv-effect",
        )
        sink.append((event_type, payload))
        return SimpleNamespace(type=event_type, payload=payload)

    return emit


@pytest.mark.parametrize("storage_fault", ["io_error", "duplicate_key_without_row"])
def test_real_effects_row_write_failure_stops_the_run_without_executing(
    tmp_path, caplog, storage_fault
):
    """AC1/AC2 (review S1): a lost durable authorization row is a persist
    failure — and the exception TYPE never grants execution rights.

    Both faults are raised by SQLite inside the real ledger write. Neither
    may be read as ``effect_in_flight`` or as a success: the run stops
    terminally, no ``tool_call`` frame exists, nothing is published, no
    effect row survives, and the external executor is never called.
    """
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, f"req-s1-{storage_fault}")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )
        _install_effects_write_fault(repo.control._conn, storage_fault)

        calls: list[int] = []
        published: list[tuple[str, dict[str, Any]]] = []
        publish = _durable_publisher(repo, run_id, published)
        with caplog.at_level(logging.ERROR, logger="ai-service.run_orchestration"):
            outcome, _frames = _drive(
                repo,
                run_id,
                identity,
                executor=_ok_executor(calls),
                emit_frame=publish,
                emit_required=publish,
            )

        # AC1 — the stable persist code, never the in-flight lie.
        assert outcome.status == "persist_failed"
        errors = [payload for event_type, payload in published if event_type == "error"]
        assert errors, published
        assert errors[-1]["code"] == "effect_persist_failed"
        assert errors[-1]["boundary_code"] == "effect_persist_failed"
        assert "effect_in_flight" not in str(published)

        # AC2 — the external effect never ran and never looked like it did.
        assert calls == []
        assert [event_type for event_type, _payload in published] == ["error"]
        success = [
            payload
            for event_type, payload in published
            if event_type == "tool_result" and payload.get("status") == "ok"
        ]
        assert success == []

        # Durable facts: no ledger row survives the failed authorization
        # write, and the only persisted frame is the stable error.
        assert repo.get_effect(identity.effect_id) is None
        assert repo.list_effects_for_run(run_id) == []
        assert _effect_row_count(repo) == 0
        stored = repo.get_events_since(run_id=run_id, from_seq=0)
        assert [event["type"] for event in stored] == ["error"]
        assert stored[0]["payload"]["code"] == "effect_persist_failed"
        assert [event for event in stored if event["type"] == "tool_call"] == []

        # The run stops terminally.
        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"

        # The fault really fired inside SQLite's own write, and the driver
        # logged it with its traceback.
        records = [
            record
            for record in caplog.records
            if "effect ledger insert" in record.getMessage()
        ]
        assert records, caplog.text
        assert isinstance(records[-1].exc_info[1], _STORAGE_FAULT_ERRORS[storage_fault])
    finally:
        _remove_effects_write_fault(repo.control._conn)
        repo.close()


def test_failed_effect_run_reads_back_through_snapshot_and_reconnect(tmp_path):
    """AC3 (review S1): the failed run's durable state IS the first stream.

    After a reconnect (independent repository over the same SQLite files)
    the snapshot and the replayed event log explain the same stopped run:
    ``error`` + ``effect_persist_failed``, no effect, no tool_call. A retry
    on the reconnected store executes nothing new.
    """
    first_stream: list[tuple[str, dict[str, Any]]] = []
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-s1-reconnect")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )
        _install_effects_write_fault(repo.control._conn, "io_error")
        calls: list[int] = []
        publish = _durable_publisher(repo, run_id, first_stream)
        outcome, _frames = _drive(
            repo,
            run_id,
            identity,
            executor=_ok_executor(calls),
            emit_frame=publish,
            emit_required=publish,
        )
        assert outcome.status == "persist_failed"
        assert calls == []
        assert [event_type for event_type, _payload in first_stream] == ["error"]
    finally:
        repo.close()

    # --- reconnect: brand new stores over the same durable files ---------
    reconnected = _repo(tmp_path)
    try:
        snapshot = reconnected.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "error"
        assert snapshot.error_code == "effect_persist_failed"
        assert snapshot.terminal_persist_failed is False
        # No phantom effect ever showed up in the replayable state.
        assert snapshot.effects == []

        replay = reconnected.get_events_since(run_id=run_id, from_seq=0)
        assert [(event["type"], event["payload"]) for event in replay] == first_stream
        assert [event["type"] for event in replay] == ["error"]

        # Retrying on the reconnected store: the same storage fault is still
        # there, so still nothing executes — and no new frame besides the
        # second stable error.
        second_calls: list[int] = []
        second_stream: list[tuple[str, dict[str, Any]]] = []
        republish = _durable_publisher(
            reconnected, run_id, second_stream, seq_start=len(replay)
        )
        outcome2, _frames2 = _drive(
            reconnected,
            run_id,
            identity,
            executor=_ok_executor(second_calls),
            emit_frame=republish,
            emit_required=republish,
        )
        assert second_calls == []
        assert outcome2.status == "persist_failed"
        assert [event_type for event_type, _payload in second_stream] == ["error"]
        assert second_stream[-1][1]["code"] == "effect_persist_failed"

        after = reconnected.get_events_since(run_id=run_id, from_seq=0)
        assert [event["type"] for event in after] == ["error", "error"]
        assert [event for event in after if event["type"] == "tool_call"] == []
        assert reconnected.get_effect(identity.effect_id) is None
        assert reconnected.list_effects_for_run(run_id) == []
        run = reconnected.get_run(run_id)
        assert run is not None and run.status == "error"
        snapshot2 = reconnected.get_snapshot(run_id)
        assert snapshot2 is not None and snapshot2.status == "error"
        assert snapshot2.effects == []
    finally:
        _remove_effects_write_fault(reconnected.control._conn)
        reconnected.close()


def test_duplicate_key_conflict_without_a_readable_original_row_fails_stably(
    tmp_path,
):
    """AC5 (review S1): a REAL duplicate-key conflict with no readable row.

    The shipped ``UNIQUE (run_id, tool_name, args_canonical)`` constraint
    rejects the ledger write because a different effect key already owns
    that logical tool call, so ``get_effect(effect_key)`` legitimately
    reads nothing back. SQLite raises the IntegrityError at the real write
    position; the driver must fail closed. An exception that LOOKS like a
    duplicate must not hand out execution rights over a row it cannot read.
    """
    repo = _repo(tmp_path)
    holder_key = "legacy-effect-key-for-the-same-tool-call"
    try:
        run_id = _make_run(repo, "req-s1-conflict")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )
        holder = repo.insert_effect(
            effect_key=holder_key,
            run_id=run_id,
            conversation_id="conv-effect",
            tool_name="generate_pdf",
            args_canonical='{"filename": "p.pdf"}',
            external_idempotency_key="legacy-external-key",
        )
        assert identity.effect_id != holder_key

        calls: list[int] = []
        outcome, events = _drive(
            repo,
            run_id,
            identity,
            executor=_ok_executor(calls),
            emit_required=_durable_emit(repo, run_id),
        )

        assert outcome.status == "persist_failed"
        errors = [payload for event_type, payload in events if event_type == "error"]
        assert errors, events
        assert errors[-1]["code"] == "effect_persist_failed"
        assert "effect_in_flight" not in str(events)
        assert calls == []
        assert [event_type for event_type, _payload in events] == ["error"]

        # No durable effect row for the requested identity and nothing was
        # published for this effect.
        assert repo.get_effect(identity.effect_id) is None
        assert repo.get_events_since(run_id=run_id, from_seq=0) == []

        # The unreadable holder row is NOT adopted: it stays authorized,
        # unclaimed and without a result — the conflict granted nothing.
        untouched = repo.get_effect(holder_key)
        assert untouched is not None
        assert untouched.status == holder.status == "authorized"
        assert untouched.claim_token is None
        assert untouched.result_json is None
        assert _effect_row_count(repo) == 1

        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"
    finally:
        repo.close()


def test_concurrent_resume_duplicate_key_conflict_reuses_the_durable_row(
    tmp_path, caplog
):
    """AC5's positive twin, at the real write position with two connections.

    A concurrent resume commits the authorization row for the SAME effect
    identity while holding the SQLite write lock. The driver under test
    therefore really raises a duplicate-key ``IntegrityError`` from its own
    ``INSERT`` and reads the original row back: the durable identity is
    reused and the CAS claim alone decides the executor. Cross-connection
    staging is the documented low-level exception for duplicate-key races.
    """
    shared_control = str(tmp_path / "control.sqlite")
    shared_checkpoints = str(tmp_path / "ckpt.sqlite")
    resumer = RunRepository(
        control=ControlStore(shared_control),
        checkpoints=CheckpointsStore(shared_checkpoints),
    )
    driver = RunRepository(
        control=ControlStore(shared_control),
        checkpoints=CheckpointsStore(shared_checkpoints),
    )
    try:
        run_id = _make_run(resumer, "req-s1-concurrent-duplicate")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
        )

        # The concurrent writer holds the write lock AND the (still
        # uncommitted) authorization row: the driver's first read cannot see
        # it, so the driver does reach its real ledger INSERT.
        writer = sqlite3.connect(shared_control, isolation_level=None)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA busy_timeout=5000")
        writer.execute("BEGIN IMMEDIATE")
        now = datetime.now(timezone.utc).isoformat()
        writer.execute(
            "INSERT INTO effects "
            "(effect_key, effect_id, idempotency_key, run_id, conversation_id, "
            " tool_name, args_canonical, status, external_idempotency_key, "
            " result_json, failure_reason, created_at, updated_at, committed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'authorized', ?, NULL, NULL, ?, ?, NULL)",
            (
                identity.effect_id,
                identity.effect_id,
                identity.idempotency_key,
                run_id,
                "conv-effect",
                "generate_pdf",
                '{"filename": "p.pdf"}',
                identity.idempotency_key,
                now,
                now,
            ),
        )

        calls: list[int] = []
        box: dict[str, Any] = {}

        def race() -> None:
            try:
                outcome, streamed = _drive(
                    driver,
                    run_id,
                    identity,
                    executor=_ok_executor(calls),
                    emit_required=_durable_emit(driver, run_id),
                )
                box["outcome"] = outcome
                box["events"] = streamed
            except BaseException as exc:  # pragma: no cover - surfaced below
                box["error"] = exc

        thread = threading.Thread(target=race)
        with caplog.at_level(logging.ERROR, logger="ai-service.run_orchestration"):
            thread.start()
            try:
                # Release the lock by committing the concurrent row; the
                # driver's own INSERT then conflicts with committed data.
                time.sleep(0.15)
                writer.execute("COMMIT")
            finally:
                thread.join(timeout=30)
        writer.close()
        assert "error" not in box, box["error"]
        assert not thread.is_alive()

        # The conflict really happened at the driver's own ledger write:
        # SQLite raised the duplicate-key IntegrityError inside the real
        # INSERT (not a stand-in), and the driver still reused the row.
        conflicts = [
            record
            for record in caplog.records
            if "effect ledger insert" in record.getMessage()
        ]
        assert conflicts, caplog.text
        assert isinstance(conflicts[-1].exc_info[1], sqlite3.IntegrityError)

        outcome = box["outcome"]
        events = box["events"]
        # The original durable row was readable, so this is NOT a persist
        # failure: the driver reuses it and the CAS gives it execution rights.
        assert outcome.status == "executed"
        assert calls == [1]
        assert "effect_persist_failed" not in str(events)
        assert "effect_in_flight" not in str(events)

        rows = resumer.list_effects_for_run(run_id)
        assert len(rows) == 1
        reused = rows[0]
        assert reused.effect_key == identity.effect_id
        assert reused.resolved_effect_id == identity.effect_id
        assert reused.resolved_idempotency_key == identity.idempotency_key
        assert reused.created_at == now  # the ORIGINAL row, not a re-insert
        assert reused.status == "committed"
        assert reused.result_json == '{"path":"p.pdf"}'
        assert reused.claim_token and reused.claim_owner
        # Both connections agree on the same durable fact.
        via_driver = driver.get_effect(identity.effect_id)
        assert via_driver is not None and via_driver.status == "committed"
    finally:
        driver.close()
        resumer.close()


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
