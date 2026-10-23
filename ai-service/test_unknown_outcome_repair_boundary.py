"""BCDEF ticket 05 — unknown-outcome reconciliation and idempotent repair.

Seam: the effect ledger (:mod:`agent_v2.effects` +
``RunRepository``), the repair orchestration
(:mod:`agent_v2.effect_repair`) and the additive HTTP repair endpoint.
All persistence is real SQLite under a temp directory; the external
executor is a counting fake. No network, no model, no sandbox.

Behavior asserted:

* every failure mode after an external effect may have run lands in
  ``outcome_unknown`` (ledger write, serialization, event write,
  post-commit publication) and never publishes a success ``tool_result``;
* a retry never re-executes the external effect and never creates a
  second effect row;
* snapshot/replay reconstruct identity, known facts, the unknown part,
  repair status/attempts, last failure reason, ``repair_age`` and the
  next action;
* manual confirm/abandon repair reuse the original run/effect identity,
  are idempotent, publish exactly one replayable final result on success,
  and never call the model or re-run the external effect;
* a failed repair stays unknown/repair_pending;
* the janitor can claim a repair without executing the effect;
* the HTTP endpoint returns stable codes and enforces ownership.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent_v2.effect_repair import RepairAction, repair_effect
from agent_v2.effects import EffectStatus, compute_effect_identity
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


# The result the external executor returns; a repair may only ever publish
# THIS value because this is what the ledger durably records.
_KNOWN_RESULT = {"path": "p.pdf", "content": "已在沙箱中生成 p.pdf"}


def _make_run(repo: RunRepository, request_id: str = "req-repair") -> str:
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    return run_id


def _insert_effect(repo: RunRepository, run_id: str, args: dict[str, Any]):
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args=args
    )
    repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id="conv-repair",
        tool_name="generate_pdf",
        args_canonical='{"filename": "p.pdf"}',
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    return identity


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
            conversation_id="conv-repair",
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


def _unknown_effect(
    tmp_path, monkeypatch, request_id: str = "req-unknown"
) -> tuple[RunRepository, str, Any, list[int]]:
    """Drive one effect into ``outcome_unknown`` via a ledger write failure."""
    repo = _repo(tmp_path)
    run_id = _make_run(repo, request_id)
    identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
    calls: list[int] = []

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("ledger write failed")

    monkeypatch.setattr(repo, "commit_effect", boom)
    outcome, _events = _drive(repo, run_id, identity, executor=_ok_executor(calls))
    assert outcome.status == "outcome_unknown"
    assert calls == [1]
    row = repo.get_effect(identity.effect_id)
    assert row is not None and row.status == EffectStatus.OUTCOME_UNKNOWN.value
    return repo, run_id, identity, calls


def _unknown_effect_with_known_result(
    tmp_path, monkeypatch, request_id: str = "req-unknown-known"
) -> tuple[RunRepository, str, Any, list[int]]:
    """Drive one effect into ``outcome_unknown`` AFTER its result was durable.

    This is the publication-failure shape: the ledger committed the external
    result, only the replayable success event was lost. It is the ONLY shape
    a ``confirm`` decision can repair — see review S5.
    """
    repo = _repo(tmp_path)
    run_id = _make_run(repo, request_id)
    identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
    calls: list[int] = []

    def fail_result(event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "tool_result":
            raise sqlite3.OperationalError("success event not durable")

    outcome, _events = _drive(
        repo,
        run_id,
        identity,
        executor=_ok_executor(calls, _KNOWN_RESULT),
        emit_required=fail_result,
    )
    assert outcome.status == "outcome_unknown"
    row = repo.get_effect(identity.effect_id)
    assert row is not None and row.status == EffectStatus.OUTCOME_UNKNOWN.value
    assert json.loads(row.result_json or "{}") == _KNOWN_RESULT
    return repo, run_id, identity, calls


def _events(repo: RunRepository, run_id: str) -> list[dict[str, Any]]:
    return repo.get_events_since(run_id=run_id, from_seq=0)


def _success_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        evt["payload"]
        for evt in events
        if evt["type"] == "tool_result" and evt["payload"].get("status") == "ok"
    ]


# ---- every post-effect failure enters outcome_unknown -----------------------


@pytest.mark.parametrize(
    "failure_kind",
    ["ledger_write", "serialization", "effect_event_write", "publication"],
)
def test_all_post_effect_failures_land_in_outcome_unknown(
    tmp_path, monkeypatch, failure_kind
):
    repo = _repo(tmp_path)
    run_id = _make_run(repo, f"req-fail-{failure_kind}")
    identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
    calls: list[int] = []
    result: dict[str, Any] = {"path": "p.pdf"}

    if failure_kind == "serialization":
        result = {"path": object()}
    if failure_kind == "ledger_write":
        def boom(*_a, **_k):
            raise sqlite3.OperationalError("ledger write failed")

        monkeypatch.setattr(repo, "commit_effect", boom)

    def strict_fail(event_type: str, payload: dict[str, Any]) -> None:
        raise sqlite3.OperationalError("event write failed")

    def strict_fail_on_result(event_type: str, payload: dict[str, Any]) -> None:
        # tool_call persists; the post-commit success event does not.
        if event_type == "tool_result":
            raise sqlite3.OperationalError("success event not durable")

    strict = None
    if failure_kind == "effect_event_write":
        strict = strict_fail
    elif failure_kind == "publication":
        strict = strict_fail_on_result

    outcome, events = _drive(
        repo,
        run_id,
        identity,
        executor=_ok_executor(calls, result),
        emit_required=strict,
    )

    types = [event_type for event_type, _payload in events]
    assert calls == [1]  # the external side effect ran exactly once
    assert outcome.status == "outcome_unknown"
    assert [
        payload
        for event_type, payload in events
        if event_type == "tool_result" and payload.get("status") == "ok"
    ] == []
    assert "effect.outcome_unknown" in types
    assert types[-1] == "error"

    row = repo.get_effect(identity.effect_id)
    assert row is not None
    assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
    assert row.repair_status == "repair_pending"
    # The publication failure keeps the known external result as evidence.
    if failure_kind == "publication":
        assert row.result_json == '{"path":"p.pdf"}'

    # A retry never re-executes and never creates a second effect row.
    retry_calls: list[int] = []
    outcome2, events2 = _drive(
        repo, run_id, identity, executor=_ok_executor(retry_calls)
    )
    assert retry_calls == []
    assert outcome2.status == "outcome_unknown"
    assert _success_results(_events(repo, run_id)) == []
    assert len(repo.list_effects_for_run(run_id)) == 1
    assert _success_results(
        [
            {"type": event_type, "payload": payload}
            for event_type, payload in events2
        ]
    ) == []
    repo.close()


# ---- snapshot / replay reconstruction --------------------------------------


def test_snapshot_reconstructs_repair_identity_facts_and_age(
    tmp_path, monkeypatch
):
    repo, run_id, identity, _calls = _unknown_effect(
        tmp_path, monkeypatch, "req-repair-snap"
    )
    try:
        # Backdate the repair clock so repair_age is measurable.
        old = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        repo.control._conn.execute(
            "UPDATE effects SET repair_updated_at = ?, updated_at = ? "
            "WHERE effect_key = ?",
            (old, old, identity.effect_id),
        )

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert len(snapshot.effects) == 1
        effect = snapshot.effects[0]

        # Identity + known facts are reconstructed.
        assert effect["effect_id"] == identity.effect_id
        assert effect["idempotency_key"] == identity.idempotency_key
        assert effect["status"] == "outcome_unknown"
        assert effect["known_facts"]["tool"] == "generate_pdf"
        assert effect["known_facts"]["args_canonical"] == '{"filename": "p.pdf"}'
        assert effect["known_facts"]["external_result_known"] is False
        # The unknown part is explicit.
        assert effect["unknown_part"]["result_known"] is False
        assert effect["unknown_part"]["requires_human"] is True
        # Repair state, attempts, last failure reason, age and next action.
        assert effect["repair_status"] == "repair_pending"
        assert effect["repair_attempts"] == 0
        assert effect["repair_last_error"]
        assert effect["repair_last_error"] == effect["failure_reason"]
        assert effect["repair_age_seconds"] >= 100
        assert effect["repair_next_action"] == "manual_confirm"
    finally:
        repo.close()


def test_publication_failure_snapshot_offers_known_result_repair(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-pub-snap")
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})

        def fail_result(event_type: str, payload: dict[str, Any]) -> None:
            if event_type == "tool_result":
                raise sqlite3.OperationalError("event write failed")

        outcome, _events_list = _drive(
            repo,
            run_id,
            identity,
            executor=_ok_executor([], {"path": "p.pdf"}),
            emit_required=fail_result,
        )
        assert outcome.status == "outcome_unknown"

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        effect = snapshot.effects[0]
        assert effect["repair_status"] == "repair_pending"
        assert effect["known_facts"]["external_result_known"] is True
        assert effect["unknown_part"]["publication_failed"] is True
        assert effect["repair_next_action"] == "confirm_known_result"
    finally:
        repo.close()


# ---- manual confirm repair --------------------------------------------------


def test_confirm_repair_commits_one_replayable_result_without_reexecution(
    tmp_path, monkeypatch
):
    repo, run_id, identity, calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-confirm"
    )
    try:
        result = {"path": "p.pdf", "content": "已在沙箱中生成 p.pdf"}
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=result,
            idempotency_key="rep-1",
            actor="operator:alice",
        )
        assert outcome.status == "repaired"
        assert outcome.payload["effect_id"] == identity.effect_id
        assert outcome.payload["run_id"] == run_id

        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.COMMITTED.value
        assert json.loads(row.result_json or "{}") == result
        assert row.repair_status == "repaired"
        assert row.repair_attempts >= 1
        assert calls == [1]  # no external re-execution

        stored = _events(repo, run_id)
        successes = _success_results(stored)
        assert len(successes) == 1
        assert successes[0]["effect_id"] == identity.effect_id
        assert successes[0]["result"] == result
        repaired_events = [
            evt
            for evt in stored
            if evt["type"] == "progress"
            and evt["payload"].get("stage") == "effect.repaired"
        ]
        assert len(repaired_events) == 1
        assert repaired_events[0]["payload"]["effect_id"] == identity.effect_id
        assert repaired_events[0]["payload"]["action"] == "confirm"

        # Retry (same or different key) is idempotent: no extra events,
        # no second committed result, no re-execution.
        retry = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=result,
            idempotency_key="rep-1",
            actor="operator:alice",
        )
        assert retry.status == "idempotent"
        retry_other = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=result,
            idempotency_key="rep-2",
            actor="operator:bob",
        )
        assert retry_other.status == "idempotent"
        assert len(_success_results(_events(repo, run_id))) == 1
        assert calls == [1]
    finally:
        repo.close()


def test_abandon_repair_fails_closed_and_is_idempotent(tmp_path, monkeypatch):
    repo, run_id, identity, calls = _unknown_effect(
        tmp_path, monkeypatch, "req-abandon"
    )
    try:
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="abandon",
            idempotency_key="ab-1",
            actor="operator:alice",
            reason="external system cannot confirm",
        )
        assert outcome.status == "abandoned"
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert row.repair_status == "abandoned"
        assert _success_results(_events(repo, run_id)) == []
        abandoned_events = [
            evt
            for evt in _events(repo, run_id)
            if evt["type"] == "progress"
            and evt["payload"].get("stage") == "effect.repair_abandoned"
        ]
        assert len(abandoned_events) == 1
        assert abandoned_events[0]["payload"]["reason"] == (
            "external system cannot confirm"
        )

        retry = repair_effect(
            repo,
            identity.effect_id,
            action="abandon",
            idempotency_key="ab-1",
            actor="operator:alice",
        )
        assert retry.status == "idempotent"
        conflict = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result={"path": "p.pdf"},
            idempotency_key="ab-2",
            actor="operator:alice",
        )
        assert conflict.status == "conflict"
        assert conflict.payload["code"] == "repair_conflict"
        assert calls == [1]
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.effects[0]["repair_next_action"] == "none"
    finally:
        repo.close()


def test_failed_repair_stays_unknown_and_retry_publishes_once(
    tmp_path, monkeypatch
):
    repo, run_id, identity, _calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-repair-fail"
    )
    try:
        real_persist = repo.persist_event
        failures = {"left": 1}

        def flaky_persist(**kwargs: Any) -> None:
            if failures["left"] > 0:
                failures["left"] -= 1
                raise sqlite3.OperationalError("checkpoints unavailable")
            real_persist(**kwargs)

        monkeypatch.setattr(repo, "persist_event", flaky_persist)

        first = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=_KNOWN_RESULT,
            idempotency_key="rep-fail",
            actor="operator:alice",
        )
        assert first.status == "failed"
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert row.repair_status == "repair_pending"
        assert "checkpoints unavailable" in (row.repair_last_error or "")
        assert _success_results(_events(repo, run_id)) == []

        retry = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=_KNOWN_RESULT,
            idempotency_key="rep-fail",
            actor="operator:alice",
        )
        assert retry.status == "repaired"
        assert len(_success_results(_events(repo, run_id))) == 1
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.repair_status == "repaired"
    finally:
        repo.close()


def test_repair_rejects_effects_that_do_not_need_it(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _make_run(repo, "req-repair-notneeded")
        identity = _insert_effect(repo, run_id, {"filename": "p.pdf"})
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result={"path": "p.pdf"},
            idempotency_key="rep-x",
        )
        assert outcome.status == "not_repairable"
        assert outcome.payload["code"] == "repair_not_needed"

        missing = repair_effect(
            repo,
            "missing-effect",
            action="confirm",
            result={"path": "p.pdf"},
            idempotency_key="rep-y",
        )
        assert missing.status == "not_found"
        assert missing.payload["code"] == "effect_not_found"
    finally:
        repo.close()


# ---- janitor takeover -------------------------------------------------------


def test_janitor_claim_repair_preserves_identity_and_does_not_execute(
    tmp_path, monkeypatch
):
    repo, run_id, identity, calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-janitor-claim"
    )
    try:
        row, won = repo.claim_effect_repair(
            identity.effect_id, owner="janitor:task-1"
        )
        assert won is True
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert row.repair_status == "repair_claimed"
        assert row.repair_claimed_by == "janitor:task-1"
        assert row.repair_attempts == 1
        assert calls == [1]

        _row2, won2 = repo.claim_effect_repair(
            identity.effect_id, owner="janitor:task-2"
        )
        assert won2 is False

        # An operator confirmation may complete a janitor-claimed repair,
        # still without re-executing the external effect.
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=_KNOWN_RESULT,
            idempotency_key="rep-janitor",
            actor="operator:alice",
        )
        assert outcome.status == "repaired"
        assert calls == [1]
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.effects[0]["repair_status"] == "repaired"
    finally:
        repo.close()


# ---- additive HTTP endpoint -------------------------------------------------


class _FakeMemory:
    def __init__(self, owns: bool = True) -> None:
        self._owns = owns

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return self._owns


def _build_app(repo: RunRepository, *, owns: bool = True) -> FastAPI:
    import api.chat_v2 as chat_v2_module

    def fake_subject(request: Any) -> Any:
        request.state.subject_id = "sub-1"
        return _FakeMemory(owns), SimpleNamespace(
            issued=False, token="tok", subject_id="sub-1"
        )

    chat_v2_module.resolve_request_subject = fake_subject  # type: ignore[assignment]
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(
        runs=repo, memory=_FakeMemory(owns)
    )
    return app


def _post(app: FastAPI, path: str, body: dict[str, Any]) -> Any:
    async def call() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, json=body)

    return asyncio.run(call())


def _get(app: FastAPI, path: str) -> Any:
    async def call() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(call())


def test_http_repair_endpoint_confirm_and_stable_codes(tmp_path, monkeypatch):
    repo, run_id, identity, _calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-http-repair"
    )
    try:
        app = _build_app(repo)
        first = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {
                "action": "confirm",
                "result": _KNOWN_RESULT,
                "idempotency_key": "rep-http-1",
            },
        )
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["status"] == "repaired"
        assert body["effect_id"] == identity.effect_id
        assert body["run_id"] == run_id
        assert body["repair_status"] == "repaired"
        assert body["result"] == _KNOWN_RESULT

        # Retry is idempotent.
        retry = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {
                "action": "confirm",
                "result": _KNOWN_RESULT,
                "idempotency_key": "rep-http-1",
            },
        )
        assert retry.status_code == 200
        assert retry.json()["status"] == "idempotent"

        # Conflicting decision after repair → stable 409 code.
        conflict = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {"action": "abandon", "idempotency_key": "rep-http-2"},
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "repair_conflict"

        # Unknown effect → 404 stable code.
        missing = _post(
            app,
            "/api/v2/chat/effects/does-not-exist/repair",
            {
                "action": "confirm",
                "result": _KNOWN_RESULT,
                "idempotency_key": "rep-http-3",
            },
        )
        assert missing.status_code == 404
        assert missing.json()["code"] == "effect_not_found"

        # Invalid action → 422.
        invalid = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {"action": "retry", "idempotency_key": "rep-http-4"},
        )
        assert invalid.status_code == 422
    finally:
        repo.close()


def test_http_repair_endpoint_enforces_conversation_ownership(
    tmp_path, monkeypatch
):
    repo, _run_id, identity, _calls = _unknown_effect(
        tmp_path, monkeypatch, "req-http-owner"
    )
    try:
        app = _build_app(repo, owns=False)
        response = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {
                "action": "confirm",
                "result": {"path": "p.pdf"},
                "idempotency_key": "rep-owner-1",
            },
        )
        assert response.status_code == 404
        # The repair did not change the ledger.
        row = repo.get_effect(identity.effect_id)
        assert row is not None and row.status == EffectStatus.OUTCOME_UNKNOWN.value
    finally:
        repo.close()


def test_http_snapshot_exposes_repair_fields(tmp_path, monkeypatch):
    repo, run_id, identity, _calls = _unknown_effect(
        tmp_path, monkeypatch, "req-http-snap"
    )
    try:
        app = _build_app(repo)
        response = _get(app, f"/api/v2/chat/snapshot/{run_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["effects"]) == 1
        effect = body["effects"][0]
        assert effect["effect_id"] == identity.effect_id
        assert effect["repair_status"] == "repair_pending"
        assert effect["repair_attempts"] == 0
        assert effect["repair_next_action"] == "manual_confirm"
        assert "known_facts" in effect and "unknown_part" in effect
        assert effect["repair_age_seconds"] >= 0
    finally:
        repo.close()


def test_repair_action_enum_is_stable():
    assert RepairAction.CONFIRM.value == "confirm"
    assert RepairAction.ABANDON.value == "abandon"


# ---- review S5: confirm is bound to the durably recorded result -----------


def test_confirm_cannot_forge_a_result_the_ledger_never_recorded(
    tmp_path, monkeypatch
):
    """A lost ledger write leaves NOTHING to confirm.

    The repair endpoint is guarded only by conversation ownership, so a
    plain subject must not be able to hand the ledger a success result it
    never recorded. ``confirm`` therefore has no authority over the result
    it publishes: it republishes the durable row or it refuses.
    """
    repo, run_id, identity, _calls = _unknown_effect(
        tmp_path, monkeypatch, "req-forge"
    )
    try:
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result={"path": "victim.pdf", "content": "伪造的成功结果"},
            idempotency_key="forge-1",
            actor="operator",
        )
        assert outcome.status == "unverified_result"
        assert outcome.payload["code"] == "repair_result_not_durable"

        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert row.repair_status in ("repair_pending", "repair_claimed")
        assert not row.result_json
        # No authoritative replayable success result reached the event log.
        assert _success_results(_events(repo, run_id)) == []

        # The fail-closed decision is still available for the same row.
        abandoned = repair_effect(
            repo,
            identity.effect_id,
            action="abandon",
            idempotency_key="forge-2",
            actor="operator",
            reason="external system cannot confirm",
        )
        assert abandoned.status == "abandoned"
        assert _success_results(_events(repo, run_id)) == []
    finally:
        repo.close()


def test_confirm_publishes_the_stored_result_not_the_supplied_one(
    tmp_path, monkeypatch
):
    repo, run_id, identity, calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-stored-result"
    )
    try:
        # A payload that disagrees with the ledger is a forgery attempt.
        forged = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result={"path": "evil.pdf"},
            idempotency_key="stored-1",
            actor="operator",
        )
        assert forged.status == "unverified_result"
        assert forged.payload["code"] == "repair_result_not_durable"
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert json.loads(row.result_json or "{}") == _KNOWN_RESULT
        assert _success_results(_events(repo, run_id)) == []

        # The legitimate decision: no payload at all.
        outcome = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            idempotency_key="stored-2",
            actor="operator",
        )
        assert outcome.status == "repaired", outcome.payload
        assert outcome.payload["result"] == _KNOWN_RESULT

        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.COMMITTED.value
        assert json.loads(row.result_json or "{}") == _KNOWN_RESULT

        successes = _success_results(_events(repo, run_id))
        assert len(successes) == 1
        assert successes[0]["result"] == _KNOWN_RESULT
        assert "evil.pdf" not in json.dumps(_events(repo, run_id))
        assert calls == [1]  # the external effect never ran again

        # A payload that MATCHES the durable row stays accepted (retries
        # written by older clients keep working, idempotently).
        retry = repair_effect(
            repo,
            identity.effect_id,
            action="confirm",
            result=dict(_KNOWN_RESULT),
            idempotency_key="stored-3",
            actor="operator",
        )
        assert retry.status == "idempotent"
        assert len(_success_results(_events(repo, run_id))) == 1
    finally:
        repo.close()


def test_http_repair_endpoint_rejects_a_result_that_was_never_recorded(
    tmp_path, monkeypatch
):
    repo, run_id, identity, _calls = _unknown_effect_with_known_result(
        tmp_path, monkeypatch, "req-http-stored"
    )
    try:
        app = _build_app(repo)
        forged = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {
                "action": "confirm",
                "result": {"path": "attacker.pdf"},
                "idempotency_key": "http-forge-1",
            },
        )
        assert forged.status_code == 409, forged.text
        assert forged.json()["code"] == "repair_result_not_durable"
        row = repo.get_effect(identity.effect_id)
        assert row is not None
        assert row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert _success_results(_events(repo, run_id)) == []

        accepted = _post(
            app,
            f"/api/v2/chat/effects/{identity.effect_id}/repair",
            {"action": "confirm", "idempotency_key": "http-forge-2"},
        )
        assert accepted.status_code == 200, accepted.text
        body = accepted.json()
        assert body["status"] == "repaired"
        assert body["result"] == _KNOWN_RESULT
        snapshot = _get(app, f"/api/v2/chat/snapshot/{run_id}")
        assert snapshot.json()["effects"][0]["status"] == "committed"
    finally:
        repo.close()
