"""BCDEF ticket 12 — the seven boundary metrics are EMITTED, not only declared.

Spec 07 names seven metrics/alerts: ``admission_failed``,
``outcome_unknown``, ``terminal_persist_failed``, ``lease_expired``,
``cleanup_pending``, ``replay_gap`` and ``repair_age``. Before this ticket
the last three existed only as enum members (``cleanup_pending`` /
``replay_gap`` in
:class:`agent_v2.execution_contract.ExecutionErrorCode`, ``repair_age``
nowhere), so a dashboard reading them would have shown a flat zero and an
operator could not tell a quiet system from an un-instrumented one.

Seam: the durable boundary operations themselves
(:meth:`RunRepository.admit`, the effect-ledger settle, the finalize
fence, :class:`SandboxLeaseGuard`, the janitor enqueue, replay and the
snapshot) plus the two read surfaces that already expose health
(``GET /api/v2/chat/health`` and ``agent_v2.health.v2_health_status``).
Real SQLite under a temp directory; no model, network or sandbox.

Assertions are always "the counter moved because the condition was
injected, and nothing else moved" — a metric that fires on a happy path is
as useless as one that never fires.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.effects import EffectStatus, compute_effect_identity
from agent_v2.execution_contract import ExecutionErrorCode
from agent_v2.health import v2_health_status
from agent_v2.leases import SandboxLeaseGuard, SandboxLeaseLostError
from agent_v2.observability import (
    BOUNDARY_METRIC_NAMES,
    BoundaryMetrics,
    MigrationThresholdMonitor,
    get_boundary_metrics,
    reset_boundary_metrics,
)
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository, TerminalPersistFailure

SEVENTH_METRICS = (
    "admission_failed",
    "outcome_unknown",
    "terminal_persist_failed",
    "lease_expired",
    "cleanup_pending",
    "replay_gap",
    "repair_age",
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_boundary_metrics()
    yield
    reset_boundary_metrics()


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _counters() -> dict[str, Any]:
    return get_boundary_metrics().snapshot()


def _counter(name: str) -> int:
    return int(_counters()[name])


def _admit(repo: RunRepository, request_id: str) -> str:
    result = repo.admit(
        request_id=request_id,
        conversation_id=None,
        profile_id="chat",
        user_message="带什么装备？",
    )
    assert result.ok and result.execute, result
    repo.mark_run_running(result.run_id)
    return result.run_id


def _unknown_effect(repo: RunRepository, run_id: str, *, effect_key: str) -> str:
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
    )
    repo.insert_effect(
        effect_key=effect_key,
        run_id=run_id,
        conversation_id=f"conv-{run_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "p.pdf"}',
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    claimed, won = repo.claim_effect(
        effect_key, owner="worker-1", claim_token="tok-1", ttl_seconds=300.0
    )
    assert won and claimed is not None
    repo.mark_effect_outcome_unknown(effect_key, "crash after external call")
    return effect_key


# ---- registry surface --------------------------------------------------------


def test_the_seven_spec_metrics_are_named_and_start_at_zero():
    assert BOUNDARY_METRIC_NAMES == SEVENTH_METRICS
    snapshot = _counters()
    for name in SEVENTH_METRICS:
        assert snapshot[name] == 0, name
    assert snapshot["repair_age_max_seconds"] == 0.0
    assert set(snapshot["last_seen_at"]) == set()


def test_registry_counts_are_independent_per_metric():
    metrics = BoundaryMetrics()
    metrics.observe("replay_gap")
    metrics.observe("replay_gap")
    metrics.observe("cleanup_pending")

    snapshot = metrics.snapshot()
    assert snapshot["replay_gap"] == 2
    assert snapshot["cleanup_pending"] == 1
    assert snapshot["admission_failed"] == 0
    assert set(snapshot["last_seen_at"]) == {"replay_gap", "cleanup_pending"}


def test_unknown_metric_names_are_rejected_not_silently_counted():
    metrics = BoundaryMetrics()
    with pytest.raises(ValueError):
        metrics.observe("policy_incompatible")


# ---- emission points (each condition injected at its durable site) ---------


def test_admission_failed_is_counted_when_the_admission_fence_fails(
    tmp_path: Path, monkeypatch
):
    repo = _repo(tmp_path)
    try:
        def boom(**_kwargs: Any):
            raise sqlite3.OperationalError("run write failed")

        monkeypatch.setattr(repo.control, "admit_run", boom)

        result = repo.admit(
            request_id="req-m-admission",
            conversation_id=None,
            profile_id="chat",
            user_message="带什么装备？",
        )

        assert result.status == "admission_failed"
        assert _counter("admission_failed") == 1
        # Only this metric moved: no phantom side-effect counters.
        snapshot = _counters()
        assert snapshot["outcome_unknown"] == 0
        assert snapshot["terminal_persist_failed"] == 0
    finally:
        repo.close()


def test_successful_admission_records_no_boundary_breach(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        _admit(repo, "req-m-clean")
        assert all(_counters()[name] == 0 for name in SEVENTH_METRICS)
    finally:
        repo.close()


def test_outcome_unknown_is_counted_when_the_ledger_settles_unknown(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-unknown")
        effect_key = f"eff-m-{run_id}"

        _unknown_effect(repo, run_id, effect_key=effect_key)

        row = repo.get_effect(effect_key)
        assert row is not None and row.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert _counter("outcome_unknown") == 1
    finally:
        repo.close()


def test_terminal_persist_failed_is_counted_when_the_terminal_event_write_fails(
    tmp_path: Path, monkeypatch
):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-terminal")

        from agent_v2.persistence import TerminalEventPersistenceError

        def boom(**_kwargs: Any):
            raise TerminalEventPersistenceError("terminal event row lost")

        monkeypatch.setattr(repo.checkpoints, "finalize_terminal", boom)

        with pytest.raises(TerminalPersistFailure):
            repo.finalize_run(
                run_id=run_id,
                request_id="req-m-terminal",
                terminal_type="done",
                payload={"code": "done"},
                sequence=5,
                assistant_content="安全第一。",
            )

        assert _counter("terminal_persist_failed") == 1
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None and snapshot.terminal_persist_failed is True
    finally:
        repo.close()


def test_lease_expired_is_counted_when_a_stale_lease_is_refused(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-lease")
        lease, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws/m-lease",
        )
        assert acquired and lease is not None
        repo.control._conn.execute(
            "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )

        guard = SandboxLeaseGuard(
            repo, run_id=run_id, fencing_token=lease.fencing_token, owner_id="worker-1"
        )
        with pytest.raises(SandboxLeaseLostError) as refused:
            guard.assert_active()
        assert refused.value.code == SandboxLeaseLostError.LEASE_EXPIRED
        assert _counter("lease_expired") == 1

        with pytest.raises(SandboxLeaseLostError):
            guard.renew()
        assert _counter("lease_expired") == 2
        # The expiry observation must not be booked as a different metric.
        assert _counter("cleanup_pending") == 0
    finally:
        repo.close()


def test_lease_expired_is_counted_once_per_durable_expiry_transition(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-lease-expire")
        repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="worker-1",
            sandbox_type="local_native",
            workspace_path="/ws/m-expire",
        )
        repo.control._conn.execute(
            "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )

        repo.expire_sandbox_lease(run_id)
        assert _counter("lease_expired") == 1
        # Already expired: no new transition, no new observation.
        repo.expire_sandbox_lease(run_id)
        assert _counter("lease_expired") == 1
    finally:
        repo.close()


def test_cleanup_pending_is_counted_once_per_new_janitor_cleanup_item(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-cleanup")

        _task, created = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id=run_id,
            evidence={"run_id": run_id, "workspace_path": "/ws/m-cleanup"},
            dedupe_key=f"cleanup:{run_id}",
        )
        assert created is True
        assert _counter("cleanup_pending") == 1

        # The dedupe key means the retry reuses the SAME pending item.
        _again, created_again = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id=run_id,
            evidence={"run_id": run_id, "workspace_path": "/ws/m-cleanup"},
            dedupe_key=f"cleanup:{run_id}",
        )
        assert created_again is False
        assert _counter("cleanup_pending") == 1

        # A repair item is not a cleanup-pending observation.
        repo.enqueue_janitor_task(
            kind="repair",
            run_id=run_id,
            evidence={"run_id": run_id},
            dedupe_key=f"repair:{run_id}",
        )
        assert _counter("cleanup_pending") == 1
        assert _counter("outcome_unknown") == 0
    finally:
        repo.close()


def test_replay_gap_is_counted_for_every_gap_a_reconnect_observes(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-gap")

        def persist(sequence: int, event_type: str) -> None:
            repo.persist_event(
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                payload={},
                request_id="req-m-gap",
                conversation_id=f"conv-req-m-gap",
            )

        persist(2, "progress")
        # sequences 3 and 4 are the dropped progress frames.
        persist(5, "text")
        persist(6, "done")

        replayed = repo.get_replay_events(run_id=run_id, from_seq=0)
        assert [event["type"] for event in replayed].count("progress.gap") == 1
        assert _counter("replay_gap") == 1

        # A second reconnect re-observes the same hole: this is a read-side
        # observation counter, so it counts again (and never invents one).
        repo.get_replay_events(run_id=run_id, from_seq=0)
        assert _counter("replay_gap") == 2
    finally:
        repo.close()


def test_a_contiguous_replay_records_no_gap(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-nogap")
        for sequence in (2, 3):
            repo.persist_event(
                run_id=run_id,
                sequence=sequence,
                event_type="progress",
                payload={},
                request_id="req-m-nogap",
                conversation_id="conv-req-m-nogap",
            )

        events = repo.get_replay_events(run_id=run_id, from_seq=0)
        assert not any(event["type"] == "progress.gap" for event in events)
        assert _counter("replay_gap") == 0
    finally:
        repo.close()


def test_repair_age_is_observed_when_a_snapshot_reports_an_open_repair(
    tmp_path: Path,
):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-repair")
        effect_key = f"eff-repair-{run_id}"
        _unknown_effect(repo, run_id, effect_key=effect_key)
        two_hours_ago = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        repo.control._conn.execute(
            "UPDATE effects SET repair_updated_at = ?, updated_at = ? "
            "WHERE effect_key = ?",
            (two_hours_ago, two_hours_ago, effect_key),
        )

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        effect = snapshot.effects[0]
        assert effect["repair_next_action"] in (
            "manual_confirm",
            "confirm_known_result",
        )
        assert effect["repair_age_seconds"] >= 7000

        counters = _counters()
        assert counters["repair_age"] == 1
        assert counters["repair_age_max_seconds"] >= 7000
        assert counters["repair_age_last_seconds"] == pytest.approx(
            effect["repair_age_seconds"], abs=5
        )
    finally:
        repo.close()


def test_a_committed_effect_records_no_repair_age(tmp_path: Path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-repair-none")
        identity = compute_effect_identity(
            run_id=run_id, tool_name="generate_pdf", args={"filename": "ok.pdf"}
        )
        effect_key = identity.effect_id
        repo.insert_effect(
            effect_key=effect_key,
            run_id=run_id,
            conversation_id="conv-req-m-repair-none",
            tool_name="generate_pdf",
            args_canonical='{"filename": "ok.pdf"}',
            external_idempotency_key=identity.idempotency_key,
            effect_id=effect_key,
            idempotency_key=identity.idempotency_key,
        )
        repo.claim_effect(
            effect_key, owner="worker-1", claim_token="tok-1", ttl_seconds=300.0
        )
        repo.commit_effect(effect_key, '{"path": "ok.pdf"}')

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None and snapshot.effects[0]["status"] == "committed"
        assert _counter("repair_age") == 0
        assert _counters()["repair_age_max_seconds"] == 0.0
    finally:
        repo.close()


def test_metric_recording_failures_never_break_the_durable_boundary(
    tmp_path: Path, monkeypatch
):
    """Counters are observability: a raising registry must stay invisible."""
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, "req-m-resilient")

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("metrics backend down")

        monkeypatch.setattr(get_boundary_metrics(), "observe", boom)

        repo.persist_event(
            run_id=run_id,
            sequence=2,
            event_type="progress",
            payload={},
            request_id="req-m-resilient",
            conversation_id="conv-req-m-resilient",
        )
        repo.persist_event(
            run_id=run_id,
            sequence=5,
            event_type="text",
            payload={},
            request_id="req-m-resilient",
            conversation_id="conv-req-m-resilient",
        )

        events = repo.get_replay_events(run_id=run_id, from_seq=0)
        assert any(event["type"] == "progress.gap" for event in events)
    finally:
        repo.close()


# ---- read surfaces -----------------------------------------------------------


def _get(app: FastAPI, path: str) -> Any:
    async def run() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(run())


def test_chat_v2_health_exposes_the_seven_counters_additively():
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_public_router, prefix="/api/v2")

    get_boundary_metrics().observe("replay_gap")
    payload = _get(app, "/api/v2/chat/health").json()

    # Existing envelope keys are untouched (no path/shape change).
    assert payload["module"] == "chat_v2"
    metrics = payload["boundary_metrics"]
    for name in SEVENTH_METRICS:
        assert name in metrics, name
    assert metrics["replay_gap"] == 1
    assert set(metrics["alert_thresholds"]) == set(SEVENTH_METRICS)


def test_full_health_report_carries_the_same_counters():
    settings = SimpleNamespace(
        agent_v2_enabled=False, agent_v2_uniharness_pin="unknown"
    )
    get_boundary_metrics().observe("admission_failed")

    payload = v2_health_status(SimpleNamespace(agent_v2_runtime=None), settings)

    assert payload["boundary_metrics"]["admission_failed"] == 1
    # Nothing sustained yet, so no escalation.
    assert payload["boundary_alerts"] == []
    # The pre-existing health semantics are unchanged.
    assert payload["status"] == "disabled"


def test_sustained_boundary_breach_becomes_an_alert(tmp_path: Path):
    monitor = MigrationThresholdMonitor()
    first = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    metrics = {
        "cleanup_pending": 8,
        # ``repair_age`` is the age gauge in seconds, not an observation
        # count — the alerting input is built by ``alerting_metrics()``.
        "repair_age": 4000.0,
        "admission_failed": 1,
    }

    # A single sample is not an alert (the existing 15-minute rule).
    assert monitor.observe(metrics, now=first) == []

    evidence = monitor.observe(metrics, now=first + timedelta(minutes=16))
    breached = {item["metric"] for item in evidence}
    assert {"cleanup_pending", "repair_age"} <= breached
    # A count under its threshold never alerts.
    assert "admission_failed" not in breached

    # Recovery clears the pending alert state (no stuck escalation).
    cleared = monitor.observe(
        {"cleanup_pending": 0, "repair_age": 0.0},
        now=first + timedelta(minutes=20),
    )
    assert cleared == []


def test_boundary_alert_helper_uses_the_live_counters():
    metrics = get_boundary_metrics()
    for _ in range(int(metrics.alert_thresholds["cleanup_pending"]) + 1):
        metrics.observe("cleanup_pending")

    now = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
    assert metrics.alerts(now=now) == []
    breached = {
        item["metric"] for item in metrics.alerts(now=now + timedelta(minutes=16))
    }
    assert "cleanup_pending" in breached

    # The age gauge feeds the same helper: an open repair older than the
    # threshold is an alertable backlog, not just a number in a snapshot.
    aged = BoundaryMetrics()
    aged.observe_repair_age(4000.0, run_id="run-x")
    assert {item["metric"] for item in aged.alerts(now=now)} == set()
    assert "repair_age" in {
        item["metric"] for item in aged.alerts(now=now + timedelta(minutes=16))
    }


def test_the_metric_names_are_the_contract_error_codes_plus_the_repair_gauge():
    """Six of the seven are contract codes; ``repair_age`` is a gauge.

    Keeping one list prevents the metric names drifting away from the
    stable codes the client and the janitor already speak.
    """
    for name in SEVENTH_METRICS:
        if name == "repair_age":
            continue
        assert ExecutionErrorCode(name) is not None, name
    assert ExecutionErrorCode.REPAIR_PENDING.value == "repair_pending"
