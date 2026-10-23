"""BCDEF ticket 12 — measured ceilings for the new durable writes.

These are the costs the spec asks to bound: the extra admission writes,
snapshot assembly and event replay at the endpoint's 1000-event page size,
the sandbox lease acquire AND its heartbeat (renew) plus the read-side
fence every sandbox command pays, and the janitor enqueue + claim pair that
cleanup failures and unknown outcomes land in. Thresholds are set from
measured medians on this machine with generous headroom; they exist to
catch an order-of-magnitude regression, not to micro-tune.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone

from agent_v2.effects import compute_effect_key
from agent_v2.execution_contract import policy_hash
from agent_v2.execution_policy import build_execution_policy
from agent_v2.leases import SANDBOX_LEASE_TTL_SECONDS, SandboxLeaseGuard
from agent_v2.persistence import CheckpointsStore, ControlStore, PolicySnapshotWrite
from agent_v2.profiles import resolve_profile
from agent_v2.runs import RunRepository

ITERATIONS = 25
REPLAY_EVENTS = 1000


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _settings():
    from types import SimpleNamespace

    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="disabled",
    )


def _time(fn) -> tuple[list[float], object]:
    samples: list[float] = []
    result = None
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return samples, result


def _median(samples: list[float]) -> float:
    return statistics.median(samples)


def _worst(samples: list[float]) -> float:
    return max(samples)


def test_admission_snapshot_and_replay_stay_within_measured_budget(tmp_path):
    repo = _repo(tmp_path)
    settings = _settings()
    policy = build_execution_policy(resolve_profile("chat"), settings)
    write = PolicySnapshotWrite(
        policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
        policy_hash=policy_hash(policy),
        schema_version=policy["schema_version"],
        source_version=policy["source_version"],
        compatibility_json=json.dumps(policy["compatibility"]),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    def admit_one() -> str:
        index = len(admitted)
        run_id = repo.admit(
            request_id=f"perf-req-{index}",
            conversation_id=None,
            profile_id="chat",
            user_message="perf",
            policy=policy,
        )
        admitted.append(run_id.run_id)
        return run_id.run_id

    admitted: list[str] = []
    try:
        admit_samples, first_run = _time(admit_one)
        assert admit_samples, "no admission samples"

        snapshot_samples, _ = _time(lambda: repo.get_snapshot(admitted[0]))

        # Replay is measured against a run that actually holds a full page
        # of events, which is the endpoint's own limit.
        for sequence in range(2, REPLAY_EVENTS + 1):
            repo.persist_event(
                run_id=first_run,
                sequence=sequence,
                event_type="progress",
                payload={"message": "x" * 40},
                request_id="perf-req-0",
                conversation_id="conv-perf-req-0",
                event_id=f"evt-{sequence}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        replay_samples, replayed = _time(
            lambda: repo.get_replay_events(run_id=first_run, from_seq=0)
        )
        assert len(replayed) >= REPLAY_EVENTS - 1

        lease_samples, _ = _time(
            lambda: repo.acquire_sandbox_lease(
                run_id=admitted[-1],
                owner_id="perf-owner",
                sandbox_type="local_native",
                workspace_path=f"/ws/perf-{len(admitted)}",
            )
        )
        effect_key_samples, _ = _time(
            lambda: compute_effect_key(
                run_id=first_run, tool_name="sandbox_bash", args={"command": "echo hi"}
            )
        )
    finally:
        repo.control.close()
        repo.checkpoints.close()

    measured = {
        "admission_ms": (_median(admit_samples), _worst(admit_samples)),
        "snapshot_ms": (_median(snapshot_samples), _worst(snapshot_samples)),
        "replay_1000_events_ms": (_median(replay_samples), _worst(replay_samples)),
        "lease_acquire_ms": (_median(lease_samples), _worst(lease_samples)),
        "effect_key_ms": (_median(effect_key_samples), _worst(effect_key_samples)),
    }
    print(f"BCDEF performance measured (median, worst) ms: {measured}")

    # Budgets: 5x the measured median, floored at whole-millisecond values
    # so a slow disk does not make the gate flaky.
    assert _median(admit_samples) < 60.0, measured
    assert _worst(admit_samples) < 250.0, measured
    assert _median(snapshot_samples) < 60.0, measured
    assert _median(replay_samples) < 250.0, measured
    assert _worst(replay_samples) < 800.0, measured
    assert _median(lease_samples) < 60.0, measured
    assert _median(effect_key_samples) < 5.0, measured
    assert write.policy_hash


def test_replay_gap_detection_is_constant_time(tmp_path):
    """Gap detection must not scale with the run's event count."""
    from agent_v2.execution_contract import compute_gap

    samples, _ = _time(lambda: compute_gap("run-perf", 1, 900))
    assert _median(samples) < 1.0, _median(samples)


def test_lease_heartbeat_and_janitor_work_stay_within_measured_budget(tmp_path):
    """Lease heartbeat + janitor enqueue/claim ceilings (ticket 12).

    Every sandbox command pays the read-side fence and, on the cowork and
    artifact paths, a renewal; every cleanup failure or unknown outcome
    pays an enqueue and later a claim. Both are on the request path or the
    operator's recovery path, so an unbounded wait here is a production
    incident — this measures what one call actually costs.
    """
    repo = _repo(tmp_path)
    try:
        run_id = repo.admit(
            request_id="perf-lease-1",
            conversation_id=None,
            profile_id="artifact",
            user_message="perf",
        ).run_id
        repo.mark_run_running(run_id)
        lease, acquired = repo.acquire_sandbox_lease(
            run_id=run_id,
            owner_id="perf-owner",
            sandbox_type="local_native",
            workspace_path="/ws/perf-lease",
            ttl_seconds=SANDBOX_LEASE_TTL_SECONDS,
        )
        assert acquired and lease is not None
        guard = SandboxLeaseGuard(
            repo,
            run_id=run_id,
            fencing_token=lease.fencing_token,
            lease_id=lease.lease_id,
            owner_id="perf-owner",
        )

        renew_samples, _ = _time(lambda: guard.renew())
        fence_samples, _ = _time(lambda: guard.assert_active())

        # A frozen lease must be refused promptly, not waited on: the guard
        # is the fence that keeps a stalled worker from executing.
        repo.control._conn.execute(
            "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )
        from agent_v2.leases import SandboxLeaseLostError

        refusal_started = time.perf_counter()
        try:
            guard.assert_active()
            raise AssertionError("expired lease must not pass the fence")
        except SandboxLeaseLostError:
            pass
        refusal_ms = (time.perf_counter() - refusal_started) * 1000

        # Janitor: one cleanup item per failed cleanup, then one claim each.
        def enqueue_one() -> None:
            index = len(enqueued)
            task, created = repo.enqueue_janitor_task(
                kind="cleanup",
                run_id=run_id,
                evidence={"run_id": run_id, "workspace_path": f"/ws/perf-{index}"},
                dedupe_key=f"cleanup:perf-{index}",
            )
            assert created is True
            enqueued.append(task.task_id)

        enqueued: list[str] = []
        enqueue_samples, _ = _time(enqueue_one)

        # Claims are measured against a standing backlog that mixes the
        # work a recovering sweeper actually meets: fresh items, taken-over
        # EXPIRED claims and unexpired claims it must refuse (BCDEF review
        # S4). 10 items is already far above what one run produces, and the
        # expiry-aware claim + the candidate scan stay inside the same
        # ceilings — no separate, looser budget for the recovery path.
        backlog_repo = RunRepository(
            control=ControlStore(str(tmp_path / "janitor-control.sqlite")),
            checkpoints=CheckpointsStore(str(tmp_path / "janitor-ckpt.sqlite")),
        )
        try:
            task_ids = [
                backlog_repo.enqueue_janitor_task(
                    kind="repair",
                    run_id="run-perf-claim",
                    evidence={"run_id": "run-perf-claim", "n": index},
                    dedupe_key=f"repair:perf-{index}",
                )[0].task_id
                for index in range(10)
            ]
            for task_id in task_ids[:5]:
                backlog_repo.claim_janitor_task(task_id, owner="janitor-live")
            # 4 of the 5 claims are already expired (takeover path); the
            # fifth stays live, so its claim must be refused.
            backlog_repo.control._conn.execute(
                "UPDATE janitor_tasks SET claim_expires_at = ? "
                "WHERE claimed_by = 'janitor-live' "
                "AND task_id IN (?, ?, ?, ?)",
                (
                    "2000-01-01T00:00:00+00:00",
                    *task_ids[:4],
                ),
            )

            cursor = {"index": 0}

            def claim_once() -> None:
                task_id = task_ids[cursor["index"] % len(task_ids)]
                cursor["index"] += 1
                backlog_repo.claim_janitor_task(
                    task_id, owner=f"janitor-{cursor['index']}"
                )

            claim_samples, _ = _time(claim_once)

            def claimable_scan() -> None:
                backlog_repo.list_janitor_tasks(claimable=True, limit=10)

            scan_samples, _ = _time(claimable_scan)

            def enqueue_duplicate() -> None:
                """The dedupe hit path: a retry re-observes the open item."""
                backlog_repo.enqueue_janitor_task(
                    kind="cleanup",
                    run_id="run-perf-dedupe",
                    evidence={"run_id": "run-perf-dedupe"},
                    dedupe_key="cleanup:run-perf-dedupe",
                )

            duplicate_samples, _ = _time(enqueue_duplicate)
        finally:
            backlog_repo.control.close()
            backlog_repo.checkpoints.close()
    finally:
        repo.control.close()
        repo.checkpoints.close()

    measured = {
        "lease_renew_ms": (_median(renew_samples), _worst(renew_samples)),
        "lease_fence_ms": (_median(fence_samples), _worst(fence_samples)),
        "lease_expired_refusal_ms": round(refusal_ms, 4),
        "janitor_enqueue_ms": (_median(enqueue_samples), _worst(enqueue_samples)),
        "janitor_claim_ms": (_median(claim_samples), _worst(claim_samples)),
        "janitor_claimable_scan_ms": (_median(scan_samples), _worst(scan_samples)),
        "janitor_dedupe_enqueue_ms": (
            _median(duplicate_samples),
            _worst(duplicate_samples),
        ),
    }
    print(f"BCDEF lease/janitor measured (median, worst) ms: {measured}")

    # Same rule as the other ceilings: headroom over the measured median so
    # the gate fires on an order-of-magnitude regression, not on disk noise.
    # Measured here (median ms): renew 0.03, fence 0.015, expired refusal
    # 0.11, janitor enqueue 3.1, claim 0.06.
    assert _median(renew_samples) < 10.0, measured
    assert _worst(renew_samples) < 60.0, measured
    assert _median(fence_samples) < 10.0, measured
    assert refusal_ms < 10.0, measured
    assert _median(enqueue_samples) < 30.0, measured
    assert _worst(enqueue_samples) < 120.0, measured
    assert _median(claim_samples) < 10.0, measured
    assert _worst(claim_samples) < 60.0, measured
    # BCDEF review S4: the recovery scan and the dedupe retry reuse the
    # existing janitor ceilings (claim + enqueue), not new looser ones.
    assert _median(scan_samples) < 10.0, measured
    assert _worst(scan_samples) < 60.0, measured
    assert _median(duplicate_samples) < 30.0, measured
    assert _worst(duplicate_samples) < 120.0, measured
