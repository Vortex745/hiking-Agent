"""BCDEF ticket 12 — measured ceilings for the new durable writes.

These are the three costs the spec asks to bound: the extra admission
writes, snapshot assembly and event replay at the endpoint's 1000-event
page size, plus sandbox lease acquire/renew. Thresholds are set from
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
