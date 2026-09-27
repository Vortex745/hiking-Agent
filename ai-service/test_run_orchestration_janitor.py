"""Ticket 41 — paused-run janitor, tested on the orchestration seam.

The sweep implementation lives in :mod:`agent_v2.run_orchestration`
(:func:`sweep_expired_paused_runs`); the startup assembly (``main.py``)
only wires it up. These tests drive the sweep directly against real
SQLite persistence under a temp directory — no ASGI shell, no sleeps.

Behavior asserted:

* An expired paused run with an active sandbox lease is reclaimed:
  lease destroyed with a failure reason, run marked ``error`` terminal,
  capacity slot released, reclaimed count returned.
* A fresh paused run (within TTL) is NOT reclaimed.
* A backdated run that is no longer paused is NOT reclaimed — the
  status re-check keeps sweep concurrent with user decision / resume.
* A paused run without an active lease is still reclaimed (mark
  terminal + capacity release, no lease destroy).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import sweep_expired_paused_runs
from agent_v2.runs import RunRepository


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _runtime(repo: RunRepository, capacity: Any = None) -> SimpleNamespace:
    return SimpleNamespace(runs=repo, capacity=capacity)


class _FakeCapacity:
    def __init__(self) -> None:
        self.released: list[str] = []

    def release_sandbox(self, request_id: str) -> None:
        self.released.append(request_id)


def _paused_run(
    repo: RunRepository,
    request_id: str,
    *,
    with_lease: bool = True,
    updated_at: str | None = None,
) -> str:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="m",
    )
    repo.mark_run_paused(run_id)
    if with_lease:
        repo.create_sandbox_lease(
            run_id=run_id,
            sandbox_type="local_native",
            workspace_path=f"/ws/{run_id}",
        )
    if updated_at is not None:
        repo.control._conn.execute(
            "UPDATE runs SET updated_at = ? WHERE run_id = ?",
            (updated_at, run_id),
        )
    return run_id


def _old_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()


def test_sweep_reclaims_expired_paused_run_and_releases_capacity(tmp_path):
    repo = _repo(tmp_path)
    capacity = _FakeCapacity()
    run_id = _paused_run(repo, "req-exp", updated_at=_old_iso())

    reclaimed = sweep_expired_paused_runs(_runtime(repo, capacity))

    assert reclaimed == 1
    assert capacity.released == ["req-exp"]
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"
    lease = repo.get_sandbox_lease(run_id)
    assert lease is not None and lease.status == "failed"
    assert "expired" in (lease.failure_reason or "")


def test_sweep_skips_fresh_paused_run(tmp_path):
    repo = _repo(tmp_path)
    capacity = _FakeCapacity()
    run_id = _paused_run(repo, "req-fresh")  # updated_at = now → within TTL

    reclaimed = sweep_expired_paused_runs(_runtime(repo, capacity))

    assert reclaimed == 0
    assert capacity.released == []
    run = repo.get_run(run_id)
    assert run is not None and run.status == "paused"
    lease = repo.get_sandbox_lease(run_id)
    assert lease is not None and lease.status == "active"


def test_sweep_skips_run_that_left_paused_concurrently(tmp_path):
    """The status re-check: a backdated run the user just resumed (or
    whose decision arrived) is no longer paused — the sweep must leave
    it alone even though the candidate query matched it."""
    repo = _repo(tmp_path)
    capacity = _FakeCapacity()
    run_id = _paused_run(repo, "req-resumed", updated_at=_old_iso())
    repo.mark_run_done(run_id)  # user decision beat the sweep

    reclaimed = sweep_expired_paused_runs(_runtime(repo, capacity))

    assert reclaimed == 0
    assert capacity.released == []
    run = repo.get_run(run_id)
    assert run is not None and run.status == "done"
    # The done run's terminal status was not clobbered by the sweep.
    assert run.terminal_type == "done"


def test_sweep_reclaims_expired_paused_run_without_lease(tmp_path):
    repo = _repo(tmp_path)
    capacity = _FakeCapacity()
    run_id = _paused_run(
        repo, "req-nolease", with_lease=False, updated_at=_old_iso()
    )

    reclaimed = sweep_expired_paused_runs(_runtime(repo, capacity))

    assert reclaimed == 1
    assert capacity.released == ["req-nolease"]
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"


def test_sweep_without_capacity_still_marks_terminal(tmp_path):
    repo = _repo(tmp_path)
    run_id = _paused_run(repo, "req-nocap", updated_at=_old_iso())

    reclaimed = sweep_expired_paused_runs(_runtime(repo))

    assert reclaimed == 1
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"
