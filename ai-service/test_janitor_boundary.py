"""BCDEF ticket 07 — approval/resume identity and cleanup janitor.

Seam: the orchestration boundary (``RunOrchestrator``), the approval
decision seam, and the durable janitor store/runner. Real SQLite under a
temp directory, fake Computer, fake kernel; no network, no real sandbox.

Behavior asserted:

* approval advances the SAME run — no second run/lease/effect/approval;
  duplicate decisions are idempotent and conflicting decisions return a
  stable conflict without changing the first decision;
* paused is resumable; done/error/cancelled/admission_failed and
  outcome_unknown never continue execution;
* resume revalidates policy, run status and the sandbox lease BEFORE any
  model/effect/sandbox work;
* the PDF resume path never re-runs the model — it uses the plan drafted
  and persisted before the pause;
* terminal cleanup is idempotent, records its status, and a cleanup
  failure creates a durable janitor work item that retains run, lease,
  effect and sandbox evidence;
* the janitor store CAS (claim/complete/fail) is durable and idempotent;
* cleanup retry reuses the ORIGINAL workspace and never creates a new
  sandbox to bypass the old lease;
* the janitor can claim a ticket-05 unknown outcome repair without
  re-executing the external effect;
* BCDEF review S4: a claim carries a fixed 600 s deadline and a generation
  (``attempts``), an unexpired claim cannot be stolen, an expired one can,
  and only the CURRENT owner+generation may settle; a completed task
  releases its ACTIVE dedupe key while keeping the historical row;
* duplicate approval/resume/cleanup add no extra side effects.

Expiry is driven by a controlled clock (never a real 600 s wait).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2 import persistence as persistence_module
from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.effect_repair import repair_effect
from agent_v2.effects import EffectStatus, compute_effect_identity
from agent_v2.execution_contract import policy_hash
from agent_v2.execution_policy import build_execution_policy
from agent_v2.janitor import run_pending_janitor_tasks
from agent_v2.persistence import (
    CheckpointsStore,
    ControlStore,
    JanitorTaskRecord,
    PolicySnapshotWrite,
)
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

# The fixed claim deadline this spec chooses for the janitor (ticket 04).
CLAIM_TTL_SECONDS = 600


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


class _Clock:
    """Controlled UTC clock: the store's only source of "now"."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def iso(self) -> str:
        return self.now.isoformat()

    def plus(self, seconds: float) -> str:
        return (self.now + timedelta(seconds=float(seconds))).isoformat()

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def elapsed_since(self, timestamp: str | None) -> float:
        assert timestamp is not None
        return (self.now - datetime.fromisoformat(timestamp)).total_seconds()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Freeze + advance the persistence clock instead of waiting 600 s."""
    frozen = _Clock(datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(persistence_module, "_now_iso", frozen.iso)
    monkeypatch.setattr(persistence_module, "_now_plus_seconds", frozen.plus)
    return frozen


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "trusted",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_frozen_policy(repo: RunRepository, run_id: str, profile_id: str) -> None:
    policy = build_execution_policy(resolve_profile(profile_id), _settings())
    write = PolicySnapshotWrite(
        policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
        policy_hash=policy_hash(policy),
        schema_version=policy["schema_version"],
        source_version=policy["source_version"],
        compatibility_json=json.dumps(policy["compatibility"]),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    repo.control.insert_policy_snapshot(run_id=run_id, policy=write)


class _FakeComputer:
    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.workspaces: list[str] = []
        self.pdfs: list[tuple[str, str, str]] = []
        self.bash_calls: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    def create_workspace(self, run_id: str) -> str:
        self.workspaces.append(run_id)
        return f"/ws/{run_id}"

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        return filename

    def run_bash(self, workspace_path, command, **_kwargs):
        self.bash_calls.append((workspace_path, command))
        return SimpleNamespace(
            exit_code=0, stdout="hi", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _fake_export(**kwargs: Any) -> dict[str, Any]:
    """Stand in for the host export seam, INCLUDING its durable evidence.

    The production export persists one artifact row per successful export
    (Ticket 10), so the fake writes the row through the repository too —
    otherwise "duplicate resume creates no extra side effects" could not
    be observed at all.
    """
    runs_repo = kwargs.get("runs_repo")
    run_id = kwargs.get("run_id")
    artifact_id = f"art-{uuid.uuid4().hex[:12]}"
    if runs_repo is not None and run_id is not None:
        runs_repo.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=kwargs.get("conversation_id") or "",
            public_filename="hiking-plan.pdf",
            mime_type="application/pdf",
            size_bytes=10,
            sha256="deadbeef",
            storage_path=f"/host/{artifact_id}.pdf",
            status="available",
            failure_reason=None,
            created_at=_now_iso(),
            expires_at="2026-11-01T00:00:00+00:00",
        )
    return {
        "artifact_id": artifact_id,
        "public_filename": "hiking-plan.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 10,
        "sha256": "deadbeef",
        "expires_at": "2026-11-01T00:00:00+00:00",
    }


def _orchestrator(
    repo: RunRepository,
    computer: _FakeComputer,
    *,
    kernel: Any = None,
    cleanup: Any = None,
    factory_calls: list[int] | None = None,
    export_calls: list[dict[str, Any]] | None = None,
):
    async def default_kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return ("# 计划正文", [])

    def default_cleanup(comp, workspace_path, runs, run_id) -> None:
        if runs is not None and run_id is not None:
            runs.destroy_sandbox_lease(run_id=run_id)

    def factory(*_a: Any, **_k: Any) -> _FakeComputer:
        if factory_calls is not None:
            factory_calls.append(1)
        return computer

    def export(**kwargs: Any) -> dict[str, Any]:
        if export_calls is not None:
            export_calls.append(kwargs)
        return _fake_export(**kwargs)

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        kernel_turn=kernel or default_kernel,
        artifact_export=export,
        computer_factory=factory,
        sandbox_cleanup=cleanup or default_cleanup,
        gate_factory=None,
    )


def _stream_chat(orch: RunOrchestrator, request_id: str, profile: str) -> list[Any]:
    req = ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id=request_id,
        profile_id=profile,
    )

    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req)]

    return asyncio.run(run())


def _resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [
            item
            async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)
        ]

    return asyncio.run(run())


def _paused_pdf_run(
    repo: RunRepository,
    request_id: str,
    *,
    prepared_content: str | None = None,
    expired_lease: bool = False,
) -> tuple[str, str]:
    """Hand-assemble the durable state after an artifact approval pause."""
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    _seed_frozen_policy(repo, run_id, "artifact")
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type="local_native",
        workspace_path=f"/ws/{run_id}",
    )
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "hiking-plan.pdf"}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "hiking-plan.pdf"}',
        risk_level="high",
        impact="生成徒步计划 PDF 并导出到宿主存储",
        created_at=_now_iso(),
        expires_at=compute_expiry(_now_iso()),
        prepared_content=prepared_content,
    )
    repo.resolve_approval(
        approval_id=approval_id,
        status="approved",
        decision_request_id=f"dec-{request_id}",
    )
    if expired_lease:
        repo.control._conn.execute(
            "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_id),
        )
    return run_id, approval_id


def _count_rows(repo: RunRepository, table: str, where: str = "1=1") -> int:
    row = repo.control._conn.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE {where}"
    ).fetchone()
    return int(row["n"])


# ---- janitor store CAS ------------------------------------------------------


def test_janitor_task_enqueue_is_idempotent_on_dedupe_key(tmp_path):
    repo = _repo(tmp_path)
    try:
        first, created = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id="run-1",
            evidence={"run_id": "run-1", "workspace_path": "/ws/run-1"},
            dedupe_key="cleanup:run-1",
        )
        second, created_again = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id="run-1",
            evidence={"run_id": "run-1"},
            dedupe_key="cleanup:run-1",
        )
        assert created is True and created_again is False
        assert first.task_id == second.task_id
        assert first.status == "pending"
        assert first.kind == "cleanup"
        assert first.evidence["workspace_path"] == "/ws/run-1"
        assert first.attempts == 0
        assert _count_rows(repo, "janitor_tasks") == 1
    finally:
        repo.close()


def test_janitor_task_claim_complete_and_fail_cas(tmp_path):
    repo = _repo(tmp_path)
    try:
        task, _ = repo.enqueue_janitor_task(
            kind="cleanup", run_id="run-cas", evidence={"run_id": "run-cas"}
        )
        claimed, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True
        assert claimed is not None
        assert claimed.status == "claimed"
        assert claimed.claimed_by == "janitor-a"
        assert claimed.attempts == 1

        _again, won_again = repo.claim_janitor_task(task.task_id, owner="janitor-b")
        assert won_again is False

        done, completed = repo.complete_janitor_task(
            task.task_id,
            owner="janitor-a",
            expected_attempt=claimed.attempts,
            result={"cleaned": True},
        )
        assert completed is True
        assert done is not None
        assert done.status == "done"
        assert done.result["cleaned"] is True
        assert done.completed_at is not None

        _dup, completed_again = repo.complete_janitor_task(
            task.task_id, owner="janitor-a", expected_attempt=claimed.attempts
        )
        assert completed_again is False

        failed_task, _ = repo.enqueue_janitor_task(
            kind="repair", run_id="run-cas", effect_id="eff-1", evidence={}
        )
        first_claim, _ = repo.claim_janitor_task(
            failed_task.task_id, owner="janitor-a"
        )
        assert first_claim is not None
        failed, failed_won = repo.fail_janitor_task(
            failed_task.task_id,
            "handler boom",
            owner="janitor-a",
            expected_attempt=first_claim.attempts,
        )
        assert failed_won is True
        assert failed is not None
        assert failed.status == "failed"
        assert failed.last_error == "handler boom"

        # A failed task can be retried (attempts increments).
        reclaimed, reclaimed_won = repo.claim_janitor_task(
            failed_task.task_id, owner="janitor-b"
        )
        assert reclaimed_won is True
        assert reclaimed is not None
        assert reclaimed.attempts == 2
        _failed, completed_claimed = repo.complete_janitor_task(
            failed_task.task_id,
            owner="janitor-b",
            expected_attempt=reclaimed.attempts,
        )
        assert completed_claimed is True
    finally:
        repo.close()


# ---- janitor claim deadline (BCDEF review S4) ------------------------------


def _pending_cleanup_task(
    repo: RunRepository, run_id: str
) -> JanitorTaskRecord:
    task, created = repo.enqueue_janitor_task(
        kind="cleanup",
        run_id=run_id,
        evidence={"run_id": run_id, "workspace_path": f"/ws/{run_id}"},
        dedupe_key=f"cleanup:{run_id}",
    )
    assert created is True
    return task


def test_claim_stamps_a_fixed_deadline_and_records_the_generation(
    tmp_path, clock: _Clock
):
    """AC1a: a claim carries the spec's fixed 600 s deadline + generation."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-deadline")
        claimed, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")

        assert won is True
        assert claimed is not None
        assert claimed.status == "claimed"
        assert claimed.claimed_by == "janitor-a"
        assert claimed.attempts == 1
        assert claimed.claim_expires_at is not None
        # The deadline is exactly the fixed claim TTL after the claim stamp.
        deadline = datetime.fromisoformat(claimed.claim_expires_at)
        claimed_at = datetime.fromisoformat(claimed.claimed_at)
        assert (deadline - claimed_at) == timedelta(seconds=CLAIM_TTL_SECONDS)
        # Durably observable, not just the returned snapshot.
        stored = repo.get_janitor_task(task.task_id)
        assert stored is not None and stored.claim_expires_at == claimed.claim_expires_at
    finally:
        repo.close()


def test_unexpired_claim_cannot_be_stolen_by_another_worker(tmp_path, clock: _Clock):
    """AC1b: before the deadline, a second worker loses the claim CAS."""
    repo = _repo(tmp_path)
    other = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        task = _pending_cleanup_task(repo, "run-no-steal")
        claimed, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True
        assert claimed is not None
        generation = claimed.attempts

        # 599 s in: still the first worker's claim.
        clock.advance(CLAIM_TTL_SECONDS - 1)
        _loser, stolen = other.claim_janitor_task(task.task_id, owner="janitor-b")
        assert stolen is False

        stored = other.get_janitor_task(task.task_id)
        assert stored is not None
        assert stored.status == "claimed"
        assert stored.claimed_by == "janitor-a"
        assert stored.attempts == generation
        assert stored.claim_expires_at == claimed.claim_expires_at
    finally:
        other.close()
        repo.close()


def test_expired_claim_is_reclaimed_at_the_boundary_and_after(
    tmp_path, clock: _Clock
):
    """AC1c: an expired claim is reclaimable AT the deadline; attempts + 1."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-expire")
        first, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True and first is not None
        deadline = first.claim_expires_at
        assert deadline is not None

        # One second short of the deadline: still not reclaimable.
        clock.advance(CLAIM_TTL_SECONDS - 1)
        _same, too_early = repo.claim_janitor_task(task.task_id, owner="janitor-b")
        assert too_early is False

        # AT the deadline the claim is expired, so the takeover wins.
        clock.advance(1)
        second, stolen = repo.claim_janitor_task(task.task_id, owner="janitor-b")
        assert stolen is True
        assert second is not None
        assert second.status == "claimed"
        assert second.claimed_by == "janitor-b"
        # Exactly one generation per successful claim — no double counting.
        assert second.attempts == first.attempts + 1
        assert second.claimed_at is not None
        assert second.claim_expires_at is not None
        assert (
            datetime.fromisoformat(second.claim_expires_at)
            - datetime.fromisoformat(second.claimed_at)
        ) == timedelta(seconds=CLAIM_TTL_SECONDS)

        # Well past that deadline too (a second crash is recoverable).
        clock.advance(CLAIM_TTL_SECONDS + 1)
        third, stolen_again = repo.claim_janitor_task(task.task_id, owner="janitor-c")
        assert stolen_again is True
        assert third is not None
        assert third.attempts == second.attempts + 1
        assert third.claimed_by == "janitor-c"
    finally:
        repo.close()


def test_two_workers_competing_for_one_expired_claim_have_one_winner(
    tmp_path, clock: _Clock
):
    """AC2: two independent connections racing an expired claim, one wins."""
    repo = _repo(tmp_path)
    worker_b = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        task = _pending_cleanup_task(repo, "run-race")
        first, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True and first is not None

        clock.advance(CLAIM_TTL_SECONDS)
        claims = [
            repo.claim_janitor_task(task.task_id, owner="janitor-a"),
            worker_b.claim_janitor_task(task.task_id, owner="janitor-b"),
        ]
        assert [ok for (_rec, ok) in claims].count(True) == 1, claims

        stored = worker_b.get_janitor_task(task.task_id)
        assert stored is not None
        assert stored.status == "claimed"
        # The one winning claim bumped the generation by exactly one.
        assert stored.attempts == first.attempts + 1
        assert stored.claimed_by in ("janitor-a", "janitor-b")
    finally:
        worker_b.close()
        repo.close()


# ---- settlement fence (BCDEF review S4) ------------------------------------


def _settled_state_assertions(
    repo: RunRepository,
    task_id: str,
    *,
    expected_owner: str,
    expected_attempt: int,
) -> JanitorTaskRecord:
    """Durable row after a REJECTED settlement: nothing the loser wrote."""
    stored = repo.get_janitor_task(task_id)
    assert stored is not None
    assert stored.status == "claimed"
    assert stored.claimed_by == expected_owner
    assert stored.attempts == expected_attempt
    # No done/failed evidence leaked out of the rejected write.
    assert stored.completed_at is None
    assert stored.result_json is None
    assert stored.last_error is None
    return stored


def test_a_taken_over_claim_locks_out_the_old_worker(tmp_path, clock: _Clock):
    """AC3a: different owner + stale generation cannot write done or failed."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-fence-1")
        old, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True and old is not None

        clock.advance(CLAIM_TTL_SECONDS)
        new, stolen = repo.claim_janitor_task(task.task_id, owner="janitor-b")
        assert stolen is True and new is not None
        assert new.attempts == old.attempts + 1

        # The dead worker finally returns and tries to settle with its OWN
        # (stale) generation: rejected on both outcomes.
        _done, completed = repo.complete_janitor_task(
            task.task_id,
            owner="janitor-a",
            expected_attempt=old.attempts,
            result={"cleaned": True},
        )
        assert completed is False
        _fail, failed = repo.fail_janitor_task(
            task.task_id,
            "late error",
            owner="janitor-a",
            expected_attempt=old.attempts,
        )
        assert failed is False
        _stale_owner, wrong_owner = repo.complete_janitor_task(
            task.task_id, owner="janitor-z", expected_attempt=new.attempts
        )
        assert wrong_owner is False
        _settled_state_assertions(
            repo,
            task.task_id,
            expected_owner="janitor-b",
            expected_attempt=new.attempts,
        )
        # The new worker still can, with the generation it was given.
        done, completed_now = repo.complete_janitor_task(
            task.task_id,
            owner="janitor-b",
            expected_attempt=new.attempts,
            result={"cleaned": True},
        )
        assert completed_now is True
        assert done is not None and done.status == "done"
    finally:
        repo.close()


def test_same_owner_stale_generation_cannot_settle(tmp_path, clock: _Clock):
    """AC3b: a reused worker name is still fenced by the generation."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-fence-2")
        first, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True and first is not None

        # The SAME owner id re-claims after the deadline (its own restart).
        clock.advance(CLAIM_TTL_SECONDS)
        second, again = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert again is True and second is not None
        assert second.attempts == first.attempts + 1
        assert second.claimed_by == "janitor-a"

        # The first generation of the same owner cannot settle.
        _done, completed = repo.complete_janitor_task(
            task.task_id,
            owner="janitor-a",
            expected_attempt=first.attempts,
            result={"cleaned": True},
        )
        assert completed is False
        _fail, failed = repo.fail_janitor_task(
            task.task_id,
            "late error",
            owner="janitor-a",
            expected_attempt=first.attempts,
        )
        assert failed is False
        _settled_state_assertions(
            repo,
            task.task_id,
            expected_owner="janitor-a",
            expected_attempt=second.attempts,
        )

        done, completed_now = repo.complete_janitor_task(
            task.task_id, owner="janitor-a", expected_attempt=second.attempts
        )
        assert completed_now is True
        assert done is not None and done.status == "done"
    finally:
        repo.close()


def test_expired_but_unreclaimed_claim_rejects_the_old_worker(
    tmp_path, clock: _Clock
):
    """AC4: past the deadline nobody took over, yet the old worker is out."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-fence-3")
        claimed, won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert won is True and claimed is not None

        # No takeover at all — only the deadline elapsed.
        clock.advance(CLAIM_TTL_SECONDS)
        _done, completed = repo.complete_janitor_task(
            task.task_id,
            owner="janitor-a",
            expected_attempt=claimed.attempts,
            result={"cleaned": True},
        )
        assert completed is False
        _fail, failed = repo.fail_janitor_task(
            task.task_id,
            "late error",
            owner="janitor-a",
            expected_attempt=claimed.attempts,
        )
        assert failed is False

        stored = repo.get_janitor_task(task.task_id)
        assert stored is not None
        assert stored.status == "claimed"
        assert stored.claimed_by == "janitor-a"
        assert stored.attempts == claimed.attempts
        assert stored.completed_at is None
        assert stored.result_json is None

        # Control: the same owner/generation settles fine one second BEFORE
        # its own deadline, so the rejection above is the deadline at work.
        control = _pending_cleanup_task(repo, "run-fence-3-control")
        live, live_won = repo.claim_janitor_task(control.task_id, owner="janitor-a")
        assert live_won is True and live is not None
        clock.advance(CLAIM_TTL_SECONDS - 1)
        control_done, in_time = repo.complete_janitor_task(
            control.task_id, owner="janitor-a", expected_attempt=live.attempts
        )
        assert in_time is True
        assert control_done is not None and control_done.status == "done"
    finally:
        repo.close()


# ---- active dedupe release (BCDEF review S4) --------------------------------


def test_completed_task_releases_its_dedupe_key_for_the_next_round(tmp_path):
    """AC7: one ACTIVE row per logical key; a new round gets a new task_id."""
    repo = _repo(tmp_path)
    try:
        key = "cleanup:run-dedupe"
        first, created = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id="run-dedupe",
            evidence={"run_id": "run-dedupe", "workspace_path": "/ws/run-dedupe"},
            dedupe_key=key,
        )
        assert created is True

        # ACTIVE duplicates (pending) still collapse onto the one row.
        again, created_again = repo.enqueue_janitor_task(
            kind="cleanup",
            run_id="run-dedupe",
            evidence={"run_id": "run-dedupe"},
            dedupe_key=key,
        )
        assert created_again is False
        assert again.task_id == first.task_id
        assert _count_rows(repo, "janitor_tasks") == 1

        claimed, won = repo.claim_janitor_task(first.task_id, owner="janitor-a")
        assert won is True and claimed is not None

        # A claimed item is still ACTIVE: dedupe keeps collapsing on it.
        during, created_during = repo.enqueue_janitor_task(
            kind="cleanup", run_id="run-dedupe", evidence={}, dedupe_key=key
        )
        assert created_during is False and during.task_id == first.task_id

        done, completed = repo.complete_janitor_task(
            first.task_id,
            owner="janitor-a",
            expected_attempt=claimed.attempts,
            result={"cleanup_status": "succeeded"},
        )
        assert completed is True and done is not None

        # The ACTIVE key is gone, so the next round opens a NEW task_id...
        nxt, created_next = repo.enqueue_janitor_task(
            kind="cleanup", run_id="run-dedupe", evidence={}, dedupe_key=key
        )
        assert created_next is True
        assert nxt.task_id != first.task_id
        assert nxt.status == "pending"
        # ...while the finished row is kept for audit, key recorded inside.
        assert _count_rows(repo, "janitor_tasks") == 2
        assert _count_rows(repo, "janitor_tasks", "status = 'done'") == 1
        historical = repo.get_janitor_task(first.task_id)
        assert historical is not None
        assert historical.status == "done"
        assert historical.dedupe_key is None
        assert historical.evidence["dedupe_key"] == key
        # The original evidence survived the release, not just the key.
        assert historical.evidence["workspace_path"] == "/ws/run-dedupe"
        assert historical.result == {"cleanup_status": "succeeded"}
        assert historical.completed_at is not None
        assert historical.attempts == 1
    finally:
        repo.close()


def test_failed_task_keeps_its_active_dedupe_key(tmp_path):
    """A failed round is still the same work item: the key stays ACTIVE."""
    repo = _repo(tmp_path)
    try:
        key = "cleanup:run-fail-key"
        task = _pending_cleanup_task(repo, "run-fail-key")
        assert task.dedupe_key == key
        claimed, _won = repo.claim_janitor_task(task.task_id, owner="janitor-a")
        assert claimed is not None

        failed, ok = repo.fail_janitor_task(
            task.task_id,
            "still locked",
            owner="janitor-a",
            expected_attempt=claimed.attempts,
        )
        assert ok is True and failed is not None
        assert failed.status == "failed"
        assert failed.dedupe_key == key
        assert failed.last_error == "still locked"
        # The deadline is cleared so the retry does not have to wait for it.
        assert failed.claim_expires_at is None
        assert repo.get_janitor_task(task.task_id) is not None

        again, created = repo.enqueue_janitor_task(
            kind="cleanup", run_id="run-fail-key", evidence={}, dedupe_key=key
        )
        assert created is False
        assert again.task_id == task.task_id
        assert _count_rows(repo, "janitor_tasks") == 1
    finally:
        repo.close()


# ---- runner recovery + claim_lost (BCDEF review S4) ------------------------


def test_runner_recovers_a_hung_claim_after_a_worker_restart(
    tmp_path, clock: _Clock
):
    """AC5a: a sweep after a crash takes over the hung claim and finishes.

    Worker 1 claims the cleanup item and dies (nothing settles). The next
    sweep — a restarted process, so a new claimer — reclaims it once the
    deadline has passed, cleans the ORIGINAL workspace and settles.
    """
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-janitor-hung")
        computer = _FakeComputer()

        def failing_cleanup(comp, workspace_path, runs, rid) -> None:
            raise RuntimeError("transient cleanup failure")

        orch = _orchestrator(repo, computer, cleanup=failing_cleanup)
        _resume(orch, run_id)
        task = repo.list_janitor_tasks(status="pending")[0]

        # The crashed worker: it claimed the item and never came back.
        hung, won = repo.claim_janitor_task(task.task_id, owner="janitor-crashed")
        assert won is True and hung is not None
        assert hung.attempts == 1

        # A restarted runner only sees the item once its deadline lapses.
        assert run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-restarted",
            computer_factory=lambda *_a, **_k: computer,
        ) == []
        clock.advance(CLAIM_TTL_SECONDS)

        results = run_pending_janitor_tasks(
            repo,
            limit=10,
            claimer="janitor-restarted",
            computer_factory=lambda *_a, **_k: computer,
        )
        assert [r.status for r in results] == ["done"], results
        assert results[0].task_id == task.task_id

        settled = repo.get_janitor_task(task.task_id)
        assert settled is not None
        assert settled.status == "done"
        assert settled.claimed_by == "janitor-restarted"
        assert settled.attempts == 2
        assert settled.result["cleanup_status"] == "succeeded"
        # Recovery touched only the original workspace, no new sandbox.
        assert computer.cleaned == [f"/ws/{run_id}"]
        assert computer.workspaces == []
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None and lease.cleanup_status == "succeeded"
        # Nothing left to do on the next sweep.
        assert (
            run_pending_janitor_tasks(
                repo, limit=10, claimer="janitor-restarted",
                computer_factory=lambda *_a, **_k: computer,
            )
            == []
        )
        assert computer.cleaned == [f"/ws/{run_id}"]
    finally:
        repo.close()


def _takeover_during_handler(
    repo: RunRepository, clock: _Clock, *, then_raise: bool
) -> Any:  # returns a cleanup handler callable
    """Cleanup handler that loses the claim before it can settle.

    Stands in for "the worker finished the work but its deadline expired
    and another worker took the item over" — the settlement CAS must then
    refuse the write, and the runner must not report done/failed.
    """
    calls: list[str] = []

    def handler(_runs_repo: RunRepository, evidence: dict[str, Any]) -> None:
        calls.append(str(evidence["run_id"]))
        clock.advance(CLAIM_TTL_SECONDS)
        ghost, won = repo.claim_janitor_task(
            evidence["task_id"], owner="janitor-ghost"
        )
        assert won is True and ghost is not None
        if then_raise:
            raise RuntimeError("handler failed after losing the claim")

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def test_runner_reports_claim_lost_when_completion_cas_loses(
    tmp_path, clock: _Clock
):
    """AC5b: a lost settlement CAS is reported as claim_lost, not done."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-claim-lost")
        handler = _takeover_during_handler(repo, clock, then_raise=False)

        def cleanup(runs_repo, evidence) -> None:
            evidence = {**evidence, "task_id": task.task_id}
            handler(runs_repo, evidence)

        results = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-loser", cleanup=cleanup
        )
        assert [r.status for r in results] == ["claim_lost"], results
        assert results[0].task_id == task.task_id
        assert results[0].kind == "cleanup"

        # The loser wrote nothing: the ghost's claim is still current.
        stored = repo.get_janitor_task(task.task_id)
        assert stored is not None
        assert stored.status == "claimed"
        assert stored.claimed_by == "janitor-ghost"
        assert stored.attempts == 2
        assert stored.result_json is None
        assert stored.completed_at is None
        assert stored.dedupe_key == "cleanup:run-claim-lost"
        # The work itself did happen once.
        assert handler.calls == ["run-claim-lost"]
    finally:
        repo.close()


def test_runner_reports_claim_lost_when_failure_cas_loses(
    tmp_path, clock: _Clock
):
    """AC5c: a failed handler that lost its claim is not recorded failed."""
    repo = _repo(tmp_path)
    try:
        task = _pending_cleanup_task(repo, "run-claim-lost-fail")
        handler = _takeover_during_handler(repo, clock, then_raise=True)

        def cleanup(runs_repo, evidence) -> None:
            handler(runs_repo, {**evidence, "task_id": task.task_id})

        results = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-loser", cleanup=cleanup
        )
        assert [r.status for r in results] == ["claim_lost"], results
        assert results[0].error == "handler failed after losing the claim"

        stored = repo.get_janitor_task(task.task_id)
        assert stored is not None
        assert stored.status == "claimed"
        assert stored.claimed_by == "janitor-ghost"
        assert stored.last_error is None
        assert stored.attempts == 2
    finally:
        repo.close()


def test_sweep_round_is_bounded_and_does_not_starve_an_expired_claim(
    tmp_path, clock: _Clock
):
    """AC6: 10 items max per round, oldest-claimable first, hung claim wins.

    A dead worker's expired claim became claimable BEFORE the fresh pending
    items, so it is picked in the FIRST round even though twelve newer
    items are queued behind it — and the round still stops at ``limit``.
    """
    repo = _repo(tmp_path)
    processed: list[str] = []

    def cleanup(_runs_repo, evidence) -> None:
        processed.append(str(evidence["run_id"]))

    try:
        hung = _pending_cleanup_task(repo, "run-hung-first")
        claimed, won = repo.claim_janitor_task(hung.task_id, owner="janitor-dead")
        assert won is True and claimed is not None

        # The claim lapses, and newer work keeps arriving afterwards.
        clock.advance(CLAIM_TTL_SECONDS)
        fresh: list[str] = []
        for index in range(12):
            clock.advance(1)
            fresh.append(f"run-fresh-{index}")
            _pending_cleanup_task(repo, fresh[-1])

        results = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-sweep", cleanup=cleanup
        )
        # Bounded round, and the expired claim is NOT starved: it leads.
        assert len(results) == 10
        assert processed[0] == "run-hung-first"
        assert processed[1:] == fresh[:9]
        assert [r.status for r in results] == ["done"] * 10
        assert results[0].task_id == hung.task_id

        # The takeover stamped a fresh deadline and bumped the generation.
        settled = repo.get_janitor_task(hung.task_id)
        assert settled is not None
        assert settled.status == "done"
        assert settled.attempts == 2
        assert settled.claimed_by == "janitor-sweep"

        # The nine items the round did not reach are still open work.
        still_open = repo.list_janitor_tasks(status="pending", limit=100)
        assert [t.run_id for t in still_open] == fresh[9:]

        second = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-sweep", cleanup=cleanup
        )
        assert len(second) == 3
        assert processed[10:] == fresh[9:]
        assert repo.list_janitor_tasks(status="pending", limit=100) == []
    finally:
        repo.close()


def test_one_sweep_issues_one_bounded_candidate_query(tmp_path):
    """AC6b: the round is a single LIMIT-bounded scan, not a per-status loop."""
    repo = _repo(tmp_path)
    statements: list[str] = []

    def cleanup(_runs_repo, evidence) -> None:
        return None

    try:
        for index in range(30):
            repo.enqueue_janitor_task(
                kind="cleanup",
                run_id=f"run-scan-{index}",
                evidence={"run_id": f"run-scan-{index}"},
                dedupe_key=f"cleanup:run-scan-{index}",
            )
        repo.control._conn.set_trace_callback(statements.append)
        try:
            results = run_pending_janitor_tasks(
                repo, limit=10, claimer="janitor-scan", cleanup=cleanup
            )
        finally:
            repo.control._conn.set_trace_callback(None)

        assert len(results) == 10
        selects = [
            statement
            for statement in statements
            if statement.strip().upper().startswith("SELECT")
            and "janitor_tasks" in statement
            and "ORDER BY" in statement.upper()
            and "LIMIT" in statement.upper()
        ]
        assert len(selects) == 1, selects
        assert "LIMIT" in selects[0].upper()

        # The statement the round actually issued is served by the
        # (status, claim_expires_at) index — no full table scan.
        plan = " ".join(
            row["detail"]
            for row in repo.control._conn.execute(
                "EXPLAIN QUERY PLAN " + selects[0]
            ).fetchall()
        )
        assert "idx_janitor_claim_candidates" in plan, plan
        assert "SCAN janitor_tasks" not in plan, plan
        indexed = repo.control._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index'"
            " AND name = 'idx_janitor_claim_candidates'"
        ).fetchone()
        assert indexed is not None
        assert "claim_expires_at" in indexed["sql"]
    finally:
        repo.close()


# ---- legacy upgrade (BCDEF review S4) --------------------------------------

_LEGACY_JANITOR_COLUMNS = (
    "task_id",
    "kind",
    "status",
    "dedupe_key",
    "run_id",
    "lease_id",
    "effect_id",
    "evidence_json",
    "attempts",
    "last_error",
    "claimed_by",
    "claimed_at",
    "result_json",
    "created_at",
    "updated_at",
    "completed_at",
)


def _legacy_control_db(tmp_path) -> str:
    """A control database whose janitor table predates the claim deadline."""
    path = str(tmp_path / "control.sqlite")
    store = ControlStore(path)
    store._conn.execute("BEGIN")
    store._conn.execute("DROP INDEX IF EXISTS idx_janitor_claim_candidates")
    store._conn.execute("ALTER TABLE janitor_tasks DROP COLUMN claim_expires_at")
    store._conn.execute("COMMIT")
    store.close()
    return path


def _insert_legacy_task(
    path: str,
    *,
    task_id: str,
    status: str,
    evidence: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    claimed_by: str | None = None,
    claimed_at: str | None = None,
    attempts: int = 0,
    last_error: str | None = None,
    result_json: str | None = None,
    completed_at: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    """Write a janitor row the way the pre-ticket-04 code did (raw SQL)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "INSERT INTO janitor_tasks (" + ", ".join(_LEGACY_JANITOR_COLUMNS) + ")"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                "cleanup",
                status,
                dedupe_key,
                str((evidence or {}).get("run_id") or ""),
                "",
                "",
                json.dumps(evidence or {}, sort_keys=True),
                attempts,
                last_error,
                claimed_by,
                claimed_at,
                result_json,
                created_at,
                created_at,
                completed_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _legacy_repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def test_legacy_claimed_rows_get_a_deadline_and_are_recovered(
    tmp_path, clock: _Clock
):
    """AC9: normal, missing and corrupt legacy claim stamps all upgrade."""
    path = _legacy_control_db(tmp_path)
    long_ago = "2026-01-01T00:00:00+00:00"
    _insert_legacy_task(
        path,
        task_id="legacy-normal",
        status="claimed",
        evidence={"run_id": "run-normal", "workspace_path": "/ws/run-normal"},
        dedupe_key="cleanup:run-normal",
        claimed_by="janitor-dead",
        claimed_at=long_ago,
        attempts=1,
    )
    _insert_legacy_task(
        path,
        task_id="legacy-missing",
        status="claimed",
        evidence={"run_id": "run-missing"},
        dedupe_key="cleanup:run-missing",
        claimed_by="janitor-dead",
        claimed_at=None,
        attempts=3,
    )
    _insert_legacy_task(
        path,
        task_id="legacy-corrupt",
        status="claimed",
        evidence={"run_id": "run-corrupt"},
        dedupe_key="cleanup:run-corrupt",
        claimed_by="janitor-dead",
        claimed_at="yesterday o'clock",
        attempts=1,
    )
    # A legacy claim that is still inside the window must stay protected.
    live_at = datetime.fromisoformat(clock.iso()) - timedelta(seconds=60)
    _insert_legacy_task(
        path,
        task_id="legacy-live",
        status="claimed",
        evidence={"run_id": "run-live"},
        dedupe_key="cleanup:run-live",
        claimed_by="janitor-alive",
        claimed_at=live_at.isoformat(),
        attempts=1,
    )

    repo = _legacy_repo(tmp_path)
    try:
        # The upgrade stamps claimed_at + the fixed TTL...
        normal = repo.get_janitor_task("legacy-normal")
        assert normal is not None
        assert normal.claim_expires_at == (
            datetime.fromisoformat(long_ago) + timedelta(seconds=CLAIM_TTL_SECONDS)
        ).isoformat()
        # ...and treats a missing or unparsable stamp as ALREADY expired,
        # so the next sweep recovers it instead of hanging.
        for task_id, expected_attempts in (
            ("legacy-missing", 3),
            ("legacy-corrupt", 1),
        ):
            recovered = repo.get_janitor_task(task_id)
            assert recovered is not None
            assert recovered.claim_expires_at is not None
            assert clock.elapsed_since(recovered.claim_expires_at) > 0
            # Nothing else about the legacy row was rewritten.
            assert recovered.attempts == expected_attempts
            assert recovered.claimed_by == "janitor-dead"
        live = repo.get_janitor_task("legacy-live")
        assert live is not None
        assert live.claim_expires_at == (
            live_at + timedelta(seconds=CLAIM_TTL_SECONDS)
        ).isoformat()

        # The recovered work is REAL work: a sweep runs it to completion,
        # while the live claim is left to its own worker.
        processed: list[str] = []

        def cleanup(_runs_repo, evidence) -> None:
            processed.append(str(evidence["run_id"]))

        results = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-upgrade", cleanup=cleanup
        )
        assert sorted(r.task_id for r in results) == [
            "legacy-corrupt",
            "legacy-missing",
            "legacy-normal",
        ]
        assert [r.status for r in results] == ["done"] * 3
        assert sorted(processed) == ["run-corrupt", "run-missing", "run-normal"]
        still_claimed = repo.get_janitor_task("legacy-live")
        assert still_claimed is not None
        assert still_claimed.status == "claimed"
        assert still_claimed.claimed_by == "janitor-alive"

        # Evidence survived the upgrade untouched (only the key release on
        # completion added the audited dedupe key).
        done = repo.get_janitor_task("legacy-normal")
        assert done is not None
        assert done.evidence["workspace_path"] == "/ws/run-normal"
        assert done.evidence["dedupe_key"] == "cleanup:run-normal"
        assert done.attempts == 2
    finally:
        repo.close()


def test_legacy_done_rows_release_their_key_once_and_history_survives(tmp_path):
    """AC8: released keys + repeated upgrades never rewrite finished rows."""
    path = _legacy_control_db(tmp_path)
    evidence = {"run_id": "run-old", "workspace_path": "/ws/run-old", "note": "第一轮的清理"}
    _insert_legacy_task(
        path,
        task_id="legacy-done",
        status="done",
        evidence=evidence,
        dedupe_key="cleanup:run-old",
        claimed_by="janitor-old",
        claimed_at="2026-01-01T00:00:00+00:00",
        attempts=2,
        last_error="earlier retry failed",
        result_json=json.dumps({"cleanup_status": "succeeded"}, sort_keys=True),
        completed_at="2026-01-01T00:05:00+00:00",
    )

    first = _legacy_repo(tmp_path)
    try:
        released = first.get_janitor_task("legacy-done")
        assert released is not None
        assert released.status == "done"
        assert released.dedupe_key is None
        assert released.evidence["dedupe_key"] == "cleanup:run-old"
        # Original evidence, result, error and stamps are all preserved.
        assert released.evidence["workspace_path"] == "/ws/run-old"
        assert released.evidence["note"] == "第一轮的清理"
        assert released.result == {"cleanup_status": "succeeded"}
        assert released.last_error == "earlier retry failed"
        assert released.attempts == 2
        assert released.completed_at == "2026-01-01T00:05:00+00:00"
        snapshot = released.evidence_json

        # The freed key lets the next round open a NEW task_id.
        nxt, created = first.enqueue_janitor_task(
            kind="cleanup",
            run_id="run-old",
            evidence={"run_id": "run-old"},
            dedupe_key="cleanup:run-old",
        )
        assert created is True
        assert nxt.task_id != "legacy-done"
        assert _count_rows(first, "janitor_tasks") == 2
    finally:
        first.close()

    # Repeated starts must not touch the finished row again.
    for _pass in range(2):
        reopened = _legacy_repo(tmp_path)
        try:
            again = reopened.get_janitor_task("legacy-done")
            assert again is not None
            assert again.evidence_json == snapshot
            assert again.dedupe_key is None
            assert _count_rows(reopened, "janitor_tasks") == 2
            assert _count_rows(reopened, "janitor_tasks", "status = 'done'") == 1
        finally:
            reopened.close()


def test_interrupted_janitor_migration_is_safely_retryable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """AC9b: an interrupted pass leaves no half-stamped state to reason about.

    The real migration runs inside one IMMEDIATE transaction, so a worker
    that dies halfway writes NOTHING: the retry simply stamps the rows the
    first attempt never got to, and already-finished rows are never
    touched twice (they no longer match ``claim_expires_at IS NULL``).
    """
    path = _legacy_control_db(tmp_path)
    for task_id in ("legacy-a", "legacy-b", "legacy-c"):
        _insert_legacy_task(
            path,
            task_id=task_id,
            status="claimed",
            evidence={"run_id": task_id},
            claimed_at="2026-01-01T00:00:00+00:00",
        )

    original_deadline = persistence_module._janitor_legacy_deadline
    calls = {"n": 0}

    def crash_on_second_row(claimed_at: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("migration interrupted")
        return original_deadline(claimed_at)

    monkeypatch.setattr(
        persistence_module, "_janitor_legacy_deadline", crash_on_second_row
    )
    with pytest.raises(RuntimeError, match="migration interrupted"):
        ControlStore(path)
    monkeypatch.setattr(
        persistence_module, "_janitor_legacy_deadline", original_deadline
    )

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        partial = conn.execute(
            "SELECT task_id FROM janitor_tasks WHERE claim_expires_at IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    assert partial == [], "a rolled-back pass must not stamp anything"

    repo = _legacy_repo(tmp_path)
    try:
        expected = (
            datetime.fromisoformat("2026-01-01T00:00:00+00:00")
            + timedelta(seconds=CLAIM_TTL_SECONDS)
        ).isoformat()
        for task_id in ("legacy-a", "legacy-b", "legacy-c"):
            row = repo.get_janitor_task(task_id)
            assert row is not None
            assert row.claim_expires_at == expected
        # The retry is convergent: one candidate index, not a rebuilt one.
        assert (
            _count_rows(
                repo,
                "sqlite_master",
                "type = 'index' AND name = 'idx_janitor_claim_candidates'",
            )
            == 1
        )
    finally:
        repo.close()


def test_takeover_recovery_touches_only_the_original_workspace(
    tmp_path, clock: _Clock
):
    """AC10: recovery after a takeover reuses run/lease/workspace identity.

    The first worker DID the cleanup but lost its claim before settling.
    The take-over sweep must therefore NOT run the cleanup a second time
    (the durable lease already says succeeded), must not provision a
    workspace, and must not re-run the model or the effect.
    """
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-janitor-takeover")
        computer = _FakeComputer()

        def failing_cleanup(comp, workspace_path, runs, rid) -> None:
            raise RuntimeError("transient cleanup failure")

        orch = _orchestrator(repo, computer, cleanup=failing_cleanup)
        _resume(orch, run_id)
        task = repo.list_janitor_tasks(status="pending")[0]
        lease_before = repo.get_sandbox_lease(run_id)
        assert lease_before is not None

        handler_calls: list[str] = []

        def cleanup(runs_repo, evidence) -> None:
            # Do the ORIGINAL workspace's cleanup, then lose the claim the
            # way a crashed worker does: the work landed, the settlement
            # never did.
            lease = runs_repo.get_sandbox_lease(evidence["run_id"])
            assert lease is not None
            handler_calls.append(lease.workspace_path)
            computer.cleanup(lease.workspace_path)
            runs_repo.destroy_sandbox_lease(run_id=evidence["run_id"])
            runs_repo.record_sandbox_cleanup(evidence["run_id"], status="succeeded")
            clock.advance(CLAIM_TTL_SECONDS)
            open_claim = runs_repo.list_janitor_tasks(status="claimed", limit=10)[0]
            _ghost, stolen = runs_repo.claim_janitor_task(
                open_claim.task_id, owner="janitor-ghost"
            )
            assert stolen is True

        results = run_pending_janitor_tasks(
            repo,
            limit=10,
            claimer="janitor-first",
            cleanup=cleanup,
        )
        assert [r.status for r in results] == ["claim_lost"], results
        assert handler_calls == [f"/ws/{run_id}"]

        mid = repo.get_janitor_task(task.task_id)
        assert mid is not None
        assert mid.status == "claimed"
        assert mid.attempts == 2

        # The take-over sweep sees a succeeded lease: nothing is re-run.
        clock.advance(CLAIM_TTL_SECONDS)
        second = run_pending_janitor_tasks(
            repo, limit=10, claimer="janitor-second", cleanup=cleanup
        )
        assert [r.status for r in second] == ["done"], second
        assert handler_calls == [f"/ws/{run_id}"], "cleanup must not run twice"
        assert computer.cleaned == [f"/ws/{run_id}"]
        assert computer.workspaces == [], "no new sandbox may be provisioned"

        settled = repo.get_janitor_task(task.task_id)
        assert settled is not None
        assert settled.status == "done"
        assert settled.attempts == 3
        assert settled.claimed_by == "janitor-second"
        assert settled.result["already_cleaned"] is True

        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None
        assert lease.lease_id == lease_before.lease_id
        assert lease.workspace_path == f"/ws/{run_id}"
        assert lease.cleanup_status == "succeeded"
        # The run's terminal state and its single effect are untouched.
        run = repo.get_run(run_id)
        assert run is not None and run.status == "done"
        assert _count_rows(repo, "effects", f"run_id = '{run_id}'") == 1
    finally:
        repo.close()


def test_takeover_repair_never_executes_an_unknown_effect(
    tmp_path, clock: _Clock
):
    """AC10b: a reclaimed repair task still only reconciles, never executes."""
    repo = _repo(tmp_path)
    try:
        run_id, identity = _unknown_effect(repo, "req-janitor-takeover-repair")
        task, _ = repo.enqueue_janitor_task(
            kind="repair",
            run_id=run_id,
            effect_id=identity.effect_id,
            evidence={"run_id": run_id, "effect_id": identity.effect_id},
            dedupe_key=f"repair:{identity.effect_id}",
        )
        hung, won = repo.claim_janitor_task(task.task_id, owner="janitor-crashed")
        assert won is True and hung is not None

        clock.advance(CLAIM_TTL_SECONDS)
        results = run_pending_janitor_tasks(repo, limit=10, claimer="janitor-new")
        assert [r.status for r in results] == ["done"], results
        assert results[0].outcome["repair_status"] == "repair_claimed"
        assert results[0].outcome["next_action"] == "manual_confirm"

        effect = repo.get_effect(identity.effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert effect.result_json is None
        # Reconciled once by the takeover; no execution, no success frame.
        assert effect.repair_attempts == 1
        assert not [
            event
            for event in repo.get_events_since(run_id=run_id, from_seq=0)
            if event["type"] == "tool_result"
            and event["payload"].get("status") == "ok"
        ]
        settled = repo.get_janitor_task(task.task_id)
        assert settled is not None and settled.status == "done"
        assert settled.attempts == 2
    finally:
        repo.close()


# ---- cleanup status + janitor wiring ---------------------------------------


def test_cleanup_success_records_status_and_repeat_is_idempotent(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-clean-ok")
        computer = _FakeComputer()
        calls: list[int] = []

        def counting_cleanup(comp, workspace_path, runs, rid) -> None:
            calls.append(1)
            runs.destroy_sandbox_lease(run_id=rid)

        orch = _orchestrator(repo, computer, cleanup=counting_cleanup)
        items = _resume(orch, run_id)
        assert items[-1].type == "done"

        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None
        assert lease.status == "destroyed"
        assert lease.cleanup_status == "succeeded"
        assert repo.list_janitor_tasks() == []
        assert calls == [1]

        # A repeated terminal cleanup is a no-op (same workspace, no task).
        orch._run_terminal_cleanup(computer, f"/ws/{run_id}", repo, run_id)
        assert calls == [1]
        assert repo.list_janitor_tasks() == []
    finally:
        repo.close()


def test_cleanup_failure_creates_janitor_item_with_evidence(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-clean-fail")
        computer = _FakeComputer()

        def failing_cleanup(comp, workspace_path, runs, rid) -> None:
            raise RuntimeError("workspace locked")

        orch = _orchestrator(repo, computer, cleanup=failing_cleanup)
        items = _resume(orch, run_id)
        assert items[-1].type == "done"  # cleanup failure does not change the terminal

        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None
        assert lease.status == "failed"
        assert lease.cleanup_status == "pending_janitor"
        assert "workspace locked" in (lease.cleanup_error or "")

        tasks = repo.list_janitor_tasks(status="pending")
        assert len(tasks) == 1
        task = tasks[0]
        assert task.kind == "cleanup"
        assert task.run_id == run_id
        assert task.lease_id == lease.lease_id
        assert task.evidence["workspace_path"] == f"/ws/{run_id}"
        assert task.evidence["sandbox_type"] == "local_native"
        assert task.evidence["run_id"] == run_id
        assert task.evidence["lease_id"] == lease.lease_id
        assert task.evidence["effect_ids"] == [
            effect.resolved_effect_id for effect in repo.list_effects_for_run(run_id)
        ]
        assert lease.janitor_task_id == task.task_id

        # The durable task is the single work item for this run.
        again = _orchestrator(repo, computer, cleanup=failing_cleanup)
        again._run_terminal_cleanup(computer, f"/ws/{run_id}", repo, run_id)
        assert len(repo.list_janitor_tasks()) == 1
    finally:
        repo.close()


def test_janitor_cleanup_retry_reuses_original_workspace_only(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-janitor-clean")
        computer = _FakeComputer()

        def failing_cleanup(comp, workspace_path, runs, rid) -> None:
            raise RuntimeError("transient cleanup failure")

        orch = _orchestrator(repo, computer, cleanup=failing_cleanup)
        _resume(orch, run_id)
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None and lease.cleanup_status == "pending_janitor"

        results = run_pending_janitor_tasks(
            repo,
            limit=10,
            computer_factory=lambda *_a, **_k: computer,
        )
        assert len(results) == 1
        assert results[0].status == "done", results[0]
        assert results[0].kind == "cleanup"
        # The janitor cleaned the ORIGINAL workspace and never provisioned
        # a new sandbox to bypass the old lease.
        assert computer.cleaned == [f"/ws/{run_id}"]
        assert computer.workspaces == []
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None
        assert lease.cleanup_status == "succeeded"
        tasks = repo.list_janitor_tasks()
        assert len(tasks) == 1 and tasks[0].status == "done"

        # Idempotent: a second sweep does nothing.
        assert run_pending_janitor_tasks(
            repo, limit=10, computer_factory=lambda *_a, **_k: computer
        ) == []
        assert computer.cleaned == [f"/ws/{run_id}"]
    finally:
        repo.close()


def test_janitor_cleanup_failure_records_error_and_retries(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-janitor-retry")
        computer = _FakeComputer()
        failures = {"left": 1}

        def cleanup(runs, evidence) -> None:
            if failures["left"] > 0:
                failures["left"] -= 1
                raise RuntimeError("still locked")
            runs.destroy_sandbox_lease(run_id=evidence["run_id"])

        def failing_cleanup(comp, workspace_path, runs, rid) -> None:
            raise RuntimeError("first failure")

        orch = _orchestrator(repo, computer, cleanup=failing_cleanup)
        _resume(orch, run_id)

        first = run_pending_janitor_tasks(repo, limit=10, cleanup=cleanup)
        assert first[0].status == "failed"
        task = repo.list_janitor_tasks()[0]
        assert task.status == "failed"
        assert task.last_error and "still locked" in task.last_error
        assert task.attempts == 1

        second = run_pending_janitor_tasks(repo, limit=10, cleanup=cleanup)
        assert second[0].status == "done"
        task = repo.list_janitor_tasks()[0]
        assert task.status == "done"
        assert task.attempts == 2
        lease = repo.get_sandbox_lease(run_id)
        assert lease is not None and lease.cleanup_status == "succeeded"
    finally:
        repo.close()


# ---- janitor repair takeover ----------------------------------------------


def _unknown_effect(repo: RunRepository, request_id: str):
    """Drive one ledger row to ``outcome_unknown`` the way a crash does.

    The CAS ladder is ``authorized → executing/claimed → outcome_unknown``
    (ticket 04), so the fixture walks it instead of forcing the status.
    """
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="m",
    )
    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "p.pdf"}
    )
    repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "p.pdf"}',
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    _record, claimed = repo.transition_effect_to_executing(identity.effect_id)
    assert claimed is True
    repo.mark_effect_outcome_unknown(identity.effect_id, reason="crash after exec")
    return run_id, identity


def test_janitor_repair_claims_without_reexecuting_effect(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, identity = _unknown_effect(repo, "req-janitor-repair")
        task, _ = repo.enqueue_janitor_task(
            kind="repair",
            run_id=run_id,
            effect_id=identity.effect_id,
            evidence={"run_id": run_id, "effect_id": identity.effect_id},
            dedupe_key=f"repair:{identity.effect_id}",
        )

        results = run_pending_janitor_tasks(repo, limit=10)
        assert len(results) == 1
        assert results[0].status == "done"
        assert results[0].outcome["repair_status"] == "repair_claimed"

        effect = repo.get_effect(identity.effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.OUTCOME_UNKNOWN.value
        assert effect.repair_status == "repair_claimed"
        assert effect.repair_claimed_by.startswith("janitor:")
        assert repo.list_effects_for_run(run_id)[0].result_json is None
        # No replayable success was published by the takeover.
        assert not [
            evt
            for evt in repo.get_events_since(run_id=run_id, from_seq=0)
            if evt["type"] == "tool_result"
            and evt["payload"].get("status") == "ok"
        ]
    finally:
        repo.close()


def test_janitor_repair_auto_confirms_known_result(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, identity = _unknown_effect(repo, "req-janitor-known")
        # A publication failure leaves the external result in the ledger.
        repo.control._conn.execute(
            "UPDATE effects SET result_json = ? WHERE effect_key = ?",
            ('{"path":"p.pdf"}', identity.effect_id),
        )
        repo.enqueue_janitor_task(
            kind="repair",
            run_id=run_id,
            effect_id=identity.effect_id,
            evidence={"run_id": run_id, "effect_id": identity.effect_id},
            dedupe_key=f"repair:{identity.effect_id}",
        )

        results = run_pending_janitor_tasks(repo, limit=10)
        assert results[0].status == "done"
        assert results[0].outcome["repair_status"] == "repaired"

        effect = repo.get_effect(identity.effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.COMMITTED.value
        assert effect.repair_status == "repaired"
        successes = [
            evt
            for evt in repo.get_events_since(run_id=run_id, from_seq=0)
            if evt["type"] == "tool_result"
            and evt["payload"].get("effect_id") == identity.effect_id
            and evt["payload"].get("status") == "ok"
        ]
        assert len(successes) == 1
    finally:
        repo.close()


# ---- approval identity ------------------------------------------------------


def _pending_approval_run(repo: RunRepository, request_id: str) -> tuple[str, str]:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="m",
    )
    _seed_frozen_policy(repo, run_id, "artifact")
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type="local_native",
        workspace_path=f"/ws/{run_id}",
    )
    repo.mark_run_paused(run_id)
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "hiking-plan.pdf"}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "hiking-plan.pdf"}',
        risk_level="high",
        impact="生成计划",
        created_at=_now_iso(),
        expires_at=compute_expiry(_now_iso()),
    )
    return run_id, approval_id


def test_approval_duplicate_is_idempotent_and_conflict_is_stable(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, approval_id = _pending_approval_run(repo, "req-approve")
        computer = _FakeComputer()
        orch = _orchestrator(repo, computer)

        first = orch.resolve_approval_decision(
            approval_id, decision="approve", decision_request_id="dec-1"
        )
        assert first.status == "recorded"
        run_after_first = repo.get_run(run_id)
        assert run_after_first is not None and run_after_first.status == "running"

        duplicate = orch.resolve_approval_decision(
            approval_id, decision="approve", decision_request_id="dec-1"
        )
        assert duplicate.status == "idempotent"
        run_after_duplicate = repo.get_run(run_id)
        assert run_after_duplicate is not None
        assert run_after_duplicate.status == "running"
        assert run_after_duplicate.updated_at == run_after_first.updated_at

        conflict = orch.resolve_approval_decision(
            approval_id, decision="reject", decision_request_id="dec-1"
        )
        assert conflict.status == "decision_conflict"
        assert conflict.payload["code"] == "decision_conflict"
        assert conflict.payload["current_status"] == "approved"
        approval = repo.get_approval(approval_id)
        assert approval is not None and approval.status == "approved"

        different = orch.resolve_approval_decision(
            approval_id, decision="reject", decision_request_id="dec-2"
        )
        assert different.status == "already_resolved"
        approval = repo.get_approval(approval_id)
        assert approval is not None and approval.status == "approved"
        assert _count_rows(repo, "approvals", f"run_id = '{run_id}'") == 1
        assert _count_rows(repo, "sandbox_leases", f"run_id = '{run_id}'") == 1
    finally:
        repo.close()


def test_approval_resume_keeps_single_run_lease_effect_identity(tmp_path):
    repo = _repo(tmp_path)
    try:
        computer = _FakeComputer()
        kernel_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        orch = _orchestrator(repo, computer, kernel=kernel)
        items = _stream_chat(orch, "req-identity", "artifact")
        assert items[-1].type == "run.paused"

        run = repo.control.get_run_by_request("req-identity")
        assert run is not None
        lease_before = repo.get_sandbox_lease(run.run_id)
        assert lease_before is not None
        approval = repo.list_pending_approvals_for_run(run.run_id)[0]

        decision = orch.resolve_approval_decision(
            approval.approval_id,
            decision="approve",
            decision_request_id="dec-identity",
        )
        assert decision.status == "recorded"
        resumed = _resume(orch, run.run_id)
        assert resumed[-1].type == "done"

        # ONE run, lease, effect and approval identity — all advancing the
        # SAME paused run.
        assert _count_rows(repo, "runs", "request_id = 'req-identity'") == 1
        assert _count_rows(repo, "sandbox_leases") == 1
        assert _count_rows(repo, "effects") == 1
        assert _count_rows(repo, "approvals") == 1
        lease_after = repo.get_sandbox_lease(run.run_id)
        assert lease_after is not None
        assert lease_after.lease_id == lease_before.lease_id
        assert lease_after.fencing_token == lease_before.fencing_token
        effect = repo.list_effects_for_run(run.run_id)[0]
        assert effect.run_id == run.run_id
        assert effect.status == EffectStatus.COMMITTED.value
        assert kernel_calls == [1]
    finally:
        repo.close()


# ---- resume status / lease / policy revalidation ---------------------------


@pytest.mark.parametrize(
    "status",
    ["done", "error", "cancelled", "admission_failed"],
)
def test_resume_of_terminal_status_is_replay_only_without_work(tmp_path, status):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, f"req-term-{status}")
        repo.control._conn.execute(
            "UPDATE runs SET status = ?, terminal_type = ?, terminal_at = ? "
            "WHERE run_id = ?",
            (status, status, _now_iso(), run_id),
        )
        stored_before = len(repo.get_events_since(run_id=run_id, from_seq=0))
        factory_calls: list[int] = []
        computer = _FakeComputer()
        kernel_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )
        items = _resume(orch, run_id)
        types = [item.type for item in items]

        # Replay-only. A terminal run with no persisted terminal frame gets
        # the DETERMINISTIC synthetic terminal (ticket 03) — that is still
        # a read, not a continuation. What must never appear is any
        # continuation step of the approved effect.
        assert not {
            "tool_call", "tool_result", "approval.resolved",
            "artifact_candidate", "artifact.available", "text", "progress",
        } & set(types)
        assert factory_calls == []
        assert kernel_calls == []
        assert computer.pdfs == []
        assert computer.workspaces == []
        assert repo.list_effects_for_run(run_id) == []
        assert len(repo.get_events_since(run_id=run_id, from_seq=0)) == stored_before
    finally:
        repo.close()


def test_resume_rejects_outcome_unknown_status_before_work(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-unknown-status")
        repo.control._conn.execute(
            "UPDATE runs SET status = 'outcome_unknown' WHERE run_id = ?", (run_id,)
        )
        factory_calls: list[int] = []
        kernel_calls: list[int] = []
        computer = _FakeComputer()

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )
        items = _resume(orch, run_id)

        # BCDEF review S3: ``outcome_unknown`` is terminal in the shared
        # contract, so ``run.is_terminal`` now stops the resume at the
        # replay boundary and the stream ends with the replay's stable
        # synthetic terminal — the same ``effect_outcome_unknown`` code the
        # live effect driver and the snapshot use — instead of the generic
        # ``run_not_resumable`` that only the late ticket-06 revalidation
        # used to reach. Either way, NOTHING continues.
        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "effect_outcome_unknown"
        assert factory_calls == []
        assert kernel_calls == []
        assert repo.list_effects_for_run(run_id) == []
    finally:
        repo.close()


def test_resume_revalidates_lease_before_model_effect_or_sandbox(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-lease-check")
        repo.destroy_sandbox_lease(run_id=run_id, failure_reason="operator")
        factory_calls: list[int] = []
        kernel_calls: list[int] = []
        computer = _FakeComputer()

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )
        items = _resume(orch, run_id)

        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "sandbox_lease_lost"
        assert factory_calls == []
        assert kernel_calls == []
        assert computer.pdfs == []
        assert repo.list_effects_for_run(run_id) == []
    finally:
        repo.close()


def test_resume_rejects_expired_lease_without_new_generation(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(
            repo, "req-lease-expired2", expired_lease=True
        )
        lease_before = repo.get_sandbox_lease(run_id)
        assert lease_before is not None
        factory_calls: list[int] = []
        computer = _FakeComputer()
        orch = _orchestrator(repo, computer, factory_calls=factory_calls)

        items = _resume(orch, run_id)

        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "lease_expired"
        assert factory_calls == []
        lease_after = repo.get_sandbox_lease(run_id)
        assert lease_after is not None
        assert lease_after.status != "active"
        assert lease_after.fencing_token == lease_before.fencing_token
        assert repo.list_effects_for_run(run_id) == []
    finally:
        repo.close()


# ---- effect-claim revalidation blocks the continuation --------------------


def test_resume_with_unknown_effect_claim_does_no_sandbox_or_model_work(tmp_path):
    """A ledger row already ``outcome_unknown`` blocks the continuation.

    The claim ladder is the only door to the side effect (ticket 04/05):
    resume re-reads it BEFORE executing, so an unknown outcome can never
    be re-run by a repeated approval/resume — the model, the sandbox tool
    and the export all stay untouched.
    """
    repo = _repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, "req-effect-claim")
        identity = compute_effect_identity(
            run_id=run_id,
            tool_name="generate_pdf",
            args={"filename": "hiking-plan.pdf"},
        )
        repo.insert_effect(
            effect_key=identity.effect_id,
            run_id=run_id,
            conversation_id=f"conv-req-effect-claim",
            tool_name="generate_pdf",
            args_canonical='{"filename": "hiking-plan.pdf"}',
            external_idempotency_key=identity.idempotency_key,
            effect_id=identity.effect_id,
            idempotency_key=identity.idempotency_key,
        )
        _record, claimed = repo.transition_effect_to_executing(identity.effect_id)
        assert claimed is True
        repo.mark_effect_outcome_unknown(identity.effect_id, reason="crash before commit")

        computer = _FakeComputer()
        kernel_calls: list[int] = []
        factory_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )
        items = _resume(orch, run_id)

        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "effect_outcome_unknown"
        assert "effect.outcome_unknown" in [item.type for item in items]
        assert kernel_calls == []
        assert computer.pdfs == []
        assert computer.workspaces == []
        assert len(repo.list_effects_for_run(run_id)) == 1
        effect = repo.get_effect(identity.effect_id)
        assert effect is not None
        assert effect.status == EffectStatus.OUTCOME_UNKNOWN.value
        run = repo.get_run(run_id)
        assert run is not None and run.status == "error"
    finally:
        repo.close()


# ---- PDF resume must not re-run the model ----------------------------------


def test_pdf_resume_uses_persisted_plan_and_never_reruns_model(tmp_path):
    repo = _repo(tmp_path)
    try:
        computer = _FakeComputer()
        kernel_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 已经起草的计划", [])

        orch = _orchestrator(repo, computer, kernel=kernel)
        items = _stream_chat(orch, "req-no-model", "artifact")
        assert items[-1].type == "run.paused"
        assert kernel_calls == [1]  # drafted BEFORE the pause

        run = repo.control.get_run_by_request("req-no-model")
        assert run is not None
        approval = repo.list_pending_approvals_for_run(run.run_id)[0]
        assert approval.prepared_content == "# 已经起草的计划"
        orch.resolve_approval_decision(
            approval.approval_id,
            decision="approve",
            decision_request_id="dec-no-model",
        )

        resumed = _resume(orch, run.run_id)
        assert resumed[-1].type == "done"
        # The resume continued the approved effect from persisted facts.
        assert kernel_calls == [1]
        assert len(computer.pdfs) == 1
        _ws, content, _filename = computer.pdfs[0]
        assert content == "# 已经起草的计划"
        effect = repo.list_effects_for_run(run.run_id)[0]
        assert effect.status == EffectStatus.COMMITTED.value
    finally:
        repo.close()


def test_legacy_pdf_resume_without_prepared_plan_never_calls_model(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id, approval_id = _paused_pdf_run(
            repo, "req-legacy-no-model", prepared_content=None
        )
        computer = _FakeComputer()
        kernel_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 不应被调用", [])

        orch = _orchestrator(repo, computer, kernel=kernel)
        items = _resume(orch, run_id)

        assert items[-1].type == "done"
        assert kernel_calls == []
        assert len(computer.pdfs) == 1
        # Deterministic fallback: the persisted user turn, never a model.
        assert "规划杭州两日徒步" in computer.pdfs[0][1]
        approval = repo.get_approval(approval_id)
        assert approval is not None and approval.prepared_content is None
    finally:
        repo.close()


def test_duplicate_resume_and_cleanup_add_no_extra_side_effects(tmp_path):
    repo = _repo(tmp_path)
    try:
        computer = _FakeComputer()
        kernel_calls: list[int] = []
        cleanup_calls: list[int] = []
        exports: list[dict[str, Any]] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划", [])

        def cleanup(comp, workspace_path, runs, rid) -> None:
            cleanup_calls.append(1)
            runs.destroy_sandbox_lease(run_id=rid)

        orch = _orchestrator(
            repo, computer, kernel=kernel, cleanup=cleanup, export_calls=exports
        )
        _stream_chat(orch, "req-duplicate", "artifact")
        run = repo.control.get_run_by_request("req-duplicate")
        assert run is not None
        approval = repo.list_pending_approvals_for_run(run.run_id)[0]
        orch.resolve_approval_decision(
            approval.approval_id,
            decision="approve",
            decision_request_id="dec-duplicate",
        )

        first = _resume(orch, run.run_id)
        assert first[-1].type == "done"
        assert len(computer.pdfs) == 1
        assert cleanup_calls == [1]

        second = _resume(orch, run.run_id)
        third = _resume(orch, run.run_id, from_seq=1)

        assert second[-1].type == "done" and third[-1].type == "done"
        assert len(computer.pdfs) == 1
        assert cleanup_calls == [1]
        assert kernel_calls == [1]
        assert len(exports) == 1
        assert _count_rows(repo, "effects") == 1
        assert _count_rows(repo, "sandbox_leases") == 1
        assert len(repo.list_artifacts_for_run(run.run_id)) == 1
    finally:
        repo.close()
