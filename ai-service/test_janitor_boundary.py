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
* duplicate approval/resume/cleanup add no extra side effects.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.effect_repair import repair_effect
from agent_v2.effects import EffectStatus, compute_effect_identity
from agent_v2.execution_contract import policy_hash
from agent_v2.execution_policy import build_execution_policy
from agent_v2.janitor import run_pending_janitor_tasks
from agent_v2.persistence import CheckpointsStore, ControlStore, PolicySnapshotWrite
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


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
            task.task_id, owner="janitor-a", result={"cleaned": True}
        )
        assert completed is True
        assert done is not None
        assert done.status == "done"
        assert done.result["cleaned"] is True
        assert done.completed_at is not None

        _dup, completed_again = repo.complete_janitor_task(
            task.task_id, owner="janitor-a"
        )
        assert completed_again is False

        failed_task, _ = repo.enqueue_janitor_task(
            kind="repair", run_id="run-cas", effect_id="eff-1", evidence={}
        )
        repo.claim_janitor_task(failed_task.task_id, owner="janitor-a")
        failed, failed_won = repo.fail_janitor_task(
            failed_task.task_id, "handler boom", owner="janitor-a"
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
            failed_task.task_id, owner="janitor-b"
        )
        assert completed_claimed is True
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
