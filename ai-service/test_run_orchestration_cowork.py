"""Ticket 38 — run orchestration module interface tests for the cowork
flow and the approval-decision state machine.

The orchestration interface is tested in-process: a fake kernel entry
and real SQLite persistence under a temp directory are injected through
the constructor. No ASGI shell, no module-global monkeypatching.

Behavior asserted (Ticket 38 AC):

* The cowork pause flow emits the same event sequence as before the
  fold: ``run_started → progress → heartbeat → approval.required →
  run.paused`` with ``run.paused`` as the single stream terminal.
* The planner kernel turn is tool-less and bound to the run identity.
* An unparsable command plan fails closed with ``bash_plan_invalid``.
* The decision state machine covers the four idempotency scenarios —
  approve, reject, expire, duplicate — plus the different-key 409 and
  the sandbox-lease fail-closed guard (spec AC5/AC7).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


TERMINALS = {"error", "done", "cancelled", "run.paused"}


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _settings(**overrides: Any) -> Any:
    from types import SimpleNamespace

    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "trusted",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeComputer:
    sandbox_type = "local_native"

    def __init__(self, *, creation_error: Exception | None = None) -> None:
        self.cleaned: list[str] = []
        self._creation_error = creation_error

    def create_workspace(self, run_id: str) -> str:
        if self._creation_error is not None:
            raise self._creation_error
        return f"/ws/{run_id}"

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _cowork_orchestrator(
    tmp_path: Path,
    kernel_calls: list[dict[str, Any]],
    *,
    plan: str = '{"command": "echo kernel-planned"}',
    computer: _FakeComputer | None = None,
    runs_repo: RunRepository | None = None,
) -> tuple[RunOrchestrator, list[str]]:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        return (plan, [])

    cleanups: list[Any] = []
    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=runs_repo if runs_repo is not None else _repo(tmp_path),
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: computer or _FakeComputer(),
        sandbox_cleanup=lambda *a, **k: cleanups.append(k),
    )
    return orch, cleanups


def _flow_req(request_id: str = "req-38") -> ChatFlowRequest:
    return ChatFlowRequest(
        message="在沙箱里创建一个文件",
        request_id=request_id,
        profile_id="cowork",
    )


def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req)]

    return asyncio.run(run())


def _paused_cowork_run(
    tmp_path: Path,
    *,
    request_id: str = "req-38",
    plan: str = '{"command": "echo kernel-planned"}',
) -> tuple[RunOrchestrator, RunRepository, list[Any], str, str]:
    """Drive one cowork stream to the paused state; return the pieces."""
    kernel_calls: list[dict[str, Any]] = []
    repo = _repo(tmp_path)
    orch, cleanups = _cowork_orchestrator(
        tmp_path, kernel_calls, plan=plan, runs_repo=repo
    )
    items = _stream(orch, _flow_req(request_id))
    approval_event = next(e for e in items if e.type == "approval.required")
    run = repo.control.get_run_by_request(request_id)
    assert run is not None
    return orch, repo, cleanups, approval_event.payload["approval_id"], run.run_id


# ---- cowork pause flow ------------------------------------------------------


def test_cowork_flow_pauses_on_high_risk_approval(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    repo = _repo(tmp_path)
    orch, cleanups = _cowork_orchestrator(
        tmp_path, kernel_calls, runs_repo=repo
    )

    items = _stream(orch, _flow_req())

    types = [e.type for e in items]
    assert types == [
        "run_started", "progress", "heartbeat", "approval.required", "run.paused",
    ]
    assert len([t for t in types if t in TERMINALS]) == 1

    # The planner turn is tool-less and bound to the run identity.
    assert len(kernel_calls) == 1
    assert kernel_calls[0]["extra_tools"] == ()
    assert kernel_calls[0]["kernel_gate"] is None
    assert "cowork 命令规划器" in kernel_calls[0]["system_prompt"]

    approval = next(e for e in items if e.type == "approval.required")
    assert approval.payload["args"]["command"] == "echo kernel-planned"
    assert approval.payload["risk_level"] == "high"

    # Durable state: approval row + paused run + active sandbox lease.
    approval_id = approval.payload["approval_id"]
    stored = repo.get_approval(approval_id)
    assert stored is not None and stored.status == "pending"
    run = repo.control.get_run_by_request("req-38")
    assert run is not None and run.status == "paused"
    lease = repo.get_sandbox_lease(run.run_id)
    assert lease is not None and lease.status == "active"

    # Paused runs keep the sandbox: no cleanup on the pause path.
    assert cleanups == []


def test_cowork_invalid_command_plan_fails_closed(tmp_path: Path):
    kernel_calls: list[dict[str, Any]] = []
    repo = _repo(tmp_path)
    orch, cleanups = _cowork_orchestrator(
        tmp_path, kernel_calls, plan="not a json blob", runs_repo=repo
    )

    items = _stream(orch, _flow_req("req-38bad"))

    types = [e.type for e in items]
    assert types[-1] == "error"
    assert items[-1].payload["code"] == "bash_plan_invalid"
    assert "approval.required" not in types
    # Behavior-unchanged note (Ticket 38): the plan-parse failure path
    # returns BEFORE the cleanup ``finally`` block is entered, exactly as
    # the pre-fold adapter code did — no cleanup happens here.
    assert cleanups == []
    run = repo.control.get_run_by_request("req-38bad")
    assert run is not None and run.status == "error"


def test_cowork_sandbox_creation_failure_is_terminal(tmp_path: Path):
    from agent_v2.computer import SandboxCreationError

    kernel_calls: list[dict[str, Any]] = []
    repo = _repo(tmp_path)
    broken = _FakeComputer(creation_error=SandboxCreationError("no disk"))
    orch, cleanups = _cowork_orchestrator(
        tmp_path, kernel_calls, computer=broken, runs_repo=repo
    )

    items = _stream(orch, _flow_req("req-38sb"))

    types = [e.type for e in items]
    assert types[-1] == "error"
    assert items[-1].payload["code"] == "sandbox_creation_failed"
    assert kernel_calls == []  # never reached the planner


# ---- approval decision state machine (four idempotency scenarios) ----------


def test_decision_approve_is_recorded_and_resumes_run(tmp_path: Path):
    orch, repo, _, approval_id, run_id = _paused_cowork_run(tmp_path)

    outcome = orch.resolve_approval_decision(
        approval_id, decision="approve", decision_request_id="dec-1"
    )

    assert outcome.status == "recorded"
    assert outcome.payload["decision"] == "approve"
    assert outcome.payload["idempotent"] is False
    assert "resume_url" not in outcome.payload  # adapter concern

    approval = repo.get_approval(approval_id)
    assert approval is not None
    assert approval.status == "approved"
    assert approval.decision_request_id == "dec-1"
    run = repo.get_run(run_id)
    assert run is not None and run.status == "running"


def test_decision_reject_is_recorded(tmp_path: Path):
    orch, repo, _, approval_id, run_id = _paused_cowork_run(tmp_path)

    outcome = orch.resolve_approval_decision(
        approval_id, decision="reject", decision_request_id="dec-r1"
    )

    assert outcome.status == "recorded"
    assert outcome.payload["decision"] == "reject"
    approval = repo.get_approval(approval_id)
    assert approval is not None and approval.status == "rejected"
    run = repo.get_run(run_id)
    assert run is not None and run.status == "running"


def test_duplicate_decision_same_request_id_is_idempotent(tmp_path: Path):
    orch, repo, _, approval_id, run_id = _paused_cowork_run(tmp_path)

    first = orch.resolve_approval_decision(
        approval_id, decision="approve", decision_request_id="dec-1"
    )
    second = orch.resolve_approval_decision(
        approval_id, decision="approve", decision_request_id="dec-1"
    )

    assert first.status == "recorded"
    assert second.status == "idempotent"
    assert second.payload["idempotent"] is True
    assert second.payload["status"] == "approved"


def test_decision_with_different_request_id_on_resolved_conflicts(tmp_path: Path):
    orch, repo, _, approval_id, _ = _paused_cowork_run(tmp_path)

    orch.resolve_approval_decision(
        approval_id, decision="approve", decision_request_id="dec-1"
    )
    conflict = orch.resolve_approval_decision(
        approval_id, decision="reject", decision_request_id="dec-2"
    )

    assert conflict.status == "already_resolved"
    assert conflict.payload["reason"] == "already_resolved"
    assert conflict.payload["current_status"] == "approved"


def test_expired_approval_returns_expired_and_marks_row(tmp_path: Path):
    orch, repo, _, _approval_id, run_id = _paused_cowork_run(tmp_path)

    # A second, already-expired approval on the same paused run.
    expired_id = "appr-expired-38"
    past = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    repo.insert_approval(
        approval_id=expired_id,
        run_id=run_id,
        conversation_id="conv-38",
        tool_name="sandbox_bash",
        args_canonical='{"command": "echo hi"}',
        risk_level="high",
        impact="test",
        created_at=past,
        expires_at=past,
    )

    outcome = orch.resolve_approval_decision(
        expired_id, decision="approve", decision_request_id="dec-x"
    )

    assert outcome.status == "expired"
    assert outcome.payload["reason"] == "expired"
    row = repo.get_approval(expired_id)
    assert row is not None and row.status == "expired"


def test_decision_fails_closed_when_sandbox_lease_lost(tmp_path: Path):
    orch, repo, _, approval_id, run_id = _paused_cowork_run(tmp_path)

    repo.destroy_sandbox_lease(run_id=run_id, failure_reason="operator sweep")

    outcome = orch.resolve_approval_decision(
        approval_id, decision="approve", decision_request_id="dec-1"
    )

    assert outcome.status == "sandbox_lease_lost"
    assert outcome.payload["reason"] == "sandbox_lease_lost"
    # No decision was persisted.
    approval = repo.get_approval(approval_id)
    assert approval is not None and approval.status == "pending"


def test_decision_unknown_approval_is_not_found(tmp_path: Path):
    orch, _repo, _, _approval_id, _run_id = _paused_cowork_run(tmp_path)

    outcome = orch.resolve_approval_decision(
        "no-such-approval", decision="approve", decision_request_id="dec-1"
    )

    assert outcome.status == "approval_not_found"
