"""Ticket 40 — run orchestration module interface tests for the resume
path: replay, terminal-run replay-only, sandbox re-execution, and the
shared effect ledger under concurrent resume.

The orchestration interface is tested in-process: a fake kernel entry,
a fake computer, and real SQLite persistence under a temp directory are
injected through the constructor. No ASGI shell, no module-global
monkeypatching.

Behavior asserted (Ticket 40 AC):

* The resume stream is driven end-to-end by the orchestration module —
  stored events replay with their ORIGINAL sequence numbers, and a
  terminal run gets replay only (no new events, no re-execution).
* The approved ``generate_pdf`` resume executes the tool through the
  injected computer factory, exports via the injected host hook, and
  ends with ``done`` + DB ``done`` + sandbox cleanup (AC4/AC6).
* A lost sandbox lease fails closed (AC7) — no new sandbox.
* A rejected decision emits ``approval.resolved`` + ``done`` without
  executing the tool (spec AC5).
* Concurrent resume: an effect left ``executing`` by a live executor
  (recent persisted ``tool_call``) returns ``effect_in_flight`` and the
  sandbox is PRESERVED (no cleanup) — the double-click mutual-delete
  fix, now asserted at the orchestration interface.
* A crashed ``executing`` effect (no live executor) becomes
  ``outcome_unknown`` and the run terminates — no auto re-execution.
* The cowork ``sandbox_bash`` resume re-executes the approved command
  through the SAME ledger and completes with ``done``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from agent_v2.effects import EffectStatus, compute_effect_key
from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository


TERMINALS = {"error", "done", "cancelled", "run.paused"}


def _repo(tmp_path) -> RunRepository:
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
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeComputer:
    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.cleaned: list[str] = []
        self.pdfs: list[tuple[str, str, str]] = []
        self.bash_calls: list[tuple[str, str]] = []

    def create_workspace(self, run_id: str) -> str:
        return f"/ws/{run_id}"

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        return filename

    def run_bash(self, workspace_path, command, *, timeout_seconds=20.0, max_output_bytes=1_048_576):
        self.bash_calls.append((workspace_path, command))
        from types import SimpleNamespace

        return SimpleNamespace(
            exit_code=0, stdout="hello", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _resume_orchestrator(
    repo: RunRepository,
    computer: _FakeComputer,
    cleanups: list[Any],
    export_calls: list[dict[str, Any]] | None = None,
    kernel_turn=None,
):
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return ("# 计划", [])

    def fake_export(**kwargs: Any) -> dict[str, Any]:
        if export_calls is not None:
            export_calls.append(kwargs)
        return {
            "artifact_id": "art-1",
            "public_filename": "hiking-plan.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 10,
            "sha256": "deadbeef",
            "expires_at": "2026-09-03T00:00:00+00:00",
        }

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        kernel_turn=kernel_turn or fake_kernel,
        artifact_export=fake_export,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=lambda *a, **k: cleanups.append(k),
        gate_factory=None,  # real gate → generate_pdf is HIGH risk → pause
    )


def _run_pause_flow(orch: RunOrchestrator, repo: RunRepository, request_id: str) -> str:
    """Drive the artifact chat flow to the durable approval pause."""
    req = ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id=request_id,
        profile_id="artifact",
    )

    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req)]

    items = asyncio.run(run())
    types = [e.type for e in items]
    assert types[-1] == "run.paused", types
    run = repo.control.get_run_by_request(request_id)
    assert run is not None and run.status == "paused"
    approval = repo.list_pending_approvals_for_run(run.run_id)[0]
    return run.run_id, approval.approval_id


def _stream_resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)]

    return asyncio.run(run())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- replay semantics --------------------------------------------------------


def test_resume_of_terminal_run_replays_original_sequence_without_new_events(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    orch = _resume_orchestrator(repo, computer, cleanups)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40term")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40term"
    )
    # Complete the run once (full resume), then resume AGAIN.
    first = _stream_resume(orch, run_id)
    assert [e.type for e in first][-1] == "done"
    stored = repo.get_events_since(run_id=run_id, from_seq=0)
    stored_seqs = [evt["sequence"] for evt in stored]
    assert stored_seqs == sorted(stored_seqs)

    second = _stream_resume(orch, run_id, from_seq=2)

    # Replay only: exactly the stored events > from_seq, same sequences,
    # nothing new, terminal included.
    assert [e.sequence for e in second] == [s for s in stored_seqs if s > 2]
    assert second[-1].type == "done"
    assert len(second) == len([s for s in stored_seqs if s > 2])
    # No re-execution, no extra cleanup.
    assert len(computer.pdfs) == 1
    assert len(cleanups) == 1


# ---- approved generate_pdf resume ---------------------------------------------


def test_approved_pdf_resume_executes_and_completes_with_cleanup(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    export_calls: list[dict[str, Any]] = []
    orch = _resume_orchestrator(repo, computer, cleanups, export_calls)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40ok")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40ok"
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    # Replay of the pause stream, then the continuation.
    assert types[:5] == [
        "run_started", "progress", "heartbeat", "approval.required", "run.paused",
    ]
    assert types[5:] == [
        "approval.resolved", "tool_call", "tool_result",
        "artifact_candidate", "artifact.available", "text", "done",
    ]
    # The replay carries the pause terminal; the continuation adds
    # exactly one terminal of its own (``done``).
    assert len([t for t in types[5:] if t in TERMINALS]) == 1

    # The tool executed with the EXACT approved args inside the leased
    # workspace; the export ran on the host; the sandbox was cleaned up.
    assert len(computer.pdfs) == 1
    ws, content, filename = computer.pdfs[0]
    assert ws == "/ws/" + run_id and filename == "hiking-plan.pdf"
    assert content == "# 计划"  # Render the generated plan, never the original request.
    assert export_calls[0]["workspace_path"] == ws
    assert len(cleanups) == 1
    run = repo.get_run(run_id)
    assert run is not None and run.status == "done"
    effect = repo.list_effects_for_run(run_id)[0]
    assert effect.status == EffectStatus.COMMITTED.value


def test_plan_generation_failure_does_not_export_request_text_or_repeat_model(tmp_path):
    from agent_v2.kernel import KernelModelFailure

    calls = []
    async def failing_kernel(**kwargs):
        calls.append(kwargs)
        raise KernelModelFailure("model_failed", "模型失败")

    repo = _repo(tmp_path)
    computer = _FakeComputer()
    orch = _resume_orchestrator(repo, computer, [], kernel_turn=failing_kernel)
    run_id, approval_id = _run_pause_flow(orch, repo, "pdf-model-failed")
    repo.resolve_approval(approval_id=approval_id, status="approved", decision_request_id="decision-failed")
    first = _stream_resume(orch, run_id)
    assert first[-1].type == "error"
    assert computer.pdfs == []
    assert repo.list_effects_for_run(run_id)[0].status == EffectStatus.FAILED.value
    assert _stream_resume(orch, run_id)[-1].type == "error"
    assert len(calls) == 1


def test_resume_with_lost_sandbox_lease_fails_closed(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    orch = _resume_orchestrator(repo, computer, cleanups)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40lost")
    repo.destroy_sandbox_lease(run_id=run_id, failure_reason="test destroyed")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40lost"
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    assert types[5:] == ["approval.resolved", "error"]
    err = items[-1]
    assert err.payload["code"] == "sandbox_lease_lost"
    # Fail closed: the tool NEVER ran, no new sandbox was created.
    assert computer.pdfs == []
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"


def test_rejected_resume_emits_resolved_and_done_without_execution(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    orch = _resume_orchestrator(repo, computer, cleanups)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40rej")
    repo.resolve_approval(
        approval_id=approval_id, status="rejected", decision_request_id="dec-40rej"
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    assert types[5:] == ["approval.resolved", "done"]
    resolved = items[5]
    assert resolved.payload["decision"] == "reject"
    assert computer.pdfs == []
    run = repo.get_run(run_id)
    assert run is not None and run.status == "done"


# ---- concurrent resume: sandbox preservation ----------------------------------


def test_concurrent_resume_preserves_sandbox_while_executor_is_live(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    orch = _resume_orchestrator(repo, computer, cleanups)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40conc")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40conc"
    )

    # Simulate the concurrent CAS winner: an effect left EXECUTING plus
    # a recent persisted tool_call proving the executor is live. The args
    # are taken from the approval row itself — the same derivation the
    # resume flow makes (Ticket 09: the canonical args now carry the
    # plan_revision discriminator) — so the simulated winner addresses the
    # effect this approval will re-derive.
    approved_args = json.loads(repo.get_approval(approval_id).args_canonical)
    effect_key = compute_effect_key(
        run_id=run_id, tool_name="generate_pdf", args=approved_args
    )
    repo.insert_effect(
        effect_key=effect_key,
        run_id=run_id,
        conversation_id="conv",
        tool_name="generate_pdf",
        args_canonical=json.dumps(approved_args, sort_keys=True, separators=(",", ":")),
        external_idempotency_key=f"ext-{effect_key}",
    )
    _record, won = repo.transition_effect_to_executing(effect_key)
    assert won
    repo.persist_event(
        run_id=run_id,
        sequence=100,
        event_type="tool_call",
        payload={"tool": "generate_pdf", "args": approved_args},
        request_id="req-40conc",
        conversation_id="conv",
        timestamp=_now_iso(),
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    # The replay includes the winner's tool_call (seq 100); the
    # continuation is approval.resolved then the in-flight error.
    assert types[-2:] == ["approval.resolved", "error"]
    assert items[-1].payload["code"] == "effect_in_flight"
    # The sandbox is PRESERVED — the live winner may still commit. No
    # cleanup ran and the effect is untouched.
    assert cleanups == []
    assert computer.pdfs == []
    effect = repo.get_effect(effect_key)
    assert effect is not None and effect.status == EffectStatus.EXECUTING.value


def test_crashed_executing_effect_becomes_outcome_unknown_on_resume(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    orch = _resume_orchestrator(repo, computer, cleanups)
    run_id, approval_id = _run_pause_flow(orch, repo, "req-40crash")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40crash"
    )

    # Crash between CAS and commit: EXECUTING effect, NO live tool_call.
    # Args come from the approval row (see the concurrent-resume test above
    # for why — Ticket 09's plan_revision rides in args_canonical).
    approved_args = json.loads(repo.get_approval(approval_id).args_canonical)
    effect_key = compute_effect_key(
        run_id=run_id, tool_name="generate_pdf", args=approved_args
    )
    repo.insert_effect(
        effect_key=effect_key,
        run_id=run_id,
        conversation_id="conv",
        tool_name="generate_pdf",
        args_canonical=json.dumps(approved_args, sort_keys=True, separators=(",", ":")),
        external_idempotency_key=f"ext-{effect_key}",
    )
    _record, won = repo.transition_effect_to_executing(effect_key)
    assert won

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    assert types[5:] == ["approval.resolved", "effect.outcome_unknown", "error"]
    assert computer.pdfs == []  # never re-executed
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"
    effect = repo.get_effect(effect_key)
    assert effect is not None and effect.status == EffectStatus.OUTCOME_UNKNOWN.value


# ---- cowork sandbox_bash resume -------------------------------------------------


def test_cowork_bash_resume_executes_approved_command(tmp_path):
    repo = _repo(tmp_path)
    computer = _FakeComputer()
    cleanups: list[Any] = []
    export_calls: list[dict[str, Any]] = []

    def fake_export(**kwargs: Any) -> dict[str, Any]:
        export_calls.append(kwargs)
        return {"artifact_id": "art-bash", "public_filename": "cowork-output.txt"}

    # Hand-assemble the paused-cowork state: run + active lease +
    # resolved approval for sandbox_bash (the planner pause flow is
    # covered by the Ticket 38 tests).
    run_id, _created = repo.create_run_idempotent(
        request_id="req-40bash",
        conversation_id="conv-40bash",
        profile_id="cowork",
        user_message="在沙箱里创建一个文件",
    )
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type=computer.sandbox_type,
        workspace_path=f"/ws/{run_id}",
    )
    expires = compute_expiry(_now_iso())
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="sandbox_bash", args={"command": "echo hello"}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id="conv-40bash",
        tool_name="sandbox_bash",
        args_canonical='{"command": "echo hello"}',
        risk_level="high",
        impact="sandbox bash",
        created_at=_now_iso(),
        expires_at=expires,
    )
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-40bash"
    )

    orch = RunOrchestrator(
        settings=_settings(agent_v2_cowork_policy="trusted"),
        runs_repo=repo,
        artifact_export=fake_export,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=lambda *a, **k: cleanups.append(k),
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    assert types == [
        "approval.resolved", "tool_call", "tool_result",
        "artifact_candidate", "artifact.available", "text", "done",
    ]
    call = items[1]
    assert call.payload["tool"] == "sandbox_bash"
    assert call.payload["args"] == {"command": "echo hello"}
    # The approved command ran inside the leased workspace; the bash
    # output surfaced in the text response; the effect committed.
    assert computer.bash_calls == [(f"/ws/{run_id}", "echo hello")]
    assert "hello" in items[5].payload["content"]
    effect = repo.list_effects_for_run(run_id)[0]
    assert effect.status == EffectStatus.COMMITTED.value
    run = repo.get_run(run_id)
    assert run is not None and run.status == "done"
    assert len(cleanups) == 1
