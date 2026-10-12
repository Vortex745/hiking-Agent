"""Ticket 39 — run orchestration module interface tests for the artifact
flow and the unified effect CAS ledger.

The orchestration interface is tested in-process: a fake kernel entry,
a fake computer, and real SQLite persistence under a temp directory are
injected through the constructor. No ASGI shell, no module-global
monkeypatching.

Behavior asserted (Ticket 39 AC):

* The artifact flow is driven end-to-end by the orchestration module —
  the same event sequence as before the fold, with the sandbox created,
  exported, and cleaned up through injected hooks.
* The artifact flow pauses on the HIGH-risk ``generate_pdf`` approval.
* The effect CAS ledger is the module's SINGLE implementation
  (:func:`drive_effect`): an effect found ``executing`` after a crash
  transitions to ``outcome_unknown``, the run terminates with a
  terminal error, and NO automatic re-execution happens (human review
  required) — including on a retry. A live concurrent executor is left
  untouched (CAS-loser semantics).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_v2.effects import EffectStatus, compute_effect_key
from agent_v2.permissions import PermissionDecision, PermissionVerdict, RiskLevel
from agent_v2.run_orchestration import (
    EFFECT_COPY_GENERATE_PDF,
    EffectDriveResult,
    RunOrchestrator,
    drive_effect,
)
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

    def create_workspace(self, run_id: str) -> str:
        return f"/ws/{run_id}"

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        return filename

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _allow_gate_factory():
    class _AllowGate:
        def evaluate(self, profile, tool_name, args=None):
            return PermissionVerdict(
                decision=PermissionDecision.ALLOW,
                risk=RiskLevel.LOW,
                reason="allow",
            )

    return _AllowGate


def _artifact_orchestrator(
    tmp_path,
    *,
    kernel_calls: list[dict[str, Any]] | None = None,
    export_calls: list[dict[str, Any]] | None = None,
    computer: _FakeComputer | None = None,
    runs_repo: RunRepository | None = None,
    plan: str = "# 徒步计划\n内核生成",
):
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        if kernel_calls is not None:
            kernel_calls.append(kwargs)
        return (plan, [])

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

    comp = computer if computer is not None else _FakeComputer()
    cleanups: list[Any] = []
    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=runs_repo if runs_repo is not None else _repo(tmp_path),
        kernel_turn=fake_kernel,
        artifact_export=fake_export,
        computer_factory=lambda *a, **k: comp,
        sandbox_cleanup=lambda *a, **k: cleanups.append(k),
        gate_factory=_allow_gate_factory(),
    )
    return orch, comp, cleanups


def _artifact_req(request_id: str = "req-39") -> Any:
    from agent_v2.run_orchestration import ChatFlowRequest

    return ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id=request_id,
        profile_id="artifact",
    )


def _stream(orch: RunOrchestrator, req: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req)]

    return asyncio.run(run())


# ---- artifact flow (module-driven) ------------------------------------------


def test_artifact_flow_is_driven_end_to_end_by_the_module(tmp_path):
    kernel_calls: list[dict[str, Any]] = []
    export_calls: list[dict[str, Any]] = []
    repo = _repo(tmp_path)
    orch, computer, cleanups = _artifact_orchestrator(
        tmp_path,
        kernel_calls=kernel_calls,
        export_calls=export_calls,
        runs_repo=repo,
    )

    items = _stream(orch, _artifact_req())

    types = [e.type for e in items]
    assert types == [
        "run_started", "progress", "heartbeat",
        "tool_call", "tool_result", "artifact_candidate",
        "artifact.available", "text", "done",
    ]
    assert len([t for t in types if t in TERMINALS]) == 1

    # The kernel turn receives the plan prompt, the run identity, and the
    # sandbox tools bridged over the product Computer.
    assert len(kernel_calls) == 1
    assert "徒步计划" in kernel_calls[0]["system_prompt"]
    assert len(kernel_calls[0]["extra_tools"]) == 2
    assert kernel_calls[0]["computer"] is not None

    call = next(e for e in items if e.type == "tool_call")
    assert call.payload["tool"] == "generate_pdf"
    # Ticket 09 AC3: the canonical args carry the plan-book revision
    # discriminator next to the unchanged filename.
    assert call.payload["args"] == {"filename": "hiking-plan.pdf", "plan_revision": 1}

    # The export hook runs on the host with the sandbox binding, and the
    # artifact.available payload carries only public metadata.
    assert len(export_calls) == 1
    assert export_calls[0]["computer"] is computer
    assert export_calls[0]["relative_path"] == "hiking-plan.pdf"
    available = next(e for e in items if e.type == "artifact.available")
    assert available.payload["artifact_id"] == "art-1"
    assert "size_bytes" in available.payload
    assert "workspace_path" not in available.payload

    # Cleanup runs on completion; the run is done in the DB.
    assert cleanups == [{"files_deleted": 0}] or len(cleanups) == 1
    run = repo.control.get_run_by_request("req-39")
    assert run is not None and run.status == "done"


def test_artifact_flow_pauses_on_high_risk_approval(tmp_path):
    repo = _repo(tmp_path)
    # No gate_factory override → the real product PermissionGate marks
    # generate_pdf HIGH risk → the durable-approval pause.
    orch, computer, cleanups = _artifact_orchestrator(tmp_path, runs_repo=repo)
    orch._gate_factory = None  # exercise the default gate path

    items = _stream(orch, _artifact_req("req-39pause"))

    types = [e.type for e in items]
    assert types == [
        "run_started", "progress", "heartbeat", "approval.required", "run.paused",
    ]
    approval = next(e for e in items if e.type == "approval.required")
    stored = repo.get_approval(approval.payload["approval_id"])
    assert stored is not None and stored.status == "pending"
    run = repo.control.get_run_by_request("req-39pause")
    assert run is not None and run.status == "paused"
    lease = repo.get_sandbox_lease(run.run_id)
    assert lease is not None and lease.status == "active"
    # Paused runs keep the sandbox: no cleanup on the pause path.
    assert len(cleanups) == 0


# ---- unified effect CAS ledger (single implementation) -----------------------


def _collect_emit(events: list[tuple[str, dict[str, Any]]]):
    def emit(event_type: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        events.append((event_type, payload))
        return (event_type, payload)

    return emit


def _run_with_executing_effect(tmp_path, tool_name: str):
    repo = _repo(tmp_path)
    run_id, _created = repo.create_run_idempotent(
        request_id="req-39eff",
        conversation_id="conv-39",
        profile_id="artifact",
        user_message="hi",
    )
    args = {"filename": "hiking-plan.pdf"} if tool_name == "generate_pdf" else {"command": "ls"}
    effect_key = compute_effect_key(
        run_id=run_id, tool_name=tool_name, args=args
    )
    repo.insert_effect(
        effect_key=effect_key,
        run_id=run_id,
        conversation_id="conv-39",
        tool_name=tool_name,
        args_canonical='{"filename": "hiking-plan.pdf"}'
        if tool_name == "generate_pdf"
        else '{"command": "ls"}',
        external_idempotency_key=f"ext-{effect_key}",
    )
    # Crash between CAS and commit: the effect is left EXECUTING.
    _record, won = repo.transition_effect_to_executing(effect_key)
    assert won
    return repo, run_id, effect_key


def _drive(
    repo: RunRepository,
    run_id: str,
    effect_key: str,
    *,
    executed: list[int],
    all_events: list[dict[str, Any]] | None = None,
    tool_args: dict[str, Any] | None = None,
) -> tuple[EffectDriveResult, list[tuple[str, dict[str, Any]]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    # Mirror production derivation: per-tool flags from the event history.
    has_tool_call = any(evt.get("type") == "tool_call" for evt in (all_events or []))
    has_tool_result_ok = any(
        evt.get("type") == "tool_result"
        and evt.get("payload", {}).get("tool") == "generate_pdf"
        and evt.get("payload", {}).get("status") == "ok"
        for evt in (all_events or [])
    )

    def execute() -> dict[str, Any]:
        executed.append(1)
        return {"path": "hiking-plan.pdf", "content": "已生成"}

    outcome = EffectDriveResult()

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=repo,
            run_id=run_id,
            conversation_id="conv-39",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=effect_key,
            external_idempotency_key=f"ext-{effect_key}",
            args_canonical='{"filename": "hiking-plan.pdf"}',
            tool_args=tool_args or {"filename": "hiking-plan.pdf"},
            all_events=all_events or [],
            has_tool_call=has_tool_call,
            has_tool_result_ok=has_tool_result_ok,
            execute=execute,
            emit=_collect_emit(events),
            outcome=outcome,
        ):
            pass

    asyncio.run(run())
    return outcome, events


def test_executing_effect_after_crash_becomes_outcome_unknown_and_never_replays(tmp_path):
    repo, run_id, effect_key = _run_with_executing_effect(tmp_path, "generate_pdf")
    executed: list[int] = []

    outcome, events = _drive(repo, run_id, effect_key, executed=executed)

    # Crash semantics: the side effect may or may not have completed —
    # outcome_unknown + terminal error, NO re-execution (spec AC3).
    assert executed == []
    assert outcome.status == "outcome_unknown"
    assert [t for t, _ in events] == ["effect.outcome_unknown", "error"]
    assert events[0][1]["effect_key"] == effect_key
    assert events[1][1]["code"] == "effect_outcome_unknown"
    effect = repo.get_effect(effect_key)
    assert effect is not None and effect.status == EffectStatus.OUTCOME_UNKNOWN.value
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"

    # A retry must NOT auto-replay either — the row is terminal and
    # requires human review.
    outcome2, events2 = _drive(repo, run_id, effect_key, executed=executed)
    assert executed == []
    assert outcome2.status == "outcome_unknown"
    assert [t for t, _ in events2] == ["effect.outcome_unknown", "error"]


def test_executing_effect_with_live_executor_is_left_alone(tmp_path):
    repo, run_id, effect_key = _run_with_executing_effect(tmp_path, "generate_pdf")
    executed: list[int] = []
    # A recent persisted tool_call → the executor is live (CAS loser).
    live_events = [
        {
            "type": "tool_call",
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "payload": {"tool": "generate_pdf", "args": {"filename": "hiking-plan.pdf"}},
        }
    ]

    outcome, events = _drive(
        repo, run_id, effect_key, executed=executed, all_events=live_events
    )

    assert executed == []
    assert outcome.status == "in_flight"
    assert outcome.preserve_sandbox is True
    assert [t for t, _ in events] == ["error"]
    assert events[0][1]["code"] == "effect_in_flight"
    # The effect is untouched — the live winner may still commit.
    effect = repo.get_effect(effect_key)
    assert effect is not None and effect.status == EffectStatus.EXECUTING.value


def test_committed_effect_recovers_result_without_reexecution(tmp_path):
    repo, run_id, effect_key = _run_with_executing_effect(tmp_path, "generate_pdf")
    repo.commit_effect(
        effect_key,
        '{"content": "已在沙箱中生成 hiking-plan.pdf", "path": "hiking-plan.pdf"}',
    )
    executed: list[int] = []

    outcome, events = _drive(repo, run_id, effect_key, executed=executed)

    assert executed == []
    assert outcome.status == "recovered"
    assert outcome.stored_result == {
        "content": "已在沙箱中生成 hiking-plan.pdf",
        "path": "hiking-plan.pdf",
    }
    # The event log was missing the pair, so it is re-emitted from the
    # stored effect — the client sees a complete event log.
    assert [t for t, _ in events] == ["tool_call", "tool_result"]
    assert events[1][1]["status"] == "ok"
    assert events[1][1]["path"] == "hiking-plan.pdf"


def test_fresh_effect_executes_once_commits_and_is_idempotent(tmp_path):
    repo, run_id, effect_key = _run_with_executing_effect(tmp_path, "generate_pdf")
    # Reset the crash fixture to a clean run: no effect row yet.
    repo.control._conn.execute("DELETE FROM effects WHERE effect_key = ?", (effect_key,))
    executed: list[int] = []

    outcome, events = _drive(repo, run_id, effect_key, executed=executed)

    assert executed == [1]  # exactly one execution
    assert outcome.status == "executed"
    assert [t for t, _ in events] == ["tool_call", "tool_result"]
    effect = repo.get_effect(effect_key)
    assert effect is not None and effect.status == EffectStatus.COMMITTED.value

    # Second drive (duplicate resume click) recovers — no second execution.
    executed.clear()
    outcome2, events2 = _drive(
        repo,
        run_id,
        effect_key,
        executed=executed,
        all_events=[
            {
                "type": "tool_call",
                "timestamp": "",
                "payload": {"tool": "generate_pdf"},
            },
            {
                "type": "tool_result",
                "timestamp": "",
                "payload": {"tool": "generate_pdf", "status": "ok"},
            },
        ],
    )
    assert executed == []
    assert outcome2.status == "recovered"
    assert events2 == []
