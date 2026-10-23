"""Ticket 31 — durable approval / effect-ledger contract for cowork Bash."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.effects import EffectStatus, compute_effect_key
from agent_v2.kernel_tools import CoworkBashTool, ProductSandboxBridge
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


class _FakeBridge:
    """Records commands; programmable outcome via self.mode."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.mode: str = "ok"

    async def run(self, command: str, *, timeout: float | None = None):
        self.commands.append(command)
        if self.mode == "boom":
            raise RuntimeError("sandbox process died mid-run")
        if self.mode == "fail":
            return SimpleNamespace(exit_code=1, stdout="", stderr="nope", duration_ms=5, truncated=False)
        return SimpleNamespace(exit_code=0, stdout="out-ok", stderr="", duration_ms=5, truncated=False)


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _authorized_tool(tmp_path: Path, command: str = "echo hi"):
    repo = _repo(tmp_path)
    run_id, _created = repo.create_run_idempotent(
        request_id="req-31",
        conversation_id="conv-31",
        profile_id="cowork",
        user_message="x",
    )
    bridge = _FakeBridge()
    audit: list[dict[str, Any]] = []
    tool = CoworkBashTool(bridge=bridge, run_id=run_id, effects=repo, audit_sink=audit.append)
    key = compute_effect_key(run_id=run_id, tool_name="sandbox_bash", args={"command": command})
    repo.insert_effect(
        effect_key=key,
        run_id=run_id,
        conversation_id="conv-31",
        tool_name="sandbox_bash",
        args_canonical='{"command": "%s"}' % command,
        external_idempotency_key="ext-1",
    )
    return tool, repo, bridge, audit, key


def test_cowork_bash_without_effect_record_is_rejected(tmp_path: Path):
    repo = _repo(tmp_path)
    tool = CoworkBashTool(bridge=_FakeBridge(), run_id="run-31", effects=repo, audit_sink=None)

    result = asyncio.run(tool(command="ls"))

    assert result.error and "缺少授权记录" in result.error
    assert repo.get_effect(
        compute_effect_key(run_id="run-31", tool_name="sandbox_bash", args={"command": "ls"})
    ) is None


def test_cowork_bash_none_ledger_fails_closed(tmp_path: Path):
    tool = CoworkBashTool(bridge=_FakeBridge(), run_id="run-31", effects=None, audit_sink=None)

    result = asyncio.run(tool(command="ls"))

    assert result.error and "effect ledger 不可用" in result.error


def test_cowork_bash_authorized_executes_once_and_commits(tmp_path: Path):
    tool, repo, bridge, audit, key = _authorized_tool(tmp_path, "echo hi")

    first = asyncio.run(tool(command="echo hi"))
    second = asyncio.run(tool(command="echo hi"))  # idempotent replay path

    assert first.output == "out-ok"
    assert second.output == "out-ok"
    assert len(bridge.commands) == 1  # committed effect replayed, NOT re-executed
    final = repo.get_effect(key)
    assert final is not None and final.status == EffectStatus.COMMITTED.value
    ok_entries = [e for e in audit if e.get("phase") == "result" and e.get("ok")]
    assert len(ok_entries) == 1  # first execution only; replay carries no execute


def test_cowork_bash_failure_marks_failed_not_committed(tmp_path: Path):
    tool, repo, bridge, audit, key = _authorized_tool(tmp_path, "bad-cmd")
    bridge.mode = "fail"

    result = asyncio.run(tool(command="bad-cmd"))

    assert result.error and "exit=1" in result.error
    final = repo.get_effect(key)
    assert final is not None and final.status == EffectStatus.FAILED.value


def test_cowork_bash_crash_marks_outcome_unknown_and_blocks_replay(tmp_path: Path):
    tool, repo, bridge, audit, key = _authorized_tool(tmp_path, "rm ./x")
    bridge.mode = "boom"

    result = asyncio.run(tool(command="rm ./x"))

    assert result.error and "outcome_unknown" in result.error
    final = repo.get_effect(key)
    assert final is not None and final.status == EffectStatus.OUTCOME_UNKNOWN.value

    # A second attempt must NOT silently replay — human reconciliation first.
    second = asyncio.run(tool(command="rm ./x"))
    assert second.error and "outcome_unknown" in second.error
    assert len(bridge.commands) == 1  # the crashed attempt was the only execution


def test_kernel_assembly_forwards_approval_callback(tmp_path: Path, monkeypatch):
    """approval_callback reaches the vendored create_agent verbatim."""
    captured: dict[str, Any] = {}

    async def _fake_create_agent(model, computer, **kwargs):
        captured.update(kwargs)
        agent = SimpleNamespace()
        agent.aclose = _async_noop
        agent.ainvoke = _async_noop_result
        agent.tools = []
        return agent

    async def _async_noop():
        return None

    async def _async_noop_result(*a, **k):
        return {"messages": []}

    import uniharness.langchain.agent as vendor_agent_module

    monkeypatch.setattr(vendor_agent_module, "create_agent", _fake_create_agent)

    from agent_v2.kernel import KernelModelFailure, run_chat_turn

    sentinel = object()

    class _Settings:
        openai_api_key = "sk-fake"
        openai_base_url = "https://fake.invalid/v1"
        openai_model = "fake"
        agent_v2_uniharness_pin = os.getenv(
            "AGENT_V2_UNIHARNESS_PIN",
            "b026a26250487678aedc2beef0d0b6ae802de5c4",
        )

    async def scenario():
        # The fake ainvoke returns no AI message → empty_model_output is the
        # expected terminal; the assertion targets the captured kwargs only.
        with pytest.raises(KernelModelFailure):
            await run_chat_turn(
                system_prompt="p",
                message="m",
                settings=_Settings(),
                extra_tools=(),
                kernel_gate=None,
                approval_callback=sentinel,
            )

    asyncio.run(scenario())
    assert captured.get("approval_callback") is sentinel


import os  # noqa: E402

# ---------------------------------------------------------------------------
# Ticket 31 second batch: cowork first stream plans command via kernel turn
# ---------------------------------------------------------------------------


def _cowork_app(tmp_path: Path) -> "Any":
    from fastapi import FastAPI

    import api.chat_v2 as chat_v2_module
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.runs import RunRepository

    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )

    class _Runtime:
        runs = repo
        memory = None
        telemetry = None
        capacity = None
        kernel_checkpoints = None

    app.state.agent_v2_runtime = _Runtime()
    return app


async def _post_cowork(app: "Any", payload: dict[str, Any]) -> list[dict[str, Any]]:
    import json as _json

    from httpx import ASGITransport, AsyncClient

    events: list[dict[str, Any]] = []
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/api/v2/chat/sse", json=payload) as resp:
            assert resp.status_code == 200
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            events.append(_json.loads(line[6:]))
    return events


class _CoworkComputer:
    sandbox_type = "local_native"

    def create_workspace(self, run_id: str) -> str:
        return f"/ws/{run_id}"

    def cleanup(self, *a: Any, **k: Any) -> dict[str, int]:
        return {"files_deleted": 0}

    def destroy_sandbox(self, *a: Any, **k: Any) -> dict[str, int]:
        return {"files_deleted": 0}


def test_cowork_first_stream_plans_command_via_kernel_turn_then_pauses(
    tmp_path: Path, monkeypatch
):
    import api.chat_v2 as chat_v2_module  # noqa: F811 - local scope for clarity

    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_cowork_policy", "trusted")
    monkeypatch.setattr(chat_v2_module, "_build_artifact_computer", lambda *a, **k: _CoworkComputer())
    # ALR ticket 03: the production cleanup adapter runs unpatched (typed
    # result, no lease mutation); the pause path preserves the sandbox.

    from agent_v2 import kernel as km

    calls: list[dict[str, Any]] = []

    async def fake_turn(**kwargs: Any):
        calls.append(kwargs)
        return '{"command": "echo kernel-planned"}', []

    monkeypatch.setattr(km, "run_chat_turn", fake_turn)

    app = _cowork_app(tmp_path)
    events = asyncio.run(_post_cowork(app, {
        "message": "在沙箱里创建一个文件",
        "request_id": "req-cowork-31",
        "profile_id": "cowork",
    }))

    types = [e["type"] for e in events]
    assert calls, "planner kernel turn was not invoked"
    # Planner turn is tool-less and binds the run identity for checkpoints.
    assert calls[0]["extra_tools"] == ()
    assert calls[0]["checkpoint_store"] is app.state.agent_v2_runtime.kernel_checkpoints
    assert "approval.required" in types
    assert types[-1] == "run.paused"  # stream terminal; run stays paused
    approval = next(e for e in events if e["type"] == "approval.required")
    assert approval["payload"]["args"]["command"] == "echo kernel-planned"


def test_cowork_kernel_invalid_command_plan_fails_closed(tmp_path: Path, monkeypatch):
    import api.chat_v2 as chat_v2_module  # noqa: F811 - local scope for clarity

    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_cowork_policy", "trusted")
    monkeypatch.setattr(chat_v2_module, "_build_artifact_computer", lambda *a, **k: _CoworkComputer())

    from agent_v2 import kernel as km

    async def fake_turn(**kwargs: Any):
        return "not a json blob", []

    monkeypatch.setattr(km, "run_chat_turn", fake_turn)

    app = _cowork_app(tmp_path)
    events = asyncio.run(_post_cowork(app, {
        "message": "随便执行点什么",
        "request_id": "req-cowork-32",
        "profile_id": "cowork",
    }))

    types = [e["type"] for e in events]
    assert types[-1] == "error"
    assert events[-1]["payload"]["code"] == "bash_plan_invalid"
    assert "approval.required" not in types  # nothing reached approval stage
