"""Ticket 01 — placeholder remote sandbox must refuse BEFORE any host access.

Spec coverage: T01 (remote adapter fail-closed) + T02 (remote orchestration).

The ``remote_e2b`` sandbox is a placeholder: no live E2B client is wired.
Before this ticket the stub mimicked the E2B lifecycle on the HOST (temp
directory + ``bash -c``), so a user selecting the remote sandbox silently
executed model-written commands on the host machine. These tests pin the
fixed behavior:

* every :class:`RemoteE2BComputer` operation raises
  :class:`SandboxUnavailableError` (a :class:`ComputerError`) before any
  host workspace, process, or file is touched — direct adapter calls and
  indirect ``ProductSandboxBridge`` calls share the same semantics;
* first-stream artifact/cowork, resume-after-approval, legacy leases,
  destroyed leases, and duplicate/concurrent resumes never execute a
  placeholder host operation; they end in exactly one terminal
  ``sandbox_unavailable`` error and record no committed effect;
* host side effects are zero, verified with sentinels that record
  ``tempfile.mkdtemp`` / ``subprocess`` / ``shutil.rmtree`` calls plus a
  temp-dir prefix scan (the repro never runs a dangerous command);
* capacity and lease bookkeeping are released on refusal, and no local
  fallback is created to bypass the refusal;
* the health endpoint distinguishes basic chat, unavailable remote
  execution, and explicitly-configured trusted local execution without
  advertising WSL as verified isolation;
* explicitly configured ``local_native`` and plain chat keep working
  through the existing contracts.

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_remote_computer_fail_closed.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.computer import (
    ComputerError,
    LocalNativeComputer,
    RemoteE2BComputer,
    SandboxUnavailableError,
    create_computer,
)
from agent_v2.effects import EffectStatus, compute_effect_key
from agent_v2.kernel_tools import ProductSandboxBridge
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator

from api.chat_v2 import _cleanup_sandbox

TERMINALS = {"error", "done", "cancelled", "run.paused"}
_SANDBOX_DIR_PREFIXES = ("v2-sandbox-", "v2-e2b-")


# ---- host-access sentinels ----------------------------------------------------


class _HostAccessSentinels:
    """Record every host-access primitive the sandbox could touch.

    Installed with :func:`_install_sentinels` for the duration of one test.
    Any nonzero count fails the test — the placeholder must refuse before
    the first host call, and the repro never runs a dangerous command.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mkdtemp_calls: list[Any] = []
        self.subprocess_calls: list[Any] = []
        self.rmtree_calls: list[Any] = []
        self._monkeypatch = monkeypatch

        real_mkdtemp = tempfile.mkdtemp
        real_run = subprocess.run
        real_popen = subprocess.Popen
        real_rmtree = shutil.rmtree

        def rec_mkdtemp(*a: Any, **k: Any):
            self.mkdtemp_calls.append(a)
            return real_mkdtemp(*a, **k)

        def rec_run(*a: Any, **k: Any):
            self.subprocess_calls.append(a)
            return real_run(*a, **k)

        def rec_popen(*a: Any, **k: Any):
            self.subprocess_calls.append(a)
            return real_popen(*a, **k)

        def rec_rmtree(*a: Any, **k: Any):
            self.rmtree_calls.append(a)
            return real_rmtree(*a, **k)

        monkeypatch.setattr(tempfile, "mkdtemp", rec_mkdtemp)
        monkeypatch.setattr(subprocess, "run", rec_run)
        monkeypatch.setattr(subprocess, "Popen", rec_popen)
        monkeypatch.setattr(shutil, "rmtree", rec_rmtree)

    @property
    def host_access_count(self) -> int:
        return (
            len(self.mkdtemp_calls)
            + len(self.subprocess_calls)
            + len(self.rmtree_calls)
        )

    def assert_no_host_access(self) -> None:
        assert self.mkdtemp_calls == [], "host temp dir creation attempted"
        assert self.subprocess_calls == [], "host process execution attempted"
        assert self.rmtree_calls == [], "host directory removal attempted"


def _sandbox_dirs_in_tempdir() -> list[str]:
    tmp = Path(tempfile.gettempdir())
    found: list[str] = []
    try:
        for entry in tmp.iterdir():
            if entry.name.startswith(_SANDBOX_DIR_PREFIXES):
                found.append(entry.name)
    except OSError:  # pragma: no cover - temp listing best effort
        pass
    return found


# ---- helpers ------------------------------------------------------------------


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "agent_v2_sandbox_type": "remote_e2b",
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _refusal_calls(computer: RemoteE2BComputer) -> list[Any]:
    """Every operation a caller could make against the placeholder."""
    return [
        lambda: computer.create_workspace("run-01"),
        lambda: computer.generate_pdf("/ws/run-01", "content", "plan.pdf"),
        lambda: computer.build_manifest("/ws/run-01", "plan.pdf", "application/pdf"),
        lambda: computer.read_artifact_bytes("/ws/run-01", "plan.pdf"),
        lambda: computer.run_bash("/ws/run-01", "echo hi"),
        lambda: computer.cleanup("/ws/run-01"),
    ]


def _stream(orch: RunOrchestrator, req: ChatFlowRequest, *, subject_id: str | None = None) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id=subject_id)]

    return asyncio.run(run())


def _stream_resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)]

    return asyncio.run(run())


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---- AC1: direct adapter + bridge refuse before any host access ---------------


def test_remote_computer_every_operation_refuses_before_host_access(
    monkeypatch: pytest.MonkeyPatch,
):
    sentinels = _HostAccessSentinels(monkeypatch)
    before_dirs = _sandbox_dirs_in_tempdir()

    computer = create_computer("remote_e2b")
    assert isinstance(computer, RemoteE2BComputer)
    assert computer.sandbox_type == "remote_e2b"

    for call in _refusal_calls(computer):
        with pytest.raises(SandboxUnavailableError):
            call()
        # Fail-closed taxonomy: a ComputerError subclass so every existing
        # flow-level mapping stays in force.
        assert issubclass(SandboxUnavailableError, ComputerError)

    sentinels.assert_no_host_access()
    assert _sandbox_dirs_in_tempdir() == before_dirs


def test_bridge_indirect_call_shares_refusal_semantics(
    monkeypatch: pytest.MonkeyPatch,
):
    sentinels = _HostAccessSentinels(monkeypatch)

    bridge = ProductSandboxBridge(RemoteE2BComputer(), "/ws/run-01")

    async def scenario() -> None:
        # The kernel-facing protocol surfaces CLIError; the important
        # contract is that NO host command ran.
        with pytest.raises(Exception):
            await bridge.run("echo kernel-planned")

    asyncio.run(scenario())
    sentinels.assert_no_host_access()


def test_invalid_sandbox_type_still_rejected():
    with pytest.raises(ValueError):
        create_computer("not-a-sandbox")  # type: ignore[arg-type]


# ---- AC2/AC4: first-stream artifact + cowork fail closed ----------------------


def _asgi_app(tmp_path: Path, *, capacity: Any | None = None) -> Any:
    from fastapi import FastAPI

    import api.chat_v2 as chat_v2_module

    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(
        runs=_repo(tmp_path),
        memory=None,
        telemetry=None,
        capacity=capacity,
        kernel_checkpoints=None,
    )
    return app


async def _post_events(app: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                            events.append(json.loads(line[6:]))
    return events


def _assert_single_sandbox_unavailable_terminal(events: list[dict[str, Any]]) -> None:
    types = [e["type"] for e in events]
    terminals = [t for t in types if t in TERMINALS]
    assert terminals == ["error"], types
    err = events[-1]
    assert err["payload"]["code"] == "sandbox_unavailable"
    # No tool ever ran and no artifact was ever published.
    assert "tool_call" not in types
    assert "artifact_candidate" not in types
    assert "artifact.available" not in types


def test_artifact_first_stream_remote_e2b_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import api.chat_v2 as chat_v2_module

    sentinels = _HostAccessSentinels(monkeypatch)
    before_dirs = _sandbox_dirs_in_tempdir()

    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_sandbox_type", "remote_e2b")
    monkeypatch.setattr(chat_v2_module.settings, "openai_api_key", "sk-test")

    app = _asgi_app(tmp_path)
    events = asyncio.run(_post_events(app, {
        "message": "规划杭州两日徒步并出 PDF 计划",
        "request_id": "req-01-art",
        "profile_id": "artifact",
    }))

    _assert_single_sandbox_unavailable_terminal(events)

    # No sandbox lease was ever persisted — nothing to clean up or resume.
    runs = app.state.agent_v2_runtime.runs
    run = runs.control.get_run_by_request("req-01-art")
    assert run is not None
    assert runs.get_sandbox_lease(run.run_id) is None

    # Replay of the same request_id yields the SAME stored events — the
    # refusal is replayable and never re-executes anything.
    replayed = asyncio.run(_post_events(app, {
        "message": "规划杭州两日徒步并出 PDF 计划",
        "request_id": "req-01-art",
        "profile_id": "artifact",
    }))
    assert [e["type"] for e in replayed] == [e["type"] for e in events]
    assert [e["payload"] for e in replayed] == [e["payload"] for e in events]

    sentinels.assert_no_host_access()
    assert _sandbox_dirs_in_tempdir() == before_dirs


def test_cowork_first_stream_remote_e2b_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import api.chat_v2 as chat_v2_module

    sentinels = _HostAccessSentinels(monkeypatch)
    before_dirs = _sandbox_dirs_in_tempdir()

    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_sandbox_type", "remote_e2b")
    monkeypatch.setattr(chat_v2_module.settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_cowork_policy", "trusted")

    app = _asgi_app(tmp_path)
    events = asyncio.run(_post_events(app, {
        "message": "在沙箱里创建一个文件",
        "request_id": "req-01-cowork",
        "profile_id": "cowork",
    }))

    _assert_single_sandbox_unavailable_terminal(events)
    assert "approval.required" not in [e["type"] for e in events]

    runs = app.state.agent_v2_runtime.runs
    run = runs.control.get_run_by_request("req-01-cowork")
    assert run is not None
    assert runs.get_sandbox_lease(run.run_id) is None

    sentinels.assert_no_host_access()
    assert _sandbox_dirs_in_tempdir() == before_dirs


# ---- AC2/AC4/AC5: resume paths refuse without host access ---------------------


def _paused_artifact_run(repo: RunRepository, request_id: str) -> tuple[str, str]:
    """Hand-assemble the post-approval artifact resume state (Ticket 40 shape)."""
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type="remote_e2b",  # legacy lease written by the stub era
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
    )
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id=f"dec-{request_id}"
    )
    return run_id, approval_id


async def _fake_kernel_turn(**kwargs: Any) -> tuple[str, list[Any]]:
    """Kernel seam for the refusal tests.

    Ticket 04 made the approved resume render the PLAN through the kernel
    seam before the tool executes, so these tests stub the model loop —
    their subject is the sandbox refusal, not content generation.
    """
    return ("# 徒步计划\n占位计划正文。", [])


def test_resume_after_approval_with_legacy_remote_lease_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinels = _HostAccessSentinels(monkeypatch)
    before_dirs = _sandbox_dirs_in_tempdir()

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-resume")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        # The real factory — production wiring, not a test double.
        computer_factory=lambda *a, **k: create_computer("remote_e2b"),
        # The REAL cleanup hook: it must absorb the refusal (the Computer
        # refuses cleanup too) and still release the lease bookkeeping.
        sandbox_cleanup=_cleanup_sandbox,
        gate_factory=None,
        kernel_turn=_fake_kernel_turn,
    )

    items = _stream_resume(orch, run_id)

    types = [e.type for e in items]
    terminals = [t for t in types if t in TERMINALS]
    assert terminals == ["error"], types
    assert items[-1].payload["code"] == "sandbox_unavailable"
    # The tool never ran: no tool_result ok, no artifact published.
    assert "artifact.available" not in types
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"
    # Deterministic refusal — recorded as failed, never committed, and NOT
    # outcome_unknown (nothing was executed, so nothing is "unknown").
    effects = repo.list_effects_for_run(run_id)
    assert effects and effects[0].status == EffectStatus.FAILED.value
    assert effects[0].failure_reason
    # The legacy lease bookkeeping is released — the row is not left
    # ``active`` (no capacity/lease leak), and the refusal reason is kept
    # for operators.
    lease = repo.get_sandbox_lease(run_id)
    assert lease is not None and lease.status != "active"
    assert lease.failure_reason

    sentinels.assert_no_host_access()
    assert _sandbox_dirs_in_tempdir() == before_dirs


def test_resume_replay_after_refusal_does_not_reexecute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinels = _HostAccessSentinels(monkeypatch)

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-replay")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        computer_factory=lambda *a, **k: create_computer("remote_e2b"),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,
        kernel_turn=_fake_kernel_turn,
    )

    first = _stream_resume(orch, run_id)
    assert first[-1].payload["code"] == "sandbox_unavailable"

    # A second resume (double click / reconnect) must not execute anything
    # new and must keep exactly one terminal per stream.
    second = _stream_resume(orch, run_id, from_seq=1)
    terminals = [e.type for e in second if e.type in TERMINALS]
    assert terminals == ["error"], [e.type for e in second]
    effects = repo.list_effects_for_run(run_id)
    assert len(effects) == 1
    assert effects[0].status == EffectStatus.FAILED.value

    sentinels.assert_no_host_access()


def test_disconnect_window_after_refusal_keeps_sandbox_unavailable_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """D1 regression: disconnect in the window AFTER the refusal ``error``
    frame is emitted but BEFORE the run is marked terminal (generator
    close). The effect row is already ``FAILED`` with the sandbox refusal
    reason; the next resume must still surface the stable
    ``sandbox_unavailable`` code with identical replayed payloads — never
    ``artifact_generation_failed``."""
    sentinels = _HostAccessSentinels(monkeypatch)

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-window")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        computer_factory=lambda *a, **k: create_computer("remote_e2b"),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,
        kernel_turn=_fake_kernel_turn,
    )

    async def interrupted_resume() -> list[Any]:
        agen = orch.stream_resume(run_id=run_id)
        items: list[Any] = []
        async for item in agen:
            items.append(item)
            if item.type == "error":
                # Stop consuming exactly in the disconnect window: the
                # error frame is persisted, ``mark_terminal_safe`` has not
                # run yet.
                break
        await agen.aclose()
        return items

    first = asyncio.run(interrupted_resume())
    first_error = first[-1]
    assert first_error.type == "error"
    assert first_error.payload["code"] == "sandbox_unavailable"

    # The window state: the refusal is recorded as FAILED, and the run did
    # NOT reach its terminal status (the disconnect cut the flow short).
    effects = repo.list_effects_for_run(run_id)
    assert effects and effects[0].status == EffectStatus.FAILED.value
    assert effects[0].failure_reason
    run = repo.get_run(run_id)
    assert run is not None and not run.is_terminal

    # Reconnect where the client left off: the refusal must stay stable.
    second = _stream_resume(orch, run_id, from_seq=first_error.sequence)
    terminals = [e.type for e in second if e.type in TERMINALS]
    assert terminals == ["error"], [e.type for e in second]
    assert second[-1].payload["code"] == "sandbox_unavailable"
    assert second[-1].payload == first_error.payload
    first_tool_result = next(e for e in first if e.type == "tool_result")
    resumed_tool_result = next(e for e in second if e.type == "tool_result")
    assert resumed_tool_result.payload == first_tool_result.payload

    # Full replay of the whole log: every recorded refusal terminal keeps
    # the stable code (no artifact_generation_failed drift).
    full = _stream_resume(orch, run_id, from_seq=0)
    error_codes = [e.payload["code"] for e in full if e.type == "error"]
    assert error_codes == ["sandbox_unavailable", "sandbox_unavailable"]

    sentinels.assert_no_host_access()


def test_concurrent_resume_remote_refusal_single_terminal_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinels = _HostAccessSentinels(monkeypatch)

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-conc")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        computer_factory=lambda *a, **k: create_computer("remote_e2b"),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,
        kernel_turn=_fake_kernel_turn,
    )

    async def scenario() -> list[list[Any]]:
        async def collect() -> list[Any]:
            return [item async for item in orch.stream_resume(run_id=run_id)]

        return list(await asyncio.gather(collect(), collect()))

    streams = asyncio.run(scenario())
    for items in streams:
        terminals = [e.type for e in items if e.type in TERMINALS]
        assert terminals == ["error"], [e.type for e in items]
        error_codes = [e.payload["code"] for e in items if e.type == "error"]
        # After the D1 fix a finalized refusal surfaces ``sandbox_unavailable``
        # even to the CAS loser, and a live winner surfaces
        # ``effect_in_flight`` — no generic per-tool failure code may appear.
        assert error_codes and all(
            code in ("sandbox_unavailable", "effect_in_flight")
            for code in error_codes
        ), error_codes
        assert items[-1].payload["code"] in ("sandbox_unavailable", "effect_in_flight")
    effects = repo.list_effects_for_run(run_id)
    assert all(e.status != EffectStatus.COMMITTED.value for e in effects)

    sentinels.assert_no_host_access()


def test_resume_with_destroyed_remote_lease_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinels = _HostAccessSentinels(monkeypatch)

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-destroyed")
    repo.destroy_sandbox_lease(run_id=run_id, failure_reason="lease destroyed")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        computer_factory=lambda *a, **k: create_computer("remote_e2b"),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,
    )

    items = _stream_resume(orch, run_id)
    types = [e.type for e in items]
    terminals = [t for t in types if t in TERMINALS]
    assert terminals == ["error"], types
    assert items[-1].payload["code"] == "sandbox_lease_lost"
    assert repo.list_effects_for_run(run_id) == []

    sentinels.assert_no_host_access()


def test_capacity_slot_released_after_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """D2: a REAL refusal through the HTTP flow returns the capacity slot.

    Exercised path: the artifact first-stream POST acquires a sandbox slot
    (``acquire_leases``), the placeholder refuses (terminal
    ``sandbox_unavailable``), and the stream's ``finally``
    (``leases.release()``, api/chat_v2.py) releases the slot once no
    ``active`` sandbox lease row remains. A refusal must not leak a slot.
    """
    import api.chat_v2 as chat_v2_module
    from agent_v2.observability import CapacityGuard

    sentinels = _HostAccessSentinels(monkeypatch)

    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_sandbox_type", "remote_e2b")
    monkeypatch.setattr(chat_v2_module.settings, "openai_api_key", "sk-test")

    async def scenario() -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
        guard = CapacityGuard(max_runs=2, max_sandboxes=1, queue_timeout_seconds=0.2)
        app = _asgi_app(tmp_path, capacity=guard)
        started = dict(guard.snapshot())
        events = await _post_events(app, {
            "message": "规划杭州两日徒步并出 PDF 计划",
            "request_id": "req-01-cap-real",
            "profile_id": "artifact",
        })
        released = dict(guard.snapshot())
        # A fresh acquisition must succeed immediately — if the refusal had
        # leaked the only sandbox slot this would queue and then raise
        # CapacityExceeded (short queue timeout).
        lease = await guard.acquire_sandbox("req-01-cap-recheck")
        lease.release()
        return started, released, events

    started, released, events = asyncio.run(scenario())
    _assert_single_sandbox_unavailable_terminal(events)
    assert started == {"active_runs": 0, "active_sandboxes": 0}
    assert released == started, "refusal leaked a capacity slot"

    sentinels.assert_no_host_access()


def test_no_local_fallback_bypasses_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Selecting remote_e2b must never silently construct a host computer."""
    constructed: list[str] = []

    real_init = LocalNativeComputer.__init__

    def recording_init(self: Any, *a: Any, **k: Any) -> None:
        constructed.append("local_native")
        real_init(self, *a, **k)

    monkeypatch.setattr(LocalNativeComputer, "__init__", recording_init)

    repo = _repo(tmp_path)
    run_id, _approval_id = _paused_artifact_run(repo, "req-01-nofallback")

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        computer_factory=lambda *a, **k: create_computer(a[0] if a else "remote_e2b"),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,
        kernel_turn=_fake_kernel_turn,
    )
    items = _stream_resume(orch, run_id)
    assert items[-1].payload["code"] == "sandbox_unavailable"
    assert constructed == [], "local fallback computer was constructed"


# ---- AC6: health endpoint honesty ---------------------------------------------


def test_health_distinguishes_chat_remote_unavailable_and_trusted_local(tmp_path: Path):
    from agent_v2.health import v2_health_status
    from agent_v2.startup import V2Runtime

    runtime = V2Runtime(context=None, uniharness_pin="pin-x", state_dir=str(tmp_path))

    remote = v2_health_status(
        SimpleNamespace(agent_v2_runtime=runtime), _settings()
    )
    assert remote["status"] == "ready"  # basic chat stays available
    sandbox = remote["dependencies"]["sandbox"]
    assert sandbox["type"] == "remote_e2b"
    assert sandbox["status"] == "unavailable"
    assert sandbox["verified_isolation"] is False

    trusted = v2_health_status(
        SimpleNamespace(agent_v2_runtime=runtime),
        _settings(agent_v2_sandbox_type="local_native"),
    )
    trusted_sandbox = trusted["dependencies"]["sandbox"]
    assert trusted_sandbox["status"] == "trusted_local"
    assert trusted_sandbox["type"] == "local_native"

    wsl = v2_health_status(
        SimpleNamespace(agent_v2_runtime=runtime),
        _settings(agent_v2_sandbox_type="wsl_localvm"),
    )
    wsl_sandbox = wsl["dependencies"]["sandbox"]
    # WSL is not advertised as verified isolation.
    assert wsl_sandbox["verified_isolation"] is False
    assert "隔离" not in json.dumps(wsl_sandbox, ensure_ascii=False).replace(
        "未验证为隔离环境", ""
    ) or wsl_sandbox["verified_isolation"] is False


# ---- AC7: trusted local + chat unaffected -------------------------------------


def test_local_native_and_chat_contracts_unaffected(tmp_path: Path):
    # Explicitly configured local_native keeps working (development only).
    computer = create_computer("local_native")
    assert isinstance(computer, LocalNativeComputer)
    workspace = computer.create_workspace("run-01-local")
    try:
        assert Path(workspace).is_dir()
    finally:
        computer.cleanup(workspace)

    # Plain chat never constructs a sandbox computer at all.
    repo = _repo(tmp_path)

    async def fake_kernel(**kwargs: Any):
        assert kwargs.get("computer") is None
        return ("你好，我是徒步助手。", [])

    orch = RunOrchestrator(
        settings=_settings(agent_v2_sandbox_type="local_native"),
        runs_repo=repo,
        # The chat profile declares memory_* tools: assembly fails closed
        # without a memory service + server-resolved subject.
        memory=SimpleNamespace(
            claim_conversation=lambda *a, **k: True,
            owns_conversation=lambda *a, **k: True,
            memory_enabled=lambda *a, **k: False,
        ),
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("chat profile must not build a sandbox computer")
        ),
        gate_factory=None,
    )
    req = ChatFlowRequest(
        message="你好",
        request_id="req-01-chat",
        profile_id="chat",
    )
    items = _stream(orch, req, subject_id="subj-01")
    types = [e.type for e in items]
    assert types[-1] == "done"
    assert "artifact_candidate" not in types
