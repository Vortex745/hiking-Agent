"""agent-lifecycle-repair ticket 02 — 同 run 独占执行，竞争恢复只读回放。

Main verification boundary (ticket + verification guide): production ASGI
HTTP over the real FastAPI routers, one real temp-directory SQLite
RunRepository shared by EVERY writer, default capacity control per
orchestrator instance, controlled model/Computer fakes and REAL SSE send
barriers — the winner is parked at a deterministic point AFTER its
``tool_call`` frame was sent, and the losing connection must exit BEFORE
that barrier releases. No arbitrary sleeps: every interleaving is
coordinated by explicit events (threading.Event / asyncio primitives).

Behavior asserted (ticket 02 ACs):

* T02-01 two overlapping resumes execute the tool ONCE; one durable
  terminal, no sequence conflicts;
* T02-02 the losing resume is a read-only observer: it replays persisted
  events after its cursor with original identity and closes before the
  winner's barrier releases — no run/effect/lease/cleanup mutation, no
  sandbox or model work;
* T02-03 first stream vs resume overlap keeps the original executor; a
  same-request re-send and terminal replay preserve identity with no
  extra model/tool calls;
* T02-04 two orchestrator instances over one shared Repository still
  yield exactly one executor, and a run paused at a barrier does not
  block a different run;
* T02-05 effect repair (HTTP) and the janitor repair claim both pass
  through the ownership gate: conflict with state unchanged while an
  owner is active, existing policy after release;
* T02-06 an old token cannot release a successor owner; normal and error
  exits both remove the owned token;
* T02-07 immediate close / disconnect deterministically close the
  connection's owned nested generators; the token is released only after
  the owned execution and effect settlement; observers never clean the
  winner's resources and unknown results are never marked success;
* T02-08 after a restart (no surviving owner) a claimed/executing effect
  converges to outcome_unknown — the recent-tool_call liveness inference
  is gone, nothing re-runs;
* T02-09 from_seq=0, tail and current-cursor replays preserve the
  original persisted event identity, and the acquired continuation uses
  the RE-READ cursor.

The ownership token API itself is asserted directly on a real repository
(on the pre-fix baseline those tests fail on the API-existence assertion
instead of an import error, per the ticket's red/green rules).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent_v2.effect_repair import repair_effect
from agent_v2.effects import EffectStatus, compute_effect_identity
from agent_v2.janitor import run_pending_janitor_tasks
from agent_v2.observability import CapacityGuard
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

import api.chat_v2 as chat_v2_module


TERMINALS = {"error", "done", "cancelled", "run.paused"}
# A read-only observer may replay a PERSISTED stream terminal (e.g. the
# pause's ``run.paused``); what it must never do is emit a terminal of
# its own beyond the replay.
NEW_TERMINALS = {"error", "done", "cancelled"}
SUBJECT_ID = "sub-owner"

_HAS_OWNERSHIP = hasattr(RunRepository, "try_acquire_run_execution")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_ownership_api() -> None:
    """Fail as a business assertion (not an import error) pre-fix."""
    assert _HAS_OWNERSHIP, (
        "RunRepository 必须提供 per-run 执行权（try_acquire_run_execution/"
        "release_run_execution，票据 02）"
    )


# ---- fakes -------------------------------------------------------------------


class _FakeMemory:
    def memory_enabled(self, subject_id: str) -> bool:
        return False

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True

    def create_candidate(self, subject_id: str, **kw):
        return None


class _BarrierComputer:
    """Counting Computer whose ``generate_pdf`` parks on a real barrier.

    The barrier is the REAL SSE send barrier of the ticket: the winner
    has already sent its ``tool_call`` frame (the emit is persisted and
    yielded BEFORE ``execute()`` runs) when it enters ``generate_pdf``.
    Parking here means the winner owns the run, holds the effect claim
    and cannot settle — exactly the moment a competing resume arrives.
    Barriers are PER WORKSPACE so two runs can overlap deterministically.
    """

    sandbox_type = "local_native"

    def __init__(self) -> None:
        self._barrier_lock = threading.Lock()
        self._barriers: dict[str, tuple[threading.Event, threading.Event]] = {}
        self.workspaces: list[str] = []
        self.pdfs: list[tuple[str, str, str]] = []
        self.bash_calls: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    def barrier(self, workspace_path: str) -> tuple[threading.Event, threading.Event]:
        with self._barrier_lock:
            pair = self._barriers.get(workspace_path)
            if pair is None:
                pair = (threading.Event(), threading.Event())
                self._barriers[workspace_path] = pair
            return pair

    def create_workspace(self, run_id: str) -> str:
        self.workspaces.append(run_id)
        return f"/ws/{run_id}"

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        entered, release = self.barrier(workspace_path)
        entered.set()
        assert release.wait(timeout=30), "winner barrier never released"
        return filename

    def run_bash(self, workspace_path, command, **_kwargs):
        self.bash_calls.append((workspace_path, command))
        return SimpleNamespace(
            exit_code=0, stdout="ok", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


class _GateKernel:
    """Async fake kernel that parks on a gate WITHOUT blocking its loop.

    The park uses the default executor so the winner's event loop stays
    responsive — required for deterministic disconnect/cancel delivery
    into the generator. ``entered`` signals the main thread; ``release``
    lets the model turn finish.
    """

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.entered = threading.Event()
        self.release = threading.Event()
        self.armed = False  # parked turns are opt-in per scenario

    async def __call__(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        self.calls.append(kwargs)
        if not self.armed:
            return ("两日行程，注意天气。", [])
        self.entered.set()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release.wait, 30.0)
        return ("两日行程，注意天气。", [])


# ---- SSE parsing --------------------------------------------------------------


def _frames(text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            frames.append(json.loads(payload))
        except ValueError:
            continue
    return frames


def _types(frames: list[dict[str, Any]]) -> list[str]:
    return [f.get("type", "") for f in frames]


# ---- harness ------------------------------------------------------------------


class OwnershipHarness:
    """One real HTTP surface over ONE shared RunRepository.

    Every HTTP request runs in its own event loop with its own
    orchestrator instance (default CapacityGuard) — the token table on
    the shared Repository is what serializes writers, which is exactly
    the cross-instance premise of ticket 02. A fresh instance per
    request also keeps each CapacityGuard bound to exactly one loop.
    Non-streaming calls (snapshot/events/decision/repair) run on
    short-lived loops each.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.repo = RunRepository(
            control=ControlStore(str(tmp_path / "control.sqlite")),
            checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
        )
        self.computer = _BarrierComputer()
        self.kernel_calls: list[dict[str, Any]] = []
        self.export_calls: list[dict[str, Any]] = []
        self.cleanup_calls: list[str] = []
        self.orchestrator_instances: list[RunOrchestrator] = []
        self.kernel: Any = _GateKernel(self.kernel_calls)
        self._build_lock = threading.Lock()

        def export(**kwargs: Any) -> dict[str, Any]:
            self.export_calls.append(kwargs)
            return {
                "artifact_id": f"art-{len(self.export_calls)}",
                "public_filename": "hiking-plan.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 12,
                "sha256": "abc123",
                "expires_at": "2027-01-01T00:00:00+00:00",
            }

        def cleanup(computer, workspace_path, runs_repo, run_id) -> None:
            self.cleanup_calls.append(workspace_path)
            computer.cleanup(workspace_path)
            if runs_repo is not None and run_id is not None:
                runs_repo.destroy_sandbox_lease(run_id=run_id)

        def build_orchestrator(runtime: Any) -> RunOrchestrator:
            runs_repo = getattr(runtime, "runs", None) if runtime else None
            with self._build_lock:
                orch = RunOrchestrator(
                    settings=_settings(),
                    runs_repo=runs_repo,
                    memory=_FakeMemory(),
                    kernel_turn=self.kernel,
                    artifact_export=export,
                    computer_factory=lambda *a, **k: self.computer,
                    sandbox_cleanup=cleanup,
                    gate_factory=None,  # real gate → generate_pdf HIGH risk → pause
                    capacity=CapacityGuard(),
                )
                self.orchestrator_instances.append(orch)
                return orch

        monkeypatch.setattr(
            chat_v2_module, "_build_run_orchestrator", build_orchestrator
        )
        monkeypatch.setattr(
            chat_v2_module,
            "resolve_request_subject",
            lambda request: (
                _FakeMemory(),
                setattr(request.state, "subject_id", SUBJECT_ID)
                or SimpleNamespace(
                    issued=False, token="tok", subject_id=SUBJECT_ID
                ),
            ),
        )

        self.app = FastAPI()
        self.app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
        self.app.include_router(
            chat_v2_module.approvals_v2_router, prefix="/api/v2"
        )
        self.app.state.agent_v2_runtime = SimpleNamespace(
            runs=self.repo, memory=_FakeMemory()
        )

    # -- barrier helpers (per-run workspace) --

    def wait_pdf_barrier(self, run_id: str) -> bool:
        entered, _release = self.computer.barrier(f"/ws/{run_id}")
        return entered.wait(timeout=20)

    def release_pdf_barrier(self, run_id: str) -> None:
        _entered, release = self.computer.barrier(f"/ws/{run_id}")
        release.set()

    def pdf_barrier_released(self, run_id: str) -> bool:
        _entered, release = self.computer.barrier(f"/ws/{run_id}")
        return release.is_set()

    # -- threaded SSE (each request = own thread + own loop + own orchestrator) --

    def sse_thread(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[threading.Thread, dict[str, Any]]:
        result: dict[str, Any] = {}

        def worker() -> None:
            async def run():
                transport = ASGITransport(app=self.app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=60.0
                ) as client:
                    if method == "POST":
                        return await client.post(path, json=body)
                    return await client.get(path)

            try:
                resp = asyncio.run(run())
                result["status"] = resp.status_code
                result["text"] = resp.text
            except BaseException as exc:  # pragma: no cover - failure evidence
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, result

    # -- short-lived loop helpers (no streaming) --

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None):
        async def run():
            transport = ASGITransport(app=self.app)
            async with AsyncClient(
                transport=transport, base_url="http://test", timeout=60.0
            ) as client:
                return await client.request(method, path, json=body)

        return asyncio.run(run())

    def pause_flow(self, request_id: str) -> tuple[str, str]:
        """Drive the artifact first stream to its durable approval pause."""
        response = self._call(
            "POST",
            "/api/v2/chat/sse",
            {
                "message": "规划杭州两日徒步并出 PDF 计划",
                "request_id": request_id,
                "profile_id": "artifact",
            },
        )
        assert response.status_code == 200, response.text
        assert _types(_frames(response.text))[-1] == "run.paused"
        run = self.repo.control.get_run_by_request(request_id)
        assert run is not None and run.status == "paused"
        approval = self.repo.list_pending_approvals_for_run(run.run_id)[0]
        return run.run_id, approval.approval_id

    def decide(self, approval_id: str, decision: str = "approve") -> dict[str, Any]:
        response = self._call(
            "POST",
            f"/api/v2/approvals/{approval_id}/decision",
            {"decision": decision, "decision_request_id": f"dec-{approval_id}"},
        )
        assert response.status_code == 200, response.text
        return response.json()

    def resume_thread(self, run_id: str, *, from_seq: int = 0):
        return self.sse_thread(
            "POST", f"/api/v2/chat/resume/{run_id}?from_seq={from_seq}"
        )

    def repair_call(
        self, effect_id: str, *, action: str, key: str, result: dict | None = None
    ):
        body: dict[str, Any] = {"action": action, "idempotency_key": key}
        if result is not None:
            body["result"] = result
        return self._call("POST", f"/api/v2/chat/effects/{effect_id}/repair", body)

    def snapshot(self, run_id: str) -> dict[str, Any]:
        response = self._call("GET", f"/api/v2/chat/snapshot/{run_id}")
        assert response.status_code == 200, response.text
        return response.json()

    # -- durable reads --

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return self.repo.get_events_since(run_id=run_id, from_seq=0)

    def run_status(self, run_id: str) -> str:
        run = self.repo.get_run(run_id)
        assert run is not None
        return run.status

    def owner_active(self, run_id: str) -> bool:
        return self.repo.run_execution_owner_active(run_id)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="disabled",
    )


def _paused_approved_run(
    harness: OwnershipHarness, request_id: str
) -> tuple[str, str]:
    run_id, approval_id = harness.pause_flow(request_id)
    harness.decide(approval_id, "approve")
    return run_id, approval_id


def _persisted_terminal_count(harness: OwnershipHarness, run_id: str) -> int:
    return sum(
        1
        for evt in harness.events(run_id)
        if evt["type"] in {"error", "done", "cancelled"}
    )


def _sequences_are_unique_and_monotonic(harness: OwnershipHarness, run_id: str) -> bool:
    seqs = [evt["sequence"] for evt in harness.events(run_id)]
    return seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


# ---- seeding helpers (direct ledger writes BEFORE the overlap starts) ---------


def _seed_unknown_effect_with_known_result(
    harness: OwnershipHarness, run_id: str, *, filename: str
) -> str:
    """Drive one effect into outcome_unknown AFTER its result was durable.

    This is the publication-failure shape a ``confirm`` repair may fix
    (review S5). Seeding happens through the real ledger driver with a
    failing strict publish, exactly like the repair-boundary tests.
    """
    from agent_v2.run_orchestration import (
        EFFECT_COPY_GENERATE_PDF,
        EffectDriveResult,
        drive_effect,
    )

    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args={"filename": filename}
    )
    harness.repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id=f"conv-{run_id}",
        tool_name="generate_pdf",
        args_canonical=json.dumps({"filename": filename}),
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    known_result = {"path": filename, "content": f"已在沙箱中生成 {filename}"}
    executed: list[int] = []

    def fail_result(event_type: str, payload: dict[str, Any]):
        if event_type == "tool_result":
            raise sqlite3.OperationalError("success event not durable")

    outcome = EffectDriveResult()

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=harness.repo,
            run_id=run_id,
            conversation_id=f"conv-{run_id}",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=identity.effect_id,
            external_idempotency_key=identity.idempotency_key,
            args_canonical=json.dumps({"filename": filename}),
            tool_args={"filename": filename},
            all_events=[],
            has_tool_call=False,
            has_tool_result_ok=False,
            execute=lambda: executed.append(1) or known_result,
            emit=lambda t, p: SimpleNamespace(type=t, payload=p),
            outcome=outcome,
            emit_required=fail_result,
        ):
            pass

    asyncio.run(run())
    assert outcome.status == "outcome_unknown"
    assert executed == [1]
    row = harness.repo.get_effect(identity.effect_id)
    assert row is not None and row.status == EffectStatus.OUTCOME_UNKNOWN.value
    return identity.effect_id


def _seed_unknown_effect_without_result(
    harness: OwnershipHarness, run_id: str, *, filename: str
) -> str:
    """Drive one effect into outcome_unknown with NO durable result.

    The janitor repair for this shape may only CLAIM the repair for an
    operator — never execute or invent a result.
    """
    from agent_v2.run_orchestration import (
        EFFECT_COPY_GENERATE_PDF,
        EffectDriveResult,
        drive_effect,
    )

    identity = compute_effect_identity(
        run_id=run_id, tool_name="generate_pdf", args={"filename": filename}
    )
    harness.repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id=f"conv-{run_id}",
        tool_name="generate_pdf",
        args_canonical=json.dumps({"filename": filename}),
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    executed: list[int] = []

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("ledger write failed")

    harness.repo.commit_effect = boom  # instance-local: seeding only
    outcome = EffectDriveResult()

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=harness.repo,
            run_id=run_id,
            conversation_id=f"conv-{run_id}",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=identity.effect_id,
            external_idempotency_key=identity.idempotency_key,
            args_canonical=json.dumps({"filename": filename}),
            tool_args={"filename": filename},
            all_events=[],
            has_tool_call=False,
            has_tool_result_ok=False,
            execute=lambda: executed.append(1) or {"path": filename},
            emit=lambda t, p: SimpleNamespace(type=t, payload=p),
            outcome=outcome,
            emit_required=lambda t, p: SimpleNamespace(type=t, payload=p),
        ):
            pass

    asyncio.run(run())
    del harness.repo.commit_effect
    assert outcome.status == "outcome_unknown"
    return identity.effect_id


# ---- T02-01 + T02-02: overlapping resumes, read-only observer ------------------


def test_overlapping_resumes_execute_tool_once_with_single_terminal(
    tmp_path, monkeypatch
):
    """T02-01: 双 resume 重叠只执行一次工具；单一持久终态，无序号冲突。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, _approval = _paused_approved_run(harness, "req-owner-1")

    winner_thread, winner_result = harness.resume_thread(run_id)
    try:
        assert harness.wait_pdf_barrier(run_id), (
            "winner never reached the generate_pdf barrier"
        )

        loser_thread, loser_result = harness.resume_thread(run_id)
        loser_thread.join(timeout=30)
        assert not loser_result.get("error"), loser_result.get("error")
        loser_frames = _frames(loser_result["text"])
        # The observer replays the persisted stream (including the
        # winner's tool_call) and closes WITHOUT any terminal of its own.
        assert "tool_call" in _types(loser_frames)
        assert not [t for t in _types(loser_frames) if t in NEW_TERMINALS]
    finally:
        harness.release_pdf_barrier(run_id)
    winner_thread.join(timeout=30)
    assert not winner_result.get("error"), winner_result.get("error")
    winner_frames = _frames(winner_result["text"])

    # The winner finishes done with exactly one terminal of its own.
    assert winner_frames[-1]["type"] == "done"
    assert harness.run_status(run_id) == "done"
    # The tool executed EXACTLY once across both connections.
    assert len(harness.computer.pdfs) == 1
    # Exactly one durable terminal row and no sequence conflicts.
    assert _persisted_terminal_count(harness, run_id) == 1
    assert _sequences_are_unique_and_monotonic(harness, run_id)
    # The token is gone once the owner settled.
    assert not harness.owner_active(run_id)
    # Two orchestrator instances (one per connection thread) were live
    # during the overlap — ownership is Repository-level, not per-instance.
    assert len(harness.orchestrator_instances) >= 2


def test_losing_resume_is_read_only_observer_exiting_before_barrier_release(
    tmp_path, monkeypatch
):
    """T02-02: 落后连接只读观察：cursor 后回放原身份事件，赢家屏障释放前退出。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, _approval = _paused_approved_run(harness, "req-owner-2")

    winner_thread, _winner_result = harness.resume_thread(run_id)
    assert harness.wait_pdf_barrier(run_id)
    # The winner is parked: effect claimed, tool_call persisted+sent.
    events_before = harness.events(run_id)
    run_status_before = harness.run_status(run_id)
    effect_before = harness.repo.list_effects_for_run(run_id)[0]
    lease_before = harness.repo.get_sandbox_lease(run_id)
    owner_before = lease_before.owner_id if lease_before else ""
    cleanups_before = list(harness.cleanup_calls)
    pdfs_before = len(harness.computer.pdfs)
    kernels_before = len(harness.kernel_calls)

    observer_thread, observer_result = harness.resume_thread(run_id)
    observer_thread.join(timeout=30)
    assert not observer_result.get("error"), observer_result.get("error")

    # The observer exited BEFORE the barrier released (it is still set).
    assert not harness.pdf_barrier_released(run_id)

    observer_frames = _frames(observer_result["text"])
    # Replay-only: every frame matches a persisted row's identity — the
    # original event_id/sequence/timestamp, never regenerated.
    persisted_by_seq = {evt["sequence"]: evt for evt in events_before}
    for frame in observer_frames:
        row = persisted_by_seq.get(frame.get("sequence"))
        assert row is not None, frame
        assert frame.get("event_id") == (row.get("event_id") or "")
        assert frame.get("timestamp") == (row.get("timestamp") or "")
        assert frame.get("type") == row["type"]
    assert [f["sequence"] for f in observer_frames] == sorted(
        f["sequence"] for f in observer_frames
    )
    # The observer's LAST frame is the last persisted event — it closed
    # right there, without emitting anything of its own.
    assert observer_frames[-1]["sequence"] == events_before[-1]["sequence"]
    assert not [t for t in _types(observer_frames) if t in NEW_TERMINALS]

    # Zero mutation of run/effect/lease/cleanup; no sandbox/model work.
    assert harness.events(run_id) == events_before
    assert harness.run_status(run_id) == run_status_before == "running"
    effect_after = harness.repo.list_effects_for_run(run_id)[0]
    assert effect_after.status == effect_before.status
    assert effect_after.claim_token == effect_before.claim_token
    lease_after = harness.repo.get_sandbox_lease(run_id)
    assert lease_after is not None and lease_after.owner_id == owner_before
    assert harness.cleanup_calls == cleanups_before
    assert len(harness.computer.pdfs) == pdfs_before
    assert len(harness.computer.workspaces) == pdfs_before
    assert len(harness.kernel_calls) == kernels_before
    # The observer never cleaned the winner's sandbox and never created
    # a workspace of its own.
    assert harness.computer.cleaned == []

    # Release the winner: it still completes done with one execution.
    harness.release_pdf_barrier(run_id)
    winner_thread.join(timeout=30)
    assert harness.run_status(run_id) == "done"
    assert len(harness.computer.pdfs) == 1
    assert _persisted_terminal_count(harness, run_id) == 1
    assert _sequences_are_unique_and_monotonic(harness, run_id)


# ---- T02-03: first stream vs resume + idempotent re-send ------------------------


def test_first_stream_overlap_and_idempotent_resend_preserve_identity(
    tmp_path, monkeypatch
):
    """T02-03: 首次流与 resume 重叠时原执行继续；同 request_id 重发不重复执行。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)

    # The first stream parks INSIDE the model turn (async gate, loop
    # responsive): run is running and owned by the first stream.
    harness.kernel.armed = True
    first_thread, first_result = harness.sse_thread(
        "POST",
        "/api/v2/chat/sse",
        {
            "message": "带什么装备？",
            "request_id": "req-owner-3",
            "profile_id": "chat",
        },
    )
    assert harness.kernel.entered.wait(timeout=20)

    run = harness.repo.control.get_run_by_request("req-owner-3")
    assert run is not None and run.status == "running"

    # A resume of the SAME run arrives while the first stream executes:
    # it must degrade to a read-only observer — the original executor
    # continues; no second model turn, no mutation.
    events_before = harness.events(run.run_id)
    observer_thread, observer_result = harness.resume_thread(run.run_id)
    observer_thread.join(timeout=30)
    assert not observer_result.get("error"), observer_result.get("error")
    observer_frames = _frames(observer_result["text"])
    assert "run_started" in _types(observer_frames)
    assert not [t for t in _types(observer_frames) if t in TERMINALS]
    assert harness.events(run.run_id) == events_before
    assert len(harness.kernel_calls) == 1
    assert harness.run_status(run.run_id) == "running"

    # A same-request re-send during execution replays without a terminal
    # and without any new write — still exactly one model turn.
    dup_thread, dup_result = harness.sse_thread(
        "POST",
        "/api/v2/chat/sse",
        {
            "message": "带什么装备？",
            "request_id": "req-owner-3",
            "profile_id": "chat",
        },
    )
    dup_thread.join(timeout=30)
    assert not dup_result.get("error"), dup_result.get("error")
    dup_frames = _frames(dup_result["text"])
    assert not [t for t in _types(dup_frames) if t in NEW_TERMINALS]
    # No new event rows, no new sequences: the idempotent admission may
    # rewrite its OWN admission event payload in place, but the identity
    # set (sequence/event_id/type) and the count are unchanged.
    after_dup = harness.events(run.run_id)
    assert len(after_dup) == len(events_before)
    assert [
        (e["sequence"], e.get("event_id"), e["type"]) for e in after_dup
    ] == [
        (e["sequence"], e.get("event_id"), e["type"]) for e in events_before
    ]
    assert len(harness.kernel_calls) == 1

    # Release the model turn: the ORIGINAL first stream finishes done.
    harness.kernel.release.set()
    first_thread.join(timeout=30)
    assert not first_result.get("error"), first_result.get("error")
    assert _frames(first_result["text"])[-1]["type"] == "done"
    assert harness.run_status(run.run_id) == "done"
    assert len(harness.kernel_calls) == 1
    assert _persisted_terminal_count(harness, run.run_id) == 1

    # Terminal replay after completion: same request_id returns the SAME
    # terminal identity (event_id/sequence), no model/tool re-run.
    replay_thread, replay_result = harness.sse_thread(
        "POST",
        "/api/v2/chat/sse",
        {
            "message": "带什么装备？",
            "request_id": "req-owner-3",
            "profile_id": "chat",
        },
    )
    replay_thread.join(timeout=30)
    assert not replay_result.get("error"), replay_result.get("error")
    replay_frames = _frames(replay_result["text"])
    replay_terminal = [f for f in replay_frames if f["type"] == "done"][-1]
    stored_terminal = [
        evt for evt in harness.events(run.run_id) if evt["type"] == "done"
    ][-1]
    assert replay_terminal["event_id"] == stored_terminal["event_id"]
    assert replay_terminal["sequence"] == stored_terminal["sequence"]
    assert replay_terminal["timestamp"] == stored_terminal["timestamp"]
    assert len(harness.kernel_calls) == 1


# ---- T02-04: cross-instance single executor; no global lock ---------------------


def test_shared_repository_two_instances_single_executor_and_no_global_lock(
    tmp_path, monkeypatch
):
    """T02-04: 两编排实例共享 Repository 仍单执行者；屏障中的 run A 不阻塞 run B。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_a, approval_a = _paused_approved_run(harness, "req-owner-4a")

    # Winner on run A parks at the barrier.
    winner_thread, _winner_result = harness.resume_thread(run_a)
    assert harness.wait_pdf_barrier(run_a)
    instances_at_overlap = len(harness.orchestrator_instances)

    # Run A still parked: a competing resume through ANOTHER instance is
    # an observer, and a DIFFERENT run completes fully in the meantime.
    observer_thread, observer_result = harness.resume_thread(run_a)
    observer_thread.join(timeout=30)
    assert not observer_result.get("error"), observer_result.get("error")
    assert not [t for t in _types(_frames(observer_result["text"])) if t in NEW_TERMINALS]

    run_b, approval_b = _paused_approved_run(harness, "req-owner-4b")
    harness.decide(approval_b, "approve")
    b_thread, b_result = harness.resume_thread(run_b)
    assert harness.wait_pdf_barrier(run_b), "run B never reached its tool"
    harness.release_pdf_barrier(run_b)
    b_thread.join(timeout=60)
    assert not b_result.get("error"), b_result.get("error")
    b_frames = _frames(b_result["text"])
    assert b_frames[-1]["type"] == "done"
    assert harness.run_status(run_b) == "done"

    # Release A: it finishes too — one tool execution per run, both done.
    harness.release_pdf_barrier(run_a)
    winner_thread.join(timeout=30)
    assert harness.run_status(run_a) == "done"
    assert len(harness.computer.pdfs) == 2  # one per run, none duplicated
    assert _persisted_terminal_count(harness, run_a) == 1
    assert _persisted_terminal_count(harness, run_b) == 1
    assert _sequences_are_unique_and_monotonic(harness, run_a)
    assert _sequences_are_unique_and_monotonic(harness, run_b)
    # Multiple orchestrator instances over the one Repository existed.
    assert instances_at_overlap >= 2


# ---- T02-05: repair + janitor repair-claim ownership gate ------------------------


def test_effect_repair_and_janitor_claim_pass_through_ownership_gate(
    tmp_path, monkeypatch
):
    """T02-05: repair 及 janitor repair claim 先经执行权；活跃时 repair_conflict。

    Phase 1 (conflict): the run's executor is parked at the barrier with
    its effect CLAIMED — an HTTP repair attempt and a janitor repair
    claim on that run are both refused with the existing
    ``repair_conflict`` semantics, state and cursor unchanged.
    Phase 2 (after release): repairs proceed per existing policy — the
    committed effect confirms idempotently from the ledger, and an
    unknown-outcome effect (seeded after the run settled) is repaired /
    claim-only through the very same entry.
    """
    _require_ownership_api()
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, _approval = _paused_approved_run(harness, "req-owner-5")

    # Winner resume parks at the barrier → the run has an active owner
    # and its generate_pdf effect is CLAIMED (tool_call already sent).
    winner_thread, _winner_result = harness.resume_thread(run_id)
    assert harness.wait_pdf_barrier(run_id)
    effects = harness.repo.list_effects_for_run(run_id)
    claimed_effect = next(
        e for e in effects if e.status == EffectStatus.EXECUTING.value
    )
    claimed_id = claimed_effect.resolved_effect_id
    events_before = harness.events(run_id)
    cursor_before = max(evt["sequence"] for evt in events_before)

    # -- phase 1a: HTTP repair while the owner is active ------------------
    response = harness.repair_call(claimed_id, action="confirm", key="rk-1")
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "repair_conflict"
    assert harness.events(run_id) == events_before
    effect_during = harness.repo.get_effect_by_effect_id(claimed_id)
    assert effect_during.status == claimed_effect.status
    assert effect_during.claim_token == claimed_effect.claim_token

    # -- phase 1b: the janitor repair claim is refused the same way -------
    task, created = harness.repo.enqueue_janitor_task(
        kind="repair",
        run_id=run_id,
        effect_id=claimed_id,
        evidence={"run_id": run_id, "effect_id": claimed_id},
    )
    assert created
    results = run_pending_janitor_tasks(harness.repo)
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "repair_conflict" in results[0].error
    effect_after_janitor = harness.repo.get_effect_by_effect_id(claimed_id)
    assert effect_after_janitor.repair_status == claimed_effect.repair_status
    assert harness.events(run_id) == events_before
    assert max(evt["sequence"] for evt in harness.events(run_id)) == cursor_before

    # -- phase 2: release; repairs proceed per existing policy -------------
    harness.release_pdf_barrier(run_id)
    winner_thread.join(timeout=30)
    assert harness.run_status(run_id) == "done"

    # The settled (committed) effect confirms idempotently from the ledger.
    response = harness.repair_call(claimed_id, action="confirm", key="rk-2")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "idempotent"
    assert harness.repo.get_effect_by_effect_id(claimed_id).status == (
        EffectStatus.COMMITTED.value
    )

    # A true unknown-outcome effect (publication-failure shape) on the now
    # settled run repairs through the SAME entry.
    repairable_id = _seed_unknown_effect_with_known_result(
        harness, run_id, filename="p-repair.pdf"
    )
    response = harness.repair_call(
        repairable_id, action="confirm", key="rk-3"
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "repaired"
    assert harness.repo.get_effect_by_effect_id(repairable_id).status == (
        EffectStatus.COMMITTED.value
    )

    # The janitor retry on the failed task now runs per policy, and a
    # claim-only repair (unknown, no durable result) claims for an
    # operator — never executes, never invents a result.
    claimable_id = _seed_unknown_effect_without_result(
        harness, run_id, filename="p-claim.pdf"
    )
    _task2, created2 = harness.repo.enqueue_janitor_task(
        kind="repair",
        run_id=run_id,
        effect_id=claimable_id,
        evidence={"run_id": run_id, "effect_id": claimable_id},
    )
    assert created2
    results = run_pending_janitor_tasks(harness.repo)
    assert len(results) == 2
    by_task = {r.task_id: r for r in results}
    statuses = sorted(r.status for r in results)
    assert statuses == ["done", "done"], results
    for result in results:
        if result.outcome.get("effect_id") == claimable_id:
            assert result.outcome["repair_status"] == "repair_claimed"
            assert result.outcome["next_action"] == "manual_confirm"
        else:
            # The retried task on the committed effect: already repaired.
            assert result.outcome.get("already_repaired") is True
    assert len(by_task) == 2
    claimed = harness.repo.get_effect_by_effect_id(claimable_id)
    assert claimed.status == EffectStatus.OUTCOME_UNKNOWN.value


# ---- T02-06: token identity, double release, error exits -------------------------


def test_old_token_double_release_cannot_release_successor(tmp_path):
    """T02-06: 旧 token 重复释放不能释放后继 owner。"""
    _require_ownership_api()
    repo = RunRepository(
        control=ControlStore(str(tmp_path / "c.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "k.sqlite")),
    )
    try:
        first = repo.try_acquire_run_execution("run-token")
        assert first is not None
        assert repo.try_acquire_run_execution("run-token") is None  # non-blocking
        assert repo.release_run_execution("run-token", first) is True

        successor = repo.try_acquire_run_execution("run-token")
        assert successor is not None
        assert successor.serial != first.serial

        # The stale token cannot release the successor, and cannot be
        # used to admit a second owner either.
        assert repo.release_run_execution("run-token", first) is False
        assert repo.try_acquire_run_execution("run-token") is None
        assert repo.run_execution_owner_active("run-token") is True

        assert repo.release_run_execution("run-token", successor) is True
        third = repo.try_acquire_run_execution("run-token")
        assert third is not None
        assert repo.release_run_execution("run-token", third) is True
        assert not repo.run_execution_owner_active("run-token")
    finally:
        repo.close()


def test_error_exit_releases_owned_token(tmp_path, monkeypatch):
    """T02-06: 错误出口同样移除所属 token（resume fail-closed 路径）。"""
    _require_ownership_api()
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, approval_id = harness.pause_flow("req-owner-6")
    harness.decide(approval_id, "approve")
    # Kill the durable lease first: the resume must fail closed with
    # ``sandbox_lease_lost`` — an ERROR exit that still releases the token.
    harness.repo.destroy_sandbox_lease(run_id=run_id, failure_reason="test")

    thread, result = harness.resume_thread(run_id)
    thread.join(timeout=30)
    assert not result.get("error"), result.get("error")
    frames = _frames(result["text"])
    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["code"] == "sandbox_lease_lost"
    assert harness.run_status(run_id) == "error"
    # The token is gone after the error exit: a new owner may take over.
    assert not harness.owner_active(run_id)
    token = harness.repo.try_acquire_run_execution(run_id)
    assert token is not None
    assert harness.repo.release_run_execution(run_id, token) is True


# ---- T02-07: deterministic generator closing + release after settlement ----------


def test_resume_connection_close_closes_owned_generators_and_releases_late(
    tmp_path, monkeypatch
):
    """T02-07: 立即关闭/断线确定关闭所属嵌套生成器；结算完成前不释放 owner。"""
    _require_ownership_api()
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, _approval = _paused_approved_run(harness, "req-owner-7")

    # Order evidence: record every token release with a sequence number.
    release_log: list[tuple[str, float]] = []
    real_release = harness.repo.release_run_execution

    def recording_release(run_id: str, token) -> bool:
        release_log.append((run_id, time.perf_counter()))
        return real_release(run_id, token)

    harness.repo.release_run_execution = recording_release

    # Call the production route function directly in-loop and close the
    # response body iterator at the continuation's first frame — the
    # deterministic equivalent of the connection being torn down while
    # the resume is mid-continuation.
    from starlette.requests import Request

    async def close_mid_continuation() -> list[bytes]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "path": f"/api/v2/chat/resume/{run_id}",
            "raw_path": f"/api/v2/chat/resume/{run_id}".encode(),
            "query_string": b"from_seq=0",
            "server": ("test", 80),
            "client": ("testclient", 123),
            "app": harness.app,
        }
        request = Request(scope)
        response = await chat_v2_module.chat_v2_resume(
            run_id, request, from_seq=0
        )
        chunks: list[str] = []
        body = response.body_iterator
        async for chunk in body:
            chunks.append(chunk)
            if "approval.resolved" in chunk:
                break
        await body.aclose()  # the connection's owned generator is closed
        return chunks

    chunks = asyncio.run(close_mid_continuation())
    assert any("approval.resolved" in c for c in chunks)

    # The close unwound the WHOLE owned chain: cleanup ran, the token was
    # released exactly once for this run, and the release happened AFTER
    # the settlement (cleanup) evidence.
    assert harness.cleanup_calls, "owned nested generator cleanup never ran"
    assert [run for run, _ in release_log] == [run_id]
    assert release_log[0][1] > 0.0
    assert not harness.owner_active(run_id)
    # No terminal was invented by the close; the run stays mid-flight.
    assert _persisted_terminal_count(harness, run_id) == 0
    stored_types = [evt["type"] for evt in harness.events(run_id)]
    assert "done" not in stored_types
    # The effect was claimed but the tool never completed — the run was
    # closed mid-execution, so nothing may publish a success result.
    effects = harness.repo.list_effects_for_run(run_id)
    assert all(
        eff.status != EffectStatus.COMMITTED.value for eff in effects
    ), "a closed mid-execution connection must not settle a success"
    del harness.repo.release_run_execution


def test_client_disconnect_closes_first_stream_generators_and_releases_token(
    tmp_path, monkeypatch
):
    """T02-07: 客户端断线（真实 ASGI http.disconnect）关闭生成器并释放 token。"""
    _require_ownership_api()
    harness = OwnershipHarness(tmp_path, monkeypatch)
    body = {
        "message": "带什么装备？",
        "request_id": "req-owner-8",
        "profile_id": "chat",
    }

    harness.kernel.armed = True
    # Raw ASGI: drive the production app; after the heartbeat frame is
    # sent, deliver http.disconnect while the first stream is parked in
    # its model turn (the async gate keeps the loop responsive, so the
    # cancellation lands INSIDE the generator deterministically).
    async def run_until_disconnect() -> list[bytes]:
        chunks: list[bytes] = []
        disconnect = asyncio.Event()
        incoming: asyncio.Queue = asyncio.Queue()
        await incoming.put(
            {
                "type": "http.request",
                "body": json.dumps(body).encode(),
                "more_body": False,
            }
        )
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "path": "/api/v2/chat/sse",
            "raw_path": b"/api/v2/chat/sse",
            "query_string": b"",
            "server": ("test", 80),
            "client": ("testclient", 123),
            "app": harness.app,
        }

        async def receive():
            if disconnect.is_set():
                return {"type": "http.disconnect"}
            try:
                return incoming.get_nowait()
            except asyncio.QueueEmpty:
                await disconnect.wait()
                return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                chunks.append(message["body"])
                if b'"heartbeat"' in message["body"]:
                    disconnect.set()

        await harness.app(scope, receive, send)
        return chunks

    chunks = asyncio.run(run_until_disconnect())
    frame_types = _types(_frames(b"".join(chunks).decode("utf-8", "replace")))
    assert "heartbeat" in frame_types
    # The kernel turn was entered and the disconnect cancelled it.
    assert harness.kernel.entered.wait(timeout=5)
    harness.kernel.release.set()  # let any parked executor thread finish

    # The generator unwind released the token; no terminal was sent or
    # persisted; the model ran exactly once (no duplicate submit).
    assert not harness.owner_active(
        harness.repo.control.get_run_by_request("req-owner-8").run_id
    )
    assert not [t for t in frame_types if t in TERMINALS]
    stored = harness.events(
        harness.repo.control.get_run_by_request("req-owner-8").run_id
    )
    assert not [evt for evt in stored if evt["type"] in {"done", "error", "cancelled"}]
    assert len(harness.kernel_calls) == 1


# ---- T02-08: restart convergence without the liveness inference ------------------


def test_restart_claimed_effect_converges_unknown_without_rerun(tmp_path, monkeypatch):
    """T02-08: 重启后无存活 owner 的 executing effect 按未知结果收敛，不重跑。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, approval_id = harness.pause_flow("req-owner-9")
    harness.decide(approval_id, "approve")

    # Durable post-crash state: the executor claimed the effect and its
    # tool_call WAS persisted moments ago. The in-process owner died with
    # the "restart"; the recent tool_call must no longer imply liveness.
    identity = compute_effect_identity(
        run_id=run_id,
        tool_name="generate_pdf",
        args={"filename": "hiking-plan.pdf"},
    )
    harness.repo.insert_effect(
        effect_key=identity.effect_id,
        run_id=run_id,
        conversation_id=f"conv-{run_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "hiking-plan.pdf"}',
        external_idempotency_key=identity.idempotency_key,
        effect_id=identity.effect_id,
        idempotency_key=identity.idempotency_key,
    )
    _record, won = harness.repo.transition_effect_to_executing(identity.effect_id)
    assert won
    harness.repo.persist_event(
        run_id=run_id,
        sequence=100,
        event_type="tool_call",
        payload={
            "tool": "generate_pdf",
            "args": {"filename": "hiking-plan.pdf"},
        },
        request_id="req-owner-9",
        conversation_id=f"conv-{run_id}",
        timestamp=_now_iso(),
    )

    thread, result = harness.resume_thread(run_id)
    thread.join(timeout=30)
    assert not result.get("error"), result.get("error")
    frames = _frames(result["text"])
    frame_types = _types(frames)

    # Converged to outcome_unknown — NOT effect_in_flight.
    assert "effect.outcome_unknown" in frame_types
    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["code"] == "effect_outcome_unknown"
    assert "effect_in_flight" not in [
        f["payload"].get("code") for f in frames if f["type"] == "error"
    ]
    # Nothing re-ran: no sandbox command, no model call beyond the
    # pause flow's single plan draft.
    assert harness.computer.pdfs == []
    assert harness.computer.bash_calls == []
    assert len(harness.kernel_calls) == 1
    # Original identity and evidence preserved; never marked success.
    effect = harness.repo.get_effect(identity.effect_id)
    assert effect is not None
    assert effect.status == EffectStatus.OUTCOME_UNKNOWN.value
    assert effect.effect_key == identity.effect_id
    stored = harness.events(run_id)
    ok_results = [
        evt
        for evt in stored
        if evt["type"] == "tool_result" and evt["payload"].get("status") == "ok"
    ]
    assert ok_results == []
    assert harness.run_status(run_id) == "error"
    assert not harness.owner_active(run_id)


# ---- T02-09: replay identity + re-read cursor ------------------------------------


def test_replay_identity_for_full_tail_and_cursor(tmp_path, monkeypatch):
    """T02-09: from_seq=0、末尾与当前 cursor 回放均保留原持久事件身份。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, _approval = _paused_approved_run(harness, "req-owner-10")
    # Complete the run once; the durable log is now fixed.
    thread, result = harness.resume_thread(run_id)
    assert harness.wait_pdf_barrier(run_id)
    harness.release_pdf_barrier(run_id)
    thread.join(timeout=30)
    assert not result.get("error"), result.get("error")
    assert harness.run_status(run_id) == "done"
    stored = harness.events(run_id)
    last_seq = max(evt["sequence"] for evt in stored)
    cursor = harness.snapshot(run_id)["sequence_cursor"]

    def assert_identity(frames: list[dict[str, Any]], from_seq: int) -> None:
        expected = [evt for evt in stored if evt["sequence"] > from_seq]
        assert [f["sequence"] for f in frames] == [
            evt["sequence"] for evt in expected
        ]
        for frame, evt in zip(frames, expected):
            assert frame["type"] == evt["type"]
            assert frame["event_id"] == (evt.get("event_id") or "")
            assert frame["timestamp"] == (evt.get("timestamp") or "")

    # Full replay.
    full_thread, full_result = harness.resume_thread(run_id, from_seq=0)
    full_thread.join(timeout=30)
    assert_identity(_frames(full_result["text"]), 0)

    # Tail replay.
    tail_thread, tail_result = harness.resume_thread(run_id, from_seq=last_seq)
    tail_thread.join(timeout=30)
    assert_identity(_frames(tail_result["text"]), last_seq)

    # Current-cursor replay: nothing new — close without a second
    # terminal and without allocating a sequence.
    cursor_thread, cursor_result = harness.resume_thread(
        run_id, from_seq=cursor
    )
    cursor_thread.join(timeout=30)
    assert not cursor_result.get("error"), cursor_result.get("error")
    assert _frames(cursor_result["text"]) == []
    assert harness.events(run_id) == stored
    assert _persisted_terminal_count(harness, run_id) == 1


def test_acquired_continuation_uses_reread_cursor(tmp_path, monkeypatch):
    """T02-09: 获权后的继续执行使用重新读取的 cursor（不重复已持久事件）。"""
    harness = OwnershipHarness(tmp_path, monkeypatch)
    run_id, approval_id = harness.pause_flow("req-owner-11")
    harness.decide(approval_id, "approve")

    # Capture the STALE view first (what this connection's pre-acquire
    # replay read will see), then let the "prior owner" durably persist
    # approval.resolved as seq 6 — the exact window between the loser's
    # replay read and its acquisition.
    real_replay = harness.repo.get_replay_events
    stale_view = real_replay(run_id=run_id, from_seq=0)
    stale_max = max((evt["sequence"] for evt in stale_view), default=0)
    harness.repo.persist_event(
        run_id=run_id,
        sequence=stale_max + 1,
        event_type="approval.resolved",
        payload={
            "approval_id": approval_id,
            "tool": "generate_pdf",
            "decision": "approve",
            "status": "approved",
            "message": "用户已批准操作，继续执行。",
        },
        request_id="req-owner-11",
        conversation_id=f"conv-{run_id}",
        event_id=f"evt-resolved-{stale_max + 1}",
        timestamp=_now_iso(),
    )
    read_calls = {"n": 0}

    def stale_then_real(*, run_id: str, from_seq: int = 0):
        read_calls["n"] += 1
        if read_calls["n"] == 1:
            return [evt for evt in stale_view if evt["sequence"] > from_seq]
        return real_replay(run_id=run_id, from_seq=from_seq)

    harness.repo.get_replay_events = stale_then_real
    try:
        thread, result = harness.resume_thread(run_id)
        assert harness.wait_pdf_barrier(run_id)
        harness.release_pdf_barrier(run_id)
        thread.join(timeout=60)
        assert not result.get("error"), result.get("error")
    finally:
        del harness.repo.get_replay_events

    frames = _frames(result["text"])
    frame_types = _types(frames)
    # The stale replay missed seq 6 and the re-read facts say it is
    # already persisted: the continuation must NOT re-emit
    # approval.resolved — it deduplicates on the RE-READ event log.
    assert frame_types.count("approval.resolved") == 0
    assert frames[-1]["type"] == "done"
    stored = harness.events(run_id)
    stored_resolved = [
        evt for evt in stored if evt["type"] == "approval.resolved"
    ]
    assert len(stored_resolved) == 1
    assert stored_resolved[0]["sequence"] == stale_max + 1
    # Sequences continue from the RE-READ cursor: unique and monotonic,
    # the tool ran exactly once, single terminal.
    assert _sequences_are_unique_and_monotonic(harness, run_id)
    assert len(harness.computer.pdfs) == 1
    assert _persisted_terminal_count(harness, run_id) == 1
    assert harness.run_status(run_id) == "done"
