"""Ticket 10 — 断线恢复、取消和幂等 for the plan book.

The recovery machinery already exists (``planV2Recovery`` on the client,
``GET /chat/snapshot``, ``GET /chat/events``, ``POST /chat/resume``, the CAS
effect ledger, ``mark_terminal_safe``). Nothing here builds a second recovery
system: these tests prove the PLAN-BOOK flow survives it, and close the gaps
that only appear once the plan draft rows, the version history and the
approval-paused PDF generation are in play.

Acceptance criteria and the seam each one is proven at:

* AC1 SSE 断线后可以通过快照或事件恢复 — the real HTTP endpoints replay a plan
  turn (including its ``plan.draft_state``) after a disconnect, after a runtime
  rebuild, and when the client only holds its ``request_id``; an abandoned plan
  turn leaves no half-written plan state and no sandbox behind.
* AC2 重复请求不会生成重复 PDF — a duplicate ``request_id`` for the plan-book PDF
  run replays and publishes nothing new (one ``artifacts`` row, one version row).
* AC3 审批后恢复不会重复执行生成 — two resumes of ONE generation execute the PDF
  once; a genuinely new ``plan_revision`` still executes (ticket 09's control).
* AC4 取消可以清理沙箱和未完成状态 — cancel destroys the lease AND releases the
  capacity slot; the deliberate ``preserve_sandbox`` leak on pause stays a leak
  until the janitor reclaims it; ``plan_cancel`` clears unfinished rows and keeps
  published version history.
* AC5 终态事件、草稿状态和 artifact 记录保持一致 — after every interruption the
  terminal event, the run row, the draft rows and the artifact/version rows
  agree.

Style follows the repo convention (``test_main_flow_resilience.py`` /
``test_route_report_export.py``): real temp SQLite, real ``LocalNativeComputer``
where the claim is about bytes, ``httpx.ASGITransport`` for the wire, a fake
kernel for the model-free plan turns. No live model, no network, no gateway.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_api
from agent_v2 import kernel
from agent_v2.computer import LocalNativeComputer
from agent_v2.memory_service import MemoryService
from agent_v2.observability import CapacityGuard
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import SUPPLEMENT_ARRANGEMENT_KEYS
from agent_v2.repositories.memory import MemoryStore
from agent_v2.run_orchestration import ChatFlowRequest
from agent_v2.runs import RunRepository
from api.chat_v2 import artifacts_v2_router, _cleanup_sandbox
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router

CONV = "conv-recovery-10"

_ARRANGEMENTS = {key: "" for key in SUPPLEMENT_ARRANGEMENT_KEYS}
_ARRANGEMENTS.update({
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
})

# notes-05 landmine 7: the offline research fixtures only speak for the demo
# route, so every plan turn in this file submits that destination.
SUBMIT_PAYLOAD = {
    "message": "提交草稿",
    "conversation_id": CONV,
    "profile_id": "chat",
    "intent": "plan_create",
    "plan_input": {
        "destination_or_route": "演示林间环线",
        "activity_date": "2026-11-08",
        "participants": 12,
        "arrangements": _ARRANGEMENTS,
    },
}


def _settings(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True, openai_api_key="test", openai_model="test",
        agent_v2_cowork_policy="disabled", agent_v2_state_dir=str(tmp_path),
        route_compare_data_mode="offline", route_decision_provider="deterministic",
        route_compare_demo_scenario="complete",
    )


def _frames(response) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _drafts(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in frames if f["type"] == "plan.draft_state"]


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """Rebuildable v2 runtime over real SQLite, plus a REAL CapacityGuard.

    ``build()`` makes a fresh app over the SAME database files, i.e. a process
    restart: everything a test asserts after ``build()`` is durable state, not
    an object cache. The capacity guard is a real one (process-local semaphores)
    because AC4's claim is about the capacity slot, not a fake's call log.
    """
    settings = _settings(tmp_path)
    monkeypatch.setattr(chat_api, "settings", settings)

    repos: list[RunRepository] = []
    model_calls: list[str] = []

    def build() -> FastAPI:
        repo = RunRepository(
            control=ControlStore(str(tmp_path / "control.sqlite")),
            checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")),
        )
        repos.append(repo)
        memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
        app = FastAPI()
        app.state.agent_v2_runtime = SimpleNamespace(
            runs=repo, memory=memory, capacity=CapacityGuard(max_runs=5, max_sandboxes=2),
        )
        app.include_router(chat_api.chat_v2_router, prefix="/api/v2")
        app.include_router(chat_api.approvals_v2_router, prefix="/api/v2")
        app.include_router(chat_api.artifacts_v2_router, prefix="/api/v2")
        app.include_router(memory_v2_router, prefix="/api/v2")
        return app

    async def fake_model(**kwargs):
        model_calls.append(kwargs["message"])
        return "普通回答。", []

    monkeypatch.setattr(kernel, "run_chat_turn", fake_model)

    app = build()
    memory = app.state.agent_v2_runtime.memory
    subject = memory.resolve_subject(None)
    yield SimpleNamespace(app=app, build=build, repo=repos[-1], memory=memory,
                          subject=subject, model_calls=model_calls, path=tmp_path)
    for repo in repos:
        repo.close()
    memory.close()


def _client(app, subject) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                       cookies={SUBJECT_COOKIE_NAME: subject.token})


def _request_id(index: int) -> str:
    return f"req-t10-{index}"


def _submit(index: int) -> dict[str, Any]:
    return {**SUBMIT_PAYLOAD, "request_id": _request_id(index)}


def _post_sse(app, subject, payload: dict[str, Any]) -> list[dict[str, Any]]:
    async def run():
        async with _client(app, subject) as client:
            return _frames(await client.post("/api/v2/chat/sse", json=payload))

    return asyncio.run(run())


#: The one advancing click for each source stage, walking a submitted draft to
#: ``final_reviewed`` — the stage ticket 07 AC4's PDF door now reads. Every turn
#: is a model-free chat-profile intent (the seam contract's ruling), so no plan
#: turn here reaches the kernel.
_ADVANCE = {
    "minimal_draft": "plan_first_review",
    "awaiting_first_review": "plan_confirm_arrangements",
    "first_reviewed": "plan_final_review",
    "awaiting_final_review": "plan_confirm_final",
}


def _advance_to_final(app, subject, repo, *, tag: str) -> None:
    """Drive the conversation's draft through the review intents to final_reviewed.

    Idempotent: a draft already at final_reviewed does nothing. Used by
    :func:`_pause_and_approve` so every plan-book publish here has genuinely
    confirmed the final review, which the export turn's confirmation door
    requires (ticket 07 AC4).
    """
    for step in range(5):
        stage = repo.get_plan_draft(CONV).stage
        if stage == "final_reviewed":
            return
        intent = _ADVANCE.get(stage)
        assert intent is not None, f"cannot advance a plan at stage {stage!r}"
        frames = _post_sse(app, subject, {
            "message": "确认", "request_id": f"{tag}-advance-{step}",
            "conversation_id": CONV, "profile_id": "chat", "intent": intent})
        assert frames[-1]["type"] == "done", (intent, frames[-1])
    raise AssertionError("plan did not reach final_reviewed")


def _get(app, subject, path: str, params: dict[str, Any] | None = None):
    async def run():
        async with _client(app, subject) as client:
            return await client.get(path, params=params or {})

    return asyncio.run(run())


# ---- AC1: a disconnected plan turn recovers from snapshot / events --------

def test_a_lost_plan_stream_replays_the_draft_state_from_the_event_log(flow):
    """AC1: the plan turn's own events are the recovery source.

    The client had the stream cut after the draft-state frame: the event log
    must hand back the SAME payload (byte-for-byte) plus the terminal, so the
    restored mirror cannot disagree with the stored draft row.
    """
    frames = _post_sse(flow.app, flow.subject, _submit(1))
    live_draft = _drafts(frames)[0]
    run_id = live_draft["run_id"]

    tail = _get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}",
                {"from_seq": live_draft["sequence"] - 1})
    assert tail.status_code == 200
    replayed = tail.json()["events"]
    assert [e["sequence"] for e in replayed] == list(
        range(live_draft["sequence"], frames[-1]["sequence"] + 1))
    replayed_draft = next(e for e in replayed if e["type"] == "plan.draft_state")
    assert replayed_draft["payload"] == live_draft["payload"]
    assert replayed[-1]["type"] == "done"

    snapshot = _get(flow.app, flow.subject, f"/api/v2/chat/snapshot/{run_id}")
    assert snapshot.status_code == 200
    body = snapshot.json()
    assert body["status"] == "done"
    assert body["terminal_type"] == "done"
    assert body["sequence_cursor"] == frames[-1]["sequence"]
    # The draft row behind the event is the same stage the replay shows.
    stored = flow.repo.get_plan_draft(CONV)
    assert stored is not None and stored.stage == live_draft["payload"]["stage"]
    # Deterministic plan flow: recovery never re-runs a model.
    assert flow.model_calls == []


def test_a_repeated_plan_request_replays_identically_after_a_runtime_restart(flow):
    """AC1 + AC2: after a restart the same ``request_id`` is a replay.

    A refresh mid-conversation re-POSTs the same idempotency key. It must
    reproduce the original frames exactly — same ``plan.draft_state``, same
    terminal — and must not advance the draft a second time or leave a second
    run row.
    """
    original = _post_sse(flow.app, flow.subject, _submit(2))
    restarted = flow.build()
    replay = _post_sse(restarted, flow.subject, _submit(2))

    assert replay == original
    assert _drafts(replay)[0]["payload"]["stage"] == "minimal_draft"
    run_id = flow.repo.resolve_run_id(_request_id(2))
    assert run_id == original[0]["run_id"]
    assert flow.model_calls == []
    # The draft row was moved by ONE run, and that run is the replayed one.
    stored = flow.repo.get_plan_draft(CONV)
    assert stored is not None and stored.updated_by_run_id == run_id


def test_a_plan_refresh_before_the_first_event_recovers_by_request_id(flow):
    """AC1: everything a client can know before the first frame is
    ``request_id`` — snapshot, events AND cancel resolve it.

    That is the "刷新发生在第一帧之前" case: without the fallback the client
    would 404, drop its recovery context and orphan a live server-side run.
    """
    payload = _submit(3)
    frames = _post_sse(flow.app, flow.subject, payload)
    request_id = payload["request_id"]
    run_id = frames[0]["run_id"]

    snapshot = _get(flow.app, flow.subject, f"/api/v2/chat/snapshot/-",
                   {"request_id": request_id})
    assert snapshot.status_code == 200
    assert snapshot.json()["run_id"] == run_id

    events = _get(flow.app, flow.subject, "/api/v2/chat/events/-",
                  {"request_id": request_id, "from_seq": 0})
    assert events.status_code == 200
    body = events.json()
    assert body["run_id"] == run_id
    assert [e["type"] for e in body["events"]] == [f["type"] for f in frames]
    assert _drafts(body["events"])[0]["payload"] == _drafts(frames)[0]["payload"]

    # Cancel by request_id resolves the same way (AC4's early-cancel half).
    async def cancel():
        async with _client(flow.app, flow.subject) as client:
            return await client.post(
                f"/api/v2/chat/cancel/{request_id}", json={"idempotency_key": "early"})

    outcome = asyncio.run(cancel()).json()
    assert outcome["run_id"] == run_id
    # A terminal run cannot be cancelled: the reason names the real terminal,
    # so the client learns the run finished rather than that it is unknown.
    assert outcome["accepted"] is False
    assert outcome["reason"] == "already_terminal:done"


def test_an_abandoned_plan_turn_leaves_no_half_written_state(flow):
    """AC1 + AC5: cutting the transport mid-stream cannot corrupt the draft.

    The consumer closes the generator right after ``run_started`` (the disconnect
    the SSE client reports as ``stream_closed_without_terminal``). Nothing after
    that point ran, so the conversation holds no draft row, no artifact, no
    version row and no sandbox lease — and the NEXT turn still works from the
    stage the server actually holds. Transport closure is not a terminal, so the
    run row stays non-terminal rather than being faked into ``done``.
    """
    async def abandon():
        orchestrator = chat_api._build_run_orchestrator(flow.app.state.agent_v2_runtime)
        stream = orchestrator.stream_chat(
            ChatFlowRequest(**{**SUBMIT_PAYLOAD, "request_id": _request_id(4)}),
            subject_id=flow.subject.subject_id,
        )
        seen: list[str] = []
        async for envelope in stream:
            seen.append(envelope.type)
            break
        await stream.aclose()
        return seen

    seen = asyncio.run(abandon())
    assert seen == ["run_started"]
    run_id = flow.repo.resolve_run_id(_request_id(4))
    assert run_id is not None
    assert flow.repo.get_plan_draft(CONV) is None
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert flow.repo.get_sandbox_lease(run_id) is None
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert [e["type"] for e in flow.repo.get_events_since(run_id=run_id, from_seq=0)] \
        == ["run_started"]

    # A later turn for the same conversation is unaffected by the orphan.
    frames = _post_sse(flow.app, flow.subject, _submit(5))
    assert _drafts(frames)[0]["payload"]["stage"] == "minimal_draft"
    assert frames[-1]["type"] == "done"
    assert flow.model_calls == []


# ---- The plan book's export run (AC2 / AC3 / AC4 / AC5 subject) -----------
#
# The plan book is published through the EXISTING trigger sites: a route_plan
# turn whose kernel calls ``route_compare`` + ``report_export``, then the same
# HIGH-risk ``generate_pdf`` approval pause. These helpers drive that once and
# let each test ask its own question about it. The computer is the real
# ``LocalNativeComputer`` (mkdtemp workspace, real ReportLab render, real
# verified host export) because AC2/AC3 are claims about bytes and rows, not
# about a mock's bookkeeping.

def _recording_computer(produced: list[str]):
    class _Computer(LocalNativeComputer):
        def generate_pdf(self, workspace_path, content, filename):
            produced.append(content)
            return super().generate_pdf(workspace_path, content, filename)

    return _Computer()


def _install_route_plan_model(monkeypatch, flow, *, meeting_point: str):
    """Fake kernel that performs the sourced-comparison + export tool pair.

    One call per run: the count doubles as the AC3 assertion that a resume does
    not re-ask the model for the plan body (the report is code-rendered).
    """

    async def model(**kwargs):
        flow.model_calls.append(kwargs["message"])
        compare = next(t for t in kwargs["extra_tools"] if t.name == "route_compare")
        export = next(t for t in kwargs["extra_tools"] if t.name == "report_export")
        result = json.loads((await compare(candidates=[{
            "name": "演示林间环线", "city": "演示城市",
            "origin": "起点", "destination": "终点"}])).output)
        await export(comparison_id=result["comparison_id"], plan_details={
            "activity_date": "2026-11-08", "meeting_time": "08:30",
            "meeting_point": meeting_point, "participants": 12,
        })
        return "模型正文不应进入计划书", []

    monkeypatch.setattr(kernel, "run_chat_turn", model)


def _export_payload(request_id: str) -> dict[str, Any]:
    """The plan book's export turn: a route_plan turn, no structured intent.

    Deliberately NOT a plan-intent request — :func:`evaluate_plan_intent` stays
    the only plan transition decider, and the PDF is queued by the existing
    ``report_export`` trigger site (ticket 09's rule: ``plan_regenerate`` alone
    never queues a PDF).
    """
    return {
        "message": "请导出带来源的计划书 PDF", "request_id": request_id,
        "conversation_id": CONV, "profile_id": "route_plan",
    }


def _pause_and_approve(flow, monkeypatch, *, request_id: str, meeting_point: str,
                       produced: list[str], submit_draft: bool = True):
    """Drive the plan book up to the pause, record the approval, do NOT resume.

    Returns ``(paused_frames, run_id, last_seq)`` — the state a client has when
    the stream closed on ``run.paused``.
    """
    if submit_draft:
        draft_frames = _post_sse(flow.app, flow.subject, _submit(f"book-{request_id}"))
        assert draft_frames[-1]["type"] == "done"
        assert _drafts(draft_frames)[0]["payload"]["draft"]["arrangements"]["meeting_point"] \
            == _ARRANGEMENTS["meeting_point"]
        # Ticket 07 AC4: the export that queues the PDF is now gated on the
        # confirmed final review, so every publish journey confirms first.
        _advance_to_final(flow.app, flow.subject, flow.repo, tag=f"book-{request_id}")

    _install_route_plan_model(monkeypatch, flow, meeting_point=meeting_point)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: _recording_computer(produced))

    async def run():
        async with _client(flow.app, flow.subject) as client:
            paused = _frames(await client.post(
                "/api/v2/chat/sse", json=_export_payload(request_id)))
            assert paused[-1]["type"] == "run.paused", paused
            approval_payload = next(
                e["payload"] for e in paused if e["type"] == "approval.required")
            decided = await client.post(
                f"/api/v2/approvals/{approval_payload['approval_id']}/decision",
                json={"decision": "approve", "decision_request_id": f"dec-{request_id}"})
            assert decided.status_code == 200
            return paused, approval_payload

    paused, approval_payload = asyncio.run(run())
    return paused, paused[-1]["run_id"], paused[-1]["sequence"], approval_payload


def _resume(app, subject, run_id: str, from_seq: int) -> list[dict[str, Any]]:
    async def run():
        async with _client(app, subject) as client:
            return _frames(await client.post(
                f"/api/v2/chat/resume/{run_id}", params={"from_seq": from_seq}))

    return asyncio.run(run())


def _pause_plan_book(flow, monkeypatch, *, request_id: str, meeting_point: str,
                     produced: list[str]) -> tuple[list[dict], list[dict]]:
    """Open the plan book once and resume it: ``(paused_frames, resume_frames)``."""
    paused, run_id, last_seq, _approval = _pause_and_approve(
        flow, monkeypatch, request_id=request_id,
        meeting_point=meeting_point, produced=produced)
    return paused, _resume(flow.app, flow.subject, run_id, last_seq)


# ---- AC4: cancel cleans the sandbox and the unfinished state ---------------

def _artifact_pause(flow, monkeypatch, request_id: str, produced: list[str]):
    """Open a plan-book PDF run on the ``artifact`` profile and stop at the pause.

    The artifact profile is the site that holds a SANDBOX capacity slot
    (:meth:`RunOrchestrator.acquire_leases` reserves one only for
    ``artifact``/``cowork``), so it is the honest seam for AC4's "release both
    the lease and the capacity key" claim.
    """
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: _recording_computer(produced))

    async def model(**kwargs):
        flow.model_calls.append(kwargs["message"])
        return "# 徒步计划\n集合时间上午八点", []

    monkeypatch.setattr(kernel, "run_chat_turn", model)
    payload = {"message": "生成计划书 PDF", "request_id": request_id,
               "conversation_id": CONV, "profile_id": "artifact"}
    frames = _post_sse(flow.app, flow.subject, payload)
    assert frames[-1]["type"] == "run.paused", frames
    assert produced == [], "the PDF was rendered before the approval"
    return frames, payload


def _cancel(app, subject, identifier: str, key: str):
    async def run():
        async with _client(app, subject) as client:
            return await client.post(
                f"/api/v2/chat/cancel/{identifier}", json={"idempotency_key": key})

    return asyncio.run(run())


def test_cancelling_a_paused_plan_book_releases_the_lease_and_the_capacity_slot(
    flow, monkeypatch
):
    """AC4: cancel destroys the preserved sandbox AND frees its capacity key.

    The pause deliberately keeps both (the next test pins that leak); cancel is
    what ends it. Nothing was published, so the version history stays empty —
    a cancelled run whose approval was never decided must not leave a row that
    "which artifact is current" could point at.
    """
    produced: list[str] = []
    request_id = "req-cancel-paused"
    frames, _payload = _artifact_pause(flow, monkeypatch, request_id, produced)
    run_id = frames[-1]["run_id"]
    capacity = flow.app.state.agent_v2_runtime.capacity

    # The deliberate leak, still in place while the run waits for a decision.
    lease = flow.repo.get_sandbox_lease(run_id)
    assert lease.status == "active"
    import os
    assert os.path.isdir(lease.workspace_path)
    assert capacity.snapshot()["active_sandboxes"] == 1

    cancelled = _cancel(flow.app, flow.subject, run_id, "cancel-1")
    assert cancelled.status_code == 200
    assert cancelled.json() == {"accepted": True, "reason": "cancelled", "run_id": run_id}

    after = flow.repo.get_sandbox_lease(run_id)
    assert after.status == "destroyed"
    assert not os.path.isdir(after.workspace_path), (
        "cancel only renamed the lease: the sandbox directory is still on disk"
    )
    assert capacity.snapshot()["active_sandboxes"] == 0, (
        "cancel released the sandbox but kept the capacity slot reserved — a "
        "paused run's slot is only freed by decision, cancel or the janitor"
    )

    # A duplicate cancel (same key) returns the recorded outcome; a fresh key on
    # the terminal run learns the real terminal instead of a second cleanup.
    repeated = _cancel(flow.app, flow.subject, run_id, "cancel-1")
    fresh_key = _cancel(flow.app, flow.subject, request_id, "cancel-2")
    assert repeated.json()["accepted"] is True
    assert fresh_key.status_code == 409
    assert fresh_key.json()["reason"] == "already_terminal:cancelled"

    # Cancel resolves by request_id too (the early-cancel path), and publishes
    # nothing: no artifact row, no version row, and the render never ran.
    assert produced == []
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert [e["payload"].get("approval_id") for e in frames
            if e["type"] == "approval.required"]

    # The cancelled run's stream is replayable and its terminal agrees.
    tail = _get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}")
    assert tail.json()["events"][-1]["type"] == "cancelled"
    assert flow.repo.get_run(run_id).status == "cancelled"


def test_a_pause_leaks_the_lease_on_purpose_and_the_janitor_reclaims_it(
    flow, monkeypatch
):
    """AC4 must assert the DELIBERATE leak, not assume destroy-always.

    ``preserve_sandbox`` keeps the lease active on pause (the approved resume
    must reuse the SAME sandbox) and on an evidence-persistence failure; the 24 h
    janitor is what reclaims it. A test that asserted cleanup on pause would be
    asserting a contract this repo does not have.
    """
    from datetime import datetime, timedelta, timezone

    from agent_v2.run_orchestration import sweep_expired_paused_runs

    produced: list[str] = []
    request_id = "req-leak-paused"
    frames, _payload = _artifact_pause(flow, monkeypatch, request_id, produced)
    run_id = frames[-1]["run_id"]
    capacity = flow.app.state.agent_v2_runtime.capacity

    lease = flow.repo.get_sandbox_lease(run_id)
    assert lease.status == "active"
    assert lease.workspace_path  # operators can still find the sandbox
    assert capacity.snapshot()["active_sandboxes"] == 1
    assert flow.repo.get_run(run_id).status == "paused"

    # Within TTL the sweep leaves it alone: the user can still approve.
    runtime = flow.app.state.agent_v2_runtime
    assert sweep_expired_paused_runs(runtime) == 0

    # Past the 24 h window the same run is reclaimed: lease failed, capacity
    # freed, terminal reconciled to error — the leak has a bounded end.
    backdated = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    runtime.runs.control._conn.execute(
        "UPDATE runs SET updated_at = ? WHERE run_id = ?", (backdated, run_id))
    runtime.runs.control._conn.commit()

    assert sweep_expired_paused_runs(runtime) == 1
    assert flow.repo.get_sandbox_lease(run_id).status == "failed"
    assert "expired" in (flow.repo.get_sandbox_lease(run_id).failure_reason or "")
    assert capacity.snapshot()["active_sandboxes"] == 0
    assert flow.repo.get_run(run_id).status == "error"
    assert produced == []
    # AC5: the sweep ends the run offline too, so the log is reconciled — a
    # replay of an expired paused run cannot still answer ``run.paused``.
    tail = _get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}")
    assert tail.json()["events"][-1]["type"] == "error"


def test_a_cancel_whose_sandbox_removal_fails_stays_honest_and_frees_the_slot(
    flow, monkeypatch
):
    """AC4's failure arm: an unremovable sandbox is RECORDED, not reported clean.

    The workspace removal raises (the real computer's injected cleanup failure);
    cancel must still end the run, still free the capacity slot, and leave the
    lease ``failed`` with the cleanup reason — the operator keeps a lead instead
    of a ``destroyed`` row pointing at a directory that is still there.
    """
    produced: list[str] = []
    request_id = "req-cancel-cleanupfail"
    frames, _payload = _artifact_pause(flow, monkeypatch, request_id, produced)
    run_id = frames[-1]["run_id"]
    workspace = flow.repo.get_sandbox_lease(run_id).workspace_path
    capacity = flow.app.state.agent_v2_runtime.capacity

    # The pause stream's computer is gone; cancel builds one from the factory,
    # so the injected cleanup failure has to come from there.
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: LocalNativeComputer(failure_mode="cleanup"))

    cancelled = _cancel(flow.app, flow.subject, run_id, "cancel-cleanup")
    assert cancelled.json()["accepted"] is True
    after = flow.repo.get_sandbox_lease(run_id)
    assert after.status == "failed"
    assert "cleanup failed" in (after.failure_reason or "")
    import os
    assert os.path.isdir(workspace)  # honest: nothing pretends the dir is gone
    assert capacity.snapshot()["active_sandboxes"] == 0
    assert flow.repo.get_run(run_id).status == "cancelled"
    assert flow.repo.list_plan_artifact_versions(CONV) == []


def test_plan_cancel_clears_unfinished_rows_and_keeps_published_history(
    flow, monkeypatch
):
    """AC4 + AC5's asymmetry: 取消 drops the unfinished state, not the evidence.

    ``plan_cancel`` clears the draft's route selection, arrangements, rule
    verdict and gate record — none of which may be inherited by the next plan.
    The version history and the artifact rows are KEPT: they are evidence of
    what was handed out, and the download rules stay theirs.
    """
    produced: list[str] = []
    first = _pause_plan_book(flow, monkeypatch, request_id="req-cancel-keep",
                             meeting_point="长坪沟北门", produced=produced)
    artifact_id = _available(first[1])[0]["payload"]["artifact_id"]

    # A research turn adds the gate record ticket 07 stores on the draft.
    # ``plan_research_route`` acts on a ``minimal_draft`` only, so the form is
    # re-opened first (the publish above confirmed through final_reviewed); the
    # re-submit is a fresh draft, and the just-published book's version rows are
    # conversation-scoped evidence that a new draft does not touch.
    _post_sse(flow.app, flow.subject, _submit("req-cancel-refill"))
    research = _post_sse(flow.app, flow.subject, {
        "message": "研究路线资料", "request_id": "req-cancel-research",
        "conversation_id": CONV, "profile_id": "chat",
        "intent": "plan_research_route",
    })
    assert research[-1]["type"] == "done"
    assert flow.repo.get_plan_safety(CONV) is not None

    cancelled = _post_sse(flow.app, flow.subject, {
        "message": "取消计划", "request_id": "req-cancel-go", "conversation_id": CONV,
        "profile_id": "chat", "intent": "plan_cancel",
    })
    assert _drafts(cancelled)[0]["payload"]["stage"] == "cancelled"
    assert cancelled[-1]["type"] == "done"

    draft = flow.repo.get_plan_draft(CONV)
    assert draft.stage == "cancelled"
    assert flow.repo.get_plan_safety(CONV) is None
    assert flow.repo.get_plan_arrangements(CONV) is None
    assert flow.repo.get_completeness_check(CONV) is None
    assert flow.repo.get_route_selection(CONV) is None

    # …while the published book survives the cancel, still downloadable.
    versions = flow.repo.list_plan_artifact_versions(CONV)
    assert [v.revision for v in versions] == [1]
    assert versions[0].artifact_id == artifact_id
    assert flow.repo.get_current_plan_artifact_version(CONV).artifact_id == artifact_id

    async def download():
        async with _client(flow.app, flow.subject) as client:
            return await client.get(f"/api/v2/artifacts/{artifact_id}")

    got = asyncio.run(download())
    assert got.status_code == 200 and got.content.startswith(b"%PDF-")
    assert flow.model_calls == ["请导出带来源的计划书 PDF"]


def _available(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f["type"] == "artifact.available"]


# ---- AC5: the terminal, the draft rows and the artifact rows agree ---------

def _drive_plan_turn_with_cancel(flow, request_id: str, stop_at: str) -> list[str]:
    """Run one plan turn, cancelling the run as soon as ``stop_at`` arrives.

    A cancel through the API is what a user pressing 取消 during a live plan
    turn looks like: the run row turns terminal while the flow is mid-body, so
    the stream has to agree with it instead of quietly finishing.
    """

    async def drive() -> list[str]:
        orchestrator = chat_api._build_run_orchestrator(flow.app.state.agent_v2_runtime)
        stream = orchestrator.stream_chat(
            ChatFlowRequest(**{**SUBMIT_PAYLOAD, "request_id": request_id}),
            subject_id=flow.subject.subject_id,
        )
        types: list[str] = []
        async for envelope in stream:
            types.append(envelope.type)
            if envelope.type == stop_at:
                flow.repo.request_cancel(
                    run_id=envelope.run_id, idempotency_key=f"mid-{request_id}")
        return types

    return asyncio.run(drive())


def test_a_plan_turn_cancelled_mid_body_ends_cancelled_and_publishes_nothing(
    flow
):
    """AC5: the stream's terminal is the run row's terminal, always.

    The cancel lands after the stage was computed but before the turn finished.
    The run must end ``cancelled`` on BOTH sides of the triangle (event log +
    run row) and deliver neither the reply text nor a side effect — the same
    rule the chat and rag streams already follow, which the plan stream was
    missing: ``mark_terminal_safe`` swallows ``RunAlreadyTerminal`` by design,
    so the row stayed ``cancelled`` while the client was handed ``done``.
    """
    types = _drive_plan_turn_with_cancel(flow, "req-mid-cancel", "plan.draft_state")

    assert types[-1] == "cancelled"
    run_id = flow.repo.resolve_run_id("req-mid-cancel")
    assert flow.repo.get_run(run_id).status == "cancelled"
    assert "text" not in types and "done" not in types
    # Nothing downstream of a plan turn ran: no artifact, no version row, no
    # sandbox. The draft row the turn had already committed is exactly what the
    # log delivered, so mirror, log and row still describe one stage.
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert flow.repo.get_sandbox_lease(run_id) is None
    stored = flow.repo.get_plan_draft(CONV)
    assert stored is not None and stored.stage == "minimal_draft"
    logged = [e for e in flow.repo.get_events_since(run_id=run_id, from_seq=0)
              if e["type"] == "plan.draft_state"]
    assert logged[-1]["payload"]["stage"] == stored.stage
    assert flow.repo.get_run(run_id).status == "cancelled"
    # The cancelled turn leaves no assistant reply behind either.
    snapshot = flow.repo.get_snapshot(run_id)
    assert [m["role"] for m in snapshot.messages] == ["user"]


def test_an_early_cancel_stops_a_plan_turn_before_any_draft_is_written(flow):
    """AC5, the other interruption point: cancel before the commit commits nothing.

    The same request cancelled at ``run_started`` must not move the conversation
    at all — the triangle then reads: cancelled run, no draft row, no event
    beyond the start frame.
    """
    types = _drive_plan_turn_with_cancel(flow, "req-early-cancel", "run_started")

    assert types == ["run_started", "cancelled"]
    run_id = flow.repo.resolve_run_id("req-early-cancel")
    assert flow.repo.get_run(run_id).status == "cancelled"
    assert flow.repo.get_plan_draft(CONV) is None
    assert flow.repo.get_events_since(run_id=run_id, from_seq=0)[-1]["type"] == "cancelled"


def test_the_plan_book_records_agree_after_a_rejected_approval(flow, monkeypatch):
    """AC5 across the whole triangle, for the interruption that publishes nothing.

    Rejecting the PDF approval ends the run ``done`` with no artifact, no
    version row and an untouched draft: "which artifact is current" must not be
    able to point at a phantom.
    """
    produced: list[str] = []
    _install_route_plan_model(monkeypatch, flow, meeting_point="长坪沟北门")
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: _recording_computer(produced))
    _post_sse(flow.app, flow.subject, _submit("req-reject"))
    _advance_to_final(flow.app, flow.subject, flow.repo, tag="req-reject")

    async def run():
        async with _client(flow.app, flow.subject) as client:
            paused = _frames(await client.post(
                "/api/v2/chat/sse", json=_export_payload("req-reject-book")))
            approval = next(e["payload"] for e in paused if e["type"] == "approval.required")
            decided = await client.post(
                f"/api/v2/approvals/{approval['approval_id']}/decision",
                json={"decision": "reject", "decision_request_id": "dec-reject"})
            resumed = _frames(await client.post(
                f"/api/v2/chat/resume/{paused[-1]['run_id']}",
                params={"from_seq": paused[-1]["sequence"]}))
            return paused, decided, resumed

    paused, decided, resumed = asyncio.run(run())
    assert decided.status_code == 200
    run_id = paused[-1]["run_id"]
    assert resumed[-1]["type"] == "done"
    assert flow.repo.get_run(run_id).status == "done"
    assert flow.repo.get_snapshot(run_id).terminal_type == "done"
    assert produced == []
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert flow.repo.get_current_plan_artifact_version(CONV) is None
    # The draft is untouched by the refusal — still the stage the last plan turn
    # wrote (final_reviewed, the confirmation the PDF door read), and no
    # safety/verdict row was cleared by a rejection.
    assert flow.repo.get_plan_draft(CONV).stage == "final_reviewed"
    # A refusal ends the run for good, so the preserved sandbox has to go with
    # it: the lease is no longer active, the workspace is off the disk, and the
    # capacity slot the pause kept reserved comes back. Nothing else ever
    # reclaimed it — the sweep only looks at ``paused`` runs, and this one is
    # ``done``.
    lease = flow.repo.get_sandbox_lease(run_id)
    assert lease is not None and lease.status == "destroyed"
    import os
    assert not os.path.isdir(lease.workspace_path)
    assert flow.app.state.agent_v2_runtime.capacity.snapshot()["active_sandboxes"] == 0


def test_the_version_history_outlives_the_run_and_reports_unavailable_not_repaired(
    flow, monkeypatch
):
    """AC5's last shape: a deleted run must not make the history a lie.

    The version table carries no FK to ``runs`` on purpose (ticket 09), so when
    the run row is swept the artifact row CASCADEs away while the history row
    stays. The answer then is 已过期/不可用 through the download rules — the
    pointer is NOT repaired, because rewriting history to hide a gone artifact
    would lose the evidence that a version was ever published.
    """
    produced: list[str] = []
    frames = _pause_plan_book(flow, monkeypatch, request_id="req-gone",
                              meeting_point="长坪沟北门", produced=produced)
    payload = _available(frames[1])[0]["payload"]
    run_id = frames[1][-1]["run_id"]
    assert flow.repo.get_artifact(payload["artifact_id"]) is not None

    flow.repo.control._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    flow.repo.control._conn.commit()

    assert flow.repo.get_artifact(payload["artifact_id"]) is None
    versions = flow.repo.list_plan_artifact_versions(CONV)
    assert [v.revision for v in versions] == [1]
    current = flow.repo.get_current_plan_artifact_version(CONV)
    assert current is not None and current.artifact_id == payload["artifact_id"]

    async def download():
        async with _client(flow.app, flow.subject) as client:
            return await client.get(f"/api/v2/artifacts/{payload['artifact_id']}")

    gone = asyncio.run(download())
    assert gone.status_code == 404  # unreadable, not repaired, not 500


# ---- AC1 / AC5 on the client: the mirror and the closure contract ----------
#
# There is no frontend test runner (``frontend/package.json`` has no test or
# lint script), so the client half is pinned by reading the sources — real
# coverage of the wiring, NOT a browser pass. ``npx tsc`` catches a missing
# field in a typed position; these catch the silent kinds: a payload read but
# never shown, a recovery context dropped on a disconnect, a draft field that
# ``isSamePlanDraft`` forgot to compare (which makes a post-refresh change read
# as "no change").

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _ts(relative: str) -> str:
    return (FRONTEND / relative).read_text(encoding="utf-8")


def _interface_body(source: str, name: str) -> str:
    """The body of ``interface X { … }`` or ``type X = { … }``."""
    block = re.search(
        rf"(?:export )?(?:interface {name}|type {name} =)\s*\{{(.*?)\n\}}", source, re.S)
    assert block, f"{name} not found in the frontend sources"
    return block.group(1)


def _function_bodies(source: str, signature: str) -> list[str]:
    """Every body that starts at an occurrence of ``signature``."""
    bodies, start = [], 0
    while True:
        index = source.find(signature, start)
        if index == -1:
            return bodies
        open_brace = source.index("{", index)
        depth, cursor = 0, open_brace
        while cursor < len(source):
            if source[cursor] == "{":
                depth += 1
            elif source[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        bodies.append(source[open_brace:cursor + 1])
        start = index + len(signature)


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth, index = 0, open_brace
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace:index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces after {signature!r}")


def test_the_artifact_version_relation_reaches_every_frontend_home():
    """AC5's surface: ticket 09's additive payload fields are read and typed.

    ``artifact.available`` carries ``plan_revision`` / ``previous_artifact_id``;
    a client that never reads them cannot show 当前版本 / 历史 at all, so the
    three homes (wire type, strict reader, the message-level mirror that is what
    survives a refresh) must move together — the same class of bug notes-02's
    landmine 1 describes for draft fields.
    """
    wire = _interface_body(_ts("api/chatV2.ts"), "V2Artifact")
    assert "plan_revision?: number" in wire
    assert "previous_artifact_id?: string" in wire

    reader = _function_body(_ts("api/v2Reducer.ts"), "export function readArtifact")
    assert "payload['plan_revision']" in reader
    assert "payload['previous_artifact_id']" in reader
    # Read strictly: a non-positive or non-integer revision is no revision.
    assert "Number.isInteger(revision)" in reader and "revision > 0" in reader

    mirrored = _ts("pages/SuperAgent.tsx")
    assert "plan_revision?: number" in _interface_body(mirrored, "AgentArtifact")
    assert "previous_artifact_id?: string" in _interface_body(mirrored, "AgentArtifact")


def test_the_plan_book_panel_shows_current_version_and_history_without_guessing():
    """AC5: the mirror shows 当前版本 / 历史 from the server's numbers only.

    The chain is derived over the artifacts the client already holds (so it is
    rebuilt from the persisted session after a refresh), an artifact without a
    revision is not counted as a version, expiry is judged from the artifact's
    own ``expires_at`` and rendered as 已过期/不可用 rather than repaired away,
    and the block rides the EXISTING restore panel instead of adding a second
    ``top-16`` absolutely-positioned surface.
    """
    source = _ts("pages/SuperAgent.tsx")

    derive = _function_body(source, "function readPlanArtifactVersions")
    assert "message.artifacts" in derive, "the chain must come from the artifacts themselves"
    assert "typeof revision !== 'number'" in derive  # no record, no claim
    assert "runs" not in derive.lower(), "version state must never be re-derived from run rows"
    assert "sort((a, b) => a.revision - b.revision)" in derive

    assert "计划书版本：当前第" in source
    assert "历史：" in source
    assert "已过期/不可用" in source
    expired = _function_body(source, "function isPlanArtifactUnavailable")
    assert "if (!version.expiresAt) return false" in expired  # unknown is not expired

    restore_panel = source[source.index("本会话有未完成的行程草稿"):source.index("取消草稿")]
    assert "计划书版本" in restore_panel, (
        "the version block must live inside the restore panel, not beside it"
    )
    version_line = restore_panel[restore_panel.index("计划书版本") - 400:restore_panel.index("计划书版本")]
    assert "absolute" not in version_line


def test_a_stream_that_ends_without_a_terminal_is_never_treated_as_success():
    """AC1's client half: transport closure is outcome_unknown, not a completion.

    The parser reports the closure marker; the page must (a) tell the user the
    outcome is unconfirmed, (b) keep the persisted run so ``planV2Recovery`` can
    reconcile via /snapshot + /events, and (c) keep clearing the run for a real
    error, so an ordinary failure does not leave a phantom to poll.
    """
    client = _ts("api/chatV2.ts")
    assert "export const V2_STREAM_CLOSED_WITHOUT_TERMINAL = " \
        "'stream_closed_without_terminal'" in client
    loop = _function_body(client, "async function readStream") if \
        "async function readStream" in client else client
    assert "options.onError(V2_STREAM_CLOSED_WITHOUT_TERMINAL)" in loop
    assert "!aborted && !terminalSeen" in loop

    page = _ts("pages/SuperAgent.tsx")
    assert "V2_STREAM_CLOSED_WITHOUT_TERMINAL" in page
    # The send path's own error handler (there are several onError callbacks —
    # the resume path's has different duties), found by what it must now know.
    handlers = _function_bodies(page, "onError: (error) =>")
    closure = [body for body in handlers if "closedWithoutTerminal" in body]
    assert len(closure) == 1, "the send path must distinguish the closure marker"
    handler = closure[0]
    assert "const closedWithoutTerminal = error === V2_STREAM_CLOSED_WITHOUT_TERMINAL" in handler
    # Only a real error drops the recovery context.
    guarded = handler[handler.index("if (!closedWithoutTerminal)"):]
    assert "saveV2PersistedRun(null)" in guarded
    assert "结果未确认" in handler
    # And the run is NOT marked terminal: nothing in the handler sets a terminal
    # v2_status, which is what keeps ``planV2Recovery`` on the poll/replay path.
    assert "v2_status" not in handler


def test_every_plan_draft_field_is_compared_by_the_restore_diff():
    """notes-02 landmine 1, widened into a guard: the four homes stay in step.

    ``isSamePlanDraft`` decides whether a restored draft differs from the live
    one. A ``V2PlanDraftState`` field it does not compare makes a post-refresh
    change read as "no change", so the effect hands back the OLD object and the
    new value silently disappears — a recovery/idempotency bug that no type
    checker can see. This pins it for every top-level field, present and future.
    """
    wire = _interface_body(_ts("api/chatV2.ts"), "V2PlanDraftState")
    fields = set(re.findall(r"^  ([a-z_]+)[?]?:", wire, re.M))
    assert fields, "no plan draft fields found — the parser is wrong, not the source"

    compare = _function_body(_ts("api/planDraftStore.ts"), "export function isSamePlanDraft")
    missing = {field for field in fields if field not in compare}
    assert missing == set(), f"isSamePlanDraft does not compare: {sorted(missing)}"

    read = _function_body(_ts("api/v2Reducer.ts"), "export function readPlanDraftState")
    unread = {field for field in fields if f"'{field}'" not in read and f'"{field}"' not in read}
    assert unread == set(), f"readPlanDraftState never reads: {sorted(unread)}"


# ---- AC2: a duplicate request cannot publish a second PDF ------------------

def test_replaying_a_published_plan_book_request_publishes_nothing_new(flow, monkeypatch):
    """AC2: the same ``request_id`` is a replay, all the way to the artifact.

    After the run has completed, re-POSTing the export request replays the whole
    stored stream — pause frames and resume continuation — and executes nothing:
    one render, one ``artifacts`` row, one version row, one download.
    """
    produced: list[str] = []
    request_id = "req-book-dup"
    paused, resumed = _pause_plan_book(
        flow, monkeypatch, request_id=request_id,
        meeting_point="北门停车场", produced=produced)
    assert len(produced) == 1
    assert _available(resumed)[0]["payload"]["plan_revision"] == 1

    again = _post_sse(flow.build(), flow.subject, _export_payload(request_id))

    assert again == paused + resumed
    assert len(_available(again)) == 1
    assert len(produced) == 1, "the duplicate request re-rendered the PDF"
    run_id = flow.repo.resolve_run_id(request_id)
    assert run_id is not None and len(flow.repo.list_artifacts_for_run(run_id)) == 1
    versions = flow.repo.list_plan_artifact_versions(CONV)
    assert [v.revision for v in versions] == [1]
    assert versions[0].artifact_id == _available(resumed)[0]["payload"]["artifact_id"]


# ---- AC3: resuming after approval must not execute the render again -------

def test_a_second_resume_of_the_same_plan_generation_executes_the_pdf_once(
    flow, monkeypatch
):
    """AC3 half 1: duplicate resumes of ONE generation render once.

    The client that lost the resume stream re-opens ``/chat/resume``. The second
    resume replays the stored continuation and stops at the run's terminal: the
    sandbox render is NOT re-run, no second ``artifacts`` row and no second
    version row appear, and the approval stays single-decision.
    """
    produced: list[str] = []
    request_id = "req-book-resume"
    _paused, run_id, last_seq, _approval = _pause_and_approve(
        flow, monkeypatch, request_id=request_id,
        meeting_point="长坪沟北门", produced=produced)
    assert produced == []  # nothing executed before the resume

    first = _resume(flow.app, flow.subject, run_id, last_seq)
    second = _resume(flow.build(), flow.subject, run_id, last_seq)

    assert first[-1]["type"] == "done" and second[-1]["type"] == "done"
    assert len(produced) == 1, (
        "the second resume re-rendered the plan book: the effect ledger or the "
        "stored-event guard stopped deduplicating this generation"
    )
    assert len(_available(second)) == 1
    assert [e["type"] for e in second if e["type"] in {"tool_result", "artifact.available"}] \
        == ["tool_result", "artifact.available"]
    assert len(flow.repo.list_artifacts_for_run(run_id)) == 1
    assert [v.revision for v in flow.repo.list_plan_artifact_versions(CONV)] == [1]
    # The approval was consumed exactly once: a second decision with the same
    # key is idempotent, and a different decision is refused.
    async def decide_again():
        async with _client(flow.app, flow.subject) as client:
            same = await client.post(
                f"/api/v2/approvals/{_approval['approval_id']}/decision",
                json={"decision": "approve", "decision_request_id": f"dec-{request_id}"})
            other = await client.post(
                f"/api/v2/approvals/{_approval['approval_id']}/decision",
                json={"decision": "reject", "decision_request_id": "dec-other"})
            return same, other

    same, other = asyncio.run(decide_again())
    assert same.status_code == 200 and same.json()["idempotent"] is True
    assert other.status_code == 409
    # The render came from the persisted report, not from a second model turn.
    assert len(flow.model_calls) == 1


def test_a_new_plan_revision_still_executes_after_the_previous_generation_committed(
    flow, monkeypatch
):
    """AC3 half 2 (ticket 09's control): replay must not be fixed away globally.

    Generation 2 of the same conversation is a NEW approval with
    ``plan_revision = 2`` inside the canonical args, so its effect key differs
    and the render really runs again — the old artifact and the old version row
    stay exactly as they were.
    """
    produced: list[str] = []
    first = _pause_plan_book(flow, monkeypatch, request_id="req-gen-1",
                             meeting_point="长坪沟北门", produced=produced)
    art1 = _available(first[1])[0]["payload"]
    assert art1["plan_revision"] == 1 and art1["previous_artifact_id"] == ""

    second = _pause_plan_book(flow, monkeypatch, request_id="req-gen-2",
                              meeting_point="长坪沟南门", produced=produced)
    art2 = _available(second[1])[0]["payload"]
    assert art2["plan_revision"] == 2
    assert art2["previous_artifact_id"] == art1["artifact_id"]
    assert art2["artifact_id"] != art1["artifact_id"]
    assert len(produced) == 2, (
        "the new revision replayed generation 1's committed effect — the "
        "discriminator is what makes a regeneration a regeneration"
    )
    versions = flow.repo.list_plan_artifact_versions(CONV)
    assert [(v.revision, v.artifact_id) for v in versions] == [
        (1, art1["artifact_id"]), (2, art2["artifact_id"])]
    assert versions[1].draft_id == versions[0].draft_id == flow.repo.get_plan_draft(CONV).draft_id

    # Generation 1 is untouched by generation 2: same row, same bytes on disk.
    stored1 = flow.repo.get_artifact(art1["artifact_id"])
    import os
    assert stored1 is not None and os.path.isfile(stored1.storage_path)
    assert stored1.sha256 != flow.repo.get_artifact(art2["artifact_id"]).sha256

    # …and re-resuming generation 2's run still adds nothing.
    again = _post_sse(flow.build(), flow.subject, _export_payload("req-gen-2"))
    assert again == second[0] + second[1]
    assert len(flow.repo.list_plan_artifact_versions(CONV)) == 2
    assert len(produced) == 2
