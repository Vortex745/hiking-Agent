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
from agent_v2.plan_workflow import PlanDraftInput, SUPPLEMENT_ARRANGEMENT_KEYS
from agent_v2.repositories.memory import MemoryStore
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
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


def _pause_plan_book(flow, monkeypatch, *, request_id: str, meeting_point: str,
                     produced: list[str]) -> tuple[list[dict], list[dict]]:
    """Open the plan book once: draft turn, then the approval-paused export run.

    Returns ``(paused_frames, resume_frames)`` with the approval already
    recorded; the PDF is rendered only by the resume stream, and the plan rows
    (draft + arrangements) are on record before it, so the version row the
    export publishes carries a real ``draft_id``.
    """
    # 1. the model-free draft turn that the book is built from.
    draft_frames = _post_sse(flow.app, flow.subject, _submit(f"book-{request_id}"))
    assert draft_frames[-1]["type"] == "done"
    assert _drafts(draft_frames)[0]["payload"]["draft"]["arrangements"]["meeting_point"] \
        == _ARRANGEMENTS["meeting_point"]

    # 2. the export run: kernel pair → pause → approve → resume renders.
    _install_route_plan_model(monkeypatch, flow, meeting_point=meeting_point)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: _recording_computer(produced))

    async def run():
        async with _client(flow.app, flow.subject) as client:
            paused = _frames(await client.post(
                "/api/v2/chat/sse", json=_export_payload(request_id)))
            assert paused[-1]["type"] == "run.paused", paused
            approval = next(e["payload"] for e in paused if e["type"] == "approval.required")
            decided = await client.post(
                f"/api/v2/approvals/{approval['approval_id']}/decision",
                json={"decision": "approve", "decision_request_id": f"dec-{request_id}"})
            assert decided.status_code == 200
            resumed = _frames(await client.post(
                f"/api/v2/chat/resume/{paused[-1]['run_id']}",
                params={"from_seq": paused[-1]["sequence"]}))
            return paused, resumed

    return asyncio.run(run())


def _available(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f["type"] == "artifact.available"]


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
