"""Main-flow HTTP contracts with real SQLite and bounded external adapters.

These are engineering regression tests, not model-quality evals. No live
model, map service, or production state is used.
"""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_api
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router
from agent_v2 import kernel
from agent_v2.computer import LocalNativeComputer, ResourceUsage
from agent_v2.memory_service import MemoryService
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.repositories.memory import MemoryStore
from agent_v2.runs import RunRepository


def events(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


@pytest.fixture
def flow(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        agent_v2_enabled=True, openai_api_key="test", openai_model="test",
        agent_v2_cowork_policy="disabled", agent_v2_state_dir=str(tmp_path),
        route_compare_data_mode="offline", route_decision_provider="deterministic",
    )
    monkeypatch.setattr(chat_api, "settings", settings)
    repos = []
    memories = []
    model_requests = []

    def build():
        repo = RunRepository(ControlStore(str(tmp_path / "control.sqlite")),
                             CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
        memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
        repos.append(repo)
        memories.append(memory)
        app = FastAPI()
        app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
        app.include_router(chat_api.chat_v2_router, prefix="/api/v2")
        app.include_router(chat_api.approvals_v2_router, prefix="/api/v2")
        app.include_router(chat_api.artifacts_v2_router, prefix="/api/v2")
        app.include_router(memory_v2_router, prefix="/api/v2")
        return app

    async def model(**kwargs):
        model_requests.append(kwargs["message"])
        return "请核对天气并准备离线地图。", []

    monkeypatch.setattr(kernel, "run_chat_turn", model)
    app = build()
    subject = app.state.agent_v2_runtime.memory.resolve_subject(None)
    yield SimpleNamespace(app=app, build=build, subject=subject,
                          model_requests=model_requests, path=tmp_path)
    for repo in repos:
        repo.close()
    for memory in memories:
        memory.close()


def client_for(app, subject):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                       cookies={SUBJECT_COOKIE_NAME: subject.token})


def test_run_storage_unavailable_refuses_execution_and_retry_can_recover(flow):
    """An idempotency record must exist before the model may execute."""
    payload = {"message": "查天气", "request_id": "locked-run", "profile_id": "rag"}
    blocker = sqlite3.connect(str(flow.path / "control.sqlite"))
    blocker.execute("BEGIN IMMEDIATE")
    try:
        async def blocked():
            async with client_for(flow.app, flow.subject) as client:
                return events(await client.post("/api/v2/chat/sse", json=payload))
        denied = asyncio.run(blocked())
    finally:
        blocker.rollback()
        blocker.close()
    assert [e["type"] for e in denied] == ["error"]
    assert denied[-1]["payload"]["code"] == "admission_failed"
    assert denied[-1]["payload"]["recoverable"] is True
    assert flow.model_requests == []

    async def retry():
        async with client_for(flow.app, flow.subject) as client:
            return events(await client.post("/api/v2/chat/sse", json=payload))
    recovered = asyncio.run(retry())
    assert recovered[-1]["type"] == "done"
    assert flow.model_requests == ["查天气"]


@pytest.mark.parametrize("profile", ["chat", "rag", "route_plan"])
def test_completed_request_replays_identically_after_runtime_restart(flow, profile):
    payload = {"message": "准备徒步", "request_id": "restart-run", "profile_id": profile}

    async def run():
        async with client_for(flow.app, flow.subject) as client:
            original = events(await client.post("/api/v2/chat/sse", json=payload))
        restarted_app = flow.build()
        async with client_for(restarted_app, flow.subject) as client:
            replay = events(await client.post("/api/v2/chat/sse", json=payload))
            snapshot = await client.get("/api/v2/chat/snapshot/restart-run")
            tail_response = await client.get(
                "/api/v2/chat/events/restart-run", params={"from_seq": original[-2]["sequence"]})
            assert tail_response.status_code == 200
            tail = tail_response.json()["events"]
        return original, replay, snapshot, tail

    original, replay, snapshot, tail = asyncio.run(run())
    assert replay == original
    assert tail == [original[-1]]
    assert snapshot.json()["status"] == "done"
    assert [e["sequence"] for e in original] == list(range(1, len(original) + 1))
    assert flow.model_requests == ["准备徒步"]


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_memory_decision_retries_store_once_and_conflicting_reuse_is_rejected(flow, decision):
    async def run():
        async with client_for(flow.app, flow.subject) as client:
            await client.put("/api/v2/memory/settings", json={"enabled": True})
            consent = events(await client.post("/api/v2/chat/sse", json={
                "message": "请记住我有过敏病史", "request_id": "memory-consent"}))
            candidate = next(e["payload"] for e in consent
                             if e["type"] == "memory.consent_required")
            url = f"/api/v2/memory/candidates/{candidate['candidate_id']}/decision"
            body = {"decision": decision, "decision_request_id": "memory-decision"}
            first = await client.post(url, json=body)
            repeated = await client.post(url, json=body)
            conflict = await client.post(url, json={
                **body, "decision": "reject" if decision == "approve" else "approve"})
            stored = await client.get("/api/v2/memories")
        return first, repeated, conflict, stored

    first, repeated, conflict, stored = asyncio.run(run())
    assert first.status_code == repeated.status_code == 200
    assert repeated.json()["idempotent"] is True
    assert conflict.status_code == 409
    assert stored.json()["total"] == (1 if decision == "approve" else 0)
    if decision == "approve":
        assert stored.json()["items"][0]["content"] == "我有过敏病史"


def test_simultaneous_duplicate_posts_do_not_start_another_model(flow, monkeypatch):
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_model(**kwargs):
            flow.model_requests.append(kwargs["message"])
            entered.set()
            await release.wait()
            return "回答", []

        monkeypatch.setattr(kernel, "run_chat_turn", slow_model)
        payload = {"message": "准备徒步", "request_id": "concurrent-run", "profile_id": "rag"}
        async with client_for(flow.app, flow.subject) as client:
            first = asyncio.create_task(client.post("/api/v2/chat/sse", json=payload))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                duplicate = events(await asyncio.wait_for(
                    client.post("/api/v2/chat/sse", json=payload), 3))
                snapshot = await client.get("/api/v2/chat/snapshot/concurrent-run")
            finally:
                release.set()
            original = events(await asyncio.wait_for(first, 3))
            recovered_response = await client.get("/api/v2/chat/events/concurrent-run")
            assert recovered_response.status_code == 200
            recovered = recovered_response.json()["events"]
        return original, duplicate, snapshot, recovered

    original, duplicate, snapshot, recovered = asyncio.run(run())
    assert snapshot.json()["status"] == "running"
    # Admission keeps one event id but refreshes its replay metadata.
    # Compare the immutable envelope and every following event exactly.
    assert duplicate[0]["payload"]["admission"]["replayed"] is True
    assert {k: v for k, v in duplicate[0].items() if k != "payload"} == {
        k: v for k, v in original[0].items() if k != "payload"}
    assert duplicate[1:] == original[1:len(duplicate)]
    assert not any(e["type"] in {"done", "error", "cancelled"} for e in duplicate)
    assert recovered == [duplicate[0], *original[1:]]
    assert flow.model_requests == ["准备徒步"]


@pytest.mark.parametrize("profile", ["artifact", "route_plan"])
def test_pdf_approval_retry_and_resume_export_only_one_artifact(flow, monkeypatch, profile):
    produced = []

    class Computer(LocalNativeComputer):
        def create_workspace(self, run_id):
            workspace = flow.path / "workspace"
            workspace.mkdir(exist_ok=True)
            return str(workspace)

        def generate_pdf(self, workspace_path, content, filename):
            produced.append(content)
            return super().generate_pdf(workspace_path, content, filename)

        def cleanup(self, workspace_path):
            # Exercise lifecycle recording without deleting retained test files.
            return ResourceUsage(duration_ms=0, files_written=1, bytes_written=0)

    monkeypatch.setattr(chat_api, "_build_artifact_computer", lambda *a, **kw: Computer())

    async def model(**kwargs):
        flow.model_requests.append(kwargs["message"])
        if profile == "route_plan":
            compare = next(t for t in kwargs["extra_tools"] if t.name == "route_compare")
            export = next(t for t in kwargs["extra_tools"] if t.name == "report_export")
            result = json.loads((await compare(candidates=[{
                "name": "演示林间环线", "city": "演示城市",
                "origin": "起点", "destination": "终点"}])).output)
            await export(comparison_id=result["comparison_id"])
        return "徒步报告", []

    monkeypatch.setattr(kernel, "run_chat_turn", model)

    async def run():
        async with client_for(flow.app, flow.subject) as client:
            paused = events(await client.post("/api/v2/chat/sse", json={
                "message": "导出徒步报告", "request_id": "pdf-run", "profile_id": profile}))
            assert paused[-1]["type"] == "run.paused"
            assert produced == []
            approval = next(e["payload"] for e in paused if e["type"] == "approval.required")
            url = f"/api/v2/approvals/{approval['approval_id']}/decision"
            body = {"decision": "approve", "decision_request_id": "pdf-decision"}
            first = await client.post(url, json=body)
            repeated = await client.post(url, json=body)
            conflict = await client.post(url, json={**body, "decision": "reject"})
            assert first.status_code == repeated.status_code == 200
            assert repeated.json()["idempotent"] is True
            assert conflict.status_code == 409
            run_id = paused[-1]["run_id"]
        # A new runtime connection must be able to continue the approved run.
        async with client_for(flow.build(), flow.subject) as client:
            resumed = events(await client.post(f"/api/v2/chat/resume/{run_id}",
                params={"from_seq": paused[-1]["sequence"]}))
            replay = events(await client.post(f"/api/v2/chat/resume/{run_id}",
                params={"from_seq": paused[-1]["sequence"]}))
            artifact = next(e["payload"] for e in resumed if e["type"] == "artifact.available")
            downloaded = await client.get(f"/api/v2/artifacts/{artifact['artifact_id']}")
        return resumed, replay, downloaded

    resumed, replay, downloaded = asyncio.run(run())
    assert resumed[-1]["type"] == "done"
    assert replay == resumed
    assert len(produced) == 1
    assert downloaded.status_code == 200 and downloaded.content.startswith(b"%PDF-")
    assert len(flow.model_requests) == 1


@pytest.mark.parametrize("profile", ["chat", "rag", "route_plan"])
def test_model_failure_is_terminal_and_retry_replays_without_execution(flow, monkeypatch, profile):
    async def failing_model(**kwargs):
        flow.model_requests.append(kwargs["message"])
        raise kernel.KernelModelFailure("model_500", "模型调用失败。")

    monkeypatch.setattr(kernel, "run_chat_turn", failing_model)

    async def run():
        payload = {"message": "查路线", "request_id": "failed-run", "profile_id": profile}
        async with client_for(flow.app, flow.subject) as client:
            original = events(await client.post("/api/v2/chat/sse", json=payload))
        async with client_for(flow.build(), flow.subject) as client:
            replay = events(await client.post("/api/v2/chat/sse", json=payload))
            snapshot = await client.get("/api/v2/chat/snapshot/failed-run")
        return original, replay, snapshot

    original, replay, snapshot = asyncio.run(run())
    assert original[-1]["type"] == "error"
    assert original[-1]["payload"]["code"] == "model_500"
    assert replay == original
    assert snapshot.json()["status"] == "error"
    assert len(flow.model_requests) == 1


@pytest.mark.parametrize("profile", ["chat", "rag", "route_plan"])
def test_cancel_during_model_execution_is_idempotent_and_recoverable(flow, monkeypatch, profile):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_model(**kwargs):
            flow.model_requests.append(kwargs["message"])
            entered.set()
            await release.wait()
            return "已取消的回答不应交付", []

        monkeypatch.setattr(kernel, "run_chat_turn", slow_model)
        payload = {"message": "查路线", "request_id": "cancel-run", "profile_id": profile}
        async with client_for(flow.app, flow.subject) as client:
            first = asyncio.create_task(client.post("/api/v2/chat/sse", json=payload))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                snapshot = await client.get("/api/v2/chat/snapshot/cancel-run")
                url = f"/api/v2/chat/cancel/{snapshot.json()['run_id']}"
                body = {"idempotency_key": "cancel-decision"}
                cancelled = await client.post(url, json=body)
                repeated = await client.post(url, json=body)
            finally:
                release.set()
            original = events(await asyncio.wait_for(first, 3))
            replay = events(await client.post("/api/v2/chat/sse", json=payload))
            final = await client.get("/api/v2/chat/snapshot/cancel-run")
        return original, replay, cancelled, repeated, final

    original, replay, cancelled, repeated, final = asyncio.run(run())
    assert cancelled.status_code == repeated.status_code == 200
    assert original[-1]["type"] == "cancelled"
    assert not any(e["type"] in {"text", "done", "error"} for e in original)
    assert replay == original
    assert final.json()["status"] == "cancelled"
    assert len(flow.model_requests) == 1


def test_other_subject_cannot_replay_resume_or_cancel_owned_run(flow):
    async def run():
        payload = {"message": "私人行程", "request_id": "owned-run", "profile_id": "rag"}
        async with client_for(flow.app, flow.subject) as client:
            original = events(await client.post("/api/v2/chat/sse", json=payload))
        other = flow.app.state.agent_v2_runtime.memory.resolve_subject(None)
        run_id = original[0]["run_id"]
        async with client_for(flow.app, other) as client:
            denied = [
                await client.get(f"/api/v2/chat/snapshot/{run_id}"),
                await client.get(f"/api/v2/chat/events/{run_id}"),
                await client.post(f"/api/v2/chat/resume/{run_id}"),
                await client.post(f"/api/v2/chat/cancel/{run_id}",
                                  json={"idempotency_key": "foreign-cancel"}),
            ]
        return denied

    denied = asyncio.run(run())
    assert all(r.status_code == 404 for r in denied)
    assert len(flow.model_requests) == 1


@pytest.mark.parametrize("profile", ["chat", "rag", "route_plan"])
@pytest.mark.parametrize("terminal", ["done", "error", "unexpected_error"])
def test_closing_stream_on_terminal_keeps_snapshot_terminal(flow, monkeypatch, profile, terminal):
    """A consumer may disconnect immediately after receiving the last frame."""
    if terminal != "done":
        async def failing_model(**kwargs):
            if terminal == "unexpected_error":
                raise RuntimeError("unexpected adapter failure")
            raise kernel.KernelModelFailure("model_500", "模型调用失败。")
        monkeypatch.setattr(kernel, "run_chat_turn", failing_model)
    expected_terminal = "done" if terminal == "done" else "error"

    async def run():
        orchestrator = chat_api._build_run_orchestrator(flow.app.state.agent_v2_runtime)
        stream = orchestrator.stream_chat(chat_api.ChatV2Request(
            message="准备徒步", request_id="terminal-close", profile_id=profile),
            subject_id=flow.subject.subject_id)
        try:
            async for envelope in stream:
                if envelope.type == expected_terminal:
                    return envelope.run_id
        finally:
            await stream.aclose()
        pytest.fail("terminal not emitted")

    run_id = asyncio.run(run())
    assert flow.app.state.agent_v2_runtime.runs.get_run(run_id).status == expected_terminal
