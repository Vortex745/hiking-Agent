"""Sourced PDF uses durable approval and the existing verified Artifact boundary."""
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent_v2.computer import LocalNativeComputer
from agent_v2.memory_service import MemoryService
from agent_v2.repositories.memory import MemoryStore
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from api.chat_v2 import artifacts_v2_router, _cleanup_sandbox
from api.memory_v2 import SUBJECT_COOKIE_NAME


def test_sourced_report_survives_approval_resume_and_isolated_download(tmp_path):
    settings = SimpleNamespace(agent_v2_enabled=True, openai_api_key="test", openai_model="test",
        agent_v2_state_dir=str(tmp_path), route_compare_data_mode="offline", route_decision_provider="deterministic")
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")))
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
    owner = memory.resolve_subject(None)
    stranger = memory.resolve_subject(None)
    memory.claim_conversation(owner.subject_id, "conv-report")
    contents = []
    class Computer(LocalNativeComputer):
        def generate_pdf(self, workspace_path, content, filename):
            contents.append(content)
            return super().generate_pdf(workspace_path, content, filename)
    computer = Computer()
    async def kernel(**kw):
        compare = next(t for t in kw["extra_tools"] if t.name == "route_compare")
        report = next(t for t in kw["extra_tools"] if t.name == "report_export")
        output = await compare.execute(compare.args_schema.model_validate({"candidates": [
            {"name":"演示林间环线", "city":"演示城市", "origin":"起点", "destination":"终点"}]}))
        result = json.loads(output.output)
        await report.execute(report.args_schema.model_validate({"comparison_id": result["comparison_id"]}))
        return "模型不应参与报告正文生成", []
    orchestrator = RunOrchestrator(settings=settings, runs_repo=repo, memory=memory,
        kernel_turn=kernel, computer_factory=lambda *args: computer, sandbox_cleanup=_cleanup_sandbox)
    async def pause():
        return [e async for e in orchestrator.stream_chat(ChatFlowRequest(message="导出比较报告",
            request_id="report-test", conversation_id="conv-report", profile_id="route_plan"), subject_id=owner.subject_id)]
    events = asyncio.run(pause())
    assert events[-1].type == "run.paused"
    assert contents == []  # Approval precedes the PDF side effect.
    approval = repo.list_pending_approvals_for_run(events[-1].run_id)[0]
    orchestrator.resolve_approval_decision(approval.approval_id, decision="approve", decision_request_id="approve-test")
    async def resume():
        return [e async for e in orchestrator.stream_resume(run_id=events[-1].run_id, from_seq=events[-1].sequence)]
    resumed = asyncio.run(resume())
    assert resumed[-1].type == "done"
    assert "offline_fixture" in contents[0]
    assert "confidence" in contents[0]
    assert "模型不应" not in contents[0]
    artifact = next(e.payload for e in resumed if e.type == "artifact.available")
    assert "storage_path" not in artifact
    app = FastAPI()
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
    app.include_router(artifacts_v2_router, prefix="/api/v2")
    async def download():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                               cookies={SUBJECT_COOKIE_NAME: owner.token}) as client:
            path = "/api/v2/artifacts/" + artifact["artifact_id"]
            good = await client.get(path)
            client.cookies.set(SUBJECT_COOKIE_NAME, stranger.token)
            denied = await client.get(path)
            return good, denied
    good, denied = asyncio.run(download())
    assert good.status_code == 200 and good.content.startswith(b"%PDF-")
    assert denied.status_code == 404
    assert repo.get_sandbox_lease(events[-1].run_id).status == "destroyed"
    asyncio.run(resume())
    assert len(contents) == 1
