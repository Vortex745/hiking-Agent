"""Explicit offline demo routing with real or fake decision providers and Artifact lifecycle."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from agent_v2.computer import LocalNativeComputer
from agent_v2.memory_service import MemoryService
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.repositories.memory import MemoryStore
from agent_v2.runs import RunRepository
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from config import settings


async def demo(args):
    state_dir = args.output_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(**vars(settings))
    config.agent_v2_enabled = True
    # The scripted demo kernel does not call the chat model. The selected
    # decision provider still uses the real configured model/key; no fallback.
    config.openai_api_key = settings.openai_api_key or ("offline-demo-router" if args.provider != "llm" else "")
    config.route_decision_provider = args.provider
    config.route_compare_data_mode = "offline"
    config.route_compare_demo_scenario = args.scenario
    config.agent_v2_state_dir = str(state_dir)
    repo = RunRepository(control=ControlStore(str(state_dir / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(state_dir / "checkpoints.sqlite")))
    memory = MemoryService(MemoryStore(str(state_dir / "memory.sqlite")))
    subject = memory.resolve_subject(None)
    comparison = None
    async def kernel(**kw):
        nonlocal comparison
        tools = {tool.name: tool for tool in kw["extra_tools"]}
        tool = tools["route_compare"]
        output = await tool.execute(tool.args_schema.model_validate({"candidates": [
            {"name": name, "city": "演示城市", "origin": "起点", "destination": "终点"}
            for name in ("演示林间环线", "演示山脊路线")]}))
        comparison = json.loads(output.output)
        if args.pdf:
            report = tools["report_export"]
            await report.execute(report.args_schema.model_validate({"comparison_id": comparison["comparison_id"]}))
        return "演示路由完成，结果由代码生成。", []
    def cleanup(computer, workspace, runs_repo, run_id):
        usage = computer.cleanup(workspace)
        runs_repo.destroy_sandbox_lease(run_id=run_id, resource_usage=usage.to_dict())
    orchestrator = RunOrchestrator(settings=config, runs_repo=repo, memory=memory, kernel_turn=kernel,
                                  computer_factory=lambda *a: LocalNativeComputer(), sandbox_cleanup=cleanup)
    request_id = "demo-" + str(uuid4())
    events = [e async for e in orchestrator.stream_chat(ChatFlowRequest(message="比较离线演示路线",
        request_id=request_id, conversation_id=request_id, profile_id="route_plan"), subject_id=subject.subject_id)]
    if args.approve_pdf and events[-1].type == "run.paused":
        approval = repo.list_pending_approvals_for_run(events[-1].run_id)[0]
        orchestrator.resolve_approval_decision(approval.approval_id, decision="approve", decision_request_id=request_id + "-approval")
        events.extend([e async for e in orchestrator.stream_resume(run_id=events[-1].run_id, from_seq=events[-1].sequence)])
    artifacts = [e.payload for e in events if e.type == "artifact.available"]
    output = {"comparison": comparison, "event_types": [e.type for e in events], "artifacts": artifacts,
              "note": "Explicit offline routing + synthetic data. llm/external decisions use real providers; no credential output."}
    (state_dir / "demo.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": comparison["status"] if comparison else "error",
                      "provider": args.provider, "artifacts": artifacts, "output": str(state_dir / "demo.json")}, ensure_ascii=False))
    memory.close()
    return 0 if comparison and events[-1].type in {"done", "run.paused"} else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["llm", "external", "deterministic"], default="deterministic")
    parser.add_argument("--scenario", choices=["complete", "partial", "empty", "severe_weather", "caution"], default="complete")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--approve-pdf", action="store_true", help="Explicitly approve the demo PDF operation")
    return asyncio.run(demo(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
