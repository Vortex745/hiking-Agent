"""MCP calls through the real agent graph, checkpoint and local transport."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_v2 import kernel
from agent_v2.kernel_persistence import KernelCheckpointStore
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from api.mcp_router import ServerInput, _save, test_mcp_server as discover_server
from config import settings
from test_mcp_management_contract import SECRET, SERVER, http_server


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self


class FakeMemory:
    def memory_enabled(self, subject_id): return True
    def claim_conversation(self, subject_id, conversation_id): return True
    def owns_conversation(self, subject_id, conversation_id): return True


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("case", ["single", "crash", "parallel", "twice", "repeat"])
def test_real_graph_mcp_executes_without_approval(tmp_path, monkeypatch, http_server, caplog, capsys, transport, case):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    effects_file = tmp_path / ("stdio-effects" if transport == "stdio" else "http-effects")
    body = ServerInput(
        name="测试服务", transport=transport,
        url=http_server if transport == "http" else "",
        command=sys.executable if transport == "stdio" else "",
        args=[str(SERVER)] if transport == "stdio" else [],
        headers={"Authorization": SECRET} if transport == "http" else {},
        env={"MCP_TEST_SECRET": SECRET, "MCP_TEST_EFFECT_FILE": str(effects_file)} if transport == "stdio" else {},
        stdio_confirmed=True, enabled=True,
    )
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def run_case():
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        server = _save(body, None)
        tool_name = f"mcp__{server['id']}__echo"
        calls = [AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": {"payload": {"text": "crash" if case == "crash" else "hello", "options": {"mode": "walk"}}},
            "id": "call-1", "type": "tool_call",
        }])]
        if case == "parallel":
            calls[0].tool_calls.append({
                "name": tool_name,
                "args": {"payload": {"text": "again", "options": {"mode": "walk"}}},
                "id": "call-2", "type": "tool_call",
            })
        if case in {"twice", "repeat"}:
            calls.append(AIMessage(content="", tool_calls=[{
                "name": tool_name,
                "args": {"payload": {"text": "hello" if case == "repeat" else "again", "options": {"mode": "walk"}}},
                "id": "call-2", "type": "tool_call",
            }]))
        calls.append(AIMessage(content="模型结果"))
        model = FakeModel(messages=iter(calls), disable_streaming=True)
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: model)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="调用服务", request_id=f"{transport}-{case}", profile_id="chat")
        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        assert events[-1].type == "done", [(event.type, event.payload) for event in events]
        assert not any(event.type == "approval.required" for event in events)
        run = repo.control.get_run_by_request(request.request_id)
        assert run.status == "done"
        assert not repo.list_pending_approvals_for_run(run.run_id)
        assert SECRET not in repr(events)
        assert SECRET not in repr(repo.get_policy_snapshot(run.run_id))
        assert SECRET not in repr(repo.get_snapshot(run.run_id))
        lines = effects_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == (2 if case in {"twice", "repeat", "parallel"} else 1)
        texts = {json.loads(line)["text"] for line in lines}
        assert texts == (
            {"hello", "again"} if case in {"twice", "parallel"}
            else {"crash"} if case == "crash"
            else {"hello"}
        )
        text = "".join(event.payload.get("content", "") for event in events if event.type == "text")
        unknown_results = [
            event for event in events
            if event.type == "tool_result"
            and event.payload.get("status") == "outcome_unknown"
        ]
        if case == "crash":
            assert "本次调用结果未确认" in text
            assert "结果未知" not in text
            assert unknown_results, [(e.type, e.payload) for e in events]
            assert unknown_results[0].payload["content"] == "本次调用结果未确认"
            assert repo.list_effects_for_run(run.run_id)[0].status == "outcome_unknown"
        else:
            assert "echo:hello:walk:" in text
            if case in {"twice", "parallel"}:
                assert "echo:again:walk:" in text
            assert not unknown_results
            assert "MCP 调用结果未知" not in text
        assert SECRET not in text
        assert SECRET not in repr(repo.list_effects_for_run(run.run_id))
        tool_results = [event for event in events if event.type == "tool_result"]
        replay = [event async for event in orch.stream_resume(run_id=run.run_id, subject_id="subject")]
        assert replay[-1].type == "done"
        # A terminal run replays durable events only; no tool is re-executed
        # and no tool_result is appended twice.
        assert [event for event in replay if event.type == "tool_result"] == tool_results
        assert len(effects_file.read_text(encoding="utf-8").splitlines()) == len(lines)

    try:
        asyncio.run(run_case())
        assert SECRET not in caplog.text
        captured = capsys.readouterr()
        assert SECRET not in captured.out + captured.err
    finally:
        checkpoints.close()
        repo.close()


@pytest.mark.parametrize(
    "scenario",
    ["repair_then_execute", "still_invalid", "text_after_repair", "mixed_calls"],
)
def test_invalid_tool_arguments_repair_boundary(tmp_path, monkeypatch, http_server, scenario):
    """Malformed tool arguments get one repair, then either execute or fail.

    The first model reply carries an unparsable call for a real local MCP
    tool. Only ``repair_then_execute`` may execute and commit exactly one
    effect; every other scenario must terminate with the public
    ``invalid_tool_arguments`` code and zero effects.
    """
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    effects_file = tmp_path / "http-effects"
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def _scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        server = _save(body, None)
        tool_name = f"mcp__{server['id']}__echo"
        malformed = {
            "name": tool_name,
            "args": '{"payload": {"text": "hello",}}',
            "id": "call-bad-1",
            "error": "Failed to parse JSON: Expecting property name",
        }
        valid_call = {
            "name": tool_name,
            "args": {"payload": {"text": "hello", "options": {"mode": "walk"}}},
            "id": "call-good-1",
            "type": "tool_call",
        }
        if scenario == "repair_then_execute":
            calls = [
                AIMessage(content="参数已确认，开始调用。", invalid_tool_calls=[malformed]),
                AIMessage(content="", tool_calls=[valid_call]),
                AIMessage(content="模型结果"),
            ]
        elif scenario == "still_invalid":
            calls = [
                AIMessage(content="参数已确认，开始调用。", invalid_tool_calls=[malformed]),
                AIMessage(content="", invalid_tool_calls=[{**malformed, "id": "call-bad-2"}]),
            ]
        elif scenario == "text_after_repair":
            calls = [
                AIMessage(content="参数已确认，开始调用。", invalid_tool_calls=[malformed]),
                AIMessage(content="抱歉，我无法继续。"),
            ]
        else:
            calls = [
                AIMessage(
                    content="参数已确认，开始调用。",
                    tool_calls=[valid_call],
                    invalid_tool_calls=[malformed],
                ),
            ]
        model = FakeModel(messages=iter(calls), disable_streaming=True)
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: model)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="帮我生成图片", request_id=f"invalid-args-{scenario}", profile_id="chat")

        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        run = repo.control.get_run_by_request(request.request_id)

        if scenario == "repair_then_execute":
            assert events[-1].type == "done", [(e.type, e.payload) for e in events]
            assert len(effects_file.read_text(encoding="utf-8").splitlines()) == 1
            assert not repo.list_pending_approvals_for_run(run.run_id)
            # The repair budget and its cleared waiting flag are durable in
            # the same checkpoint thread that executed the repaired call.
            saver = await checkpoints.get()
            snapshot = await saver.aget_tuple({"configurable": {"thread_id": f"v2:chat:{run.run_id}"}})
            channel_values = snapshot.checkpoint["channel_values"]
            assert channel_values["invalid_tool_call_repair_count"] == 1
            assert channel_values["awaiting_tool_call_repair"] is False
            return

        assert events[-1].type == "error", [(e.type, e.payload) for e in events]
        assert events[-1].payload["code"] == "invalid_tool_arguments", events[-1].payload
        assert run.status == "error"
        assert not effects_file.exists()
        assert not repo.list_pending_approvals_for_run(run.run_id)

    try:
        asyncio.run(_scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_repair_after_committed_effect_keeps_history_and_does_not_replay(tmp_path, monkeypatch, http_server):
    """A repair later in the run must not re-execute an already committed call."""
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    effects_file = tmp_path / "http-effects"
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def _scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        server = _save(body, None)
        tool_name = f"mcp__{server['id']}__echo"
        first_call = {
            "name": tool_name,
            "args": {"payload": {"text": "first", "options": {"mode": "walk"}}},
            "id": "call-first",
            "type": "tool_call",
        }
        malformed = {
            "name": tool_name,
            "args": '{"payload": {"text": "second",}}',
            "id": "call-bad",
            "error": "Failed to parse JSON",
        }
        repaired_call = {
            "name": tool_name,
            "args": {"payload": {"text": "second", "options": {"mode": "walk"}}},
            "id": "call-second",
            "type": "tool_call",
        }
        model = FakeModel(messages=iter([
            AIMessage(content="", tool_calls=[first_call]),
            AIMessage(content="继续调用。", invalid_tool_calls=[malformed]),
            AIMessage(content="", tool_calls=[repaired_call]),
            AIMessage(content="模型结果"),
        ]), disable_streaming=True)
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: model)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="调用服务两次", request_id="repair-after-effect", profile_id="chat")

        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        assert events[-1].type == "done", [(e.type, e.payload) for e in events]
        run = repo.control.get_run_by_request(request.request_id)
        assert not repo.list_pending_approvals_for_run(run.run_id)
        lines = effects_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["text"] for line in lines] == ["first", "second"]
        saver = await checkpoints.get()
        snapshot = await saver.aget_tuple({"configurable": {"thread_id": f"v2:chat:{run.run_id}"}})
        assert snapshot.checkpoint["channel_values"]["invalid_tool_call_repair_count"] == 1

    try:
        asyncio.run(_scenario())
    finally:
        checkpoints.close()
        repo.close()


_IMAGE_BIG = "https://cdn.fixture.test/images/white-cloud-big.jpg?x=fixture"
_IMAGE_PREVIEW = "https://preview.fixture.test/images/white-cloud.jpg!qt_kuan320"
_IMAGE_ORIGIN = "https://origin.fixture.test/images/white-cloud.jpg?token=fixture-secret"
_IMAGE_DOWNLOAD = "https://origin.fixture.test/download/white-cloud.jpg"


def test_image_versions_merge_and_unknown_call_does_not_hide_success(
    tmp_path, monkeypatch, http_server
):
    """One asset shows one public preview; an earlier unknown call stays unknown."""
    from agent_v2.mcp_config_store import load_servers, save_servers

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def scenario():
        body = ServerInput(name="图片服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        saved = _save(body, None)
        assert "search_images" in saved["tools"]

        # A second, unreachable service stays advertised so this turn can show
        # an unknown call followed by a successful image list.
        servers = load_servers()
        offline = {
            **servers[0],
            "id": "custom_offline_image",
            "url": "http://127.0.0.1:1/mcp",
            "revision": "offline-rev",
        }
        save_servers([*servers, offline])
        offline_tool = "mcp__custom_offline_image__echo"
        image_tool = f"mcp__{saved['id']}__search_images"

        model = FakeModel(messages=iter([
            AIMessage(content="", tool_calls=[{
                "name": offline_tool,
                "args": {"payload": {"text": "before", "options": {"mode": "walk"}}},
                "id": "unknown-call", "type": "tool_call",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": image_tool,
                "args": {"query": {"keyword": "白云山"}},
                "id": "image-call", "type": "tool_call",
            }]),
            AIMessage(content="图片如下。"),
        ]), disable_streaming=True)
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: model)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="看看图片", request_id="image-display", profile_id="chat")
        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        assert events[-1].type == "done", [(event.type, event.payload) for event in events]
        run = repo.control.get_run_by_request(request.request_id)

        text = "".join(event.payload.get("content", "") for event in events if event.type == "text")
        assert "图片如下。" in text
        assert text.count("![") == 1
        assert _IMAGE_BIG in text
        assert _IMAGE_PREVIEW not in text
        assert _IMAGE_ORIGIN not in text
        assert _IMAGE_DOWNLOAD not in text
        assert "共 1 张图片" not in text
        assert "MCP 调用结果未知" not in text

        unknown = [
            event for event in events
            if event.type == "tool_result"
            and event.payload.get("status") == "outcome_unknown"
        ]
        assert [event.payload["tool"] for event in unknown] == [offline_tool]
        assert unknown[0].payload["content"] == "本次调用结果未确认"

        effects = {effect.tool_name: effect for effect in repo.list_effects_for_run(run.run_id)}
        assert effects[offline_tool].status == "outcome_unknown"
        assert effects[image_tool].status == "committed"

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_newly_advertised_tool_is_not_automatically_authorized(tmp_path, monkeypatch, http_server):
    from agent_v2.capability_projection import build_capability_projection
    from agent_v2.execution_policy import build_execution_policy
    from agent_v2.mcp_config_store import LocalMcpClient, client_config, load_servers
    from agent_v2.profiles import resolve_profile

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    body = ServerInput(name="测试服务", transport="http", url=http_server,
                       headers={"Authorization": SECRET}, enabled=True)

    async def scenario():
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        saved = _save(body, None)
        connection = client_config(load_servers()[0])
        async with LocalMcpClient(saved["id"], connection) as client:
            echo = next(tool for tool in client.tools if tool.name.endswith("__echo"))
            await echo.execute(echo.args_schema.model_validate({
                "payload": {"text": "enable_extra", "options": {"mode": "walk"}},
            }))
        async with LocalMcpClient(saved["id"], connection) as client:
            advertised = [tool.name.split("__", 2)[-1] for tool in client.tools]
        assert set(advertised) == {"echo", "new_tool", "search_images"}
        projection = build_capability_projection(build_execution_policy(resolve_profile("chat"), settings), run_id="new-tool")
        result = projection.discover_mcp_tools(saved["id"], advertised)
        assert {item["tool"] for item in result.discovered} == {
            f"mcp__{saved['id']}__echo",
            f"mcp__{saved['id']}__search_images",
        }
        assert [item["tool"] for item in result.denied] == [f"mcp__{saved['id']}__new_tool"]

    asyncio.run(scenario())


def test_route_custom_server_does_not_replace_builtin_evidence(tmp_path, monkeypatch, http_server):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.delenv("AMAP_API_KEY", raising=False)
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        _save(body, None)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="规划路线", request_id="route-no-evidence", profile_id="route_plan")
        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        assert events[-1].type == "error"
        assert events[-1].payload["code"] == "evidence_unavailable"
        assert not (tmp_path / "http-effects").exists()

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


def test_route_custom_tool_executes_with_builtin_source(tmp_path, monkeypatch, http_server):
    from agent_v2 import kernel_tools

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    monkeypatch.setattr(kernel_tools, "build_profile_mcp_servers", lambda *args, **kwargs: (
        {"amap_map": {"type": "http", "url": http_server}}, []
    ))
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        server = _save(body, None)
        model = FakeModel(messages=iter([
            AIMessage(content="", tool_calls=[{
                "name": f"mcp__{server['id']}__echo",
                "args": {"payload": {"text": "route", "options": {"mode": "walk"}}},
                "id": "route-call", "type": "tool_call",
            }]),
            AIMessage(content="路线结果"),
        ]), disable_streaming=True)
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: model)
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="规划路线", request_id="route-custom", profile_id="route_plan")
        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        # route_plan closes the SSE stream without a done event (existing
        # behavior); the run row still reaches the done terminal.
        assert not any(event.type in {"error", "run.paused", "approval.required"} for event in events), [
            (event.type, event.payload) for event in events
        ]
        assert any(event.type == "tool_result" and event.payload.get("status") == "ok" for event in events)
        run = repo.control.get_run_by_request(request.request_id)
        assert run.status == "done"
        assert not repo.list_pending_approvals_for_run(run.run_id)
        assert json.loads((tmp_path / "http-effects").read_text(encoding="utf-8"))["text"] == "route"

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()


@pytest.mark.parametrize("count", [0, 10])
def test_offline_plugins_do_not_connect_until_called(tmp_path, monkeypatch, http_server, count):
    from agent_v2.mcp_config_store import load_servers, save_servers

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.setattr(settings, "agent_v2_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "agent_v2_thoughts_enabled", False)
    repo = RunRepository(control=ControlStore(str(tmp_path / "control.sqlite")),
                         checkpoints=CheckpointsStore(str(tmp_path / "events.sqlite")))
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")

    async def scenario():
        body = ServerInput(name="测试服务", transport="http", url=http_server, enabled=True)
        tested = await discover_server(body)
        body.test_id = tested["test_id"]
        _save(body, None)
        template = load_servers()[0]
        save_servers([{**template, "id": f"custom_offline_{i}", "url": "http://127.0.0.1:1/mcp"} for i in range(count)])
        monkeypatch.setattr(kernel, "build_chat_model", lambda _: FakeModel(
            messages=iter([AIMessage(content="模型结果")]), disable_streaming=True
        ))
        orch = RunOrchestrator(settings=settings, runs_repo=repo, memory=FakeMemory(), kernel_checkpoints=checkpoints)
        request = ChatFlowRequest(message="你好", request_id=f"offline-{count}", profile_id="chat")
        start = time.perf_counter()
        events = [event async for event in orch.stream_chat(request, subject_id="subject")]
        elapsed = time.perf_counter() - start
        print(f"offline MCP servers={count}: chat turn without calls {elapsed:.3f}s")
        assert events[-1].type == "done", [(event.type, event.payload) for event in events]

    try:
        asyncio.run(scenario())
    finally:
        checkpoints.close()
        repo.close()
