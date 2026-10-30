"""Agent process tracking — live activity timeline contract.

Ticket 01: a tool that already finished publishes its ``tool_result`` on
the next polling tick of the running kernel turn. Ordering is proven with
an event barrier (the model's second round cannot finish before the test
observed the result), never with an arbitrary sleep.

Seam: real ``POST /api/v2/chat/sse`` over ASGI, a temp SQLite run
repository, the real UniHarness graph, and a scripted model plus a
controlled tool.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import api.chat_v2 as chat_v2_module
from agent_v2 import kernel
from agent_v2.kernel_persistence import KernelCheckpointStore
from agent_v2.kernel_tools import BoundedDomainTool, _KnowledgeSearchParams
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from test_run_orchestration_chat import _FakeMemory, _repo, _settings

TERMINALS = {"error", "done", "cancelled", "run.paused"}

SEARCH_TOOL = "hiking_knowledge_search"


class _Barrier:
    """Handshake between the scripted model thread and the SSE consumer.

    The model's first call returns a tool decision; the second call parks
    in a worker thread until the consumer observed the tool result. If the
    result were still batched at turn end, no frame would ever arrive and
    the barrier would time out instead of releasing the turn.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.release = threading.Event()

    def before_model_call(self) -> None:
        with self.lock:
            self.calls += 1
            blocking = self.calls > 1
        if blocking and not self.release.wait(timeout=15):
            raise RuntimeError("activity barrier timed out: no tool_result arrived")


_BARRIER = _Barrier()


class _BarrierModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201 - LangChain hook
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        _BARRIER.before_model_call()

        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _scripted_model(calls: list[AIMessage]) -> _BarrierModel:
    return _BarrierModel(messages=iter(calls), disable_streaming=True)


class _ControlledSearchTool(BoundedDomainTool):
    """The real product tool shape, with a deterministic retrieval result."""

    name = SEARCH_TOOL
    description = "检索徒步知识库。"
    args_schema = _KnowledgeSearchParams
    delegate_name = SEARCH_TOOL

    def _success_audit_fields(self, raw: Any) -> dict[str, Any]:
        return {"evidence": {"has_chunks": True, "chunks": ["c1"], "sources": ["safety.md"]}}


def _search_call(query: str = "冬季露营装备") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": SEARCH_TOOL,
            "args": {"query": query},
            "id": "call-search-1",
            "type": "tool_call",
        }],
    )


def _mount_tool(monkeypatch, seen: list[str]) -> None:
    """Mount one controlled domain tool through the real builder seam."""

    def _fake_extra_tools(profile, *, run_id, audit_sink, **_kwargs):  # noqa: ANN001
        async def _delegate(query: str, k: int | None = None) -> str:
            seen.append(query)
            return "找到 1 条资料：冬季露营需带四季帐。"

        return [_ControlledSearchTool(
            run_id=run_id, audit_sink=audit_sink, delegate=_delegate
        )]

    monkeypatch.setattr("agent_v2.kernel_tools.build_chat_extra_tools", _fake_extra_tools)


def _build_app(repo: RunRepository, checkpoints: KernelCheckpointStore) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = SimpleNamespace(
        runs=repo,
        kernel_checkpoints=checkpoints,
        capacity=None,
        telemetry=None,
        memory=None,
    )
    return app


class _LiveSseServer:
    """A real HTTP server so SSE frames are observed as they are written.

    ``httpx``'s ASGITransport buffers the whole response before the caller
    can iterate it, which would hide exactly the timing this contract is
    about. The test therefore binds a loopback port and streams over TCP.
    """

    def __init__(self, app: FastAPI) -> None:
        import uvicorn

        self.port = _free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_LiveSseServer":
        self._thread.start()
        deadline = time.time() + 30
        while not self._server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self._server.started, "uvicorn did not start"
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=30)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _collect_sse(
    server: _LiveSseServer,
    payload: dict[str, Any],
    on_event=None,  # noqa: ANN001
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    base_url = f"http://127.0.0.1:{server.port}"
    async with AsyncClient(base_url=base_url, timeout=60) as client:
        async with client.stream("POST", "/api/v2/chat/sse", json=payload) as response:
            assert response.status_code == 200, response.status_code
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            event = json.loads(line[len("data: "):])
                            events.append(event)
                            if on_event is not None:
                                on_event(event)
    return events


def test_tool_result_is_published_before_the_kernel_turn_ends(tmp_path: Path, monkeypatch):
    """The finished tool's result is on the wire while the model still runs.

    Barrier: the second model round only completes after the consumer saw
    the ``tool_result`` frame. Batching the result at turn end deadlocks
    the round and the barrier raises instead of the test passing late.
    """
    global _BARRIER
    _BARRIER = _Barrier()
    repo = _repo(tmp_path)
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")
    seen: list[str] = []
    _mount_tool(monkeypatch, seen)

    def _fake_model(_settings_obj):
        return _scripted_model([
            _search_call(),
            AIMessage(content="根据已检索资料，冬季露营应带四季帐。"),
        ])

    monkeypatch.setattr(kernel, "build_chat_model", _fake_model)

    app = _build_app(repo, checkpoints)

    def _on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_result":
            _BARRIER.release.set()

    async def _run(server: _LiveSseServer) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            _collect_sse(
                server,
                {"message": "冬季露营带什么", "request_id": "req-t01-live", "profile_id": "chat"},
                _on_event,
            ),
            timeout=60,
        )

    try:
        with _LiveSseServer(app) as server:
            events = asyncio.run(_run(server))
    finally:
        checkpoints.close()
        repo.close()
    types = [event["type"] for event in events]

    assert "tool_result" in types, types
    result_index = types.index("tool_result")
    # The result arrives BEFORE the terminal and before the final answer.
    assert result_index < types.index("done")
    assert result_index < types.index("text")
    assert [t for t in types if t in TERMINALS] == ["done"]
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))

    result = events[result_index]
    assert result["payload"]["tool"] == SEARCH_TOOL
    assert result["payload"]["status"] == "ok"
    # A finished tool source was really executed by the mounted tool.
    assert seen == ["冬季露营装备"]
    # Published exactly once, never re-sent at turn end.
    assert types.count("tool_result") == 1
    # Ticket 02: the call line precedes its result and carries the args.
    call_index = types.index("tool_call")
    assert call_index < result_index
    call = events[call_index]
    assert call["payload"]["tool"] == SEARCH_TOOL
    assert call["payload"]["args"] == {"query": "冬季露营装备"}
    assert call["payload"]["step"] is not None
    assert types.count("tool_call") == 1


def test_parallel_same_tool_calls_pair_in_arrival_order(tmp_path: Path, monkeypatch):
    """Two parallel calls to one tool stay distinct and keep their order."""
    global _BARRIER
    _BARRIER = _Barrier()
    repo = _repo(tmp_path)
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")
    seen: list[str] = []
    _mount_tool(monkeypatch, seen)

    parallel = AIMessage(
        content="",
        tool_calls=[
            {
                "name": SEARCH_TOOL,
                "args": {"query": "雪线"},
                "id": "call-1",
                "type": "tool_call",
            },
            {
                "name": SEARCH_TOOL,
                "args": {"query": "冰裂缝"},
                "id": "call-2",
                "type": "tool_call",
            },
        ],
    )

    def _fake_model(_settings_obj):
        return _scripted_model([
            parallel,
            AIMessage(content="根据已检索资料，雪线需冰爪与结组绳。"),
        ])

    monkeypatch.setattr(kernel, "build_chat_model", _fake_model)
    app = _build_app(repo, checkpoints)

    def _on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_result":
            _BARRIER.release.set()

    async def _run(server: _LiveSseServer) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            _collect_sse(
                server,
                {"message": "雪线要带什么", "request_id": "req-t02-par", "profile_id": "chat"},
                _on_event,
            ),
            timeout=60,
        )

    try:
        with _LiveSseServer(app) as server:
            events = asyncio.run(_run(server))
    finally:
        checkpoints.close()
        repo.close()

    types = [event["type"] for event in events]
    calls = [event for event in events if event["type"] == "tool_call"]
    results = [event for event in events if event["type"] == "tool_result"]

    assert len(calls) == 2, types
    assert len(results) == 2, types
    assert [call["payload"]["args"]["query"] for call in calls] == ["雪线", "冰裂缝"]
    # Both calls precede both results: nothing was serialized behind the
    # first result.
    assert max(call["sequence"] for call in calls) < min(
        result["sequence"] for result in results
    )
    assert types.count("tool_call") == len({call["sequence"] for call in calls})
    assert sorted(seen) == ["冰裂缝", "雪线"]


def test_tool_call_arguments_are_redacted_and_bounded():
    """The public argument summary drops secrets and truncates long text."""
    from agent_v2.run_orchestration import _public_tool_args

    public = _public_tool_args({
        "query": "鳌太线 冬季 露营 装备清单",
        "latitude": 33.9,
        "user_longitude": 108.1,
        "address": "西安市碑林区",
        "token": "abc123",
        "memory": "用户的私人备注",
        "note": "Authorization: Bearer sk-live-abcdefghijk",
        "long": "x" * 400,
        "nested": {"city": "西安", "lat": 1.0},
        "count": 3,
        "flag": True,
    })

    assert public["query"] == "鳌太线 冬季 露营 装备清单"
    # Location, credential and memory keys never reach the wire.
    assert not {"latitude", "user_longitude", "address", "token", "memory"} & set(public)
    assert "[REDACTED]" in public["note"]
    assert "sk-live" not in public["note"]
    assert len(public["long"]) < 400 and public["long"].endswith("chars]")
    assert public["nested"]["lat"] == "[REDACTED]"
    assert public["count"] == 3 and public["flag"] is True
    # Key count is bounded.
    wide = _public_tool_args({f"k{i}": i for i in range(40)})
    assert len(wide) == 8


def test_subagent_namespace_and_sink_failures_stay_out_of_the_timeline():
    """| namespaces are ignored; a failing sink cannot break the tool."""
    from observability.callbacks import ToolActivityCallbackHandler

    pushed: list[tuple[str, dict[str, Any]]] = []
    handler = ToolActivityCallbackHandler(lambda kind, payload: pushed.append((kind, payload)))
    handler.on_tool_start(
        {"name": "hiking_knowledge_search"},
        {"query": "雪线"},
        run_id="r1",
        metadata={"langgraph_checkpoint_ns": "tools:abc|sub:1", "langgraph_step": 3},
    )
    assert pushed == []

    handler.on_tool_start(
        {"name": "hiking_knowledge_search"},
        {"query": "雪线"},
        run_id="r2",
        metadata={"langgraph_checkpoint_ns": "", "langgraph_step": 3},
    )
    assert pushed == [(
        "tool_start",
        {"tool": "hiking_knowledge_search", "args": {"query": "雪线"}, "step": 3},
    )]

    def _explode(kind: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("activity channel down")

    broken = ToolActivityCallbackHandler(_explode)
    # Swallowed: a reporting failure is not a tool failure.
    broken.on_tool_start({"name": "gear_checklist"}, {"days": 2}, run_id="r3", metadata={})


def test_model_rounds_surround_the_tool_in_execution_order(tmp_path: Path, monkeypatch):
    """Two main-model rounds bracket one tool call, in real order."""
    global _BARRIER
    _BARRIER = _Barrier()
    repo = _repo(tmp_path)
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")
    seen: list[str] = []
    _mount_tool(monkeypatch, seen)

    def _fake_model(_settings_obj):
        return _scripted_model([
            _search_call(),
            AIMessage(content="根据已检索资料，冬季露营应带四季帐。"),
        ])

    monkeypatch.setattr(kernel, "build_chat_model", _fake_model)
    app = _build_app(repo, checkpoints)

    def _on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_result":
            _BARRIER.release.set()

    async def _run(server: _LiveSseServer) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            _collect_sse(
                server,
                {"message": "冬季露营带什么", "request_id": "req-t03", "profile_id": "chat"},
                _on_event,
            ),
            timeout=60,
        )

    try:
        with _LiveSseServer(app) as server:
            events = asyncio.run(_run(server))
    finally:
        checkpoints.close()
        repo.close()

    rounds = [
        event for event in events
        if event["type"] == "progress" and event["payload"].get("stage") == "model"
    ]
    assert [round["payload"]["state"] for round in rounds] == [
        "started", "finished", "started", "finished",
    ]
    starts = [round for round in rounds if round["payload"]["state"] == "started"]
    finishes = [round for round in rounds if round["payload"]["state"] == "finished"]
    # The legacy progress label rides the START only.
    assert [start["payload"]["message"] for start in starts] == ["模型第 1 轮", "模型第 2 轮"]
    assert all("message" not in finish["payload"] for finish in finishes)
    assert all(isinstance(finish["payload"]["duration_ms"], (int, float)) for finish in finishes)
    # Round 1 closes when it decides to call the tool; round 2 starts after it.
    first_finish = finishes[0]["sequence"]
    second_start = starts[1]["sequence"]
    calls = [event["sequence"] for event in events if event["type"] == "tool_call"]
    results = [event["sequence"] for event in events if event["type"] == "tool_result"]
    assert starts[0]["sequence"] < first_finish < calls[0] < results[0] < second_start
    assert events[-1]["type"] == "done"
    assert [t for t in (e["type"] for e in events) if t in TERMINALS] == ["done"]


def test_model_failure_still_closes_the_round_with_one_terminal(tmp_path: Path, monkeypatch):
    """A model error mid-turn closes the open round and keeps one terminal."""
    repo = _repo(tmp_path)
    checkpoints = KernelCheckpointStore(tmp_path / "graph.sqlite")
    seen: list[str] = []
    _mount_tool(monkeypatch, seen)

    class _BoomThenFail(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            if len(messages) > 2:
                raise RuntimeError("provider exploded")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    monkeypatch.setattr(
        kernel,
        "build_chat_model",
        lambda _s: _BoomThenFail(messages=iter([_search_call()]), disable_streaming=True),
    )
    app = _build_app(repo, checkpoints)

    async def _run(server: _LiveSseServer) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            _collect_sse(
                server,
                {"message": "随便问问", "request_id": "req-t03-err", "profile_id": "chat"},
            ),
            timeout=60,
        )

    try:
        with _LiveSseServer(app) as server:
            events = asyncio.run(_run(server))
    finally:
        checkpoints.close()
        repo.close()

    types = [event["type"] for event in events]
    rounds = [
        event for event in events
        if event["type"] == "progress" and event["payload"].get("stage") == "model"
    ]
    # Round 1 opened and was closed by its own tool decision. The second
    # model call failed before emitting any chunk, so it never opened a
    # round — a round that produced output is never left dangling, and no
    # phantom round is invented for a call with no output.
    assert [round["payload"]["state"] for round in rounds] == ["started", "finished"]
    assert seen == ["冬季露营装备"]
    assert types[-1] == "error"
    assert [t for t in types if t in TERMINALS] == ["error"]
    # Nothing is published after the terminal.
    assert rounds[-1]["sequence"] < events[-1]["sequence"]


def test_tool_result_statuses_keep_their_semantics_and_publish_once(tmp_path: Path):
    """Failed / denied / unknown results keep today's payload semantics."""
    audits = [
        {"tool": "gear_checklist", "phase": "result", "ok": False, "code": "tool_timeout"},
        {"tool": "file_write", "phase": "denied", "code": "denied_by_gate"},
        {
            "tool": "mcp__demo__echo",
            "phase": "result",
            "ok": True,
            "status": "outcome_unknown",
        },
    ]

    async def fake_kernel(**_kwargs: Any):
        return ("回答", audits)

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=_repo(tmp_path),
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )

    async def _run() -> list[Any]:
        return [
            item
            async for item in orch.stream_chat(
                ChatFlowRequest(message="x", request_id="req-t01-status"),
                subject_id="sub-1",
            )
        ]

    items = asyncio.run(_run())
    results = [item for item in items if item.type == "tool_result"]

    assert [event.payload["tool"] for event in results] == [
        "gear_checklist",
        "file_write",
        "mcp__demo__echo",
    ]
    failed, denied, unknown = (event.payload for event in results)
    assert failed["status"] == "tool_timeout"
    assert denied["status"] == "denied" and denied["reason"] == "denied_by_gate"
    # Unknown is never reported as success.
    assert unknown["status"] == "outcome_unknown"
    assert unknown["content"] == "本次调用结果未确认"
    assert len(results) == len({event.sequence for event in results})
    assert [item.type for item in items][-1] == "done"


def test_completed_tool_results_survive_the_artifact_approval_pause(tmp_path: Path):
    """A completed step is published before the pause closes the stream."""
    audits = [{"tool": "gear_checklist", "phase": "result", "ok": True, "duration_ms": 40}]

    async def fake_kernel(**_kwargs: Any):
        return ("# 计划", audits)

    class _Computer:
        sandbox_type = "local_native"

        def create_workspace(self, run_id: str) -> str:
            return f"/ws/{run_id}"

        def generate_pdf(self, workspace_path, content, filename):  # noqa: ANN001
            return filename

        def cleanup(self, workspace_path):  # noqa: ANN001
            return {"files_deleted": 0}

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=_repo(tmp_path),
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: _Computer(),
    )

    async def _run() -> list[Any]:
        return [
            item
            async for item in orch.stream_chat(
                ChatFlowRequest(
                    message="规划两日徒步并出 PDF",
                    request_id="req-t01-pause",
                    profile_id="artifact",
                )
            )
        ]

    items = asyncio.run(_run())
    types = [item.type for item in items]

    assert "tool_result" in types
    assert types.index("tool_result") < types.index("approval.required")
    assert types.index("approval.required") < types.index("run.paused")
    assert [t for t in types if t in TERMINALS] == ["run.paused"]
    sequences = [item.sequence for item in items]
    assert sequences == sorted(sequences)
