"""LLM_LOG per-step coverage contract: the REAL kernel loop, canned transport.

Existing ``test_llm_log.py`` proves the sink's unit behaviour by calling
``llm_log.generation/tool/...`` and the callback handler by hand. That cannot
answer "does a real turn record every step?" — the records a live request
produces come out of ``assemble_kernel_agent`` → real ``create_agent`` → real
LangChain callback manager → real model decorator.

Only the transport-level ``ChatOpenAI`` class is canned here, so assembly,
pin re-check, the graph loop, permission gate, tool adapters, callbacks and
the generation decorator all run exactly as they do over HTTP.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

from config import settings
from agent_v2.kernel import run_chat_turn
from agent_v2.kernel_tools import (
    GearChecklistTool,
    RiskAssessmentTool,
    build_chat_kernel_gate,
)
from observability import llm_log, logging_setup, tracing

TOOL_CALL_TURN = {"kind": "tool", "name": "gear_checklist", "args": {"days": 2}}
FINAL_TURN = {"kind": "text", "content": "两天露营的保守装备清单已生成。"}


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch):
    """Keep the OTLP exporter out of the process; LLM_LOG is the subject."""
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_provider", None)


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "llm.log"
    monkeypatch.setattr(settings, "llm_log_enabled", True)
    monkeypatch.setattr(settings, "llm_log_path", str(path))
    llm_log.close()
    llm_log.setup()
    yield path
    llm_log.close()


@pytest.fixture()
def kernel_settings():
    """Settings the kernel needs to assemble: creds are never used live here."""
    return SimpleNamespace(
        openai_api_key="sk-canned",
        openai_base_url="https://canned.invalid/v1",
        openai_model="canned-model",
        agent_v2_uniharness_pin=settings.agent_v2_uniharness_pin,
        langfuse_enabled=False,
    )


def _message_for(step: dict[str, Any], index: int) -> AIMessage:
    if step["kind"] == "tool":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": step["name"],
                    "args": dict(step["args"]),
                    "id": f"call-{index}",
                }
            ],
            usage_metadata={
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
            },
        )
    return AIMessage(
        content=step["content"],
        usage_metadata={
            "input_tokens": 13,
            "output_tokens": 6,
            "total_tokens": 19,
        },
    )


def _patch_model(monkeypatch, script: list[dict[str, Any]]) -> dict[str, int]:
    """Replace the transport class inside ``langchain_openai``.

    ``build_chat_model`` imports ``ChatOpenAI`` at call time and wraps it with
    ``trace_chat_model``, so patching the module attribute keeps BOTH the model
    factory and the generation decorator on the real code path.
    """
    import langchain_openai

    calls = {"n": 0}
    real = langchain_openai.ChatOpenAI

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        step = script[index] if index < len(script) else script[-1]
        return ChatResult(
            generations=[ChatGeneration(message=_message_for(step, index))]
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        step = script[index] if index < len(script) else script[-1]
        text = step.get("content", "")
        half = max(1, len(text) // 2)
        first = AIMessageChunk(
            content=text[:half],
            usage_metadata={"input_tokens": 13, "output_tokens": 0, "total_tokens": 13},
        )
        yield ChatGenerationChunk(message=first)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=text[half:],
                usage_metadata={"input_tokens": 13, "output_tokens": 6, "total_tokens": 19},
            )
        )

    canned = type(
        "_CannedChat",
        (real,),
        {"_agenerate": _agenerate, "_astream": _astream},
    )
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", canned)
    return calls


def _profile(allowed: tuple[str, ...]) -> Any:
    return SimpleNamespace(
        profile_id=SimpleNamespace(value="chat"), allowed_tools=allowed
    )


def _gear_delegate(**kwargs: Any) -> dict[str, Any]:
    async def _delegate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "days": _kwargs.get("days", 1),
            "checklist": {"基础": ["登山鞋", "雨衣"]},
            "message": "保守清单。",
        }

    return _delegate


def _failing_delegate(**kwargs: Any) -> dict[str, Any]:
    async def _delegate(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("internal detail must not leak")

    return _delegate


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(monkeypatch, *, run_id: str, script: list[dict[str, Any]], log_file: Path,
         kernel_settings: Any, tools=(), gate=None, on_text=None, history=None):
    calls = _patch_model(monkeypatch, script)
    text, audit = asyncio.run(
        run_chat_turn(
            system_prompt="你是徒步助手。",
            message="帮我出清单",
            settings=kernel_settings,
            run_id=run_id,
            extra_tools=tuple(tools),
            kernel_gate=gate,
            on_text=on_text,
            history=history,
        )
    )
    return text, audit, calls, _records(log_file)


def test_real_turn_writes_one_record_per_step_in_order(log_file, kernel_settings, monkeypatch):
    """tool-call turn: start → gen → tool → gen → completed, nothing missing."""
    audit: list[dict[str, Any]] = []
    tool = GearChecklistTool(
        run_id="run-steps", audit_sink=audit.append, delegate=_gear_delegate()
    )
    text, _audit, calls, records = _run(
        monkeypatch,
        run_id="run-steps",
        script=[TOOL_CALL_TURN, FINAL_TURN],
        log_file=log_file,
        kernel_settings=kernel_settings,
        tools=[tool],
        gate=build_chat_kernel_gate(
            _profile(("gear_checklist",)), run_id="run-steps", audit_sink=None
        ),
    )

    assert text == FINAL_TURN["content"]
    assert calls["n"] == 2
    assert [r["event"] for r in records] == [
        "llm.turn.started",
        "llm.generation",
        "llm.tool",
        "llm.generation",
        "llm.turn.completed",
    ]

    started, gen1, tool_rec, gen2, completed = records
    assert started["model"] == "canned-model"
    assert started["message_chars"] == len("帮我出清单")

    # Step counter advances per model call; the tool rides the step that called it.
    assert (gen1["step"], gen2["step"]) == (1, 2)
    assert tool_rec["step"] == 1
    assert tool_rec["langgraph_step"] is not None
    assert tool_rec["tool"] == "gear_checklist"
    assert tool_rec["status"] == "ok" and tool_rec["level"] == "INFO"
    assert tool_rec["output_chars"] > 0

    # Generation payload carries the model-facing detail an operator needs.
    assert gen1["usage"] == {"input_tokens": 11, "output_tokens": 4}
    assert gen2["usage"] == {"input_tokens": 13, "output_tokens": 6}
    assert gen2["output"]["text"] == FINAL_TURN["content"]
    assert "gear_checklist" in gen1["tools"]
    roles = [m["role"] for m in gen2["input"]["messages"]]
    assert "human" in roles and "ai" in roles and "tool" in roles

    assert completed["status"] == "ok"
    assert completed["generations"] == 2 and completed["tools"] == 1
    assert completed["duration_ms"] >= 0


def test_every_record_of_one_turn_shares_the_correlation_keys(
    log_file, kernel_settings, monkeypatch
):
    """run_id / turn_id / schema / service must be present on EVERY step record."""
    tool = GearChecklistTool(
        run_id="run-corr", audit_sink=None, delegate=_gear_delegate()
    )
    _run(
        monkeypatch,
        run_id="run-corr",
        script=[TOOL_CALL_TURN, FINAL_TURN],
        log_file=log_file,
        kernel_settings=kernel_settings,
        tools=[tool],
        gate=build_chat_kernel_gate(
            _profile(("gear_checklist",)), run_id="run-corr", audit_sink=None
        ),
    )

    records = _records(log_file)
    assert len(records) == 5
    turn_ids = {r["turn_id"] for r in records}
    assert len(turn_ids) == 1 and "" not in turn_ids
    assert {r["run_id"] for r in records} == {"run-corr"}
    for record in records:
        assert record["schema"] == llm_log.SCHEMA_VERSION
        assert record["service"] == "ai-service"
        assert record["logger"] == "ai-service.llm_log"
        assert record["ts"].endswith("Z")


def test_denied_tool_is_recorded_as_its_own_step_without_executing(
    log_file, kernel_settings, monkeypatch
):
    """Out-of-profile tool: a ``llm.tool.denied`` record, never a ``llm.tool``."""
    audit: list[dict[str, Any]] = []
    tool = GearChecklistTool(
        run_id="run-deny", audit_sink=audit.append, delegate=_gear_delegate()
    )
    _run(
        monkeypatch,
        run_id="run-deny",
        script=[
            {"kind": "tool", "name": "Bash", "args": {"command": "rm -rf /"}},
            FINAL_TURN,
        ],
        log_file=log_file,
        kernel_settings=kernel_settings,
        tools=[tool],
        gate=build_chat_kernel_gate(
            _profile(("gear_checklist",)), run_id="run-deny", audit_sink=None
        ),
    )

    records = _records(log_file)
    events = [r["event"] for r in records]
    assert events[0] == "llm.turn.started"
    assert events[-1] == "llm.turn.completed"
    denied = [r for r in records if r["event"] == "llm.tool.denied"]
    assert len(denied) == 1
    assert denied[0]["tool"] == "Bash"
    assert denied[0]["level"] == "WARNING"
    assert denied[0]["step"] == 1
    assert denied[0]["reason"]
    assert denied[0]["run_id"] == "run-deny"
    # Denied work never executes the adapter: no tool record, and the tool's
    # own audit sink stays empty (the gate records the denial on the log only).
    assert not [r for r in records if r["event"] == "llm.tool"]
    assert audit == []


def test_failing_tool_is_an_error_step_yet_the_turn_still_closes(
    log_file, kernel_settings, monkeypatch
):
    tool = RiskAssessmentTool(
        run_id="run-fail", audit_sink=None, delegate=_failing_delegate()
    )
    _run(
        monkeypatch,
        run_id="run-fail",
        script=[
            {"kind": "tool", "name": "risk_assessment", "args": {"destination": "鳌太线"}},
            FINAL_TURN,
        ],
        log_file=log_file,
        kernel_settings=kernel_settings,
        tools=[tool],
        gate=build_chat_kernel_gate(
            _profile(("risk_assessment",)), run_id="run-fail", audit_sink=None
        ),
    )

    records = _records(log_file)
    tool_rec = [r for r in records if r["event"] == "llm.tool"][0]
    assert tool_rec["status"] == "error"
    assert tool_rec["level"] == "ERROR"
    # The adapter turns the exception into a public failure wording, so the log
    # must classify it as a failure instead of laundering it as a success.
    assert tool_rec["error_type"] == "ToolError"
    assert [r["event"] for r in records][-1] == "llm.turn.completed"


def test_streaming_turn_records_generation_with_ttft(
    log_file, kernel_settings, monkeypatch
):
    """The astream seam must emit the same per-step records, plus first-token time."""
    deltas: list[str] = []

    async def _drive():
        async def _on_text(delta: str, _audit: list) -> None:
            deltas.append(delta)

        return await run_chat_turn(
            system_prompt="你是徒步助手。",
            message="讲讲三口分组",
            settings=kernel_settings,
            run_id="run-stream",
            on_text=_on_text,
        )

    calls = _patch_model(monkeypatch, [FINAL_TURN])
    text, _audit = asyncio.run(_drive())

    assert "".join(deltas) == FINAL_TURN["content"]
    assert text == FINAL_TURN["content"]
    records = _records(log_file)
    assert [r["event"] for r in records] == [
        "llm.turn.started",
        "llm.generation",
        "llm.turn.completed",
    ]
    gen = records[1]
    assert gen["step"] == 1
    assert gen["ttft_ms"] is not None and gen["ttft_ms"] >= 0
    assert gen["duration_ms"] >= gen["ttft_ms"]
    assert gen["usage"] == {"input_tokens": 13, "output_tokens": 6}
    assert records[-1]["generations"] == calls["n"] == 1


def test_history_is_visible_in_the_generation_input_records(
    log_file, kernel_settings, monkeypatch
):
    """Server-read history is a step input, not a separate turn."""
    calls = _patch_model(monkeypatch, [FINAL_TURN])
    asyncio.run(
        run_chat_turn(
            system_prompt="你是徒步助手。",
            message="继续",
            settings=kernel_settings,
            run_id="run-hist",
            history=[
                {"role": "user", "content": "前一天问了装备"},
                {"role": "assistant", "content": "前一天答了清单"},
            ],
        )
    )

    records = _records(log_file)
    started = records[0]
    gen = records[1]
    assert started["history_count"] == 2
    # What the model actually received: system prompt + bounded history + the
    # current message appended exactly once.
    assert gen["input"]["messages_total"] == 4
    assert [m["role"] for m in gen["input"]["messages"]] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert gen["input"]["messages"][0]["content"] == "你是徒步助手。"
    assert gen["input"]["messages"][-1]["content"] == "继续"
    assert calls["n"] == 1


def test_two_turns_on_one_run_do_not_blend_their_step_counters(
    log_file, kernel_settings, monkeypatch
):
    tool = GearChecklistTool(
        run_id="run-two", audit_sink=None, delegate=_gear_delegate()
    )
    gate = build_chat_kernel_gate(
        _profile(("gear_checklist",)), run_id="run-two", audit_sink=None
    )
    for _ in range(2):
        _run(
            monkeypatch,
            run_id="run-two",
            script=[TOOL_CALL_TURN, FINAL_TURN],
            log_file=log_file,
            kernel_settings=kernel_settings,
            tools=[tool],
            gate=gate,
        )

    records = _records(log_file)
    assert [r["event"] for r in records] == [
        "llm.turn.started", "llm.generation", "llm.tool", "llm.generation",
        "llm.turn.completed",
        "llm.turn.started", "llm.generation", "llm.tool", "llm.generation",
        "llm.turn.completed",
    ]
    assert len({r["turn_id"] for r in records}) == 2
    # run_id stays stable across turns; each turn restarts its own step counter.
    assert {r["run_id"] for r in records} == {"run-two"}
    steps = [r["step"] for r in records if r["event"] == "llm.generation"]
    assert steps == [1, 2, 1, 2]


def test_step_records_carry_the_http_request_correlation(
    log_file, kernel_settings, monkeypatch
):
    """Inside a request context the log links to request_id / trace_id / session.

    Out of HTTP these are empty by design; this test pins the populated side so
    the per-step log can be joined with the access log and the trace tree.
    """
    request_token = logging_setup.request_id_var.set("20261021T170000000-abcdef")
    trace_token = logging_setup.trace_id_var.set("0" * 31 + "1")
    session_token = tracing.set_session("sess-42")
    try:
        _run(
            monkeypatch,
            run_id="run-req",
            script=[TOOL_CALL_TURN, FINAL_TURN],
            log_file=log_file,
            kernel_settings=kernel_settings,
            tools=[
                GearChecklistTool(
                    run_id="run-req", audit_sink=None, delegate=_gear_delegate()
                )
            ],
            gate=build_chat_kernel_gate(
                _profile(("gear_checklist",)), run_id="run-req", audit_sink=None
            ),
        )
    finally:
        tracing.reset_session(session_token)
        logging_setup.trace_id_var.reset(trace_token)
        logging_setup.request_id_var.reset(request_token)

    records = _records(log_file)
    assert len(records) == 5
    for record in records:
        assert record["request_id"] == "20261021T170000000-abcdef"
        assert record["trace_id"] == "0" * 31 + "1"
        assert record["session"] == "sess-42"


def test_generation_steps_have_no_graph_step_yet(    log_file, kernel_settings, monkeypatch
):
    """Documented capability boundary, deliberately pinned.

    Tools carry ``langgraph_step`` (the callback metadata supplies it); model
    calls are recorded by the decorator, which has no graph-step view, so the
    field stays absent there. If generation ever gains graph-step attribution,
    this test must be updated on purpose — that is a schema change, not a
    drive-by edit.
    """
    tool = GearChecklistTool(
        run_id="run-bound", audit_sink=None, delegate=_gear_delegate()
    )
    _run(
        monkeypatch,
        run_id="run-bound",
        script=[TOOL_CALL_TURN, FINAL_TURN],
        log_file=log_file,
        kernel_settings=kernel_settings,
        tools=[tool],
        gate=build_chat_kernel_gate(
            _profile(("gear_checklist",)), run_id="run-bound", audit_sink=None
        ),
    )

    records = _records(log_file)
    generations = [r for r in records if r["event"] == "llm.generation"]
    tools = [r for r in records if r["event"] == "llm.tool"]
    assert all("langgraph_step" not in r for r in generations)
    assert all(r.get("langgraph_step") is not None for r in tools)


def _enable_in_memory_tracing(monkeypatch):
    """Real tracer provider, in-memory exporter: no network, real span ids."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test", raising=False)
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test", raising=False)
    return exporter


def _run_turn_over_http(kernel_settings, headers: dict[str, str] | None = None):
    """Mirror the product path: middleware → chat.request span → kernel turn."""
    from observability.logging_setup import RequestContextMiddleware

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.post("/turn")
    async def turn(request: Request):
        parent = tracing.extract_traceparent(request.headers.get("traceparent"))
        with tracing.start_span(
            "chat.request", {"http.route": "/turn"}, context=parent
        ):
            text, _ = await run_chat_turn(
                system_prompt="你是徒步助手。",
                message="hi",
                settings=kernel_settings,
                run_id="run-trace-http",
            )
        return {"text": text}

    async def go():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/turn", headers=headers or {})
            assert response.status_code == 200

    asyncio.run(go())


def test_turn_start_record_inherits_the_agent_turn_trace(
    log_file, kernel_settings, monkeypatch
):
    """Without an incoming traceparent the span is the only source of the id.

    Before the fix ``begin_turn`` ran before ``start_span``, so this first
    record carried ``trace_id=""`` while its own children carried the span id.
    """
    exporter = _enable_in_memory_tracing(monkeypatch)
    _patch_model(monkeypatch, [FINAL_TURN])
    _run_turn_over_http(kernel_settings)

    records = _records(log_file)
    agent_turn = [s for s in exporter.get_finished_spans() if s.name == "agent.turn"]
    assert len(agent_turn) == 1
    span_trace = format(agent_turn[0].context.trace_id, "032x")

    assert records[0]["event"] == "llm.turn.started"
    assert records[0]["trace_id"] == span_trace
    assert {r["trace_id"] for r in records} == {span_trace}


def test_turn_records_follow_the_incoming_traceparent(
    log_file, kernel_settings, monkeypatch
):
    exporter = _enable_in_memory_tracing(monkeypatch)
    _patch_model(monkeypatch, [FINAL_TURN])
    incoming = "4bf92f3577b34da6a3ce929d0e0e4736"
    _run_turn_over_http(
        kernel_settings,
        headers={"traceparent": f"00-{incoming}-00f067aa0ba902b7-01"},
    )

    records = _records(log_file)
    assert records[0]["event"] == "llm.turn.started"
    assert {r["trace_id"] for r in records} == {incoming}
    agent_turn = [s for s in exporter.get_finished_spans() if s.name == "agent.turn"]
    assert format(agent_turn[0].context.trace_id, "032x") == incoming


def test_subagent_tag_attributes_the_generation(
    log_file, kernel_settings, monkeypatch
):
    """The vendored runner tags subagent runs; the decorator must consume it.

    Full subagent turns cannot be exercised in this build (the vendored
    subagent prompt raises unresolved placeholders against the chat tool
    surface), so the contract is pinned at the tag boundary the runner sets.
    """
    import langchain_openai
    from observability.generation import trace_chat_model

    _patch_model(monkeypatch, [FINAL_TURN, FINAL_TURN])
    model = trace_chat_model(
        langchain_openai.ChatOpenAI(
            model="canned-model", api_key="sk-canned", max_retries=0
        )
    )

    async def go():
        sub_token = llm_log.begin_turn(run_id="run-sub-tag")
        await model.ainvoke(
            [HumanMessage(content="hi")],
            config={"tags": ["uniharness:subagent:abc123"]},
        )
        llm_log.end_turn(sub_token, status="ok", final_text="ok")

        plain_token = llm_log.begin_turn(run_id="run-main-tag")
        await model.ainvoke([HumanMessage(content="hi")])
        llm_log.end_turn(plain_token, status="ok", final_text="ok")

    asyncio.run(go())

    generations = [r for r in _records(log_file) if r["event"] == "llm.generation"]
    assert [g["agent"] for g in generations] == ["subagent:abc123", "main"]


def test_builtin_agent_tool_is_recorded_with_its_real_outcome(
    log_file, kernel_settings, monkeypatch
):
    """Builtin (``task`` group) tools ride the same callback as domain tools."""
    from uniharness.harness.definition import AgentDefinition

    _patch_model(
        monkeypatch,
        [
            {
                "kind": "tool",
                "name": "Agent",
                "args": {
                    "description": "检查路线",
                    "prompt": "给出一句话结论",
                    "subagent_type": "scout",
                },
            },
            FINAL_TURN,
        ],
    )
    asyncio.run(
        run_chat_turn(
            system_prompt="你是徒步助手。",
            message="查一下",
            settings=kernel_settings,
            run_id="run-builtin",
            subagent_definitions={
                "scout": AgentDefinition(
                    description="scout", system_prompt="你是子代理。", tools=()
                )
            },
        )
    )

    tools = [r for r in _records(log_file) if r["event"] == "llm.tool"]
    assert len(tools) == 1
    record = tools[0]
    assert record["tool"] == "Agent"
    assert record["output_chars"] > 0
    # The vendored subagent prompt cannot resolve against the chat tool
    # surface, and its <error> result must not be laundered as ok.
    assert record["status"] == "error"
    assert record["error_type"] == "ToolError"


def test_kernel_turn_writes_nothing_when_the_channel_is_disabled(
    tmp_path, monkeypatch, kernel_settings
):
    """Disabled is the shipped default: a real turn must not touch the disk."""
    path = tmp_path / "llm.log"
    monkeypatch.setattr(settings, "llm_log_enabled", False)
    monkeypatch.setattr(settings, "llm_log_path", str(path))
    llm_log.close()
    llm_log.setup()
    try:
        calls = _patch_model(monkeypatch, [TOOL_CALL_TURN, FINAL_TURN])
        text, _audit = asyncio.run(
            run_chat_turn(
                system_prompt="你是徒步助手。",
                message="帮我出清单",
                settings=kernel_settings,
                run_id="run-off",
                extra_tools=(
                    GearChecklistTool(
                        run_id="run-off", audit_sink=None, delegate=_gear_delegate()
                    ),
                ),
                kernel_gate=build_chat_kernel_gate(
                    _profile(("gear_checklist",)), run_id="run-off", audit_sink=None
                ),
            )
        )
        assert text and calls["n"] == 2
        assert not path.exists()
        assert llm_log.enabled() is False
    finally:
        llm_log.close()
