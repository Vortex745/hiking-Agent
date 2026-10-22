"""LLM_LOG contract tests: local JSONL execution log (ticket 01)."""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from config import settings
from observability import llm_log, tracing
from observability.generation import trace_chat_model


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch):
    """Never build the real OTLP exporter from a unit test."""
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    monkeypatch.setattr(tracing, "_tracer", None)
    monkeypatch.setattr(tracing, "_provider", None)


class _FakeModel:
    model_name = "gpt-test"
    temperature = 0
    _llm_type = "openai"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
            llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        )


class _FakeStreamModel:
    model_name = "gpt-test"
    temperature = 0
    _llm_type = "openai"

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield AIMessageChunk(
            content="he",
            usage_metadata={"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
        )
        yield AIMessageChunk(
            content="llo",
            usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        )


class _FakeSyncModel:
    model_name = "gpt-test"
    temperature = 0
    _llm_type = "openai"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="sync-ok"))]],
            llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield AIMessageChunk(
            content="he",
            usage_metadata={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
        )
        yield AIMessageChunk(content="llo")


def _lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "llm.log"
    monkeypatch.setattr(settings, "llm_log_enabled", True)
    monkeypatch.setattr(settings, "llm_log_path", str(path))
    llm_log.close()
    llm_log.setup()
    yield path
    llm_log.close()


def test_disabled_creates_no_file(tmp_path, monkeypatch):
    path = tmp_path / "llm.log"
    monkeypatch.setattr(settings, "llm_log_enabled", False)
    monkeypatch.setattr(settings, "llm_log_path", str(path))
    llm_log.close()
    llm_log.setup()

    token = llm_log.begin_turn(run_id="run-off")
    llm_log.generation(model="m", messages=[HumanMessage(content="hi")], output="x")
    llm_log.tool(tool="search", args={"q": "x"})
    llm_log.end_turn(token, status="ok", final_text="x")

    assert not path.exists()
    assert llm_log.enabled() is False
    llm_log.close()


def test_turn_and_generation_records(log_file):
    token = llm_log.begin_turn(
        run_id="run-1", model="gpt-test", message_chars=5, history_count=1
    )
    llm_log.generation(
        model="gpt-test",
        provider="openai",
        temperature=0,
        tools=["search"],
        messages=[HumanMessage(content="hello")],
        output="world",
        input_tokens=7,
        output_tokens=3,
        ttft_ms=12.5,
        duration_ms=34.0,
    )
    llm_log.end_turn(token, status="ok", final_text="world")

    lines = _lines(log_file)
    assert [line["event"] for line in lines] == [
        "llm.turn.started",
        "llm.generation",
        "llm.turn.completed",
    ]
    assert all(line["run_id"] == "run-1" for line in lines)
    assert all(line["level"] == "INFO" for line in lines)
    assert all(line["service"] == "ai-service" for line in lines)
    assert all(line["logger"] == "ai-service.llm_log" for line in lines)
    assert all(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", line["ts"])
        for line in lines
    )
    turn_id = lines[0]["turn_id"]
    assert turn_id and all(line["turn_id"] == turn_id for line in lines)
    generation = lines[1]
    assert generation["step"] == 1
    assert generation["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert "hello" in json.dumps(generation["input"]["messages"], ensure_ascii=False)
    assert generation["output"]["text"] == "world"
    assert generation["status"] == "ok"
    assert lines[2]["final_text"] == "world"
    assert lines[2]["generations"] == 1
    # Schema 2: v1 lines on disk predate level/logger/service (and used a
    # +00:00 timestamp), so they cannot be trusted to share this shape.
    assert lines[2]["schema"] == 2


def test_same_run_two_turns_get_distinct_turn_ids(log_file):
    first = llm_log.begin_turn(run_id="run-2")
    llm_log.end_turn(first, status="ok")
    second = llm_log.begin_turn(run_id="run-2")
    llm_log.end_turn(second, status="ok")

    started = [line for line in _lines(log_file) if line["event"] == "llm.turn.started"]
    assert len(started) == 2
    assert started[0]["turn_id"] != started[1]["turn_id"]


def test_cancelled_turn_still_closes_pair(log_file):
    import asyncio

    token = llm_log.begin_turn(run_id="run-cancel")
    llm_log.end_turn(token, status="error", error=asyncio.CancelledError())

    completed = [line for line in _lines(log_file) if line["event"] == "llm.turn.completed"][0]
    assert completed["status"] == "cancelled"
    assert completed["error_type"] == ""
    assert completed["level"] == "WARNING"


def test_sensitive_values_are_redacted(log_file):
    content = (
        "{'lat': 39.9, 'lng': 116.3} address=文三路 123 号 "
        "latitude=30.123456 Authorization: Bearer super-secret "
        "Cookie: session=abc123 sk-abcdefgh12345678"
    )
    token = llm_log.begin_turn(run_id="run-redact")
    llm_log.generation(model="m", messages=[HumanMessage(content=content)])
    llm_log.tool(tool="search", args={"address": "文三路 123 号", "lat": 39.9})
    llm_log.end_turn(token, status="ok")

    raw = log_file.read_text(encoding="utf-8")
    for secret in ("39.9", "116.3", "文三路", "30.123456", "super-secret", "abc123", "sk-abcdefgh12345678"):
        assert secret not in raw
    assert "[REDACTED]" in raw


def test_input_messages_are_truncated_per_message(log_file):
    messages = [HumanMessage(content="A" * 3000)]
    messages += [HumanMessage(content=f"m{i}" + "x" * 3000) for i in range(59)]

    token = llm_log.begin_turn(run_id="run-trunc")
    llm_log.generation(model="m", messages=messages)
    llm_log.end_turn(token, status="ok")

    generation = [line for line in _lines(log_file) if line["event"] == "llm.generation"][0]
    sent = generation["input"]["messages"]
    assert len(sent) == 50
    assert generation["input"]["messages_total"] == 60
    assert sent[0]["content"].startswith("A" * 100)
    assert "m58" in sent[-1]["content"]
    assert all(len(item["content"]) <= 2048 for item in sent)


def test_tool_args_are_truncated(log_file):
    token = llm_log.begin_turn(run_id="run-tool")
    llm_log.tool(tool="search", args={"query": "x" * 5000}, duration_ms=12, output_chars=99)
    llm_log.end_turn(token, status="ok")

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool"][0]
    assert len(record["args"]) <= llm_log.MAX_TOOL_ARGS_CHARS + 40
    assert "[truncated" in record["args"]
    assert record["output_chars"] == 99


def test_denied_tool_is_recorded(log_file):
    llm_log.tool_denied(tool="bash", reason="high_risk_requires_approval", args={"command": "rm -rf /"})

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool.denied"][0]
    assert record["tool"] == "bash"
    assert record["reason"] == "high_risk_requires_approval"
    assert "rm -rf" in record["args"]
    assert record["level"] == "WARNING"


def test_error_events_are_leveled(log_file):
    token = llm_log.begin_turn(run_id="run-levels")
    llm_log.generation(model="m", error=RuntimeError("boom"))
    llm_log.tool(tool="search", status="error", error=TimeoutError("slow"))
    llm_log.end_turn(token, status="error", error=RuntimeError("turn"))

    by_event = {line["event"]: line for line in _lines(log_file)}
    assert by_event["llm.generation"]["level"] == "ERROR"
    assert by_event["llm.tool"]["level"] == "ERROR"
    assert by_event["llm.turn.completed"]["level"] == "ERROR"


def test_rotation_keeps_newest_record_readable(tmp_path, monkeypatch):
    path = tmp_path / "llm.log"
    monkeypatch.setattr(settings, "llm_log_enabled", True)
    monkeypatch.setattr(settings, "llm_log_path", str(path))
    llm_log.close()
    llm_log.setup(max_bytes=600, backup_count=1)

    token = llm_log.begin_turn(run_id="run-rotate")
    for _ in range(6):
        llm_log.generation(model="m", output="y" * 400)
    llm_log.end_turn(token, status="ok")

    assert (tmp_path / "llm.log.1").exists()
    assert _lines(path)
    llm_log.close()


def test_unwritable_path_degrades_without_raising(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(settings, "llm_log_enabled", True)
    monkeypatch.setattr(settings, "llm_log_path", str(blocker / "llm.log"))
    llm_log.close()
    llm_log.setup()

    assert llm_log.enabled() is False
    llm_log.generation(model="m", messages=[], output="x")
    llm_log.tool(tool="t")
    llm_log.close()


def test_write_failure_disables_sink(log_file, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(llm_log.logger, "info", _boom)
    llm_log.generation(model="m", messages=[], output="x")

    assert llm_log.enabled() is False
    llm_log.generation(model="m", messages=[], output="y")
    assert not log_file.exists()


def test_concurrent_writers_keep_valid_lines(log_file):
    def worker(run_id):
        token = llm_log.begin_turn(run_id=run_id)
        for _ in range(10):
            llm_log.generation(model="m", messages=[], output="x")
        llm_log.end_turn(token, status="ok")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, [f"run-{i % 2}" for i in range(8)]))

    lines = _lines(log_file)
    assert len(lines) == 8 * 12
    assert {line["run_id"] for line in lines} == {"run-0", "run-1"}


def test_non_ascii_round_trips(log_file):
    token = llm_log.begin_turn(run_id="run-unicode")
    llm_log.generation(model="m", output="中文回复 ✓ 完成 ✗ 失败")
    llm_log.end_turn(token, status="ok", final_text="中文回复 ✓")

    raw = log_file.read_text(encoding="utf-8")
    assert "中文回复 ✓ 完成 ✗ 失败" in raw


def test_ansi_sequences_are_stripped_but_line_breaks_survive(log_file):
    """ANSI must not reach the file; D8's newline preservation still holds."""
    token = llm_log.begin_turn(run_id="run-ansi")
    llm_log.generation(
        model="m",
        messages=[HumanMessage(content="\x1b[31mred\x1b[0m\nsecond line")],
        output="\x1b[32mgreen\x1b[0m\npara2",
    )
    llm_log.end_turn(token, status="ok", final_text="\x1b[1mbold\x1b[0m")

    raw = log_file.read_text(encoding="utf-8")
    assert "\x1b" not in raw
    records = _lines(log_file)
    generation = records[1]
    assert generation["output"]["text"] == "green\npara2"
    assert generation["input"]["messages"][0]["content"] == "red\nsecond line"
    assert records[2]["final_text"] == "bold"


@pytest.mark.asyncio
async def test_generation_decorator_writes_record(log_file):
    token = llm_log.begin_turn(run_id="run-decorator")
    model = trace_chat_model(_FakeModel())
    await model._agenerate([HumanMessage(content="hello")])
    llm_log.end_turn(token, status="ok", final_text="ok")

    generation = [line for line in _lines(log_file) if line["event"] == "llm.generation"][0]
    assert generation["model"] == "gpt-test"
    assert generation["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert generation["step"] == 1
    assert generation["output"]["text"] == "ok"


@pytest.mark.asyncio
async def test_stream_decorator_records_ttft(log_file):
    token = llm_log.begin_turn(run_id="run-stream")
    model = trace_chat_model(_FakeStreamModel())
    [chunk async for chunk in model._astream([HumanMessage(content="hi")])]
    llm_log.end_turn(token, status="ok", final_text="hello")

    generation = [line for line in _lines(log_file) if line["event"] == "llm.generation"][0]
    assert generation["ttft_ms"] is not None
    assert generation["output"]["text"] == "hello"
    assert generation["usage"] == {"input_tokens": 7, "output_tokens": 2}


def test_sync_generation_path_records(log_file):
    token = llm_log.begin_turn(run_id="run-sync")
    model = trace_chat_model(_FakeSyncModel())
    model._generate([HumanMessage(content="hi")])
    llm_log.end_turn(token, status="ok", final_text="sync-ok")

    generation = [line for line in _lines(log_file) if line["event"] == "llm.generation"][0]
    assert generation["output"]["text"] == "sync-ok"
    assert generation["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_sync_stream_path_records(log_file):
    token = llm_log.begin_turn(run_id="run-sync-stream")
    model = trace_chat_model(_FakeSyncModel())
    list(model._stream([HumanMessage(content="hi")]))
    llm_log.end_turn(token, status="ok", final_text="hello")

    generation = [line for line in _lines(log_file) if line["event"] == "llm.generation"][0]
    assert generation["output"]["text"] == "hello"
    assert generation["ttft_ms"] is not None


# -- ticket 02: tools and denials -------------------------------------------


def test_tool_log_callback_records_success(log_file):
    from observability.callbacks import ToolLogCallbackHandler

    handler = ToolLogCallbackHandler()
    handler.on_tool_start(
        {"name": "search_hiking_knowledge"},
        '{"query": "西湖"}',
        run_id="lc-1",
        metadata={"langgraph_step": 3},
    )
    handler.on_tool_end("x" * 5000, run_id="lc-1")

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool"][0]
    assert record["tool"] == "search_hiking_knowledge"
    assert record["status"] == "ok"
    assert record["output_chars"] == 5000
    assert record["langgraph_step"] == 3
    assert "query" in record["args"]


def test_tool_log_callback_records_error(log_file):
    from observability.callbacks import ToolLogCallbackHandler

    handler = ToolLogCallbackHandler()
    handler.on_tool_start({"name": "bash"}, {"command": "ls"}, run_id="lc-2")
    handler.on_tool_error(RuntimeError("boom"), run_id="lc-2")

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool"][0]
    assert record["tool"] == "bash"
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"


def test_domain_tool_timeout_output_is_marked_error(log_file):
    from langchain_core.messages import ToolMessage

    from observability.callbacks import ToolLogCallbackHandler

    handler = ToolLogCallbackHandler()
    handler.on_tool_start({"name": "hiking_knowledge_search"}, "{}", run_id="t-timeout")
    handler.on_tool_end(
        ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "工具 hiking_knowledge_search 执行超时，请稍后重试。",
                }
            ],
            tool_call_id="c-timeout",
        ),
        run_id="t-timeout",
    )

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool"][0]
    assert record["status"] == "error"
    assert record["error_type"] == "ToolTimeout"
    assert record["output_chars"] > 0


@pytest.mark.asyncio
async def test_parallel_tool_records_are_not_lost(log_file):
    import asyncio

    from observability.callbacks import ToolLogCallbackHandler

    handler = ToolLogCallbackHandler()

    async def one(run_id, name):
        handler.on_tool_start({"name": name}, "{}", run_id=run_id)
        await asyncio.sleep(0)
        handler.on_tool_end("ok", run_id=run_id)

    await asyncio.gather(one("a", "tool_a"), one("b", "tool_b"))

    records = [line for line in _lines(log_file) if line["event"] == "llm.tool"]
    assert len(records) == 2
    assert {record["tool"] for record in records} == {"tool_a", "tool_b"}
    assert all(record["status"] == "ok" for record in records)


def test_callbacks_include_log_handler_without_tracing(log_file, monkeypatch):
    from agent_v2.kernel import _callbacks_for_turn

    monkeypatch.setattr(tracing, "tracing_enabled", lambda: False)
    names = [type(handler).__name__ for handler in _callbacks_for_turn(None)]
    assert names == ["ToolLogCallbackHandler"]


def test_callbacks_keep_span_handler_with_tracing(log_file, monkeypatch):
    from agent_v2.kernel import _callbacks_for_turn

    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    names = [type(handler).__name__ for handler in _callbacks_for_turn(object())]
    assert names == ["ToolSpanCallbackHandler", "ToolLogCallbackHandler"]


def test_callbacks_are_empty_when_log_is_off(log_file, monkeypatch):
    from agent_v2.kernel import _callbacks_for_turn

    monkeypatch.setattr(tracing, "tracing_enabled", lambda: False)
    monkeypatch.setattr(llm_log, "enabled", lambda: False)
    assert _callbacks_for_turn(None) == []


def test_gate_denial_is_recorded_without_audit_sink(log_file):
    from agent_v2.kernel_tools import ProductGateRule

    gate = object.__new__(ProductGateRule)
    gate._run_id = "run-denied"
    gate._audit_sink = None
    gate._record_denial("bash", "high_risk_requires_approval", {"command": "rm -rf /"})

    record = [line for line in _lines(log_file) if line["event"] == "llm.tool.denied"][0]
    assert record["tool"] == "bash"
    assert record["reason"] == "high_risk_requires_approval"
    assert record["run_id"] == "run-denied"
    assert "rm -rf" in record["args"]
