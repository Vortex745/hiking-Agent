"""LLM_LOG coverage for MCP tools (ticket 04).

The mcp client stack is not importable under the default interpreter on this
machine, so the module skips there and runs under ``ai-service/.venv`` (which
is what CI's full requirements install provides).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import langchain_openai  # noqa: E402
from agent_v2.kernel import run_chat_turn  # noqa: E402
from config import settings  # noqa: E402
from observability import llm_log, tracing  # noqa: E402

STUB_SERVER = Path(__file__).resolve().parent / "test_fixtures" / "logging_mcp_server.py"
REAL_CHATOPENAI = ChatOpenAI


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
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


def test_mcp_tool_call_is_recorded(log_file, monkeypatch):
    """The model-facing MCP tool executes and produces one llm.tool record."""
    calls = {"n": 0}

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        calls["n"] += 1
        names: list[str] = []
        for tool in kwargs.get("tools") or []:
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = (fn or {}).get("name") or tool.get("name")
            if name:
                names.append(str(name))
        if calls["n"] == 1:
            target = next(
                (n for n in names if "echo" in n.lower()),
                names[0] if names else "",
            )
            assert target, f"no MCP tool exposed to the model: {names}"
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": target, "args": {"text": "hi"}, "id": "call-mcp"}
                ],
            )
        else:
            message = AIMessage(content="MCP 调用完成")
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(
        langchain_openai,
        "ChatOpenAI",
        type("_McpCannedChat", (REAL_CHATOPENAI,), {"_agenerate": _agenerate}),
    )

    kernel_settings = SimpleNamespace(
        openai_api_key="sk-canned",
        openai_base_url="https://canned.invalid/v1",
        openai_model="canned-model",
        agent_v2_uniharness_pin=settings.agent_v2_uniharness_pin,
        langfuse_enabled=False,
    )
    text, _ = asyncio.run(
        run_chat_turn(
            system_prompt="你是徒步助手。",
            message="用 echo 工具",
            settings=kernel_settings,
            run_id="run-mcp",
            mcp_servers={
                "stub": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(STUB_SERVER)],
                }
            },
        )
    )

    assert text == "MCP 调用完成"
    records = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tools = [r for r in records if r["event"] == "llm.tool"]
    assert len(tools) == 1, records
    assert "echo" in tools[0]["tool"].lower()
    assert tools[0]["status"] == "ok"
    assert tools[0]["output_chars"] > 0
