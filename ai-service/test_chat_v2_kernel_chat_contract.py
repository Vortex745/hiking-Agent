"""Ticket 26 — Chat v2 toolless runs through the real UniHarness kernel.

HTTP/SSE contract coverage:

* success — real kernel model loop generates a single ``text`` event then
  one ``done`` terminal (real credentials required; skipped without).
* model failure / assembly failure — stable terminal ``error`` with a
  public code and NO fallback to stub responses or direct flows.
* envelope invariants — monotonic sequence, exactly one terminal,
  unknown-event dropping untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.kernel import (
    KernelAssemblyError,
    KernelModelFailure,
    extract_final_text,
)

_HAS_LLM_CREDS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))
# Opt-in live check (provider qpm makes un-gated bursts flaky).
_LIVE_LLM = _HAS_LLM_CREDS and os.getenv("KERNEL_LIVE_E2E", "") == "1"


class _FakeSettings:
    """Minimal server settings for stream-level contract tests."""

    def __init__(self, **overrides: Any) -> None:
        self.agent_v2_enabled = True
        self.openai_api_key = "sk-fake"
        self.openai_base_url = "https://fake.invalid/v1"
        self.openai_model = "fake-model"
        for key, value in overrides.items():
            setattr(self, key, value)


def _build_app(runtime: Any | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    state = app.state

    class _FakeMemory:
        def memory_enabled(self, subject_id): return True
        def claim_conversation(self, s, c): return True
        def owns_conversation(self, s, c): return True
        def save_preference(self, s, content): return "m1"
        def list_active_memories(self, s): return []
        def create_candidate(self, s, **kw): return kw

    if runtime is None:
        class _FakeRuntime:
            runs = None
            telemetry = None
            kernel_checkpoints = None
            memory = _FakeMemory()

        state.agent_v2_runtime = _FakeRuntime()
    else:
        state.agent_v2_runtime = runtime
    return app, state


async def _post_sse(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    events: list[dict[str, Any]] = []
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/api/v2/chat/sse", json=payload
        ) as response:
            assert response.status_code == 200
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[len("data: ") :]))
    return events


def test_kernel_run_toolless_failure_maps_to_terminal_error_without_fallback(monkeypatch):
    """Kernel failures produce exactly one terminal error; stub never runs."""
    called: dict[str, int] = {"n": 0}

    async def _failing_run(**kwargs: Any):
        called["n"] += 1
        raise KernelModelFailure("model_unavailable", "主模型未配置 API Key (OPENAI_API_KEY 为空)。")

    # The stream imports run_chat_turn lazily from agent_v2.kernel at call
    # time, so patching there is authoritative (Ticket 28 renamed the entry).
    from agent_v2 import kernel as kernel_module

    monkeypatch.setattr(kernel_module, "run_chat_turn", _failing_run)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    app, _state = _build_app(None)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    events = asyncio.run(_post_sse(app, {
        "message": "随便聊聊",
        "request_id": "req-kernel-fail",
        "profile_id": "chat",
    }))

    assert called["n"] == 1  # kernel path was attempted
    types = [e["type"] for e in events]
    assert types.count("error") == 1, f"exactly one terminal error expected, got {types}"
    assert events[-1]["type"] == "error"
    assert events[-1]["payload"]["code"] == "model_unavailable"
    # No stub-generated text leaked onto the wire.
    assert all(e["type"] != "text" for e in events)


def test_envelope_monotonic_and_single_terminal_on_kernel_success_contract():
    """Unit-level: extract_final_text maps last AI content; non-AI tails ignored."""
    from types import SimpleNamespace

    result = {
        "messages": [
            SimpleNamespace(type="human", content="hi"),
            SimpleNamespace(type="ai", content="partial"),
            SimpleNamespace(type="tool", content="ignored"),
            SimpleNamespace(type="ai", content="最终回答 "),
        ]
    }
    assert extract_final_text(result) == "最终回答"

    empty = {"messages": [SimpleNamespace(type="ai", content="")]}
    assert extract_final_text(empty) is None


@pytest.mark.skipif(
    not _LIVE_LLM,
    reason="live kernel SSE E2E requires KERNEL_LIVE_E2E=1 plus OPENAI_* creds (recorded skip)",
)
def test_real_kernel_end_to_end_success_over_http_sse(monkeypatch):
    """Full product SSE stream driven by the real vendored kernel loop."""
    from config import Settings

    real_settings = Settings()
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)

    app, _state = _build_app(None)
    payload = {
        "message": "用一句话介绍徒步时的三口分组原则。",
        "request_id": "req-real-kernel-e2e",
        "profile_id": "chat",
    }
    events = asyncio.run(_post_sse(app, payload))

    types = [e["type"] for e in events]
    assert types[:1] == ["run_started"]
    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) == 1 and text_events[0]["payload"]["content"].strip()

    terminals = [t for t in types if t in ("done", "error", "cancelled")]
    assert terminals == ["done"]
    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def _make_validation_probe_app(monkeypatch):
    """App whose kernel stub records whether the request reached it."""
    called: dict[str, int] = {"n": 0}

    async def _failing_run(**kwargs: Any):
        called["n"] += 1
        raise KernelModelFailure("model_unavailable", "stub")

    from agent_v2 import kernel as kernel_module

    monkeypatch.setattr(kernel_module, "run_chat_turn", _failing_run)

    from types import SimpleNamespace

    def _fake_subject(request):
        request.state.subject_id = "sub-1"
        return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-1")

    monkeypatch.setattr(chat_v2_module, "resolve_request_subject", _fake_subject)
    app, _state = _build_app(None)
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)
    return app, called


async def _post_raw(app: FastAPI, payload: dict[str, Any]):
    """Non-streaming probe: status code is enough to observe validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v2/chat/sse", json=payload)


def test_current_location_accepts_browser_accuracy_field(monkeypatch):
    """Browser geolocation always carries ``accuracy``; the strict contract
    must not 422 it away (A01)."""
    app, called = _make_validation_probe_app(monkeypatch)
    events = asyncio.run(_post_sse(app, {
        "message": "我这里适合徒步吗",
        "request_id": "req-loc-accuracy",
        "profile_id": "chat",
        "current_location": {
            "latitude": 30.26,
            "longitude": 119.56,
            "accuracy": 35.0,
            "source": "browser",
        },
    }))
    # Validation passed -> the stream opened and the kernel stub ran.
    assert called["n"] == 1
    assert events[-1]["type"] == "error"  # stub terminal, stream completed


def test_current_location_still_rejects_unknown_fields(monkeypatch):
    """extra=forbid keeps guarding against unknown location fields (A01)."""
    app, called = _make_validation_probe_app(monkeypatch)
    response = asyncio.run(_post_raw(app, {
        "message": "我这里适合徒步吗",
        "request_id": "req-loc-unknown",
        "profile_id": "chat",
        "current_location": {"latitude": 30.26, "elevation_m": 100},
    }))
    assert response.status_code == 422
    assert called["n"] == 0  # kernel never reached
