"""Spec #13 phase 2 (ticket #18) — RAG-page SSE adapter contract (seam A).

The route POST /api/v1/rag/query runs a kernel run (profile=rag) behind a
translation adapter. External behavior asserted over HTTP:

* The legacy frame contract is preserved: only process / documents /
  text / done / error frames appear; the stream ends with exactly one
  terminal (done — the legacy contract has no cancelled frame).
* A knowledge tool result with evidence renders the documents panel
  frame (metadata.searched_count / matched_chunks / documents[]).
* A search without usable chunks renders the fail-closed 资料不足 text.
* Missing v2 runtime → HTTP 503 before the stream starts.

Same harness as ``test_chat_v2_kernel_chat_contract.py``: FastAPI app +
fake runtime in app.state + patched kernel entry, driven over ASGI.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.rag as rag_module
from agent_v2 import kernel as kernel_module


_LEGACY_TYPES = {"process", "documents", "text", "done", "error"}


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(rag_module.rag_router, prefix="/api/v1")

    class _FakeMemory:
        def memory_enabled(self, subject_id): return True
        def claim_conversation(self, s, c): return True
        def owns_conversation(self, s, c): return True

    class _FakeRuntime:
        runs = None
        telemetry = None
        kernel_checkpoints = None
        memory = _FakeMemory()

    app.state.agent_v2_runtime = _FakeRuntime()
    return app


def _fake_subject(request):
    request.state.subject_id = "sub-rag"
    return None, SimpleNamespace(issued=False, token="tok", subject_id="sub-rag")


def _kernel_result(text: str, audit: list[dict[str, Any]]):
    async def _run(**kwargs: Any):
        return text, audit

    return _run


async def _post_sse(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    transport = ASGITransport(app=app)
    events: list[dict[str, Any]] = []
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/api/v1/rag/query", json=payload) as response:
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


def test_query_streams_legacy_frames_with_documents_panel(monkeypatch) -> None:
    monkeypatch.setattr(rag_module, "resolve_request_subject", _fake_subject)
    monkeypatch.setattr(rag_module.settings, "agent_v2_enabled", True)

    audit = [
        {
            "tool": "hiking_knowledge_search",
            "phase": "result",
            "ok": True,
            "evidence": {
                "has_chunks": True,
                "sources": ["鳌太线攻略.md"],
                "chunks": [
                    {
                        "title": "鳌太线攻略",
                        "source": "鳌太线攻略.md",
                        "preview": "需要帐篷、睡袋。",
                    }
                ],
            },
        }
    ]
    monkeypatch.setattr(
        kernel_module,
        "run_chat_turn",
        _kernel_result("根据知识库证据：需要帐篷（来源：鳌太线攻略.md）。", audit),
    )

    events = asyncio.run(
        _post_sse(_build_app(), {"question": "鳌太线带什么装备？", "chat_id": "c-rag-1"})
    )

    types = [e["type"] for e in events]
    assert set(types) <= _LEGACY_TYPES, f"legacy contract broken: {types}"
    assert types[-1] == "done"
    assert types.count("done") == 1

    docs = [e for e in events if e["type"] == "documents"]
    assert len(docs) == 1, f"one documents panel frame expected, got {types}"
    meta = docs[0]["metadata"]
    assert meta["matched_chunks"] == 1
    assert meta["documents"][0]["source"] == "鳌太线攻略.md"
    assert "帐篷" in meta["documents"][0]["content"]

    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert "帐篷" in text


def test_query_renders_fail_closed_text_when_search_has_no_chunks(monkeypatch) -> None:
    monkeypatch.setattr(rag_module, "resolve_request_subject", _fake_subject)
    monkeypatch.setattr(rag_module.settings, "agent_v2_enabled", True)

    audit = [
        {
            "tool": "hiking_knowledge_search",
            "phase": "result",
            "ok": True,
            "evidence": {"has_chunks": False, "sources": [], "chunks": []},
        }
    ]
    monkeypatch.setattr(
        kernel_module,
        "run_chat_turn",
        _kernel_result("鳌太线全程约120公里。", audit),
    )

    events = asyncio.run(
        _post_sse(_build_app(), {"question": "查鳌太线精确里程", "chat_id": "c-rag-2"})
    )

    types = [e["type"] for e in events]
    assert set(types) <= _LEGACY_TYPES
    assert types[-1] == "done"
    assert all(e["type"] != "documents" for e in events)

    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"
    assert "120公里" not in text


def test_query_returns_503_when_v2_runtime_missing() -> None:
    app = _build_app()
    app.state.agent_v2_runtime = None

    async def run() -> Any:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/rag/query", json={"question": "q", "chat_id": "c"}
            )

    response = asyncio.run(run())
    assert response.status_code == 503
