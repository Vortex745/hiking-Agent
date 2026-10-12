"""Ticket 01 — 计划书入口与最小草稿: request-contract tests.

Seam under test: ``POST /api/v2/chat/sse`` (the only public chat surface),
driven through ``httpx.ASGITransport`` against a rebuilt FastAPI app with
``app.state.agent_v2_runtime`` — the same harness style as
``test_main_flow_resilience.py``. No live model, no gateway, no network.

Acceptance criteria proven here:

* AC3 前端入口和服务端请求使用白名单 intent — the four whitelisted intents
  are accepted by the strict request model.
* AC5 服务端拒绝未知 intent — an intent outside the whitelist is rejected
  by the model (422) BEFORE any streaming or run creation happens.
"""

from __future__ import annotations

import asyncio
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_api
from agent_v2 import kernel
from agent_v2.memory_service import MemoryService
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.repositories.memory import MemoryStore
from agent_v2.runs import RunRepository
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router

PLAN_INTENTS = ("plan_create", "plan_confirm", "plan_cancel", "plan_regenerate")


def _frames(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def flow(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="test",
        openai_model="test",
        agent_v2_cowork_policy="disabled",
        agent_v2_state_dir=str(tmp_path),
        route_compare_data_mode="offline",
        route_decision_provider="deterministic",
    )
    monkeypatch.setattr(chat_api, "settings", settings)

    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")),
    )
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))

    app = FastAPI()
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
    app.include_router(chat_api.chat_v2_router, prefix="/api/v2")
    app.include_router(memory_v2_router, prefix="/api/v2")

    model_calls: list[str] = []

    async def fake_model(**kwargs):
        model_calls.append(kwargs["message"])
        return "好的。", []

    monkeypatch.setattr(kernel, "run_chat_turn", fake_model)

    subject = memory.resolve_subject(None)
    yield SimpleNamespace(app=app, repo=repo, memory=memory, subject=subject,
                          model_calls=model_calls)
    repo.close()
    memory.close()


def _post(app, subject, payload):
    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: subject.token},
        ) as client:
            return await client.post("/api/v2/chat/sse", json=payload)
    return asyncio.run(run())


def test_unknown_intent_is_rejected_before_streaming(flow):
    """AC5: an intent outside the whitelist never reaches the stream."""
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": "req-unknown-intent",
            "conversation_id": "conv-01",
            "intent": "plan_teleport",
        },
    )

    assert response.status_code == 422
    assert not response.text.startswith("data:")
    # Rejection happens before the run row exists and before any model call.
    assert flow.repo.resolve_run_id("req-unknown-intent") is None
    assert flow.model_calls == []


def test_missing_intent_keeps_the_plain_chat_contract(flow):
    """AC3: ``intent`` is optional — ordinary chat is untouched."""
    response = _post(
        flow.app,
        flow.subject,
        {"message": "带什么装备？", "request_id": "req-no-intent", "conversation_id": "conv-01"},
    )

    assert _frames(response)[-1]["type"] == "done"
    assert flow.model_calls == ["带什么装备？"]


@pytest.mark.parametrize("intent", PLAN_INTENTS)
def test_whitelisted_plan_intents_are_accepted(flow, intent):
    """AC3: the server accepts exactly the whitelisted plan intents."""
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": f"req-{intent}",
            "conversation_id": "conv-01",
            "intent": intent,
        },
    )

    frames = _frames(response)
    assert frames, "the whitelisted intent must produce a stream"
    assert frames[-1]["type"] in {"done", "error"}
