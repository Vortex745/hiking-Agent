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
* AC6 最小字段校验 — the dialog's ``plan_input`` crosses the wire on the
  strict model, so the fields the entry dialog collects are validated at the
  boundary and a fourth key cannot be smuggled into a draft.
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


# ---- AC6: the dialog's fields cross the wire on the strict model ----------

_VALID_DRAFT = {
    "destination_or_route": "四姑娘山长坪沟",
    "activity_date": "2026-11-08",
    "participants": 12,
}


def _draft_frames(frames):
    return [f for f in frames if f["type"] == "plan.draft_state"]


def test_dialog_submit_over_http_creates_a_minimal_draft(flow):
    """AC6 + AC8: ``plan_input`` is a whitelisted field, not an extra.

    ``ChatV2Request`` is ``extra="forbid"``, so without the field the entry
    dialog's submit would 422 before streaming and the minimal draft would
    be unreachable over HTTP no matter what the orchestrator supports.
    """
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": "req-draft-submit",
            "conversation_id": "conv-01",
            "intent": "plan_create",
            "plan_input": _VALID_DRAFT,
        },
    )

    frames = _frames(response)
    drafts = _draft_frames(frames)
    assert [d["payload"]["stage"] for d in drafts] == ["minimal_draft"]
    assert drafts[0]["payload"]["draft"] == _VALID_DRAFT
    assert drafts[0]["payload"]["intent"] == "plan_create"
    assert drafts[0]["payload"]["source"] == "structured_intent"
    assert frames[-1]["type"] == "done"

    stored = flow.repo.get_plan_draft("conv-01")
    assert (stored.stage, stored.destination_or_route, stored.activity_date,
            stored.participants) == ("minimal_draft", "四姑娘山长坪沟", "2026-11-08", 12)
    # Deterministic entry: no model call decided anything (spec AC4).
    assert flow.model_calls == []


def test_minimal_draft_preview_reaches_the_stream(flow):
    """AC1: submitting the three fields yields a previewable draft text."""
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": "req-draft-preview",
            "conversation_id": "conv-01",
            "intent": "plan_create",
            "plan_input": _VALID_DRAFT,
        },
    )

    text = "".join(
        f["payload"].get("content", "") for f in _frames(response) if f["type"] == "text"
    )
    assert "四姑娘山长坪沟" in text and "2026-11-08" in text and "12" in text
    # Nothing uncollected is invented: 待确认 for arrangements, 待核验 for
    # route facts (docs/plan-pdf-format.md 内容边界).
    assert "待确认" in text and "待核验" in text


@pytest.mark.parametrize("plan_input", [
    {**_VALID_DRAFT, "activity_date": "2026-02-30"},
    {**_VALID_DRAFT, "activity_date": "下周六"},
    {**_VALID_DRAFT, "participants": "12"},
    {**_VALID_DRAFT, "participants": 0},
    {**_VALID_DRAFT, "destination_or_route": ""},
    {**_VALID_DRAFT, "budget": "每人 300 元"},
])
def test_malformed_or_smuggled_plan_input_is_rejected_before_streaming(flow, plan_input):
    """AC6 + AC5: the boundary validates the draft; nothing partial survives.

    The last case is the one that matters most: ``plan_input`` is itself
    ``extra="forbid"``, so a fourth key (a price, a route fact, a config
    blob) never lands in a draft row.
    """
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": "req-bad-draft",
            "conversation_id": "conv-01",
            "intent": "plan_create",
            "plan_input": plan_input,
        },
    )

    assert response.status_code == 422
    assert not response.text.startswith("data:")
    assert flow.repo.resolve_run_id("req-bad-draft") is None
    assert flow.repo.get_plan_draft("conv-01") is None
    assert flow.model_calls == []


def test_plan_input_on_a_non_collecting_intent_is_rejected_by_the_flow(flow):
    """AC5: the state machine still owns the field-carrying rules.

    ``plan_cancel`` is whitelisted at the boundary but must not silently
    carry draft data; the in-depth rejection lands as a stable reason on the
    error terminal rather than a draft row.
    """
    response = _post(
        flow.app,
        flow.subject,
        {
            "message": "生成行程",
            "request_id": "req-cancel-with-fields",
            "conversation_id": "conv-01",
            "intent": "plan_cancel",
            "plan_input": _VALID_DRAFT,
        },
    )

    frames = _frames(response)
    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["reason"] == "plan_input_not_allowed"
    assert _draft_frames(frames) == []
    assert flow.repo.get_plan_draft("conv-01") is None
