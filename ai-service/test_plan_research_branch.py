"""Ticket 04 — 自由目的地/路线的研究分支与通用草案降级.

Seams under test (the ones the seam contract names for this ticket):

* :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` driven
  in-process with a fake kernel over real temp SQLite — the only writer of
  the draft row and the only emitter of ``plan.draft_state``;
* the research leaf in :mod:`agent_v2.route_tools`, beside ticket 03's
  selection leaf: it goes through the SAME deterministic gate
  (:func:`route_compare.service.compare_routes` / ``assess_candidate``) and
  therefore never needs a second one;
* :func:`agent_v2.plan_workflow.draft_summary` — the one renderer whose text
  the stream, the payload and a restored draft all read, and where a generic
  draft's "this is not a route plan" disclaimer lives;
* ``POST /api/v2/chat/sse`` for the wire half (``plan_type`` must exist on
  ``ChatV2Request`` *and* ``ChatFlowRequest``, or the flow silently sees
  ``None`` and ``extra="forbid"`` never gets exercised);
* the frontend sources, pinned the way tickets 01-03 pinned the draft
  vocabulary (there is no frontend test runner).

Live research (``data_mode="live"``, the real ``route_research`` web search)
is NOT exercised here: it needs network + MCP credentials. The insufficient
branch is driven through the offline fixtures in
:func:`route_compare.fixtures.fixture_candidates`, which is the same
``collect``-free path the leaf uses, so "not enough evidence" is tested
against the real gate rather than a stubbed one.

Acceptance criteria:

* AC1 自由输入可以触发路线研究.
* AC2 研究成功后形成带证据的路线候选.
* AC3 研究失败或资料不足时不直接生成正式路线计划.
* AC4 用户可以选择继续生成通用草案或取消.
* AC5 草案状态在摘要和计划书数据中明确标记.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import (
    PLAN_INTENTS,
    PLAN_TYPES,
    PlanDraftInput,
    PlanDraftStage,
    PlanIntent,
    PlanType,
    draft_summary,
    evaluate_plan_intent,
)
from agent_v2.route_tools import RouteSession
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

CONVERSATION = "conv-plan-04"
SUBJECT_ID = "sub-plan-04"

#: The destination the offline ``complete`` fixture actually holds, so research
#: through the real gate succeeds; anything else is a route the fixtures know
#: nothing about, which is the insufficient-evidence branch (AC3).
KNOWN_DESTINATION = "演示林间环线"
UNKNOWN_DESTINATION = "雨崩上村"

RESEARCH_SCENARIOS = ("partial", "empty", "caution", "severe_weather")

#: Ticket 04 adds ``tool_result`` to the plan stream: the research candidates
#: are the server's own evidence, and the existing comparison card is the only
#: place they may be shown (it re-reads them through the leaf on selection).
PLAN_FLOW_EVENT_TYPES = {
    "run_started",
    "plan.draft_state",
    "text",
    "tool_result",
    "error",
    "done",
    "cancelled",
}


def _settings(scenario: str = "complete") -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="disabled",
        agent_v2_state_dir="/tmp/unused",
        route_compare_data_mode="offline",
        route_decision_provider="deterministic",
        route_compare_demo_scenario=scenario,
    )


@pytest.fixture
def plan_repo(tmp_path):
    store = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    yield store
    store.close()


def _drive(repo: RunRepository, *requests: ChatFlowRequest, scenario: str = "complete") -> dict[str, Any]:
    """Drive requests through one orchestrator over the same repository."""
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orchestrator = RunOrchestrator(
        settings=_settings(scenario), runs_repo=repo, memory=_FakeMemory(), kernel_turn=fake_kernel
    )

    async def drive() -> list[list[Any]]:
        return [
            [e async for e in orchestrator.stream_chat(req, subject_id=SUBJECT_ID)]
            for req in requests
        ]

    return {"streams": asyncio.run(drive()), "kernel_calls": kernel_calls, "repo": repo}


class _FakeMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True


def _req(request_id: str, *, message: str = "生成行程", intent: str | None = None,
         plan_input: Any | None = None, route_selection: Any | None = None,
         plan_type: str | None = None,
         conversation_id: str | None = CONVERSATION,
         profile_id: str = "chat") -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message,
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id=profile_id,
        intent=intent,
        plan_input=plan_input,
        route_selection=route_selection,
        plan_type=plan_type,
    )


def _submit(request_id: str, destination: str = KNOWN_DESTINATION) -> ChatFlowRequest:
    """The free destination/route the user typed into the entry dialog (AC1)."""
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate({
            "destination_or_route": destination,
            "activity_date": "2026-11-08",
            "participants": 12,
        }),
    )


def _draft_events(events):
    return [e for e in events if e.type == "plan.draft_state"]


def _draft_payload(events) -> dict:
    return _draft_events(events)[-1].payload


def _texts(events) -> str:
    return "".join(e.payload.get("content", "") for e in events if e.type == "text")


def _comparisons(events) -> list[dict]:
    return [e.payload["route_comparison"] for e in events
            if e.type == "tool_result" and isinstance(e.payload.get("route_comparison"), dict)]


def _errors(events) -> list[dict]:
    return [e.payload for e in events if e.type == "error"]


# ---- AC1/AC2: free input triggers research that yields evidenced candidates --


def test_the_free_destination_triggers_a_route_research_turn(plan_repo):
    """AC1: a whitelisted ``plan_research_route`` intent runs the research leaf.

    Entry is structured, never a sentence: ``_resolve_plan_entry`` stays
    intent-only on ``route_plan`` and chat-only for natural language, so typed
    text cannot hijack a run (ticket 03's rule still holds here).
    """
    result = _drive(plan_repo, _submit("req-submit"),
                    _req("req-research", message="研究路线资料", intent="plan_research_route"))
    events = result["streams"][1]

    comparisons = _comparisons(events)
    assert len(comparisons) == 1, "one research turn, one research result"
    assert comparisons[0]["status"] == "recommended"
    assert [a["candidate"]["name"] for a in comparisons[0]["candidates"]] == [KNOWN_DESTINATION]

    # Deterministic path: the kernel is never asked to decide a research turn.
    assert result["kernel_calls"] == []
    assert set(e.type for e in events) <= PLAN_FLOW_EVENT_TYPES
    assert _errors(events) == []


def test_research_success_yields_route_candidates_that_carry_their_sources(plan_repo):
    """AC2: the candidates are evidence-backed, i.e. every fact names its source.

    The payload is the stored ``ComparisonResult`` — the same shape ticket 03's
    leaf re-reads — so the evidence shown to the user is what the deterministic
    gate certified, not a summary the plan flow wrote.
    """
    result = _drive(plan_repo, _submit("req-submit"),
                    _req("req-research", message="研究路线资料", intent="plan_research_route"))
    comparison = _comparisons(result["streams"][1])[0]

    candidate = comparison["candidates"][0]
    assert candidate["evidence_status"] == "complete"
    assert candidate["risk_status"] == "clear"
    sources = {source["id"] for source in candidate["candidate"]["sources"]}
    assert sources, "an evidenced candidate cites at least one source"
    for field, fact in candidate["candidate"]["facts"].items():
        assert fact is not None, field
        assert fact["source_id"] in sources, field
        assert fact["source_field"] == field, field

    # The reference the user can later select is a real one: the stored
    # comparison is reachable through the leaf for THIS conversation.
    session = RouteSession(settings=_settings(), run_id="reader", audit_sink=None,
                           runs_repo=plan_repo, conversation_id=CONVERSATION)
    outcome = session.select_candidate(comparison["comparison_id"], candidate["candidate"]["id"])
    assert outcome.status == "ok"
    assert outcome.basis is not None and outcome.basis.candidate_name == KNOWN_DESTINATION


# ---- AC3: insufficient material never becomes a formal route plan ----------


@pytest.mark.parametrize("scenario", RESEARCH_SCENARIOS)
def test_insufficient_research_parks_the_draft_for_a_choice_instead_of_a_plan(
    plan_repo, scenario
):
    """AC3: the gate's own verdict is the only thing that decides the branch.

    Every scenario here is real fixture material that
    :func:`route_compare.service.assess_candidate` refuses — missing weather or
    risk (``partial``), no facts at all (``empty``), moderate risk (``caution``)
    and severe weather (``severe_weather``) — so the flow reacts to the existing
    deterministic gate, not to a second threshold invented for the plan path.
    The draft parks for the user's choice, and nothing labels it a route plan.
    """
    result = _drive(plan_repo, _submit("req-submit"),
                    _req("req-research", message="研究路线资料", intent="plan_research_route"),
                    scenario=scenario)
    events = result["streams"][1]
    payload = _draft_payload(events)
    comparison = _comparisons(events)[0]

    assert comparison["status"] != "recommended"
    assert comparison["report_allowed"] is False
    assert payload["stage"] == "awaiting_generic_choice"
    assert payload["plan_type"] == "generic_draft", "no basis, so never a route plan"
    assert payload["route_selection"] is None
    assert plan_repo.get_route_selection(CONVERSATION) is None, "nothing was promoted"

    text = _texts(events)
    assert "不会据此生成正式路线计划" in text
    assert "route_plan" not in text and "路线核验事实" not in text
    assert result["kernel_calls"] == []


def test_research_of_a_destination_no_source_knows_reports_its_gaps(plan_repo):
    """AC3 honestly: an unknown free destination comes back as 资料不足 with the gap.

    The offline sample set holds two demo routes; a real place name is exactly
    the case where collection produced nothing, so the comparison that reaches
    the user carries the reason it was refused instead of a quiet empty plan.
    """
    result = _drive(plan_repo, _submit("req-submit", UNKNOWN_DESTINATION),
                    _req("req-research", message="研究路线资料", intent="plan_research_route"))
    events = result["streams"][1]
    comparison = _comparisons(events)[0]

    assert [a["candidate"]["name"] for a in comparison["candidates"]] == [UNKNOWN_DESTINATION]
    assert comparison["status"] == "insufficient_evidence"
    assert any("离线样例未包含该路线" in reason
               for a in comparison["candidates"] for reason in a["reasons"])
    assert _draft_payload(events)["plan_type"] == "generic_draft"
    # The arrangements the user gave are retained untouched — a refused research
    # turn must not blank the form or invent a replacement value.
    draft = plan_repo.get_plan_draft(CONVERSATION)
    assert (draft.destination_or_route, draft.activity_date, draft.participants) == (
        UNKNOWN_DESTINATION, "2026-11-08", 12)


def test_research_without_a_submitted_draft_is_a_stable_refusal(plan_repo):
    """AC1/AC3 guard: research reads the draft's destination, so it needs a draft."""
    result = _drive(plan_repo,
                    _req("req-open", message="生成行程", intent="plan_create"),
                    _req("req-research", message="研究路线资料", intent="plan_research_route"))
    events = result["streams"][1]

    assert [e["reason"] for e in _errors(events)] == ["route_research_without_draft"]
    assert _draft_events(events) == [], "a refusal shows no draft state"
    assert _comparisons(events) == [], "and runs no research"
    assert plan_repo.get_plan_draft(CONVERSATION).stage == "awaiting_input"


# ---- AC4: the user chooses — generic draft, or nothing -----------------------


def _parked(plan_repo, request_id: str = "req-research") -> list:
    """Submit a draft, then research a destination nothing can evidence."""
    result = _drive(plan_repo, _submit("req-submit", UNKNOWN_DESTINATION),
                    _req(request_id, message="研究路线资料", intent="plan_research_route"))
    assert _draft_payload(result["streams"][1])["stage"] == "awaiting_generic_choice"
    return result["streams"]


def test_the_user_can_continue_to_a_marked_generic_draft(plan_repo):
    """AC4 + AC5: 「继续生成通用草案」is a conversational turn, and it is marked.

    The choice travels as a whitelisted intent plus the ONE draft type a client
    may name. The result is a draft whose summary and payload both say
    ``generic_draft`` — the streamed text, the payload's summary and the payload's
    marker come from the same render, so they cannot disagree (AC5).
    """
    _parked(plan_repo)
    result = _drive(plan_repo, _req(
        "req-generic", message="继续生成通用草案", intent="plan_generic_draft",
        plan_type="generic_draft",
    ))
    events = result["streams"][0]
    payload = _draft_payload(events)

    assert payload["stage"] == "minimal_draft"
    assert payload["plan_type"] == "generic_draft"
    assert payload["route_selection"] is None
    assert payload["summary"] == _texts(events)

    marked = payload["summary"]
    assert "通用草案" in marked and "generic_draft" in marked
    assert "不是路线计划" in marked
    assert "待核验" in marked and "待确认" in marked
    assert "路线核验事实" not in marked, "no verified-fact section without a basis"
    assert plan_repo.get_plan_draft(CONVERSATION).stage == "minimal_draft"
    assert result["kernel_calls"] == []


def test_the_user_can_cancel_the_parked_choice_instead(plan_repo):
    """AC4: 取消 is the other answer, and cancelling stays inert (02's split)."""
    _parked(plan_repo)
    result = _drive(plan_repo, _req("req-cancel", message="取消", intent="plan_cancel"))
    events = result["streams"][0]

    assert _draft_payload(events)["stage"] == "cancelled"
    assert _draft_payload(events)["plan_type"] == "generic_draft"
    assert _comparisons(events) == [], "a cancel starts no research and no comparison"
    assert plan_repo.get_route_selection(CONVERSATION) is None
    assert result["kernel_calls"] == []


@pytest.mark.parametrize("intent,plan_type,reason", [
    ("plan_generic_draft", "route_plan", "invalid_plan_type"),
    ("plan_generic_draft", None, "invalid_plan_type"),
    ("plan_cancel", "generic_draft", "plan_type_not_allowed"),
    ("plan_create", "generic_draft", "plan_type_not_allowed"),
])
def test_a_draft_type_can_only_be_claimed_by_the_choice_that_means_it(
    plan_repo, intent, plan_type, reason
):
    """AC3/AC5 at the state machine: ``route_plan`` is never a client claim.

    Even the sanctioned intent refuses a ``route_plan`` declaration, and a draft
    type on any other intent is rejected — the label is a statement about
    evidence the server holds, so the only field a client may set says "make
    this the generic branch".
    """
    _parked(plan_repo)
    result = _drive(plan_repo, _req(
        "req-claim", message="声明类型", intent=intent, plan_type=plan_type))
    events = result["streams"][0]

    assert [e["reason"] for e in _errors(events)] == [reason]
    assert _draft_events(events) == []
    assert plan_repo.get_plan_draft(CONVERSATION).stage == "awaiting_generic_choice"


def test_a_research_intent_on_the_route_plan_profile_stays_intent_only(plan_repo):
    """Landmine: the widened entry is structured — a sentence never hijacks one.

    A ``route_plan`` run carrying no plan intent still reaches the kernel, so
    ticket 04's research branch cannot be entered by typed text alone, and a
    plan intent on that profile is honoured without touching the kernel.
    """
    result = _drive(plan_repo,
                    _req("req-kernel", message="雨崩上村怎么走", profile_id="route_plan"),
                    _req("req-noop", message="研究路线资料", intent=None, profile_id="route_plan"))

    assert result["kernel_calls"] == ["雨崩上村怎么走", "研究路线资料"]
    assert _comparisons(result["streams"][0]) == []


# ---- AC5: the marker is one renderer's answer, in both draft types ---------


def test_a_draft_without_a_basis_is_rendered_as_a_generic_draft():
    """AC5: the disclaimer lives in the one renderer, not in a component."""
    rendered = draft_summary(
        stage=PlanDraftStage.MINIMAL_DRAFT,
        destination_or_route=UNKNOWN_DESTINATION,
        activity_date="2026-11-08",
        participants=12,
    )

    assert "通用草案" in rendered
    assert "草案类型：generic_draft（通用草案，不是路线计划）" in rendered
    assert "不是已核验的路线计划" in rendered
    assert "待核验" in rendered and "待确认" in rendered
    assert "路线核验事实" not in rendered


def test_a_draft_with_a_re_read_basis_is_the_only_one_labelled_route_plan():
    """AC5 the other way: the label flips only when a caller hands in evidence.

    ``draft_summary`` takes no ``plan_type`` argument at all, so there is no
    parameter a caller could use to claim ``route_plan`` — the marker is read
    off the basis object the comparison leaf re-ran the gate on.
    """
    from agent_v2.plan_workflow import RouteBasis

    basis = RouteBasis(comparison_id="run:1", candidate_id="demo-a",
                       candidate_name=KNOWN_DESTINATION, role="recommended",
                       data_mode="offline")
    rendered = draft_summary(
        stage=PlanDraftStage.MINIMAL_DRAFT,
        destination_or_route=KNOWN_DESTINATION,
        activity_date="2026-11-08",
        participants=12,
        route=basis,
    )

    assert "草案类型：route_plan" in rendered
    assert "通用草案" not in rendered
    assert "**路线核验事实" in rendered
    assert "待核验（无服务器证据）" in rendered, "dimensions the gate could not certify stay unknown"


def test_the_parked_research_stage_renders_its_own_refusal_copy():
    """AC3/AC4: the stage the user reads must state the choice it is asking for."""
    rendered = draft_summary(
        stage=PlanDraftStage.AWAITING_GENERIC_CHOICE,
        destination_or_route=UNKNOWN_DESTINATION,
        activity_date="2026-11-08",
        participants=12,
    )

    assert "不会据此生成正式路线计划" in rendered
    assert "通用草案" in rendered and "取消" in rendered


def test_research_and_generic_draft_are_whitelisted_intents():
    """AC1/AC4: both new steps are structured, and derived from one vocabulary."""
    assert PLAN_INTENTS == {
        "plan_create", "plan_confirm", "plan_cancel", "plan_regenerate",
        "plan_select_route", "plan_research_route", "plan_generic_draft",
        # Ticket 05's three review steps; see test_plan_missing_fields_first_review.
        "plan_first_review", "plan_more_input", "plan_confirm_arrangements",
    }
    assert PLAN_TYPES == set(get_args(PlanType)) == {"route_plan", "generic_draft"}
    assert set(get_args(PlanIntent)) == PLAN_INTENTS

    # And each is still decided by the one pure transition function.
    ok = evaluate_plan_intent(intent="plan_research_route",
                              current_stage=PlanDraftStage.MINIMAL_DRAFT,
                              plan_input=None, conversation_id="c")
    assert ok.status == "ok" and ok.stage is PlanDraftStage.MINIMAL_DRAFT
    assert ok.fallback_stage is PlanDraftStage.AWAITING_GENERIC_CHOICE

    choice = evaluate_plan_intent(intent="plan_generic_draft",
                                  current_stage=PlanDraftStage.AWAITING_GENERIC_CHOICE,
                                  plan_input=None, conversation_id="c",
                                  plan_type="generic_draft")
    assert choice.status == "ok" and choice.stage is PlanDraftStage.MINIMAL_DRAFT
    assert choice.plan_type == "generic_draft"


# ---- The HTTP boundary (landmine: flow tests cannot see this) --------------

import api.chat_v2 as chat_api  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_v2 import kernel  # noqa: E402
from agent_v2.memory_service import MemoryService  # noqa: E402
from agent_v2.repositories.memory import MemoryStore  # noqa: E402
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router  # noqa: E402

HTTP_CONVERSATION = "conv-plan-04-http"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """``POST /api/v2/chat/sse`` over ASGI with a fake kernel — the only
    harness that can catch a plan field missing from ``ChatV2Request`` or from
    ``ChatFlowRequest``: the request model is ``extra="forbid"``, so an unwired
    field is a 422 here and a silently-absent value in every flow-level test.
    """
    settings = SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="test",
        openai_model="test",
        agent_v2_cowork_policy="disabled",
        agent_v2_state_dir=str(tmp_path),
        route_compare_data_mode="offline",
        route_decision_provider="deterministic",
        route_compare_demo_scenario="complete",
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


def _frames(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def _draft_frames(frames):
    return [f for f in frames if f["type"] == "plan.draft_state"]


def _comparison_frames(frames):
    return [f for f in frames if f["type"] == "tool_result"
            and isinstance(f["payload"].get("route_comparison"), dict)]


def _http_send(flow, request_id: str, **extra):
    body = {"message": "生成行程", "request_id": request_id,
            "conversation_id": HTTP_CONVERSATION, **extra}
    return _frames(_post(flow.app, flow.subject, body))


def test_a_free_destination_reaches_the_flow_over_http_and_never_the_kernel(flow):
    """AC1/AC2 on the wire: research runs server-side, kernel stays untouched."""
    _http_send(flow, "req-http-submit", intent="plan_create", plan_input={
        "destination_or_route": KNOWN_DESTINATION,
        "activity_date": "2026-11-08",
        "participants": 12,
    })
    frames = _http_send(flow, "req-http-research", intent="plan_research_route")

    assert frames[-1]["type"] == "done"
    assert flow.model_calls == []
    comparison = _comparison_frames(frames)[-1]["payload"]["route_comparison"]
    assert comparison["status"] == "recommended"
    assert _draft_frames(frames)[-1]["payload"]["plan_type"] == "generic_draft"


def test_the_generic_choice_reaches_the_flow_over_http(flow):
    """AC4 + landmine 2: ``plan_type`` must exist on BOTH request models.

    A field on ``ChatV2Request`` that never reached ``ChatFlowRequest`` would
    make this turn come back as ``invalid_plan_type`` instead of the draft,
    while every flow-level test stayed green — so this is the boundary proof.
    """
    _http_send(flow, "req-http-park-submit", intent="plan_create", plan_input={
        "destination_or_route": UNKNOWN_DESTINATION,
        "activity_date": "2026-11-08",
        "participants": 12,
    })
    _http_send(flow, "req-http-park", intent="plan_research_route")
    frames = _http_send(flow, "req-http-generic", intent="plan_generic_draft",
                        message="继续生成通用草案", plan_type="generic_draft")

    payload = _draft_frames(frames)[-1]["payload"]
    assert payload["stage"] == "minimal_draft"
    assert payload["plan_type"] == "generic_draft"
    assert "通用草案" in payload["summary"]
    # A confirmation is a plain conversational turn: the approval machinery
    # (300 s TTL + sandbox lease) never appears on the plan path.
    assert [f["type"] for f in frames if f["type"] == "approval.required"] == []


@pytest.mark.parametrize("plan_type", [
    "计划", "ROUTE_PLAN", "RoutePlan", 1, True, 0, 1.0, {"a": 1}, ["generic_draft"], "", " ",
])
def test_a_malformed_or_smuggled_plan_type_is_rejected_before_streaming(flow, plan_type):
    """AC5 at the boundary: only the two declared type names can even be parsed.

    ``ChatV2Request`` is ``extra="forbid"`` and annotates the field with
    :data:`agent_v2.plan_workflow.PlanType`, so anything that is not exactly one
    of the two members is a 422 before a run exists, and a smuggled fourth
    field (``distance_km``, ``report_allowed``) never gets a wire shape at all.
    """
    response = _post(flow.app, flow.subject, {
        "message": "声明类型",
        "request_id": f"req-bad-type-{abs(hash(str(plan_type))) % 9973}",
        "conversation_id": HTTP_CONVERSATION,
        "intent": "plan_generic_draft",
        "plan_type": plan_type,
    })

    assert response.status_code == 422
    assert not response.text.startswith("data:")
    assert flow.model_calls == []


def test_a_route_plan_claim_over_http_is_refused_by_the_state_machine(flow):
    """AC3 on the wire: a well-formed ``route_plan`` is still not a licence.

    ``route_plan`` parses — it is a real member of the vocabulary — and is
    refused one layer deeper, as a stable reason with no draft state and no
    write. The draft's type is derived from the basis the gate re-read, never
    from what a client said.
    """
    _http_send(flow, "req-http-claim-submit", intent="plan_create", plan_input={
        "destination_or_route": UNKNOWN_DESTINATION,
        "activity_date": "2026-11-08",
        "participants": 12,
    })
    _http_send(flow, "req-http-claim-park", intent="plan_research_route")
    frames = _http_send(flow, "req-http-claim", message="声明类型",
                        intent="plan_generic_draft", plan_type="route_plan")

    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["reason"] == "invalid_plan_type"
    assert _draft_frames(frames) == []
    assert flow.repo.get_plan_draft(HTTP_CONVERSATION).stage == "awaiting_generic_choice"


def test_a_smuggled_research_flag_or_route_fact_is_rejected_before_streaming(flow):
    """AC3: research takes no client-supplied evidence — there is no wire shape.

    A distance, an ascent, a weather string or a ``report_allowed`` flag next to
    a research ask is a 422, exactly as ticket 03 made a smuggled fact next to
    a selection impossible. ``plan_type`` is the only extra field the boundary
    knows about, and it names a draft type, never a route fact.
    """
    response = _post(flow.app, flow.subject, {
        "message": "研究路线资料",
        "request_id": "req-smuggle-research",
        "conversation_id": HTTP_CONVERSATION,
        "intent": "plan_research_route",
        "distance_km": 8.0,
    })

    assert response.status_code == 422
    assert flow.repo.resolve_run_id("req-smuggle-research") is None


# ---- Frontend wiring, pinned from the sources (no test runner exists) ------

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _ts(path: str) -> str:
    return (FRONTEND / path).read_text(encoding="utf-8")


def _ts_callback(source: str, name: str) -> str:
    """The body of a ``const <name> = useCallback(...)`` handler, verbatim."""
    start = source.index(f"const {name} = useCallback(")
    depth, index = 0, start + len("const ")
    for offset in range(index, len(source)):
        if source[offset] == "(":
            depth += 1
        elif source[offset] == ")":
            depth -= 1
            if depth == 0:
                return source[start:offset + 1]
    raise AssertionError(f"unbalanced useCallback for {name}")


def test_plan_type_is_read_compared_and_typed_on_the_frontend():
    """Landmine: the draft payload has FOUR frontend places, three of them silent.

    ``V2PlanDraftState`` (compile time), ``readPlanDraftState`` (the single
    validator for the SSE payload AND for a localStorage mirror), and
    ``isSamePlanDraft`` — which decides whether a restore is a no-op. Miss the
    last one and a route_plan → generic_draft flip after a refresh hands back the
    OLD object, so the marker the user is owed (AC5) silently vanishes.
    """
    wire = _ts("src/api/chatV2.ts")
    reducer = _ts("src/api/v2Reducer.ts")
    store = _ts("src/api/planDraftStore.ts")

    assert re.search(r"export type V2PlanType = 'route_plan' \| 'generic_draft'", wire)
    assert re.search(r"export const V2_PLAN_TYPES: ReadonlySet<V2PlanType> = new Set\(\[\n"
                     r"  'route_plan',\n  'generic_draft',\n\]\)", wire)
    assert "plan_type: V2PlanType" in wire, "required on the state type, never optional"

    block = reducer[reducer.index("export function readPlanDraftState"):]
    assert "plan_type: readPlanType(payload['plan_type'], routeSelection)" in block
    # An unknown value never guesses its way into route_plan; it falls back to
    # what the reference demonstrably supports.
    read_body = reducer[reducer.index("function readPlanType"):
                        reducer.index("export function readPlanDraftState")]
    assert "V2_PLAN_TYPES.has" in read_body
    assert "return reference ? 'route_plan' : 'generic_draft'" in read_body

    same = store[store.index("export function isSamePlanDraft"):]
    assert "a.plan_type === b.plan_type" in same


def test_the_parked_choice_is_answered_by_two_plain_intent_sends():
    """AC4 in the UI: continue as generic draft, or cancel — both are turns.

    Pinned from the sources because there is no frontend runner: the affordance
    exists for the stage the server parks on, the affirmative send is the ONLY
    place this page names a draft type, and it names ``generic_draft``. The
    negative answer reuses ticket 02's single ``plan_cancel`` exit, and closing
    the dialog still only parks (never cancels).
    """
    page = _ts("src/pages/SuperAgent.tsx")

    assert "planDraft?.stage === 'awaiting_generic_choice'" in page
    affirm = _ts_callback(page, "handleContinueAsGenericDraft")
    assert "intent: 'plan_generic_draft'" in affirm
    assert "planType: 'generic_draft'" in affirm
    assert "route_plan" not in affirm

    # The send is a conversational turn: no approval machinery is wired here.
    assert "approval" not in affirm
    # And the dialog's dismissal still parks rather than cancels (ticket 02).
    assert "plan_cancel" not in _ts_callback(page, "handlePlanDialogDismiss")


def test_the_research_affordance_sends_the_intent_and_nothing_else():
    """AC1 in the UI: research is asked for, never supplied with evidence.

    The send carries the whitelisted intent alone — the query is the destination
    the draft already holds and every fact comes back from the server, so no
    search term, distance, weather string or draft type can leave this page.
    """
    page = _ts("src/pages/SuperAgent.tsx")
    research = _ts_callback(page, "handleResearchRoute")

    assert "intent: 'plan_research_route'" in research
    for smuggled in ("planType", "planInput", "routeSelection", "profileId"):
        assert smuggled not in research
    assert "handleResearchRoute}" in page, "the affordance is rendered, not just defined"
    assert "研究路线资料" in page


def test_the_page_never_claims_a_route_plan_for_the_user():
    """AC3/AC5 in the UI: the only labels this page can send say generic_draft."""
    page = _ts("src/pages/SuperAgent.tsx")

    assert "planType: 'route_plan'" not in page
    assert "plan_type: 'route_plan'" not in page
    # And the type the user reads is translated from the server's marker, so a
    # generic draft is never dressed up.
    labels = page[page.index("const PLAN_TYPE_LABELS"):]
    assert "generic_draft: '通用草案（无已核验路线依据）'" in labels
    assert "PLAN_TYPE_LABELS[planDraft.plan_type]" in page


def test_every_plan_request_field_exists_on_both_sides_of_the_boundary():
    """Landmine 2 in one assertion: ChatV2Request ↔ ChatFlowRequest field sets.

    ``extra="forbid"`` makes the pair invisible to flow tests: a field the
    adapter validates but the flow never declares is accepted on the wire and
    read as ``None`` in ``_stream_plan_flow`` — every flow-level test stays green
    while the feature is dead. The sets are compared, not sampled.
    """
    import dataclasses

    import api.chat_v2 as chat_api

    wire_fields = set(chat_api.ChatV2Request.model_fields)
    flow_fields = {f.name for f in dataclasses.fields(ChatFlowRequest)}

    assert {"intent", "plan_input", "route_selection", "plan_type"} <= wire_fields
    assert flow_fields <= wire_fields | {"current_location"}
    for plan_field in ("intent", "plan_input", "route_selection", "plan_type"):
        assert plan_field in flow_fields, f"{plan_field} never reaches the flow"
        assert plan_field in wire_fields, f"{plan_field} never reaches the adapter"


