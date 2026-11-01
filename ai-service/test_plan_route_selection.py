"""Ticket 03 — 复用已有路线比较结果.

Seams under test (the ones the context pack names for this ticket):

* :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` driven
  in-process with a fake kernel over real temp SQLite — the only writer of
  the draft row and the only emitter of ``plan.draft_state``;
* :class:`agent_v2.route_tools.RouteSession` as the *only* way the plan flow
  reaches a comparison result: a selection stores
  ``comparison_id`` + ``candidate_id``, never a copy of the facts;
* :func:`agent_v2.plan_workflow.draft_summary` — the one renderer whose text
  the stream, the payload and a restored draft all read;
* ``POST /api/v2/chat/sse`` for the wire half (``route_selection`` must exist
  on ``ChatV2Request`` *and* ``ChatFlowRequest``, or the flow silently sees
  ``None``);
* the frontend sources, pinned the way tickets 01/02 pinned the draft
  vocabulary (there is no frontend test runner).

Acceptance criteria:

* AC1 用户可以选择推荐路线或备选路线.
* AC2 距离、爬升、耗时、天气和开放状态保留来源信息.
* AC3 用户安排与路线核验事实在草稿中分开保存.
* AC4 没有可用比较结果时不展示伪造路线事实.
* AC5 路线选择结果可以进入后续摘要确认.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import (
    PlanDraftInput,
    PlanDraftStage,
    RouteBasis,
    RouteFactLine,
    draft_summary,
)
from agent_v2.route_tools import RouteSession
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

CONVERSATION = "conv-plan-03"
SUBJECT_ID = "sub-plan-03"

#: The offline fixture comparison has exactly these two gate-passed candidates,
#: and the deterministic provider ranks them in request order — so ``demo-a``
#: is the recommendation and ``demo-b`` the alternative (AC1's two choices).
RECOMMENDED_ID = "demo-a"
ALTERNATIVE_ID = "demo-b"

PLAN_FLOW_EVENT_TYPES = {
    "run_started",
    "plan.draft_state",
    "text",
    "error",
    "done",
    "cancelled",
}

_DEMO_QUERIES = [
    {"name": "演示林间环线", "city": "演示城市", "origin": "起点", "destination": "终点"},
    {"name": "演示山脊路线", "city": "演示城市", "origin": "起点", "destination": "终点"},
]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="disabled",
        agent_v2_state_dir="/tmp/unused",
        route_compare_data_mode="offline",
        route_decision_provider="deterministic",
        route_compare_demo_scenario="complete",
    )


@pytest.fixture
def plan_repo(tmp_path):
    store = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    yield store
    store.close()


def _drive(repo: RunRepository, *requests: ChatFlowRequest) -> dict[str, Any]:
    """Drive requests through one orchestrator over the same repository."""
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orchestrator = RunOrchestrator(
        settings=_settings(), runs_repo=repo, memory=_FakeMemory(), kernel_turn=fake_kernel
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
    )


def _submit(request_id: str) -> ChatFlowRequest:
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate({
            "destination_or_route": "演示林间环线",
            "activity_date": "2026-11-08",
            "participants": 12,
        }),
    )


def _selection(comparison_id: str, candidate_id: str) -> dict:
    return {"comparison_id": comparison_id, "candidate_id": candidate_id}


def _seed_comparison(repo: RunRepository, conversation_id: str,
                     *, scenario: str = "complete", request_id: str = "req-compare") -> dict:
    """Produce a gate-passed comparison the way the kernel tool does.

    The comparison is written into its OWN ``route_plan`` run and persisted as
    that run's ``tool_result`` event, which is exactly the shape
    :meth:`RouteSession.load_result` reads back. Nothing here hands the plan
    flow a fact: the only thing the later turns can legitimately reach is the
    reference.
    """
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="route_plan",
        user_message="比较演示路线",
    )
    settings = _settings()
    settings.route_compare_demo_scenario = scenario
    session = RouteSession(settings=settings, run_id=run_id, audit_sink=None)
    result = asyncio.run(session.compare(list(_DEMO_QUERIES)))
    repo.persist_event(
        run_id=run_id, sequence=1, event_type="tool_result",
        request_id=request_id, conversation_id=conversation_id,
        payload={"route_comparison": result},
    )
    return result


def _draft_events(events):
    return [e for e in events if e.type == "plan.draft_state"]


def _texts(events) -> str:
    return "".join(e.payload.get("content", "") for e in events if e.type == "text")


# ---- AC1: the user can pick the recommendation or an alternative -----------


def test_the_recommended_route_from_an_existing_comparison_can_be_selected(plan_repo):
    """AC1: a ``plan_select_route`` intent stores a reference to the candidate.

    The comparison lives in another run; the plan flow reaches it through
    :meth:`RouteSession.load_result` and keeps only ``comparison_id`` +
    ``candidate_id`` on the draft — the invariant the whole ticket rests on.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    assert comparison["recommended_id"] == RECOMMENDED_ID

    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
    )
    events = result["streams"][1]
    payload = _draft_events(events)[-1].payload

    assert payload["stage"] == "minimal_draft"
    assert payload["intent"] == "plan_select_route"
    assert payload["route_selection"] == {
        "comparison_id": comparison["comparison_id"],
        "candidate_id": RECOMMENDED_ID,
    }

    draft = plan_repo.get_plan_draft(CONVERSATION)
    stored = plan_repo.get_route_selection(CONVERSATION)
    assert (stored.comparison_id, stored.candidate_id) == (comparison["comparison_id"], RECOMMENDED_ID)
    assert stored.draft_id == draft.draft_id, "the reference belongs to this draft"

    # Deterministic path: the kernel is never asked to decide a selection.
    assert result["kernel_calls"] == []
    assert set(e.type for e in events) <= PLAN_FLOW_EVENT_TYPES


def test_an_alternative_route_can_be_selected_instead(plan_repo):
    """AC1: 备选路线 is a first-class choice, not a re-comparison."""
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    assert ALTERNATIVE_ID in comparison["alternatives"]

    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select-alt", message="选用备选路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], ALTERNATIVE_ID)),
    )
    payload = _draft_events(result["streams"][1])[-1].payload

    assert payload["route_selection"]["candidate_id"] == ALTERNATIVE_ID
    summary = payload["summary"]
    assert "演示山脊路线" in summary and "备选" in summary
    # The alternative's own verified numbers, from the stored result.
    assert "12" in summary and "650" in summary
    assert result["repo"].get_route_selection(CONVERSATION).candidate_id == ALTERNATIVE_ID


def test_a_selection_needs_a_draft_to_carry_it(plan_repo):
    """AC3 precondition, enforced as a stable reason: facts join a draft.

    Selecting before the arrangements exist would put a verified route card in
    front of an empty form, and a cancelled draft has nothing to join. Both are
    refused by the state machine rather than guessed at.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _req("req-select-cold", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
    )
    events = result["streams"][0]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "route_selection_without_draft"
    assert _draft_events(events) == []
    assert result["repo"].get_plan_draft(CONVERSATION) is None
    assert result["repo"].get_route_selection(CONVERSATION) is None
    assert result["kernel_calls"] == []


# ---- AC2: the five facts keep their provenance -----------------------------


def test_distance_ascent_duration_weather_and_open_status_keep_their_sources(plan_repo):
    """AC2: each carried fact names the source that makes it verified.

    Provenance here is the candidate's own ``source_id`` + ``source_field``
    joined to that source's ``tool`` and ``observed_at`` — the same quartet the
    comparison card shows, read back through the session rather than copied.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    source = comparison["candidates"][0]["candidate"]["sources"][0]

    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
    )
    payload = _draft_events(result["streams"][1])[-1].payload
    summary = payload["summary"]

    assert "**路线核验事实" in summary
    for label, value, field in (
        ("距离", "8", "distance_km"),
        ("爬升", "300", "ascent_m"),
        ("耗时", "4", "duration_hours"),
        ("天气", "晴", "weather"),
        ("开放状态", "开放", "trail_status"),
    ):
        assert label in summary, label
        assert field in summary, field

    # One provenance pattern for every line: the source id, its tool, the field
    # it was read from, and when it was observed.
    assert f"[{source['id']}]" in summary
    assert source["tool"] in summary
    assert source["observed_at"] in summary
    # Offline fixture data must never read as live: the card says it, so must
    # the draft that reuses it.
    assert "离线演示" in summary


# ---- AC3: arrangements and verified facts stay separate -------------------


def test_user_arrangements_and_verified_route_facts_are_stored_separately(plan_repo):
    """AC3: two storage places, two render sections, one pointer between them.

    Asserted through the repository's public reads rather than table internals:
    the draft row holds only the three arrangements (and no fact value appears
    anywhere in it), the selection row holds only ids, and the summary keeps
    them under separate headings — an unfilled arrangement stays 待确认 even
    though the route's distance and ascent are known.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
    )
    summary = _draft_events(result["streams"][1])[-1].payload["summary"]

    draft = result["repo"].get_plan_draft(CONVERSATION)
    selection = result["repo"].get_route_selection(CONVERSATION)

    def values_of(record) -> list[Any]:
        return list(asdict(record).values())

    # The draft row is arrangements only: no reference columns, and nothing in
    # it carries a verified fact value.
    assert [draft.destination_or_route, draft.activity_date, draft.participants] == [
        "演示林间环线", "2026-11-08", 12,
    ]
    assert not any(key in asdict_keys(draft) for key in ("comparison_id", "candidate_id")), (
        "the reference is not stored on the draft row"
    )
    for fact_value in ("8.0", "300.0", "晴", "open"):
        assert fact_value not in [str(value) for value in values_of(draft)]

    # The selection row is the reference only: ids, never facts.
    for fact_value in ("8.0", "300.0", "晴", "open", "low", "4.0"):
        assert fact_value not in [str(value) for value in values_of(selection)]

    # Rendered as two sections: the user's own unfilled arrangements stay
    # 待确认 while the route facts below them are verified.
    arrangements, facts = summary.split("**路线核验事实")
    assert "**最小计划草稿" in arrangements and "待确认" in arrangements
    assert "集合时间" in arrangements and "费用预算" in arrangements
    assert "距离" in facts and "来源" in facts
    assert "待确认" not in facts.split("路线事实由服务器复核")[0], (
        "verified facts are not presented as arrangements awaiting confirmation"
    )


def asdict_keys(record) -> list[str]:
    return list(asdict(record).keys())


# ---- AC4: nothing usable, nothing shown -----------------------------------


def test_no_comparison_result_means_no_route_facts_anywhere(plan_repo):
    """AC4: with no comparison at all, the draft shows nothing fabricated."""
    result = _drive(plan_repo, _submit("req-submit"))
    payload = _draft_events(result["streams"][0])[-1].payload

    assert payload["route_selection"] is None
    summary = payload["summary"]
    assert "路线核验事实" not in summary
    assert "待核验" in summary
    assert "来源 [" not in summary
    # No number that could be read as a distance or an ascent. Ticket 05
    # itemised the unverified dimensions from ``FACT_DIMENSIONS`` instead of the
    # old paraphrased blanket line, so the labels are now named — each one with
    # its 待核验 marker and never with a value, which is the assertion below.
    assert "km" not in summary
    assert "- 爬升：待核验（无服务器证据）" in summary
    assert not re.search(r"(距离|爬升|耗时|天气)[:：]\s*[\d.]", summary)
    assert result["repo"].get_route_selection(CONVERSATION) is None


@pytest.mark.parametrize("scenario,bogus_comparison,candidate_id,reason", [
    # A comparison id that was never issued.
    ("complete", True, RECOMMENDED_ID, "comparison_not_found"),
    # A real comparison, a candidate nobody offered.
    ("complete", False, "demo-z", "route_not_selectable"),
    # A real comparison whose gate refused every recommendation.
    ("severe_weather", False, RECOMMENDED_ID, "route_not_selectable"),
])
def test_a_selection_that_is_not_usable_is_refused_without_inventing_anything(
    plan_repo, scenario, bogus_comparison, candidate_id, reason
):
    """AC4: every "no usable comparison" shape ends in a reason, not a fact.

    The draft keeps the stage and the fields it had, no reference row appears,
    and the summary the user still has is the 待核验 one — the server never
    fills a route in to make a turn look productive.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION, scenario=scenario)
    chosen = (
        "run-never-issued:00000000-0000-0000-0000-000000000000"
        if bogus_comparison else comparison["comparison_id"]
    )

    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用路线", intent="plan_select_route",
             route_selection=_selection(chosen, candidate_id)),
    )
    events = result["streams"][1]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == reason
    assert _draft_events(events) == []
    assert result["repo"].get_route_selection(CONVERSATION) is None
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "minimal_draft"
    assert result["kernel_calls"] == []
    assert "路线核验事实" not in _texts(events)


def test_another_conversations_comparison_cannot_be_selected(plan_repo):
    """AC4 + the ownership rule: a reference is scoped to this conversation.

    ``load_result`` only returns a comparison whose owning run belongs to the
    same conversation, so a copied id from someone else's chat resolves to
    nothing and no fact of theirs can enter this draft.
    """
    other = _seed_comparison(plan_repo, "conv-plan-03-stranger", request_id="req-stranger")
    _drive(plan_repo, _submit("req-submit"))

    result = _drive(plan_repo,
        _req("req-select", message="选用别人的路线", intent="plan_select_route",
             route_selection=_selection(other["comparison_id"], RECOMMENDED_ID)),
    )
    events = result["streams"][0]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "comparison_not_found"
    assert result["repo"].get_route_selection(CONVERSATION) is None
    assert "路线核验事实" not in _texts(events)


def test_a_stored_reference_is_re_read_and_re_verified_on_every_use(plan_repo):
    """The reference-not-copy invariant, as observable behaviour.

    After the selection lands, the stored evidence is edited under the SAME
    ``comparison_id`` so one fact is no longer supported by its source. The next
    plan turn reads through :class:`RouteSession` again and the gate refuses
    that candidate, so the draft must lose the fact — a draft that had copied
    the numbers at selection time would keep showing them.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
    )
    assert "路线核验事实" in _draft_events(result["streams"][1])[-1].payload["summary"]

    # Corrupt the evidence the reference points at: the weather claim loses the
    # evidence value behind it, so the deterministic gate nulls the fact.
    tampered = json.loads(json.dumps(comparison))
    candidate = tampered["candidates"][0]["candidate"]
    candidate["facts"]["weather"] = None
    candidate["evidence_values"][candidate["facts"]["distance_km"]["source_id"]].pop("weather")
    plan_repo.persist_event(
        run_id=comparison["comparison_id"].rsplit(":", 1)[0],
        sequence=2, event_type="tool_result", request_id="req-compare",
        conversation_id=CONVERSATION, payload={"route_comparison": tampered},
    )

    later = _drive(plan_repo, _req("req-regen", message="重新生成", intent="plan_regenerate"))
    payload = _draft_events(later["streams"][0])[-1].payload

    # The draft still points where the user aimed it...
    assert payload["route_selection"] == {
        "comparison_id": comparison["comparison_id"], "candidate_id": RECOMMENDED_ID,
    }
    # ...but nothing is shown that the evidence no longer supports.
    assert "路线核验事实" not in payload["summary"]
    assert "待核验" in payload["summary"]
    assert "来源 [" not in payload["summary"]


def test_a_selection_is_only_rendered_on_the_draft_stage_that_owns_it(plan_repo):
    """The basis belongs to ``minimal_draft``, not to every stage message.

    An ambiguous planning ask can land in a conversation that already has a
    selected route; the answer is still the confirmation question, so verified
    facts must not be pasted underneath it.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
        _req("req-amb", message="周末想去爬山，还没定", intent=None),
    )
    ask = _draft_events(result["streams"][2])[-1].payload

    assert ask["stage"] == "awaiting_confirmation"
    assert "路线核验事实" not in ask["summary"]
    assert "是否" in ask["summary"]


def test_cancelling_the_draft_clears_the_route_reference(plan_repo):
    """Ticket 02 AC4/AC5 extended: a cancelled plan state leaves no reference.

    Cancelled is terminal, so a dangling route pointer would be exactly the
    "unfinished plan state" 02 promised to clean up — and it must still fire
    nothing (no kernel, no tool, no artifact).
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
        _req("req-cancel", message="先不用了", intent="plan_cancel"),
    )
    cancel_events = result["streams"][2]

    assert _draft_events(cancel_events)[-1].payload["stage"] == "cancelled"
    assert _draft_events(cancel_events)[-1].payload["route_selection"] is None
    assert "路线核验事实" not in _texts(cancel_events)
    assert result["repo"].get_route_selection(CONVERSATION) is None
    assert result["kernel_calls"] == []


# ---- AC5: the selection feeds the summary confirmation ---------------------


def test_the_selection_enters_the_next_plan_turn_without_being_resent(plan_repo):
    """AC5: the reference is part of the draft, so the summary turn reads one.

    Ticket 05's first-summary confirmation re-renders the stored draft. It must
    get the arrangements AND the route basis from the draft alone — the client
    resending ``route_selection`` is not part of the contract, and the text the
    stream shows stays the same string the payload carries (ticket 02's rule
    that a restored draft cannot disagree).
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-select", message="选用推荐路线", intent="plan_select_route",
             route_selection=_selection(comparison["comparison_id"], RECOMMENDED_ID)),
        _req("req-regen", message="重新生成", intent="plan_regenerate"),
    )
    third = result["streams"][2]
    payload = _draft_events(third)[-1].payload

    assert payload["stage"] == "minimal_draft"
    assert payload["route_selection"] == {
        "comparison_id": comparison["comparison_id"], "candidate_id": RECOMMENDED_ID,
    }
    summary = payload["summary"]
    assert "演示林间环线" in summary and "2026-11-08" in summary and "12 人" in summary
    assert "路线核验事实" in summary and "推荐路线" in summary
    assert _texts(third) == summary
    assert result["kernel_calls"] == []


def test_draft_summary_renders_the_basis_as_its_own_section_with_the_disclaimer():
    """AC2/AC3/AC4 at the pure renderer: the copy lives in ``draft_summary``.

    Built from a hand-written ``RouteBasis`` so the expected text is
    independent of the leaf, and pinned because a disclaimer that lived in a
    component could disagree with the stream and the restored draft.
    """
    basis = RouteBasis(
        comparison_id="run-9:cand-9",
        candidate_id="demo-a",
        candidate_name="示范环线",
        role="recommended",
        data_mode="live",
        facts=(RouteFactLine(
            field="distance_km", label="距离", display_value="8",
            source_id="src-1", source_field="distance_km",
            source_tool="amap_map", observed_at="2026-11-01T08:00:00+08:00",
        ),),
        missing_labels=("天气",),
    )
    rendered = draft_summary(
        stage=PlanDraftStage.MINIMAL_DRAFT,
        destination_or_route="示范环线",
        activity_date="2026-11-08",
        participants=12,
        route=basis,
    )

    assert "**最小计划草稿" in rendered and "**路线核验事实" in rendered
    assert "距离：8 km（来源 [src-1] amap_map · 字段 distance_km · 观测 2026-11-01T08:00:00+08:00）" in rendered
    # An unverified fact in a verified section still renders as 待核验.
    assert "天气：待核验（无服务器证据）" in rendered
    # The separation is stated, and the draft is still a draft.
    assert "两者分开保存" in rendered and "尚未通过计划审核" in rendered
    assert "reference" not in rendered

    # Without a basis the same renderer keeps every fact 待核验 (AC4).
    plain = draft_summary(
        stage=PlanDraftStage.MINIMAL_DRAFT,
        destination_or_route="示范环线",
        activity_date="2026-11-08",
        participants=12,
    )
    assert "路线核验事实" not in plain
    assert "待核验" in plain


# ---- The HTTP boundary (landmine: flow tests cannot see this) --------------

import api.chat_v2 as chat_api  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_v2 import kernel  # noqa: E402
from agent_v2.memory_service import MemoryService  # noqa: E402
from agent_v2.repositories.memory import MemoryStore  # noqa: E402
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router  # noqa: E402

HTTP_CONVERSATION = "conv-plan-03-http"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """``POST /api/v2/chat/sse`` over ASGI with a fake kernel — the only
    harness that can catch a plan field missing from ``ChatV2Request``: the
    request model is ``extra="forbid"``, so an unwired field is a 422 here and
    a silently-absent value in every flow-level test.
    """
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


def _frames(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def _draft_frames(frames):
    return [f for f in frames if f["type"] == "plan.draft_state"]


def _http_submit(flow, request_id: str) -> None:
    response = _post(flow.app, flow.subject, {
        "message": "生成行程",
        "request_id": request_id,
        "conversation_id": HTTP_CONVERSATION,
        "intent": "plan_create",
        "plan_input": {
            "destination_or_route": "演示林间环线",
            "activity_date": "2026-11-08",
            "participants": 12,
        },
    })
    assert _frames(response)[-1]["type"] == "done"


def test_a_route_selection_reaches_the_flow_over_http_and_never_the_kernel(flow):
    """AC1 + AC4 on the wire: the reference is accepted, resolved server-side.

    It also pins ticket 03's widening — a plan intent sent with profile
    ``route_plan`` (the profile that produced the comparison) enters the plan
    flow instead of the kernel.
    """
    comparison = _seed_comparison(flow.repo, HTTP_CONVERSATION)
    _http_submit(flow, "req-http-submit")
    flow.model_calls.clear()

    response = _post(flow.app, flow.subject, {
        "message": "选用推荐路线",
        "request_id": "req-http-select",
        "conversation_id": HTTP_CONVERSATION,
        "profile_id": "route_plan",
        "intent": "plan_select_route",
        "route_selection": {
            "comparison_id": comparison["comparison_id"],
            "candidate_id": RECOMMENDED_ID,
        },
    })

    frames = _frames(response)
    drafts = _draft_frames(frames)
    assert [d["payload"]["stage"] for d in drafts] == ["minimal_draft"]
    assert drafts[0]["payload"]["route_selection"] == {
        "comparison_id": comparison["comparison_id"], "candidate_id": RECOMMENDED_ID,
    }
    # Only the plan-flow event vocabulary, and no model was consulted.
    assert set(f["type"] for f in frames) <= PLAN_FLOW_EVENT_TYPES
    assert flow.model_calls == []
    assert flow.repo.get_route_selection(HTTP_CONVERSATION).candidate_id == RECOMMENDED_ID


def test_a_route_plan_run_without_a_plan_intent_still_reaches_the_kernel(flow):
    """The widening is one-directional: no plan intent, no plan flow.

    A route question on profile ``route_plan`` must keep its kernel turn — the
    entry may not hijack it because the words look planning-shaped.
    """
    response = _post(flow.app, flow.subject, {
        "message": "比较演示路线",
        "request_id": "req-http-kernel",
        "conversation_id": HTTP_CONVERSATION,
        "profile_id": "route_plan",
    })

    assert _frames(response)[-1]["type"] == "done"
    assert _draft_frames(_frames(response)) == []
    assert flow.model_calls == ["比较演示路线"]


@pytest.mark.parametrize("route_selection", [
    # A fact smuggled alongside the reference: the draft stores ids only.
    {"comparison_id": "run:1", "candidate_id": "demo-a", "distance_km": 8.0},
    # The client cannot assert report eligibility either.
    {"comparison_id": "run:1", "candidate_id": "demo-a", "report_allowed": True},
    {"comparison_id": "", "candidate_id": "demo-a"},
    {"comparison_id": "run:1", "candidate_id": ""},
    {"comparison_id": "x" * 151, "candidate_id": "demo-a"},
    {"comparison_id": "run:1", "candidate_id": "x" * 81},
    {"candidate_id": "demo-a"},
    "run:1",
])
def test_a_malformed_or_smuggled_route_selection_is_rejected_before_streaming(
    flow, route_selection
):
    """AC3/AC4 at the boundary: only two ids can ever travel, and they validate.

    ``RouteSelectionInput`` is ``extra="forbid"``, so a distance, an ascent or a
    ``report_allowed`` flag next to the reference is a 422 — there is no wire
    shape in which a client supplies a verified fact.
    """
    response = _post(flow.app, flow.subject, {
        "message": "选用路线",
        "request_id": "req-bad-selection",
        "conversation_id": HTTP_CONVERSATION,
        "intent": "plan_select_route",
        "route_selection": route_selection,
    })

    assert response.status_code == 422
    assert not response.text.startswith("data:")
    assert flow.repo.resolve_run_id("req-bad-selection") is None
    assert flow.repo.get_route_selection(HTTP_CONVERSATION) is None
    assert flow.model_calls == []


def test_a_route_selection_on_another_intent_is_rejected_by_the_flow(flow):
    """AC5: in-depth — the state machine, not the model, owns who may select."""
    response = _post(flow.app, flow.subject, {
        "message": "取消",
        "request_id": "req-cancel-with-selection",
        "conversation_id": HTTP_CONVERSATION,
        "intent": "plan_cancel",
        "route_selection": {"comparison_id": "run:1", "candidate_id": "demo-a"},
    })

    frames = _frames(response)
    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["reason"] == "route_selection_not_allowed"
    assert _draft_frames(frames) == []
    assert flow.repo.get_route_selection(HTTP_CONVERSATION) is None


# ---- Frontend wiring, pinned from the sources (no test runner exists) ------

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _source(*parts: str) -> str:
    return (FRONTEND.joinpath(*parts)).read_text(encoding="utf-8")


def test_the_comparison_reader_carries_the_plan_side_fields():
    """``routeComparison.ts`` must describe what the server already sends.

    ``plan_details`` (the user's own arrangements), per-candidate ``gaps``,
    ``ranking`` and the top-level ``required_action`` were all missing, so a
    card built from this type could not tell a selectable alternative from a
    route with evidence gaps, and the draft had nothing to point at.
    """
    reader = _source("src", "api", "routeComparison.ts")

    assert "plan_details" in reader and "ranking" in reader
    assert "required_action" in reader and "gaps" in reader
    assert "RoutePlanDetails" in reader or "plan_details: RoutePlanDetails" in reader
    # The reference a selection sends is exactly the pair the server stores.
    assert "comparison_id" in reader and "candidate_id" in reader


def test_the_comparison_card_offers_a_selection_and_nothing_else():
    """AC1/AC4 on the card: a select affordance, only for what was offered.

    The selection belongs on the existing card (no parallel card), and it must
    not appear for a candidate the comparison did not recommend or list, nor
    when there is no usable result at all.
    """
    card = _source("src", "components", "RouteComparison.tsx")

    assert "onSelect" in card, "the card gains a select action"
    assert "RouteComparisonCard" in card
    assert "recommended" in card and "alternate" in card
    # Only the recommendation / alternatives branch may offer it.
    assert "selectable" in card
    assert "选择该路线进入计划书草稿" in card or "选用该路线" in card
    assert "disabled={busy" in card, "a select action must respect the in-flight guard"
    # The provenance display ticket 02 shipped stays intact.
    assert "safeSourceUrl" in card and "observed_at" in card


def test_the_route_reference_travels_on_the_draft_wire_and_the_store():
    """Landmine: a new draft field has FOUR places, and the fourth is silent.

    ``V2PlanDraftState`` + ``readPlanDraftState`` + ``planDraftStore.isSamePlanDraft``
    + the send path. Missing the comparison in ``isSamePlanDraft`` makes a
    restore return the OLD object, so the reference quietly vanishes after a
    refresh and looks like a store bug.
    """
    wire = _source("src", "api", "chatV2.ts")
    reducer = _source("src", "api", "v2Reducer.ts")
    store = _source("src", "api", "planDraftStore.ts")
    page = _source("src", "pages", "SuperAgent.tsx")

    assert "'plan_select_route'" in wire, "intent union"
    assert "route_selection" in wire and "V2PlanRouteReference" in wire
    assert "route_selection" in reducer and "readPlanDraftState" in reducer
    assert "a.route_selection" in store and "b.route_selection" in store, "isSamePlanDraft"
    assert "comparison_id" in store and "candidate_id" in store

    # The send path is structured: intent + reference, never a sentence with a
    # pseudo id in it for the server to re-parse.
    assert "plan_select_route" in page
    assert "route_selection" in page
    assert "onSelectRoute" in page
