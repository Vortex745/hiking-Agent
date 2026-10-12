"""Ticket 05 — 缺失信息决策与第一次摘要确认.

Seams under test (the ones notes-04 names for this ticket):

* :func:`agent_v2.plan_workflow.evaluate_plan_intent` — still the ONLY
  transition decider: 继续补充 / 直接生成草案 / 确认摘要 are three whitelisted
  intents and the stages they may or may not leave the draft on;
* :func:`agent_v2.plan_workflow.draft_summary` + :func:`render_minimal_draft`
  — the one renderer whose text the stream, the ``plan.draft_state`` payload
  and a restored draft all read. The 待确认 / 待核验 boundary and the
  no-invention rule (AC4) live here, so they cannot be re-decided by a
  component;
* :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` driven
  in-process with a fake kernel over real temp SQLite — the only writer of
  the draft row and the only emitter of ``plan.draft_state``; the review
  stages must re-read the stored route basis through
  :meth:`agent_v2.route_tools.RouteSession.select_candidate` (notes-04:
  otherwise the first summary prints 待核验 for facts the draft holds);
* ``POST /api/v2/chat/sse`` for the wire half — a review decision is a plain
  conversational turn, never the tool-approval pause (300 s TTL + sandbox
  lease), and a smuggled field next to it is a 422 before a run exists;
* the frontend sources, pinned the way tickets 01-04 pinned the draft
  vocabulary (there is no frontend test runner).

Acceptance criteria:

* AC1 缺失字段清单明确展示.
* AC2 支持「继续补充」和「直接生成草案」.
* AC3 第一次摘要只确认用户填写的安排和输入.
* AC4 系统不自动补充日期、费用、电话或撤退地点.
* AC5 用户确认结果可被恢复和审计.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import (
    ARRANGEMENT_FIELDS,
    BLOCK_MESSAGE,
    FACT_DIMENSIONS,
    PLAN_INTENTS,
    SUPPLEMENT_ARRANGEMENT_KEYS,
    PlanDraftInput,
    PlanDraftStage,
    RouteBasis,
    RouteFactLine,
    UNCONFIRMED,
    UNVERIFIED,
    draft_summary,
    evaluate_plan_completeness,
    evaluate_plan_intent,
    missing_plan_fields,
    render_minimal_draft,
)
from agent_v2.route_tools import RouteSession
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from route_compare.models import PlanDetails

CONVERSATION = "conv-plan-05"
SUBJECT_ID = "sub-plan-05"

#: Ticket 05 adds no event type: a review turn is the same plan stream. A
#: ``tool_call`` / ``artifact_candidate`` / ``run.paused`` here would mean the
#: summary confirmation started work it does not own (spec AC5 + the ruling in
#: ``seam-contract.md`` that confirmations are never the approval machinery).
PLAN_FLOW_EVENT_TYPES = {
    "run_started",
    "plan.draft_state",
    "text",
    "error",
    "done",
    "cancelled",
}

_DEMO_ROUTE = "演示林间环线"
_DESTINATION = "四姑娘山长坪沟"
_DATE = "2026-11-08"
_PEOPLE = 12

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _ts(path: str) -> str:
    return (FRONTEND / path).read_text(encoding="utf-8")


def _ts_union_members(source: str, type_name: str) -> set[str]:
    """Members of ``export type <name> = ...``, comments stripped.

    The block is read up to the next ``export`` and its /* */ comments removed
    first: those comments contain apostrophes ("the draft's own destination"),
    which a naive quote scan pairs into fake members.
    """
    start = source.index(f"export type {type_name} =")
    block = re.sub(r"/\*.*?\*/", "", source[start:], flags=re.S)
    block = re.split(r"\nexport ", block, maxsplit=1)[0]
    return set(re.findall(r"'([^']+)'", block))


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
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orchestrator = RunOrchestrator(
        settings=_settings(), runs_repo=repo, memory=_FakeMemory(),
        kernel_turn=fake_kernel,
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


def _submit(request_id: str, *, arrangements: dict[str, str] | None = None) -> ChatFlowRequest:
    payload: dict[str, Any] = {
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
    }
    # Ticket 06 turned 集合时间/集合地点/应急联络 into CORE arrangements: 「直接
    # 生成草案」 over a draft missing one bounces to 补充流程 (fail-closed AC3).
    # The review-flow tests below need a draft that clears the gate, so they pass
    # the core in; the missing-field-list tests keep the bare submit.
    if arrangements is not None:
        payload["arrangements"] = arrangements
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate(payload),
    )


#: The three arrangements ticket 06 marks core — filled in a valid format, the
#: draft can reach the first summary; missing one bounces the click to 补充流程.
_CORE = {
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
}


def _supplement(**overrides: str) -> dict[str, str]:
    """The supplementary record with only the core filled by default."""
    values = {key: "" for key in SUPPLEMENT_ARRANGEMENT_KEYS}
    values.update(_CORE)
    values.update(overrides)
    return values


def _draft_events(events):
    return [e for e in events if e.type == "plan.draft_state"]


def _draft_payload(events) -> dict:
    return _draft_events(events)[-1].payload


def _texts(events) -> str:
    return "".join(e.payload.get("content", "") for e in events if e.type == "text")


def _errors(events) -> list[dict]:
    return [e.payload for e in events if e.type == "error"]


def _evaluate(intent, stage=None, **kwargs):
    return evaluate_plan_intent(
        intent=intent, current_stage=stage, plan_input=kwargs.get("plan_input"),
        conversation_id=kwargs.get("conversation_id", "conv-05"),
        route_selection=kwargs.get("route_selection"), plan_type=kwargs.get("plan_type"),
        completeness=kwargs.get("completeness"),
    )


# ---- AC2: the two decisions, and the confirmation, are state-machine moves --


def test_the_first_review_intents_are_whitelisted_and_derive_from_one_list():
    """AC2/AC3: three new structured steps, no new wire field to forget.

    ``ChatV2Request.intent``, ``V2PlanIntent`` and ``PLAN_INTENT_LABELS`` all
    derive from :data:`PlanIntent`, so the whitelist is the only place a plan
    intent is declared (ticket 03/04's rule, unchanged).
    """
    assert PLAN_INTENTS >= {
        "plan_first_review", "plan_more_input", "plan_confirm_arrangements",
    }

    wire = _ts("src/api/chatV2.ts")
    assert _ts_union_members(wire, "V2PlanIntent") == set(PLAN_INTENTS)


def test_direct_generation_moves_a_submitted_draft_to_the_first_review():
    """AC2: 「直接生成草案」 is the decision that produces the first summary.

    Ticket 06 makes the verdict a required input and fails the gate CLOSED: the
    move only reaches the summary once the CORE arrangements are present, so the
    draft here is judged with them filled (a core-missing draft bounces to 补充
    流程 — ticket 06's own test covers that branch)."""
    ready = evaluate_plan_completeness(
        destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        arrangements=_supplement(), route=None,
    )
    assert ready.verdict != "supplement_required"
    transition = _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT,
                           completeness=ready)

    assert transition.status == "ok"
    assert transition.stage is PlanDraftStage.AWAITING_FIRST_REVIEW


def test_continuing_to_supplement_returns_the_draft_to_the_form():
    """AC2: 「继续补充」 is the other answer, and it opens the form again."""
    for stage in (PlanDraftStage.MINIMAL_DRAFT, PlanDraftStage.AWAITING_FIRST_REVIEW):
        transition = _evaluate("plan_more_input", stage)
        assert transition.status == "ok", stage
        assert transition.stage is PlanDraftStage.AWAITING_INPUT


def test_the_first_summary_can_only_be_confirmed_while_it_is_pending():
    """AC3/AC5: a confirmation is recorded only against a summary that was shown.

    Confirming before the first summary exists would let the flow claim the
    user approved a draft they never saw, so the only legal source stage is
    ``awaiting_first_review`` — and the move is recorded, not inferred.
    """
    ok = _evaluate("plan_confirm_arrangements", PlanDraftStage.AWAITING_FIRST_REVIEW)
    assert ok.status == "ok" and ok.stage is PlanDraftStage.FIRST_REVIEWED

    for stage in (None, PlanDraftStage.AWAITING_INPUT, PlanDraftStage.MINIMAL_DRAFT,
                  PlanDraftStage.AWAITING_GENERIC_CHOICE, PlanDraftStage.FIRST_REVIEWED,
                  PlanDraftStage.CANCELLED):
        blocked = _evaluate("plan_confirm_arrangements", stage)
        assert blocked.blocked, stage
        assert blocked.reason == "arrangements_not_pending", stage


@pytest.mark.parametrize("intent,reason", [
    ("plan_first_review", "first_review_without_draft"),
    ("plan_more_input", "supplement_without_draft"),
])
def test_a_review_decision_without_a_submitted_draft_is_a_stable_refusal(intent, reason):
    """The decisions need arrangements: from ``awaiting_input`` there is nothing
    to review and nothing to supplement, so the reason is stable and stored
    nothing happens (the same shape as tickets 03/04's refusals)."""
    transition = _evaluate(intent, PlanDraftStage.AWAITING_INPUT)

    assert transition.blocked and transition.reason == reason
    assert BLOCK_MESSAGE[reason]


def test_a_review_decision_carries_no_fields_reference_or_type_claim():
    """AC3 at the boundary: the decision intents accept no payload at all.

    The three guards tickets 01/03/04 built already cover the new intents:
    a draft field, a route reference or a draft type next to a review decision
    is refused rather than silently ignored — so a client cannot smuggle an
    arrangement (a date, a cost, a phone number) alongside the click that asks
    for the summary.
    """
    assert _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT,
                     plan_input={"destination_or_route": "x",
                                 "activity_date": "2026-11-08",
                                 "participants": 3}).reason == "plan_input_not_allowed"
    assert _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT,
                     route_selection={"comparison_id": "r:1",
                                      "candidate_id": "demo-a"}).reason == (
        "route_selection_not_allowed")
    assert _evaluate("plan_confirm_arrangements", PlanDraftStage.AWAITING_FIRST_REVIEW,
                     plan_type="generic_draft").reason == "plan_type_not_allowed"


# ---- AC1: the missing-field list is one enumeration, not a second list ------

#: The fields ticket 05's own wording names as the ones the system must never
#: fill in: 日期 / 费用 / 电话 / 撤退地点, plus the arrangements the plan document
#: collects (docs/plan-pdf-format.md). Spelled out here as literals from the
#: spec, not recomputed from the module under test.
_MUST_STAY_BLANK = (
    "meeting_time", "meeting_point", "leader", "transport", "accommodation",
    "budget", "emergency_contact", "retreat_plan",
)


def _missing(**overrides) -> list[tuple[str, str, str]]:
    fields = {
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "route": None,
    }
    fields.update(overrides)
    return [(f.key, f.label, f.status) for f in missing_plan_fields(**fields)]


def test_the_missing_list_names_exactly_the_fields_the_draft_does_not_hold():
    """AC1: the enumeration is the draft's own gaps, in the renderer's order.

    The three collected arrangements are supplied, so they are absent from the
    list; every remaining PlanDetails arrangement is on it as 待确认, and every
    route-fact dimension is on it as 待核验 because no basis was re-read. No
    third status exists, and no entry carries a value.
    """
    assert _missing() == [
        ("meeting_time", "集合时间", UNCONFIRMED),
        ("meeting_point", "集合地点", UNCONFIRMED),
        ("leader", "领队", UNCONFIRMED),
        ("transport", "交通", UNCONFIRMED),
        ("accommodation", "住宿", UNCONFIRMED),
        ("budget", "费用预算", UNCONFIRMED),
        ("emergency_contact", "应急联络", UNCONFIRMED),
        ("retreat_plan", "撤退方案", UNCONFIRMED),
        ("distance_km", "距离", UNVERIFIED),
        ("ascent_m", "爬升", UNVERIFIED),
        ("duration_hours", "耗时", UNVERIFIED),
        ("weather", "天气", UNVERIFIED),
        ("map", "地图", UNVERIFIED),
        ("trail_status", "开放状态", UNVERIFIED),
        ("risk", "风险资料", UNVERIFIED),
    ]


def test_an_unsubmitted_arrangement_is_listed_and_a_supplied_one_is_not():
    """AC1 honestly: the list tracks the stored value, including an empty one."""
    listed = {key for key, _label, _status in _missing(activity_date="")}
    assert "activity_date" in listed, "an empty 活动日期 is a missing field, not a blank"

    with_basis = RouteBasis(
        comparison_id="run:1", candidate_id="demo-a", candidate_name=_DESTINATION,
        role="recommended", data_mode="offline",
        facts=(RouteFactLine(field="distance_km", label="距离", display_value="8.2",
                             source_id="s1", source_field="distance_km",
                             source_tool="hiking_domain", observed_at="2026-11-01T00:00:00Z"),),
    )
    keys = {key for key, _label, _status in _missing(route=with_basis)}
    assert "distance_km" not in keys, "the gate certified it on this turn"
    assert "weather" in keys, "and it did not certify this one"


def test_the_arrangement_enumeration_is_the_plan_arrangement_record():
    """AC1/AC4 against a drift: the list has no private vocabulary of its own.

    :data:`ARRANGEMENT_FIELDS` is the same tuple :func:`render_minimal_draft`
    draws its lines from, and every key but the draft's own
    ``destination_or_route`` is a field of
    :class:`route_compare.models.PlanDetails` — the arrangement record the seam
    contract names. So a field cannot be added to the checklist without adding
    it to the record, and cannot exist in the record without being checked.
    """
    keys = {key for key, _label in ARRANGEMENT_FIELDS}
    assert keys - {"destination_or_route"} <= set(PlanDetails.model_fields)
    assert set(_MUST_STAY_BLANK) | {"activity_date", "participants"} <= keys
    # ``budget`` is the arrangement list, not a computed total.
    assert "budget" in keys and "budget_total" not in keys


def test_the_first_summary_shows_the_missing_list_and_the_payload_agrees(plan_repo):
    """AC1 through the seam: one enumeration, shown as text and as data.

    The payload list and the rendered 缺失清单 are the same call, so a client
    that shows the checklist and a client that shows the bubble cannot present
    different gaps — and each entry's label really is in the text beside its
    marker.
    """
    result = _drive(plan_repo, _submit("req-submit"))
    payload = _draft_payload(result["streams"][0])
    summary = payload["summary"]

    assert [
        (f["key"], f["label"], f["status"]) for f in payload["missing_fields"]
    ] == _missing()
    for entry in payload["missing_fields"]:
        assert f"{entry['label']}：" in summary, entry
        assert entry["status"] in summary, entry
    # The supplied arrangements are named as supplied, not as gaps.
    assert _DESTINATION in summary and _DATE in summary and f"{_PEOPLE} 人" in summary
    assert result["kernel_calls"] == []


# ---- AC2: the two answers, driven through the flow --------------------------


def _reviewed(plan_repo) -> dict:
    """Submit the arrangements, then answer the list with 直接生成草案.

    Ticket 06 requires the core arrangements filled for the click to reach the
    summary (a core-missing draft bounces to 补充流程), so the submit carries them.
    """
    return _drive(plan_repo,
                  _submit("req-submit", arrangements=_supplement()),
                  _req("req-review", message="直接生成草案", intent="plan_first_review"))


def test_generating_the_draft_directly_produces_the_first_summary(plan_repo):
    """AC2 + AC3: 直接生成草案 lands on the stage whose text IS the first summary."""
    result = _reviewed(plan_repo)
    events = result["streams"][1]
    payload = _draft_payload(events)

    assert payload["stage"] == "awaiting_first_review"
    assert payload["intent"] == "plan_first_review"
    assert payload["summary"] == _texts(events)
    assert payload["plan_type"] == "generic_draft"
    assert "第一次摘要确认" in payload["summary"]
    assert "由你提供的安排" in payload["summary"]
    assert "尚未填写的字段" in payload["summary"]
    assert _DESTINATION in payload["summary"] and _DATE in payload["summary"]
    assert result["kernel_calls"] == []


def test_continuing_to_supplement_reopens_the_form_and_keeps_the_draft(plan_repo):
    """AC2: 继续补充 is the other answer — the form reopens, nothing is lost.

    The stored arrangements survive the trip (an upsert keeps what was
    submitted), so "keep collecting" never means "start over", and no cancel is
    involved: the park/cancel split from ticket 02 still holds.
    """
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_supplement()),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"),
                    _req("req-more", message="继续补充", intent="plan_more_input"))
    events = result["streams"][2]
    payload = _draft_payload(events)

    assert payload["stage"] == "awaiting_input"
    # The stored arrangements survive the trip and come back from the row on the
    # reopened form (ticket 06 AC5) — so the draft now carries them too.
    assert payload["draft"] == {
        "destination_or_route": _DESTINATION, "activity_date": _DATE, "participants": _PEOPLE,
        "arrangements": _supplement(),
    }
    stored = result["repo"].get_plan_draft(CONVERSATION)
    assert (stored.stage, stored.destination_or_route) == ("awaiting_input", _DESTINATION)
    assert result["kernel_calls"] == []


def test_a_review_decision_cannot_be_made_before_the_draft_exists(plan_repo):
    """The refusals are flow-level too, and store nothing (tickets 01-04's shape)."""
    result = _drive(plan_repo,
                    _req("req-open", intent="plan_create"),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"))
    events = result["streams"][1]

    assert [e["reason"] for e in _errors(events)] == ["first_review_without_draft"]
    assert _draft_events(events) == []
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "awaiting_input"


# ---- AC3: the first summary confirms the user's own arrangements only -------


def _basis_of(plan_repo, comparison_id: str, candidate_id: str):
    """Re-read the basis through the leaf, the way the flow does (not a copy)."""
    session = RouteSession(settings=_settings(), run_id="reader", audit_sink=None,
                           runs_repo=plan_repo, conversation_id=CONVERSATION)
    return session.select_candidate(comparison_id, candidate_id)


def _seed_comparison(repo: RunRepository, conversation_id: str,
                     *, scenario: str = "complete",
                     request_id: str = "req-compare") -> dict:
    """A gate-passed comparison persisted as its own run's ``tool_result``.

    Copied from ticket 03's harness: this is the only shape
    :meth:`RouteSession.load_result` can find, so the review turn re-reads real
    stored evidence rather than a fixture dict.
    """
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="route_plan",
        user_message="比较演示路线",
    )
    session = RouteSession(settings=_settings(), run_id=run_id, audit_sink=None)
    result = asyncio.run(session.compare([
        {"name": _DEMO_ROUTE, "city": "演示城市", "origin": "起点", "destination": "终点"},
    ]))
    repo.persist_event(
        run_id=run_id, sequence=1, event_type="tool_result",
        request_id=request_id, conversation_id=conversation_id,
        payload={"route_comparison": result},
    )
    return result


def test_the_first_summary_keeps_showing_the_route_basis_it_points_at(plan_repo):
    """notes-04's landmine, as an AC3 test: a review stage must re-read the basis.

    If ``awaiting_first_review`` is missing from the flow's re-read condition,
    the summary the user confirms prints 待核验 for facts the draft actually
    holds, and the missing-field list names dimensions the gate certified — a
    first confirmation built on facts the server never denied having.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    candidate = comparison["candidates"][0]["candidate"]
    result = _drive(plan_repo,
                    _submit("req-submit"),
                    _req("req-select", message="选用该路线", intent="plan_select_route",
                         route_selection={"comparison_id": comparison["comparison_id"],
                                          "candidate_id": candidate["id"]}),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"))
    events = result["streams"][1]
    payload = _draft_payload(events)

    outcome = _basis_of(plan_repo, comparison["comparison_id"], candidate["id"])
    assert outcome.status == "ok" and outcome.basis is not None
    certified = {line.field for line in outcome.basis.facts}
    listed = {entry["key"] for entry in payload["missing_fields"]}

    assert certified, "the offline complete fixture certifies something"
    assert not (certified & listed), "a certified fact is never a missing field"
    assert {field for field, _l, _u in FACT_DIMENSIONS} - certified <= listed
    assert "**路线核验事实" in payload["summary"], "the basis stays in the summary"
    assert payload["plan_type"] == "route_plan"
    for entry in payload["missing_fields"]:
        assert f"{entry['label']}：" in payload["summary"], entry
    for field, label, _unit in FACT_DIMENSIONS:
        if field in certified:
            assert f"{label}：{UNVERIFIED}" not in payload["summary"], label


def test_the_first_summary_confirms_nothing_the_user_did_not_supply(plan_repo):
    """AC3's split, on the streamed text: the confirmed half is the user's own."""
    result = _reviewed(plan_repo)
    summary = _draft_payload(result["streams"][1])["summary"]
    gaps = _draft_payload(result["streams"][1])["missing_fields"]

    supplied, _, rest = summary.partition("**尚未填写的字段")
    assert "由你提供的安排" in supplied
    assert _DESTINATION in supplied and _DATE in supplied and f"{_PEOPLE} 人" in supplied
    assert UNCONFIRMED not in supplied and UNVERIFIED not in supplied
    for gap in gaps:
        assert gap["label"] in rest, gap


def test_confirming_the_first_summary_is_recorded_with_its_scope(plan_repo):
    """AC3 + AC5: the confirmation says what it covered, and still names gaps.

    ``first_reviewed`` is the durable record of "the user confirmed the
    arrangements they supplied". It is not a completeness claim, not a route
    verification and not the final review, so the render after it keeps naming
    every gap — deleting them from the text would read as an approval of them.
    """
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_supplement()),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"),
                    _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"))
    events = result["streams"][2]
    payload = _draft_payload(events)

    assert payload["stage"] == "first_reviewed"
    assert payload["intent"] == "plan_confirm_arrangements"
    assert payload["summary"] == _texts(events)
    assert "已确认" in payload["summary"] and _DESTINATION in payload["summary"]
    assert "不代表" in payload["summary"], "the scope of the confirmation is stated"
    for key, label, _status in _missing():
        assert label in payload["summary"], key
    assert payload["missing_fields"]
    assert result["kernel_calls"] == []


def test_the_first_summary_cannot_be_confirmed_that_was_never_shown(plan_repo):
    """AC5 as a guard: no recorded approval without a summary on screen."""
    result = _drive(plan_repo,
                    _submit("req-submit"),
                    _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"))
    events = result["streams"][1]

    assert [e["reason"] for e in _errors(events)] == ["arrangements_not_pending"]
    assert _draft_events(events) == []
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "minimal_draft"


# ---- AC4: nothing is ever filled in for the user ----------------------------

#: A date, a money amount or a phone number in the summary would be an invented
#: arrangement. ``docs/plan-pdf-format.md``: 未知安排留空显示"待确认"，预算只合计
#: 用户提供的条目，不根据总耗时推算时刻.
_INVENTED_PATTERNS = (
    (r"\d{1,3}(?:\.\d+)?\s*(?:元|¥|￥)|[¥￥]\s*\d", "a cost"),
    (r"(?:电话|联系|手机)[^\n]{0,12}\d{3,}", "a phone number"),
    (r"(?:撤退|下撤|撤离)[^\n]{0,20}\d", "a timed retreat"),
    (r"\d{1,2}:\d{2}", "a clock time"),
)


def _review_render(**fields) -> str:
    base = {
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "route": None,
        "review": "pending",
    }
    base.update(fields)
    return render_minimal_draft(**base)


def test_the_first_summary_invents_no_date_no_cost_no_phone_and_no_retreat_point():
    """AC4 at the renderer: an unfilled arrangement can only render as absent.

    The scan covers the whole first summary, and the only date in it is the one
    the user supplied. A retreat plan, a budget line or an emergency contact
    appearing here would have had to come from this module, which is exactly
    what the spec forbids.
    """
    rendered = _review_render()

    gaps = _missing()
    for key, label, _status in gaps:
        assert f"{label}：" in rendered, key
    for line in rendered.splitlines():
        if any(f"{label}：" in line for _key, label, _status in gaps):
            assert line.rstrip().endswith(UNCONFIRMED) or UNVERIFIED in line, line

    assert re.findall(r"\d{4}-\d{2}-\d{2}", rendered) == [_DATE]
    for pattern, what in _INVENTED_PATTERNS:
        found = [m.group(0) for m in re.finditer(pattern, rendered)]
        assert not found, f"{what} appeared without the user supplying it: {found}"


def test_an_empty_draft_has_nothing_to_confirm_and_nothing_filled():
    """AC4's other half: with no stored value, every arrangement is absent.

    Even the three collected fields render 待确认 when the row is empty, so the
    review copy can never be reached carrying a value the user did not give.
    """
    rendered = _review_render(destination_or_route="", activity_date="", participants=None)

    assert "尚无填写" in rendered
    for _key, label in ARRANGEMENT_FIELDS:
        assert f"{label}：{UNCONFIRMED}" in rendered, label
    assert _DATE not in rendered
    assert UNVERIFIED in rendered


def test_the_renderer_has_no_parameter_a_value_could_be_smuggled_through():
    """AC4 inexpressible: the draft renderer takes the stored fields and a basis.

    There is no argument for a date, a cost, a phone number, a retreat point or
    a derived segment time, and no ``plan_type`` argument either (ticket 04's
    rule, still load-bearing for the review render): the only way a value can
    reach the summary is by being in the draft row or certified by the gate.
    """
    params = set(inspect.signature(render_minimal_draft).parameters)
    # Ticket 06 adds ``arrangements`` — but it is a mapping read only for the
    # enumerated ARRANGEMENT_FIELDS keys (see ``_arrangement_values``), so no
    # smuggled route fact / derived time / computed cost can enter through it.
    assert params == {
        "destination_or_route", "activity_date", "participants", "route", "review",
        "arrangements",
    }
    assert not (params & set(_MUST_STAY_BLANK))


# ---- AC5: the confirmation is recoverable and auditable ---------------------


def _confirm_flow(plan_repo) -> dict:
    """Submit → 直接生成草案 → 确认安排, the three turns one user actually does.

    Ticket 06: the submit carries the core arrangements so the review reaches
    ``awaiting_first_review`` (a core-missing draft bounces the click back to the
    form) — the confirmation is then a real confirmation of a shown summary.
    """
    return _drive(plan_repo,
                  _submit("req-submit", arrangements=_supplement()),
                  _req("req-review", message="直接生成草案", intent="plan_first_review"),
                  _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"))


def test_the_confirmation_outlives_its_run_and_points_at_the_run_that_recorded_it(
    plan_repo
):
    """AC5: the durable half — on the conversation axis, run-scoped by reference.

    ``V2PersistedRun`` is nulled at every terminal (ticket 02's finding), so the
    audit answer to "did the user confirm, when, and on which draft?" has to come
    from the row and the events, not from a live run object.
    """
    result = _confirm_flow(plan_repo)
    events = result["streams"][2]
    run_id = events[-1].run_id
    draft_id = _draft_payload(events)["draft_id"]

    stored = plan_repo.get_plan_draft(CONVERSATION)
    assert stored.stage == "first_reviewed"
    assert stored.draft_id == draft_id
    assert stored.updated_by_run_id == run_id
    # The run itself is terminal: the record survived it.
    assert plan_repo.get_run(run_id).is_terminal
    # And the reference stayed with the draft, not with the run.
    assert plan_repo.get_plan_draft(CONVERSATION).draft_id == draft_id


def test_the_confirmation_is_auditable_in_the_stored_events_and_the_history(plan_repo):
    """AC5: the audit trail is ``run_events`` plus the conversation history.

    Both the machine-readable state (stage, intent, draft id, the checklist the
    user was deciding from, the summary text) and the text the user read are
    stored, so a reviewer can reconstruct which summary was confirmed without a
    model and without the live client.
    """
    result = _confirm_flow(plan_repo)
    confirm_run = result["streams"][2][-1].run_id
    payload = _draft_payload(result["streams"][2])

    stored = [
        e for e in plan_repo.get_events_since(run_id=confirm_run, from_seq=0)
        if e["type"] == "plan.draft_state"
    ]
    assert len(stored) == 1
    event = stored[0]["payload"]
    assert event["stage"] == "first_reviewed"
    assert event["intent"] == "plan_confirm_arrangements"
    assert event["draft_id"] == payload["draft_id"]
    assert event["conversation_id"] == CONVERSATION
    assert event["missing_fields"] == payload["missing_fields"]
    assert event["summary"] == payload["summary"]

    history = [
        m.content for m in plan_repo.checkpoints.get_messages(conversation_id=CONVERSATION)
        if m.role == "assistant"
    ]
    assert payload["summary"] in history, "the user's own confirmed text is in the history"


def test_a_review_turn_is_inert_and_never_an_approval_pause(plan_repo):
    """AC5 + the standing ruling: confirming is a conversational turn.

    No kernel call, no research, no comparison, no PDF, and above all no
    ``approval.required``: the pause machinery has a 300 s TTL and holds a
    sandbox lease, so a user reading a summary for six minutes would expire an
    approval that was never the right tool for a plan confirmation.
    """
    result = _confirm_flow(plan_repo)
    review, confirm = result["streams"][1], result["streams"][2]

    for events in (review, confirm):
        assert set(e.type for e in events) <= PLAN_FLOW_EVENT_TYPES, [e.type for e in events]
        assert "approval.required" not in [e.type for e in events]
    assert result["kernel_calls"] == []
    assert [e for e in review + confirm if e.type == "tool_result"] == []
    for events in (review, confirm):
        assert plan_repo.get_sandbox_lease(events[-1].run_id) is None


@pytest.mark.parametrize("step", [2, 3])
def test_cancelling_still_works_from_either_review_stage(plan_repo, step):
    """The park/cancel split holds on the new stages (ticket 02's contract).

    ``awaiting_first_review`` and ``first_reviewed`` are both open plan states,
    so 取消 still reaches the server from either and still clears everything —
    and neither cancel started a comparison or a research turn on the way out.
    """
    turns = [_submit("req-submit"),
             _req("req-review", message="直接生成草案", intent="plan_first_review")]
    if step == 3:
        turns.append(
            _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"))
    turns.append(_req("req-cancel", message="取消", intent="plan_cancel"))

    result = _drive(plan_repo, *turns)
    events = result["streams"][-1]

    assert _draft_payload(events)["stage"] == "cancelled"
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "cancelled"
    assert result["kernel_calls"] == []
    assert [e for e in events if e.type == "tool_result"] == []


# ---- The HTTP boundary ------------------------------------------------------

import api.chat_v2 as chat_api  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_v2 import kernel  # noqa: E402
from agent_v2.memory_service import MemoryService  # noqa: E402
from agent_v2.repositories.memory import MemoryStore  # noqa: E402
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router  # noqa: E402

HTTP_CONVERSATION = "conv-plan-05-http"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """``POST /api/v2/chat/sse`` over ASGI with a fake kernel.

    The only harness that sees the wire half: ``ChatV2Request`` is
    ``extra="forbid"``, so a field the adapter does not declare is a 422 here
    and invisible to every flow-level test.
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
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")]


def _http_send(flow, request_id: str, **extra):
    body = {"message": "生成行程", "request_id": request_id,
            "conversation_id": HTTP_CONVERSATION, **extra}
    return _frames(_post(flow.app, flow.subject, body))


def _draft_frames(frames):
    return [f for f in frames if f["type"] == "plan.draft_state"]


def test_the_missing_list_and_the_review_stages_reach_the_client_over_http(flow):
    """AC1/AC2/AC5 on the wire, with no model in sight.

    A restore after a refresh reads exactly this payload shape, so the
    checklist and the review stage have to survive the HTTP boundary — and a
    confirmation turn must never fall through to the kernel.
    """
    _http_send(flow, "req-http-submit", intent="plan_create", plan_input={
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        # Ticket 06: the core arrangements arrive nested in plan_input, and the
        # review gate only reaches the summary once they are present.
        "arrangements": _supplement(),
    })
    frames = _http_send(flow, "req-http-review", message="直接生成草案",
                        intent="plan_first_review")
    payload = _draft_frames(frames)[-1]["payload"]

    assert payload["stage"] == "awaiting_first_review"
    # The three core arrangements were supplied, so they LEAVE the checklist;
    # the still-missing non-core arrangements lead it (ticket 06 AC3).
    assert [entry["key"] for entry in payload["missing_fields"]][:3] == [
        "leader", "transport", "accommodation",
    ]
    assert payload["summary"] == "".join(
        f["payload"]["content"] for f in frames if f["type"] == "text")

    confirmed = _http_send(flow, "req-http-confirm", message="确认安排",
                           intent="plan_confirm_arrangements")
    assert _draft_frames(confirmed)[-1]["payload"]["stage"] == "first_reviewed"
    assert flow.model_calls == []


@pytest.mark.parametrize("intent", [
    "plan_first_review", "plan_more_input", "plan_confirm_arrangements",
])
def test_a_review_decision_cannot_carry_an_arrangement_over_http(flow, intent):
    """AC4 at the boundary: the decision click has nowhere to put a value.

    A draft field next to a review decision is refused by the state machine
    with a stable reason and no write, and a field the request model never
    declared (``meeting_time``, ``budget``, ``emergency_contact``) is a 422
    before a run exists — so an "auto-filled" arrangement cannot arrive from
    the client either.
    """
    frames = _http_send(flow, f"req-smuggle-{intent}", intent=intent, plan_input={
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
    })
    assert frames[-1]["type"] == "error", frames[-1]
    assert frames[-1]["payload"]["reason"] == "plan_input_not_allowed"
    assert _draft_frames(frames) == []

    smuggled = _post(flow.app, flow.subject, {
        "message": "确认安排", "request_id": f"req-extra-{intent}",
        "conversation_id": HTTP_CONVERSATION, "intent": intent,
        "meeting_time": "08:00", "budget": 300, "emergency_contact": "13800000000",
    })
    assert smuggled.status_code == 422
    assert flow.repo.resolve_run_id(f"req-extra-{intent}") is None


# ---- Frontend wiring, pinned from the sources (no test runner exists) ------


def test_the_missing_field_list_travels_on_the_wire_and_the_mirror():
    """Landmine (notes-02 #1): a new draft payload field has FOUR frontend homes.

    ``V2PlanDraftState`` (compile time), ``readPlanDraftState`` (the single
    validator for BOTH the SSE payload and the localStorage mirror),
    ``isSamePlanDraft`` (which decides whether a restore is a no-op) and the
    panel that shows it. Miss the third and a checklist that changed looks like
    "no change", so the restore hands back the OLD object and the gaps the user
    is deciding from silently stale after a refresh.
    """
    wire = _ts("src/api/chatV2.ts")
    reducer = _ts("src/api/v2Reducer.ts")
    store = _ts("src/api/planDraftStore.ts")
    page = _ts("src/pages/SuperAgent.tsx")

    assert "export interface V2PlanMissingField" in wire
    assert "missing_fields: V2PlanMissingField[]" in wire, "required, never optional"
    assert "missing_fields: readMissingPlanFields(payload['missing_fields'])" in reducer
    assert "isSameMissingFields(a.missing_fields, b.missing_fields)" in store
    assert "planDraft.missing_fields" in page

    # The stages are on the wire in all three vocabularies, and the two review
    # stages are reopenable — while the parked research stage stays out, so no
    # second top-16 panel stacks on this one (notes-04 landmine 5).
    assert "'awaiting_first_review'" in wire and "'first_reviewed'" in wire
    open_block = store[store.index("OPEN_PLAN_DRAFT_STAGES"):store.index("isDurablePlanDraftStage")]
    assert "'awaiting_first_review'" in open_block and "'first_reviewed'" in open_block
    assert "'awaiting_generic_choice'" not in open_block
    assert "'awaiting_confirmation'" not in open_block


def test_the_page_offers_both_answers_and_a_confirmation_as_plain_turns():
    """AC1/AC2/AC3 in the UI, pinned by reading the handlers.

    One panel carries the checklist and the decisions (继续补充 / 直接生成草案 /
    确认安排), each send is the whitelisted intent ALONE — no fields, no draft
    type, no route reference, no profile override — and dismissal still only
    parks: closing the dialog must never send ``plan_cancel`` (ticket 02).
    """
    page = _ts("src/pages/SuperAgent.tsx")

    review = _ts_callback(page, "handleGenerateDraftNow")
    more = _ts_callback(page, "handleContinueSupplementing")
    confirm = _ts_callback(page, "handleConfirmArrangements")
    assert "intent: 'plan_first_review'" in review
    assert "intent: 'plan_more_input'" in more
    assert "intent: 'plan_confirm_arrangements'" in confirm
    for handler in (review, more, confirm):
        for smuggled in ("planInput", "planType", "routeSelection", "profileId", "approval"):
            assert smuggled not in handler

    # Rendered, not merely defined.
    for name in ("handleGenerateDraftNow", "handleContinueSupplementing",
                 "handleConfirmArrangements"):
        assert f"onClick={{{name}}}" in page, name
    assert "直接生成草案" in page and "继续补充" in page and "确认我提供的安排" in page
    assert "缺失字段：" in page

    # Still parks on dismiss, still one cancel exit.
    assert "plan_cancel" not in _ts_callback(page, "handlePlanDialogDismiss")
    assert "plan_cancel" in _ts_callback(page, "handlePlanDialogCancel")

    labels = page[page.index("const PLAN_INTENT_LABELS"):page.index("const PLAN_TYPE_LABELS")]
    assert "plan_first_review: '直接生成草案'" in labels
