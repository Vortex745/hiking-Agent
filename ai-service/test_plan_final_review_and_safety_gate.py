"""Ticket 07 — 最终计划审阅与安全门.

Seams under test (the ones notes-04/05/06 name for this ticket):

* :func:`agent_v2.plan_workflow.evaluate_plan_intent` — still the ONLY
  transition decider: 「查看最终摘要」 and 「确认最终摘要」 are two whitelisted
  intents with a source-stage guard, so a confirmation can never precede the
  summary it confirms, and a dangerous state can never advance;
* :func:`agent_v2.plan_workflow.draft_summary` + :func:`render_minimal_draft`
  — the SECOND summary is the same renderer in a ``review=`` mode, so 路线核验结果,
  用户安排, 缺失项 and 计划类型 cannot disagree with the draft they summarise;
* :func:`route_compare.service.assess_candidate` / :func:`render_report` — the
  gate owns report eligibility: ticket 07 reads its answer, it never restates it
  (``PlanCompleteness`` deliberately has no ``report_allowed``);
* :meth:`agent_v2.route_tools.RouteSession.research_destination` — the 危险封闭
  split lives in the LEAF's ``status``/``reason``, not in the flow;
* :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` in-process over
  real temp SQLite — the only writer of the draft row and the only emitter of
  ``plan.draft_state`` (which carries the safety block reason to the UI *and* into
  the event record, AC5);
* ``POST /api/v2/chat/sse`` and the frontend sources, pinned the way tickets
  01-06 pinned the draft vocabulary (there is no frontend test runner).

Acceptance criteria:

* AC1 第二次摘要包含路线核验结果、用户安排、缺失项和计划类型.
* AC2 no_go、严重天气、路线关闭和证据门失败时阻止正式路线计划.
* AC3 普通安排缺失时允许生成通用草案并突出待确认项.
* AC4 用户必须明确确认后才能进入 PDF 审批.
* AC5 安全阻止原因可以在前端和事件记录中解释.
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
    SAFETY_MESSAGE,
    SUPPLEMENT_ARRANGEMENT_KEYS,
    PlanBlockReason,
    PlanDraftInput,
    PlanDraftStage,
    PlanSafety,
    PlanSafetyReason,
    RecordedCompleteness,
    RouteBasis,
    UNCONFIRMED,
    UNVERIFIED,
    draft_summary,
    evaluate_plan_completeness,
    evaluate_plan_intent,
    pdf_approval_block_reason,
    render_minimal_draft,
    safety_lines,
)
from agent_v2.route_tools import RouteSession
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

CONVERSATION = "conv-plan-07"
SUBJECT_ID = "sub-plan-07"

#: Ticket 07 adds no event type: the second summary and its confirmation are the
#: same plan stream, never the tool-approval pause (300 s TTL + sandbox lease) and
#: never a PDF (that is ticket 09's, through the artifact lifecycle).
PLAN_FLOW_EVENT_TYPES = {
    "run_started",
    "plan.draft_state",
    "text",
    "error",
    "done",
    "cancelled",
    "tool_result",
}

_DEMO_ROUTE = "演示林间环线"
_DESTINATION = "四姑娘山长坪沟"
_DATE = "2026-11-08"
_PEOPLE = 12

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _ts(path: str) -> str:
    return (FRONTEND / path).read_text(encoding="utf-8")


def _ts_union_members(source: str, type_name: str) -> set[str]:
    start = source.index(f"export type {type_name} =")
    block = re.sub(r"/\*.*?\*/", "", source[start:], flags=re.S)
    block = re.split(r"\nexport ", block, maxsplit=1)[0]
    return set(re.findall(r"'([^']+)'", block))


def _ts_string_set_literal(source: str, const_name: str) -> set[str]:
    block = re.search(rf"{const_name}[^=]*=\s*new Set\(\[(.*?)\]\)", source, re.S)
    assert block, f"{const_name} not found in the frontend sources"
    return set(re.findall(r"'([^']+)'", block.group(1)))


def _ts_callback(source: str, name: str) -> str:
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


def _drive(repo: RunRepository, *requests: ChatFlowRequest,
           scenario: str = "complete") -> dict[str, Any]:
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orchestrator = RunOrchestrator(
        settings=_settings(scenario), runs_repo=repo, memory=_FakeMemory(),
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
    if arrangements is not None:
        payload["arrangements"] = arrangements
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate(payload),
    )


#: The three arrangements ticket 06 marks core. Without them the fail-closed gate
#: bounces 直接生成草案 to 补充流程 (notes-06 landmine 2), so every flow test here
#: that clicks through to a summary submits them in a valid format.
_CORE = {
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
}


def _supplement(**overrides: str) -> dict[str, str]:
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


def _verdict(**kwargs) -> Any:
    """A rule verdict over a draft that clears the core gate."""
    return evaluate_plan_completeness(
        destination_or_route=kwargs.get("destination_or_route", _DESTINATION),
        activity_date=kwargs.get("activity_date", _DATE),
        participants=kwargs.get("participants", _PEOPLE),
        arrangements=kwargs.get("arrangements", _supplement()),
        route=kwargs.get("route"),
    )


def _recorded(**kwargs) -> RecordedCompleteness:
    """The stored verdict row, as the domain sees it."""
    return RecordedCompleteness(
        draft_id=kwargs.get("draft_id", "draft-07"),
        verdict=kwargs.get("verdict", "confirmation_required"),
        reasons=kwargs.get("reasons", ()),
        checked_by_run_id=kwargs.get("checked_by_run_id", "run-07"),
    )


def _evaluate(intent, stage=None, **kwargs):
    return evaluate_plan_intent(
        intent=intent, current_stage=stage, plan_input=kwargs.get("plan_input"),
        conversation_id=kwargs.get("conversation_id", "conv-07"),
        route_selection=kwargs.get("route_selection"), plan_type=kwargs.get("plan_type"),
        completeness=kwargs.get("completeness"),
        recorded_completeness=kwargs.get("recorded_completeness"),
    )


# ---- AC4: the second summary and its confirmation are state-machine moves ---


def test_the_second_summary_stages_are_the_pre_assigned_vocabulary():
    """AC1/AC4: three members, the exact names ``seam-contract.md`` assigned to 07.

    Adding a stage means six places at once (enum, store, wire union, wire set,
    stage label, reopenable set); the cross-checks in ``test_plan_draft_restore.py``
    hold the other five, this one pins the enum itself against the contract so a
    rename cannot quietly become a second vocabulary.
    """
    assert {stage.value for stage in PlanDraftStage} == {
        "awaiting_confirmation",
        "awaiting_input",
        "minimal_draft",
        "awaiting_generic_choice",
        "awaiting_first_review",
        "first_reviewed",
        "awaiting_final_review",
        "final_reviewed",
        "blocked_safety",
        "cancelled",
    }


def test_the_second_summary_is_requested_and_confirmed_as_two_intents():
    """AC1/AC4: one 「show me the final summary」 intent, one 「confirm」 intent."""
    assert PLAN_INTENTS >= {"plan_final_review", "plan_confirm_final"}
    # Both are conversational turns, so neither may carry a draft field, a route
    # reference or a plan type (tickets 03-05's rule, unchanged).
    assert "plan_final_review" in PLAN_INTENTS and "plan_confirm_final" in PLAN_INTENTS


def test_the_second_summary_is_asked_for_only_after_the_first_is_confirmed():
    """AC4: ``first_reviewed`` is the precondition — and only that one."""
    transition = _evaluate("plan_final_review", PlanDraftStage.FIRST_REVIEWED,
                           completeness=_verdict(), recorded_completeness=_recorded())

    assert transition.status == "ok"
    assert transition.stage is PlanDraftStage.AWAITING_FINAL_REVIEW
    assert transition.completeness is not None


@pytest.mark.parametrize("stage,reason", [
    (None, "first_review_not_confirmed"),
    (PlanDraftStage.AWAITING_INPUT, "first_review_not_confirmed"),
    (PlanDraftStage.MINIMAL_DRAFT, "first_review_not_confirmed"),
    (PlanDraftStage.AWAITING_GENERIC_CHOICE, "first_review_not_confirmed"),
    (PlanDraftStage.AWAITING_FIRST_REVIEW, "first_review_not_confirmed"),
    (PlanDraftStage.AWAITING_FINAL_REVIEW, "first_review_not_confirmed"),
    (PlanDraftStage.FINAL_REVIEWED, "first_review_not_confirmed"),
    (PlanDraftStage.CANCELLED, "first_review_not_confirmed"),
    (PlanDraftStage.BLOCKED_SAFETY, "safety_blocked"),
])
def test_a_second_summary_cannot_be_requested_before_the_first_was_confirmed(
    stage, reason
):
    """The source-stage guard, enumerated (05's discipline for the same rule)."""
    transition = _evaluate("plan_final_review", stage,
                           completeness=_verdict(), recorded_completeness=_recorded())
    assert transition.status == "blocked"
    assert transition.reason == reason
    assert reason in BLOCK_MESSAGE, "a stable reason must have user-facing copy"


def test_a_second_summary_without_a_recorded_verdict_fails_closed():
    """AC4/AC5: the rules' answer must be ON RECORD before a review or a PDF.

    A flow that forgot to run the rules — or a conversation restored from a build
    that has none — must not honour the click: a missing rule input is never a
    passing one (ticket 06's fail-closed rule, extended to the second summary).
    """
    no_recomputed = _evaluate("plan_final_review", PlanDraftStage.FIRST_REVIEWED,
                              completeness=None, recorded_completeness=_recorded())
    no_stored_row = _evaluate("plan_final_review", PlanDraftStage.FIRST_REVIEWED,
                              completeness=_verdict(), recorded_completeness=None)

    assert no_recomputed.reason == no_stored_row.reason == "final_review_without_verdict"
    assert no_recomputed.status == no_stored_row.status == "blocked"


def test_a_supplement_required_draft_bounces_the_second_summary_to_the_form():
    """AC3 vs AC2, kept apart: a missing ordinary arrangement is not a safety block.

    The bounce reuses ticket 06's mechanism (``awaiting_input`` + the stable reason
    codes) rather than inventing a second decider — and it is NOT
    ``blocked_safety``, because 「缺少集合时间」 is not a dangerous state.
    """
    lacking = _verdict(arrangements={**_supplement(), "meeting_time": ""})
    assert lacking.supplement_required

    transition = _evaluate("plan_final_review", PlanDraftStage.FIRST_REVIEWED,
                           completeness=lacking, recorded_completeness=_recorded())

    assert transition.status == "ok"
    assert transition.stage is PlanDraftStage.AWAITING_INPUT
    assert transition.stage is not PlanDraftStage.BLOCKED_SAFETY
    assert transition.completeness is not None


def test_the_second_summary_can_only_be_confirmed_while_it_is_pending():
    """AC4 enumerated: no recorded approval of a summary that was never shown."""
    allowed = _evaluate("plan_confirm_final", PlanDraftStage.AWAITING_FINAL_REVIEW,
                        completeness=_verdict(), recorded_completeness=_recorded())
    assert allowed.status == "ok"
    assert allowed.stage is PlanDraftStage.FINAL_REVIEWED

    for stage in (
        None, PlanDraftStage.AWAITING_INPUT, PlanDraftStage.MINIMAL_DRAFT,
        PlanDraftStage.AWAITING_GENERIC_CHOICE, PlanDraftStage.AWAITING_FIRST_REVIEW,
        PlanDraftStage.FIRST_REVIEWED, PlanDraftStage.FINAL_REVIEWED,
        PlanDraftStage.CANCELLED,
    ):
        refused = _evaluate("plan_confirm_final", stage,
                            completeness=_verdict(), recorded_completeness=_recorded())
        assert refused.status == "blocked", stage
        assert refused.reason == "final_review_not_pending", stage

    dangerous = _evaluate("plan_confirm_final", PlanDraftStage.BLOCKED_SAFETY,
                          completeness=_verdict(), recorded_completeness=_recorded())
    assert dangerous.reason == "safety_blocked"


def test_a_confirmation_recorded_at_final_reviewed_is_not_reopenable():
    """AC4: re-confirmation must go through a NEW summary, not a repeat click."""
    already = _evaluate("plan_confirm_final", PlanDraftStage.FINAL_REVIEWED,
                        completeness=_verdict(), recorded_completeness=_recorded())
    assert already.status == "blocked"

    again = _evaluate("plan_final_review", PlanDraftStage.FINAL_REVIEWED,
                      completeness=_verdict(), recorded_completeness=_recorded())
    assert again.status == "blocked"


def test_a_dangerous_state_cannot_reach_a_formal_route_plan():
    """AC2 in the state machine: 危险封闭 refuses every advancing intent.

    The dangerous stage is not 「资料不足」: the generic-draft escape ticket 04
    offers for missing material is deliberately NOT offered here, so a no-go
    destination cannot be laundered into a plan that advances. Only 取消 and a
    re-submitted form (a different destination) leave it.
    """
    generic = _evaluate("plan_generic_draft", PlanDraftStage.BLOCKED_SAFETY,
                        plan_type="generic_draft")
    first = _evaluate("plan_first_review", PlanDraftStage.BLOCKED_SAFETY,
                      completeness=_verdict(), recorded_completeness=_recorded())
    final = _evaluate("plan_final_review", PlanDraftStage.BLOCKED_SAFETY,
                      completeness=_verdict(), recorded_completeness=_recorded())
    confirm = _evaluate("plan_confirm_final", PlanDraftStage.BLOCKED_SAFETY,
                        completeness=_verdict(), recorded_completeness=_recorded())

    for transition in (generic, first, final, confirm):
        assert transition.status == "blocked", transition
        assert transition.reason == PlanBlockReason.SAFETY_BLOCKED

    # Cancellation still works from the hard stop (ticket 02's exit is never blocked).
    cancelled = _evaluate("plan_cancel", PlanDraftStage.BLOCKED_SAFETY)
    assert cancelled.status == "ok" and cancelled.stage is PlanDraftStage.CANCELLED
    # Re-entering the form is legal from every stage, and that is the user's only
    # other way out: a new destination is judged by the gate again.
    reentry = _evaluate("plan_create", PlanDraftStage.BLOCKED_SAFETY,
                        plan_input=PlanDraftInput.model_validate(
                            {"destination_or_route": "演示林间环线",
                             "activity_date": _DATE, "participants": _PEOPLE}))
    assert reentry.status == "ok"


# ---- AC4: PDF approval requires the explicit confirmation, and nothing less --


def test_pdf_approval_is_entered_only_from_a_final_confirmation():
    """AC4: 「进入 PDF 审批」 has exactly one door, and it is the user's click."""
    assert pdf_approval_block_reason(
        stage=PlanDraftStage.FINAL_REVIEWED, recorded_completeness=_recorded(), safety=None,
    ) is None

    for stage in (
        None, PlanDraftStage.AWAITING_INPUT, PlanDraftStage.MINIMAL_DRAFT,
        PlanDraftStage.AWAITING_GENERIC_CHOICE, PlanDraftStage.AWAITING_FIRST_REVIEW,
        PlanDraftStage.FIRST_REVIEWED, PlanDraftStage.AWAITING_FINAL_REVIEW,
        PlanDraftStage.CANCELLED,
    ):
        assert pdf_approval_block_reason(
            stage=stage, recorded_completeness=_recorded(), safety=None,
        ) == PlanBlockReason.FINAL_REVIEW_NOT_PENDING, stage

    assert pdf_approval_block_reason(
        stage=PlanDraftStage.FINAL_REVIEWED, recorded_completeness=None, safety=None,
    ) == PlanBlockReason.FINAL_REVIEW_WITHOUT_VERDICT

    danger = PlanSafety(
        hard_stop=True, codes=(PlanSafetyReason.SEVERE_WEATHER,), comparison_id="c",
        comparison_status="no_go", gate_reasons=(), required_action="",
    )
    assert pdf_approval_block_reason(
        stage=PlanDraftStage.FINAL_REVIEWED, recorded_completeness=_recorded(),
        safety=danger,
    ) == PlanSafetyReason.SEVERE_WEATHER


# ---- AC1: the second summary is the draft, re-rendered, with what it owes ----


def _second_summary(**kwargs) -> str:
    base = {
        "stage": PlanDraftStage.AWAITING_FINAL_REVIEW,
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "arrangements": _supplement(),
    }
    base.update(kwargs)
    return draft_summary(**base)


def _route_basis(**overrides) -> RouteBasis:
    """A gate-certified basis, shaped exactly as the comparison leaf builds one."""
    from agent_v2.plan_workflow import RouteFactLine

    facts = overrides.get("facts") or (
        RouteFactLine(
            field="distance_km", label="距离", display_value="8",
            source_id="demo-a-fixture", source_field="distance_km",
            source_tool="offline_fixture", observed_at="2026-11-01T00:00:00+00:00",
        ),
        RouteFactLine(
            field="weather", label="天气", display_value="晴",
            source_id="demo-a-fixture", source_field="weather",
            source_tool="offline_fixture", observed_at="2026-11-01T00:00:00+00:00",
        ),
    )
    return RouteBasis(
        comparison_id=overrides.get("comparison_id", "run-07:cmp-07"),
        candidate_id=overrides.get("candidate_id", "demo-a"),
        candidate_name=overrides.get("candidate_name", _DEMO_ROUTE),
        role="recommended", data_mode="offline", facts=facts,
    )


def test_the_second_summary_names_the_route_result_the_arrangements_the_gaps_and_the_type():
    """AC1, all four halves in one render — the same render the payload carries."""
    rendered = _second_summary(route=_route_basis(), completeness=_verdict())

    assert "最终摘要" in rendered
    assert "草案类型：route_plan" in rendered, "计划类型"
    assert _DESTINATION in rendered and _DATE in rendered and f"{_PEOPLE} 人" in rendered
    assert "**由你提供的安排" in rendered
    assert "**尚未填写的字段" in rendered
    assert "**路线核验事实" in rendered, "路线核验结果 comes from the re-read basis"
    assert "距离：8" in rendered
    assert "完整度规则结论" in rendered, "the rules' verdict is part of what is reviewed"
    assert "PDF" in rendered, "the render says what the confirmation is for"


def test_a_basis_less_second_summary_stays_a_generic_draft_and_highlights_the_gaps():
    """AC3: an ordinary missing arrangement yields a generic draft, not a plan."""
    rendered = _second_summary(completeness=_verdict())

    assert "草案类型：generic_draft" in rendered and "不是路线计划" in rendered
    assert "**路线核验事实" not in rendered
    for key, label in ARRANGEMENT_FIELDS:
        assert f"{label}：" in rendered, key
    assert UNVERIFIED in rendered, "uncertified facts stay 待核验 in the second summary"
    assert "待确认" in rendered


@pytest.mark.parametrize("mode", ["pending", "confirmed", "final_pending", "final_confirmed"])
def test_the_second_summary_is_a_mode_of_the_same_renderer(mode):
    """One renderer, four modes: the 待确认 / 待核验 split cannot drift.

    Ticket 07 was told to add a ``review=`` mode rather than a second render
    function, and this is the pin: the parameter set is still exactly the one
    ticket 05/06 pinned, so no value can be smuggled into the final summary
    either — the only thing a caller can change is WHICH scope of confirmation is
    being drawn.
    """
    rendered = render_minimal_draft(
        destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        arrangements=_supplement(), review=mode,
    )
    assert rendered.strip()
    assert mode.startswith("final") == ("最终摘要" in rendered)
    assert mode.startswith("final") == (mode in ("final_pending", "final_confirmed"))

    params = set(inspect.signature(render_minimal_draft).parameters)
    assert params == {
        "destination_or_route", "activity_date", "participants", "route", "review",
        "arrangements",
    }


_INVENTED_PATTERNS = (
    (r"\d{1,3}(?:\.\d+)?\s*(?:元|¥|￥)|[¥￥]\s*\d", "a cost"),
    (r"(?:电话|联系|手机)[^\n]{0,12}\d{3,}", "a phone number"),
    (r"(?:撤退|下撤|撤离)[^\n]{0,20}\d", "a timed retreat"),
    (r"\d{1,2}:\d{2}", "a clock time"),
)


def test_the_second_summary_invents_no_date_no_cost_no_phone_and_no_retreat_point():
    """AC4's no-invention rule holds for the FINAL summary too.

    The render here has NO 集合时间 in the record, so a single clock time anywhere
    in it would have been written by the system. (When the user supplies one it
    appears verbatim — that is the other test below.)
    """
    blank_time = _supplement(meeting_time="")
    rendered = _second_summary(
        arrangements=blank_time, completeness=_verdict(arrangements=blank_time)
    )

    assert re.findall(r"\d{4}-\d{2}-\d{2}", rendered) == [_DATE]
    for pattern, what in _INVENTED_PATTERNS:
        found = [m.group(0) for m in re.finditer(pattern, rendered)]
        assert not found, f"{what} appeared without the user supplying it: {found}"


def test_a_supplied_arrangement_is_repeated_verbatim_in_the_second_summary():
    """The mirror half of AC4: what the user gave is shown as given, not rewritten."""
    rendered = _second_summary(completeness=_verdict())

    assert "集合时间：08:30" in rendered
    assert "应急联络：13800000000" in rendered
    assert "集合地点：长坪沟景区北门停车场" in rendered


# ---- AC2/AC5: the safety decision is data, copy and an explanation ----------


def _danger(**overrides) -> PlanSafety:
    base = {
        "hard_stop": True,
        "codes": (PlanSafetyReason.NO_GO,),
        "comparison_id": "run-07:cmp-07",
        "comparison_status": "no_go",
        "gate_reasons": ("严重天气、封闭路线或高风险条件：禁止推荐",),
        "required_action": "取消该路线；等待官方解除风险后重新查询。",
    }
    base.update(overrides)
    return PlanSafety(**base)


def test_every_safety_block_reason_has_stable_copy_and_a_code():
    """AC5: the reason is a stable code PLUS readable copy, never prose alone."""
    assert {
        PlanSafetyReason.NO_GO, PlanSafetyReason.SEVERE_WEATHER,
        PlanSafetyReason.ROUTE_CLOSED, PlanSafetyReason.EVIDENCE_GATE_FAILED,
    } <= set(SAFETY_MESSAGE)
    for code, message in SAFETY_MESSAGE.items():
        assert code == code.strip() and message.strip(), code

    rendered = safety_lines(_danger())
    assert PlanSafetyReason.NO_GO in rendered
    assert SAFETY_MESSAGE[PlanSafetyReason.NO_GO] in rendered
    assert "严重天气、封闭路线或高风险条件：禁止推荐" in rendered, "the gate's own words"
    assert "等待官方解除风险" in rendered, "the gate's own required action"


def test_an_evidence_gate_failure_refuses_the_formal_plan_without_a_hard_stop():
    """AC2/AC3 apart: no basis means NO formal route plan, but the generic draft
    the user chose in ticket 04 stays available — and the refusal is explainable."""
    gate = _danger(hard_stop=False, codes=(PlanSafetyReason.EVIDENCE_GATE_FAILED,),
                   comparison_status="insufficient_evidence",
                   gate_reasons=("weather: 资料不足或证据不支持",))

    assert gate.hard_stop is False
    assert gate.blocks_formal_route_plan is True
    payload = gate.to_payload()
    assert payload["hard_stop"] is False
    assert payload["codes"] == [PlanSafetyReason.EVIDENCE_GATE_FAILED]
    assert payload["message"] == SAFETY_MESSAGE[PlanSafetyReason.EVIDENCE_GATE_FAILED]
    assert payload["gate_reasons"] == ["weather: 资料不足或证据不支持"]

    danger_payload = _danger().to_payload()
    assert danger_payload["hard_stop"] is True
    assert danger_payload["required_action"] == "取消该路线；等待官方解除风险后重新查询。"


def test_the_hard_stop_stage_renders_the_safety_reason_and_nothing_else():
    """AC5: the stage the draft lands on explains itself in the streamed text."""
    rendered = draft_summary(
        stage=PlanDraftStage.BLOCKED_SAFETY, destination_or_route=_DESTINATION,
        activity_date=_DATE, participants=_PEOPLE, safety=_danger(),
    )
    assert PlanSafetyReason.NO_GO in rendered and "正式路线计划" in rendered
    assert _DESTINATION not in rendered, "the hard stop is not a draft render"

    without = draft_summary(
        stage=PlanDraftStage.BLOCKED_SAFETY, destination_or_route=_DESTINATION,
        activity_date=_DATE, participants=_PEOPLE,
    )
    from agent_v2.plan_workflow import STAGE_MESSAGE
    assert STAGE_MESSAGE[PlanDraftStage.BLOCKED_SAFETY] == without

    review = _second_summary(safety=_danger(
        hard_stop=False, codes=(PlanSafetyReason.EVIDENCE_GATE_FAILED,),
        comparison_status="confirmation_required",
    ))
    assert PlanSafetyReason.EVIDENCE_GATE_FAILED in review


# ---- AC1/AC4: the two turns, driven through the flow -----------------------


def _seed_comparison(repo: RunRepository, conversation_id: str,
                     *, scenario: str = "complete",
                     request_id: str = "req-compare") -> dict:
    """A gate-passed comparison persisted as its own run's ``tool_result``.

    Copied from tickets 03-05's harness: this is the only shape
    :meth:`RouteSession.load_result` can find, so the final review re-reads real
    stored evidence rather than a fixture dict. The demo route name is deliberate
    (notes-05 landmine 7) — the offline fixtures only know it.
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


def _basis_of(plan_repo, comparison_id: str, candidate_id: str):
    session = RouteSession(settings=_settings(), run_id="reader", audit_sink=None,
                           runs_repo=plan_repo, conversation_id=CONVERSATION)
    return session.select_candidate(comparison_id, candidate_id)


def _turns_to_first_reviewed(*, after_submit=()) -> list[ChatFlowRequest]:
    """The turns a user actually does before a final review is even askable.

    ``after_submit`` is where a route selection belongs: it is only honoured on a
    ``minimal_draft``, i.e. before 直接生成草案.
    """
    return [
        _submit("req-submit", arrangements=_supplement()),
        *after_submit,
        _req("req-review", message="直接生成草案", intent="plan_first_review"),
        _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"),
    ]


def _turns_to_final_review(*, after_submit=()) -> list[ChatFlowRequest]:
    return [
        *_turns_to_first_reviewed(after_submit=after_submit),
        _req("req-final", message="查看最终摘要", intent="plan_final_review"),
    ]


def _turns_to_final_confirmed(*, after_submit=()) -> list[ChatFlowRequest]:
    return [
        *_turns_to_final_review(after_submit=after_submit),
        _req("req-confirm-final", message="确认最终摘要", intent="plan_confirm_final"),
    ]


def _to_first_reviewed(plan_repo, *, after_submit=()) -> dict:
    return _drive(plan_repo, *_turns_to_first_reviewed(after_submit=after_submit))


def _to_final_review(plan_repo, *, after_submit=()) -> dict:
    return _drive(plan_repo, *_turns_to_final_review(after_submit=after_submit))


def _to_final_confirmed(plan_repo, *, after_submit=()) -> dict:
    return _drive(plan_repo, *_turns_to_final_confirmed(after_submit=after_submit))


def test_the_second_summary_is_produced_by_its_own_turn(plan_repo):
    """AC1 through the seam: the turn lands on the stage whose text IS the summary."""
    result = _to_final_review(plan_repo)
    events = result["streams"][3]
    payload = _draft_payload(events)

    assert payload["stage"] == "awaiting_final_review"
    assert payload["intent"] == "plan_final_review"
    assert payload["summary"] == _texts(events)
    assert "最终摘要" in payload["summary"]
    assert payload["plan_type"] == "generic_draft"
    # AC1's four inputs, all from the server: route result, arrangements, gaps, type.
    assert payload["missing_fields"] and payload["draft"]["arrangements"]["meeting_time"] == "08:30"
    assert payload["rules"] is not None and payload["rules"]["verdict"]
    assert result["kernel_calls"] == []


def test_the_second_summary_keeps_showing_the_route_basis_it_points_at(plan_repo):
    """AC1's route half, and notes-06's documented silent bug, one stage later.

    Forget the two new stages in the flow's stored-reference re-read condition and
    the final summary prints 待核验 for facts the draft actually holds — a user
    confirming a plan built on gaps the server never denied having.
    """
    comparison = _seed_comparison(plan_repo, CONVERSATION)
    candidate = comparison["candidates"][0]["candidate"]
    result = _to_final_review(
        plan_repo,
        after_submit=(_req("req-select", message="选用该路线", intent="plan_select_route",
                           route_selection={"comparison_id": comparison["comparison_id"],
                                            "candidate_id": candidate["id"]}),),
    )
    events = result["streams"][-1]
    payload = _draft_payload(events)

    outcome = _basis_of(plan_repo, comparison["comparison_id"], candidate["id"])
    assert outcome.status == "ok" and outcome.basis is not None
    certified = {line.field for line in outcome.basis.facts}
    listed = {entry["key"] for entry in payload["missing_fields"]}

    assert certified and not (certified & listed)
    assert "**路线核验事实" in payload["summary"]
    assert payload["plan_type"] == "route_plan"
    assert payload["route_selection"] == {
        "comparison_id": comparison["comparison_id"], "candidate_id": candidate["id"]}
    for field, label, _unit in FACT_DIMENSIONS:
        if field in certified:
            assert f"{label}：{UNVERIFIED}" not in payload["summary"], label


def test_the_final_confirmation_is_recorded_and_names_what_it_covered(plan_repo):
    """AC4/AC5: the explicit confirmation is the durable record, gaps and all."""
    result = _to_final_confirmed(plan_repo)
    events = result["streams"][-1]
    payload = _draft_payload(events)

    assert payload["stage"] == "final_reviewed"
    assert payload["intent"] == "plan_confirm_final"
    assert payload["summary"] == _texts(events)
    assert "已明确确认本次最终摘要" in payload["summary"]
    assert "PDF 审批" in payload["summary"]
    assert "不代表" in payload["summary"], "the scope of the confirmation is stated"
    assert payload["missing_fields"], "confirming does not fill a single gap"
    assert result["kernel_calls"] == []

    stored = result["repo"].get_plan_draft(CONVERSATION)
    assert stored.stage == "final_reviewed"


@pytest.mark.parametrize("intent,reason", [
    ("plan_final_review", "first_review_not_confirmed"),
    ("plan_confirm_final", "final_review_not_pending"),
])
def test_a_second_summary_or_its_confirmation_cannot_jump_the_queue(
    plan_repo, intent, reason
):
    """AC4's guard at the flow level: no summary shown, no approval recorded."""
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_supplement()),
                    _req("req-jump", message="查看最终摘要", intent=intent))
    events = result["streams"][1]

    assert [e["reason"] for e in _errors(events)] == [reason]
    assert _draft_events(events) == [], "a refusal shows no draft state"
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "minimal_draft"


def test_a_second_summary_without_the_stored_rule_verdict_is_refused(plan_repo):
    """AC4 fails CLOSED on the record: no stored verdict, no final review.

    The row is deleted after the fact here to stand in for a conversation written
    by a build that ran no rules — the click must not be honoured from the stage
    alone.
    """
    reached = _to_first_reviewed(plan_repo)
    assert reached["repo"].get_plan_draft(CONVERSATION).stage == "first_reviewed"
    reached["repo"].clear_completeness_check(CONVERSATION)

    result = _drive(reached["repo"], _req("req-final", message="查看最终摘要",
                                          intent="plan_final_review"))
    events = result["streams"][0]

    assert [e["reason"] for e in _errors(events)] == ["final_review_without_verdict"]
    assert _draft_events(events) == []


def test_a_stale_recorded_verdict_never_carries_the_final_review(plan_repo):
    """The row is a precondition, never the answer (notes-06's recompute rule).

    The stored verdict still says 可审阅 while 集合时间 has since been cleared, so
    trusting the row would honour the click. The fresh verdict bounces it to the
    form with ticket 06's reason codes instead.
    """
    reached = _to_first_reviewed(plan_repo)
    repo = reached["repo"]
    draft = repo.get_plan_draft(CONVERSATION)
    stale = repo.get_completeness_check(CONVERSATION)
    assert stale is not None and stale.verdict != "supplement_required"

    emptied = {**_supplement(), "meeting_time": ""}
    repo.save_plan_arrangements(conversation_id=CONVERSATION, draft_id=draft.draft_id,
                                arrangements=emptied, updated_by_run_id="req-tamper")

    result = _drive(repo, _req("req-final", message="查看最终摘要",
                               intent="plan_final_review"))
    events = result["streams"][0]
    payload = _draft_payload(events)

    assert payload["stage"] == "awaiting_input", "the fresh verdict decides, not the row"
    assert payload["rules"]["verdict"] == "supplement_required"
    assert "core_arrangements_missing" in json.dumps(payload["rules"])
    assert repo.get_completeness_check(CONVERSATION).verdict == "supplement_required"


def test_a_final_confirmation_cannot_be_recorded_twice(plan_repo):
    """AC4: re-confirmation goes through a NEW summary, never a repeat click."""
    confirmed = _to_final_confirmed(plan_repo)
    result = _drive(confirmed["repo"], _req("req-again", message="再确认一次",
                                            intent="plan_confirm_final"))
    events = result["streams"][0]

    assert [e["reason"] for e in _errors(events)] == ["final_review_not_pending"]
    assert _draft_events(events) == []


def test_a_missing_ordinary_arrangement_never_reads_as_a_safety_block(plan_repo):
    """AC3 kept apart from AC2 at the flow level: a gap bounces, it does not block.

    缺少 撤退方案 (non-core) still gets a reviewable draft; 缺少 集合时间 goes back
    to the form. Neither of those is ``blocked_safety``, which only the
    deterministic gate can put a draft on.
    """
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_supplement(retreat_plan="")),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"),
                    _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"),
                    _req("req-final", message="查看最终摘要", intent="plan_final_review"))
    payload = _draft_payload(result["streams"][-1])

    assert payload["stage"] == "awaiting_final_review"
    assert "retreat_plan" in [entry["key"] for entry in payload["missing_fields"]]
    assert payload["plan_type"] == "generic_draft", "AC3: a generic draft, not a plan"
    assert "待确认" in payload["summary"] or UNCONFIRMED in payload["summary"]


def test_a_final_review_turn_is_inert_and_never_an_approval_pause(plan_repo):
    """The ruling, one stage later: a confirmation is a conversational turn.

    No ``run.paused``, no approval row, no artifact, no sandbox lease and no model
    call on either final-review turn — 进入 PDF 审批 is a stage the user reached,
    and the approval machinery (300 s TTL + lease) stays ticket 09's.
    """
    result = _to_final_confirmed(plan_repo)
    for stream in result["streams"][3:]:
        run_id = stream[-1].run_id
        assert set(e.type for e in stream if e.type != "tool_result") <= PLAN_FLOW_EVENT_TYPES
        assert "run.paused" not in [e.type for e in stream]
        assert result["repo"].list_approvals_for_run(run_id) == []
        assert result["repo"].list_artifacts_for_run(run_id) == []
        assert result["repo"].get_sandbox_lease(run_id) is None
    assert result["kernel_calls"] == []


def test_the_second_summary_and_its_confirmation_survive_their_run(plan_repo):
    """AC5: what the user confirmed is readable from the events after the fact."""
    result = _to_final_confirmed(plan_repo)
    repo = result["repo"]
    events = result["streams"][-1]
    run_id = events[-1].run_id

    stored = repo.get_events_since(run_id=run_id, from_seq=0)
    draft_states = [e for e in stored if e["type"] == "plan.draft_state"]
    assert [e["payload"]["stage"] for e in draft_states] == ["final_reviewed"]
    assert draft_states[0]["payload"]["intent"] == "plan_confirm_final"

    history = [m.content for m in repo.checkpoints.get_messages(conversation_id=CONVERSATION)
               if m.role == "assistant"]
    assert draft_states[0]["payload"]["summary"] in history, \
        "the audit text says what was confirmed"
    assert any("最终摘要" in content and "PDF 审批" in content for content in history)


def test_a_final_review_turn_reports_no_safety_block_when_the_gate_refused_nothing(plan_repo):
    """AC5's negative half: no refusal, no invented reason.

    ``safety`` is absent (the reader materialises it as ``null``) whenever the gate
    refused nothing, so the panel cannot render a block the server never claimed —
    and a draft the user reached ``final_reviewed`` with is exactly that case.
    """
    result = _to_final_review(plan_repo)
    payload = _draft_payload(result["streams"][-1])

    assert "safety" not in payload
    assert payload["stage"] == "awaiting_final_review"


@pytest.mark.parametrize("step", [3, 4])
def test_cancelling_still_works_from_either_final_stage(plan_repo, step):
    """The park/cancel split survives the new stages (ticket 02's rule)."""
    requests = [
        _submit("req-submit", arrangements=_supplement()),
        _req("req-review", message="直接生成草案", intent="plan_first_review"),
        _req("req-confirm", message="确认安排", intent="plan_confirm_arrangements"),
        _req("req-final", message="查看最终摘要", intent="plan_final_review"),
        _req("req-confirm-final", message="确认最终摘要", intent="plan_confirm_final"),
    ][:step] + [_req("req-cancel", message="先不用了", intent="plan_cancel")]
    result = _drive(plan_repo, *requests)
    events = result["streams"][-1]

    assert _draft_payload(events)["stage"] == "cancelled"
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "cancelled"


# ---- The HTTP boundary ------------------------------------------------------

import api.chat_v2 as chat_api  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_v2 import kernel  # noqa: E402
from agent_v2.memory_service import MemoryService  # noqa: E402
from agent_v2.repositories.memory import MemoryStore  # noqa: E402
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router  # noqa: E402

HTTP_CONVERSATION = "conv-plan-07-http"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """``POST /api/v2/chat/sse`` over ASGI with a fake kernel."""
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
                          settings=settings, model_calls=model_calls)
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


def test_the_second_summary_and_its_confirmation_reach_the_client_over_http(flow):
    """AC1/AC4/AC5 on the wire, with no model in sight.

    A restore after a refresh reads exactly this payload shape, so the final
    review stage has to survive the HTTP boundary — and neither turn may fall
    through to the kernel or open an approval.
    """
    _http_send(flow, "req-http-submit", intent="plan_create", plan_input={
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "arrangements": _supplement(),
    })
    _http_send(flow, "req-http-review", message="直接生成草案", intent="plan_first_review")
    _http_send(flow, "req-http-confirm", message="确认安排", intent="plan_confirm_arrangements")

    frames = _http_send(flow, "req-http-final", message="查看最终摘要",
                        intent="plan_final_review")
    payload = _draft_frames(frames)[-1]["payload"]
    assert payload["stage"] == "awaiting_final_review"
    assert "最终摘要" in payload["summary"]
    assert payload["summary"] == "".join(
        f["payload"]["content"] for f in frames if f["type"] == "text")

    confirmed = _http_send(flow, "req-http-confirm-final", message="确认最终摘要",
                           intent="plan_confirm_final")
    assert _draft_frames(confirmed)[-1]["payload"]["stage"] == "final_reviewed"
    assert flow.model_calls == []


@pytest.mark.parametrize("intent", ["plan_final_review", "plan_confirm_final"])
def test_a_final_review_decision_cannot_carry_an_arrangement_over_http(flow, intent):
    """AC4 at the boundary: the decision click has nowhere to put a value."""
    frames = _http_send(flow, f"req-smuggle-{intent}", intent=intent, plan_input={
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
    })
    assert frames[-1]["type"] == "error", frames[-1]
    assert frames[-1]["payload"]["reason"] == "plan_input_not_allowed"
    assert _draft_frames(frames) == []

    smuggled = _post(flow.app, flow.subject, {
        "message": "确认最终摘要", "request_id": f"req-extra-{intent}",
        "conversation_id": HTTP_CONVERSATION, "intent": intent,
        "meeting_time": "08:00", "budget": 300, "emergency_contact": "13800000000",
        "report_allowed": True,
    })
    assert smuggled.status_code == 422
    assert flow.repo.resolve_run_id(f"req-extra-{intent}") is None


# ---- AC2: the leaf's 危险封闭 split, and the hard stop in the flow ----------


def _danger_submit(request_id: str = "req-submit") -> ChatFlowRequest:
    """A draft whose destination the offline fixtures DO hold.

    notes-05 landmine 7: researching an unknown place name produces
    ``insufficient_evidence``, not ``no_go`` — the severe-weather fixture only
    speaks for the demo route it is built from.
    """
    return _req(request_id, message="提交草稿", intent="plan_create",
                plan_input=PlanDraftInput.model_validate({
                    "destination_or_route": _DEMO_ROUTE, "activity_date": _DATE,
                    "participants": _PEOPLE, "arrangements": _supplement()}))


def _research(destination: str, scenario: str) -> Any:
    session = RouteSession(settings=_settings(scenario), run_id="leaf-07", audit_sink=None)
    return asyncio.run(session.research_destination(destination))


@pytest.mark.parametrize("scenario,expected_status", [
    ("severe_weather", "blocked"),
    ("partial", "insufficient"),
    ("empty", "insufficient"),
    ("caution", "insufficient"),
])
def test_the_leaf_splits_danger_from_merely_thin_material(scenario, expected_status):
    """The split ticket 04 left open, made in the LEAF's status/reason.

    ``reason`` keeps ``ComparisonResult.status`` verbatim, so ``no_go`` stays
    distinguishable from ``insufficient_evidence`` exactly as the gate said it;
    what changes is that a refusal is now its own status rather than a shared
    parking lot. The leaf judges nothing new — it reads ``assess_candidate``.
    """
    outcome = _research(_DEMO_ROUTE, scenario)

    assert outcome.status == expected_status
    assert outcome.result is not None
    assert outcome.result.report_allowed is False, "AC2: the gate owns eligibility"
    assert outcome.safety is not None
    assert outcome.safety.hard_stop is (scenario == "severe_weather")
    assert outcome.safety.blocks_formal_route_plan is True
    assert outcome.safety.comparison_id == outcome.result.comparison_id
    assert outcome.safety.comparison_status == outcome.result.status

    if scenario == "severe_weather":
        assert outcome.reason == "no_go"
        assert PlanSafetyReason.SEVERE_WEATHER in outcome.safety.codes
        assert any("禁止推荐" in line for line in outcome.safety.gate_reasons), \
            "the gate's own words, copied"
        assert "取消该路线" in outcome.safety.required_action
    else:
        assert PlanSafetyReason.EVIDENCE_GATE_FAILED in outcome.safety.codes


def test_a_gate_passed_research_carries_no_safety_record():
    """The negative half: nothing refused, so nothing to report."""
    outcome = _research(_DEMO_ROUTE, "complete")

    assert outcome.status == "ok"
    assert outcome.safety is None


def test_a_dangerous_research_lands_on_the_hard_stop_and_says_why(plan_repo):
    """AC2/AC5 through the seam: the stage, the payload and the event record."""
    result = _drive(plan_repo, _danger_submit(),
                    _req("req-research", message="研究路线资料",
                         intent="plan_research_route"),
                    scenario="severe_weather")
    events = result["streams"][1]
    payload = _draft_payload(events)

    assert payload["stage"] == "blocked_safety"
    assert payload["plan_type"] == "generic_draft", "never promoted to a route plan"
    assert payload["route_selection"] is None
    assert result["repo"].get_route_selection(CONVERSATION) is None
    assert payload["safety"]["hard_stop"] is True
    assert PlanSafetyReason.SEVERE_WEATHER in payload["safety"]["codes"]
    assert SAFETY_MESSAGE[PlanSafetyReason.SEVERE_WEATHER] in payload["safety"]["message"]
    assert payload["safety"]["comparison_status"] == "no_go"
    assert payload["safety"]["formal_route_plan_allowed"] is False
    assert "危险封闭" in payload["summary"] and "正式路线计划" in payload["summary"]
    assert payload["summary"] == _texts(events)
    assert result["kernel_calls"] == []

    stored = result["repo"].get_events_since(run_id=events[-1].run_id, from_seq=0)
    record = [e for e in stored if e["type"] == "plan.draft_state"][0]["payload"]
    assert record["safety"]["codes"] == payload["safety"]["codes"], \
        "the reason is in the event record, not only in the stream"


@pytest.mark.parametrize("intent,plan_type", [
    ("plan_generic_draft", "generic_draft"),
    ("plan_first_review", None),
    ("plan_final_review", None),
    ("plan_confirm_final", None),
])
def test_the_hard_stop_refuses_every_advancing_click_in_the_flow(plan_repo, intent,
                                                                 plan_type):
    """AC2: no click walks past 危险封闭, including the generic-draft escape."""
    reached = _drive(plan_repo, _danger_submit(),
                     _req("req-research", message="研究路线资料",
                          intent="plan_research_route"),
                     scenario="severe_weather")
    assert reached["repo"].get_plan_draft(CONVERSATION).stage == "blocked_safety"

    result = _drive(reached["repo"], _req("req-advance", intent=intent,
                                          plan_type=plan_type))
    events = result["streams"][0]

    assert [e["reason"] for e in _errors(events)] == ["safety_blocked"]
    assert _draft_events(events) == [], "a refusal stores no new state"
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "blocked_safety"


def test_a_re_searched_destination_is_judged_again_from_scratch(plan_repo):
    """The exits are the honest ones: cancel, or change the route and re-verify."""
    reached = _drive(plan_repo, _danger_submit(),
                     _req("req-research", message="研究路线资料",
                          intent="plan_research_route"),
                     scenario="severe_weather")

    after_cancel = _drive(reached["repo"],
                          _req("req-cancel", message="先不用了", intent="plan_cancel"))
    assert _draft_payload(after_cancel["streams"][0])["stage"] == "cancelled"

    reentry = _drive(reached["repo"], _danger_submit("req-reopen"),
                     _req("req-research2", message="研究路线资料",
                          intent="plan_research_route"),
                     scenario="complete")
    assert _draft_payload(reentry["streams"][1])["stage"] == "minimal_draft", \
        "the gate got the last word again, on the new material"


def test_a_dangerous_state_reaches_the_client_and_stays_there_over_http(flow):
    """AC5 on the wire, and AC2 at the boundary: the generic-draft click is refused."""
    flow.settings.route_compare_demo_scenario = "severe_weather"
    _http_send(flow, "req-http-submit", intent="plan_create", plan_input={
        "destination_or_route": _DEMO_ROUTE,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "arrangements": _supplement(),
    })
    frames = _http_send(flow, "req-http-research", message="研究路线资料",
                        intent="plan_research_route")
    payload = _draft_frames(frames)[-1]["payload"]

    assert payload["stage"] == "blocked_safety"
    assert payload["safety"]["hard_stop"] is True
    assert PlanSafetyReason.SEVERE_WEATHER in payload["safety"]["codes"]

    refused = _http_send(flow, "req-http-generic", message="继续生成通用草案",
                         intent="plan_generic_draft", plan_type="generic_draft")
    assert refused[-1]["type"] == "error"
    assert refused[-1]["payload"]["reason"] == "safety_blocked"
    assert _draft_frames(refused) == []
    assert flow.model_calls == []


# ---- Frontend wiring, pinned from the sources (no test runner exists) ------


def test_the_final_review_intents_and_stages_are_on_both_sides_of_the_wire():
    """AC1/AC4's vocabulary: one list on the server, mirrored exactly in TS."""
    wire = _ts("src/api/chatV2.ts")
    assert _ts_union_members(wire, "V2PlanIntent") == set(PLAN_INTENTS)
    assert _ts_union_members(wire, "V2PlanDraftStage") == {s.value for s in PlanDraftStage}
    assert _ts_string_set_literal(wire, "V2_PLAN_DRAFT_STAGES") == {
        s.value for s in PlanDraftStage}

    page = _ts("src/pages/SuperAgent.tsx")
    for intent in sorted(PLAN_INTENTS):
        assert f"{intent}:" in page, f"PLAN_INTENT_LABELS has no label for {intent}"
    for stage in sorted(s.value for s in PlanDraftStage):
        assert f"{stage}:" in page, f"PLAN_STAGE_LABELS has no label for {stage}"


def test_the_safety_record_has_all_four_frontend_homes():
    """Notes-06's rule for a new draft payload field, applied to ``safety``.

    ``V2PlanDraftState`` (compile time) + the reader (the single validator for
    BOTH the SSE payload and the localStorage mirror) + ``isSamePlanDraft`` (which
    decides whether a restore is a no-op) + the panel that shows it. Miss the
    third and a safety block that appeared after a refresh reads as "no change",
    so the panel keeps the OLD (absent) refusal — the one piece of state a user
    must never read stale, because it is the difference between 可继续 and 已阻止.
    """
    wire = _ts("src/api/chatV2.ts")
    reducer = _ts("src/api/v2Reducer.ts")
    store = _ts("src/api/planDraftStore.ts")
    page = _ts("src/pages/SuperAgent.tsx")

    assert "export interface V2PlanSafety" in wire
    assert "safety: V2PlanSafety | null" in wire, "required in state, optional on the wire"
    assert "safety: readPlanSafety(payload['safety'])" in reducer
    assert "isSamePlanSafety(a.safety, b.safety)" in store
    assert "planDraft.safety" in page and "阻止原因：" in page

    # An unknown code is dropped rather than rendered as an unexplained string,
    # and no code list here may drift from the server's.
    assert "V2_PLAN_SAFETY_CODES" in reducer
    from agent_v2.plan_workflow import SAFETY_MESSAGE

    codes_block = re.search(
        r"V2_PLAN_SAFETY_CODES: readonly string\[\] = \[(.*?)\]", wire, re.S)
    assert codes_block, "the reader's code list is declared once, in chatV2.ts"
    assert set(re.findall(r"'([^']+)'", codes_block.group(1))) == set(SAFETY_MESSAGE)


def test_the_page_offers_the_second_summary_and_treats_danger_as_no_choice():
    """AC1/AC2/AC4 in the UI, pinned by reading the handlers and the panels.

    查看最终摘要 is offered only on ``first_reviewed`` and 确认最终摘要 only while
    the second summary is pending (the server refuses both elsewhere), each send
    carries the whitelisted intent ALONE, the ``blocked_safety`` panel offers no
    generic-draft affordance at all, and the final-review stages are reopenable
    inside the ONE existing panel while the hard stop keeps its own — so no second
    ``top-16`` panel stacks on this one (notes-04 landmine 5).
    """
    page = _ts("src/pages/SuperAgent.tsx")
    store = _ts("src/api/planDraftStore.ts")

    final = _ts_callback(page, "handleShowFinalSummary")
    confirm = _ts_callback(page, "handleConfirmFinalSummary")
    assert "intent: 'plan_final_review'" in final
    assert "intent: 'plan_confirm_final'" in confirm
    for handler in (final, confirm):
        for smuggled in ("planInput", "planType", "routeSelection", "profileId", "approval",
                         "arrangements"):
            assert smuggled not in handler
    for name in ("handleShowFinalSummary", "handleConfirmFinalSummary"):
        assert f"onClick={{{name}}}" in page, name

    assert "planDraft.stage === 'first_reviewed'" in page
    assert "planDraft.stage === 'awaiting_final_review'" in page
    assert "查看最终摘要" in page and "确认最终摘要" in page

    # The hard stop: no 继续生成通用草案 inside the blocked_safety panel.
    danger_panel = page[page.index("planDraft?.stage === 'blocked_safety'"):]
    danger_panel = danger_panel[:danger_panel.index("planDraft && !planDialogOpen")]
    assert "继续生成通用草案" not in danger_panel
    assert "handlePlanDecline" in danger_panel, "取消 is the plan_cancel exit"
    assert "handleRestorePlanDraft" in danger_panel, "修改安排 reopens the form locally"

    open_block = store[store.index("OPEN_PLAN_DRAFT_STAGES"):store.index("isDurablePlanDraftStage")]
    assert "'awaiting_final_review'" in open_block and "'final_reviewed'" in open_block
    assert "'blocked_safety'" not in open_block, "answered by its own panel, not a reopened form"


def test_the_safety_panel_adds_no_new_tailwind_class():
    """AGENTS.md's Tailwind rule, checked the only way this repo can.

    There is no frontend test runner and the dev server may not be relied on, so
    the classes the new panels use are pinned against a generated stylesheet
    instead of a screenshot: every utility the ticket 07 markup introduces must
    already be produced by this project's own config.
    """
    page = _ts("src/pages/SuperAgent.tsx")
    for cls in ("max-w-sm", "rounded-md", "border-border", "text-text-muted",
                "text-xs", "mt-1", "mt-3", "justify-end", "gap-2", "flex-wrap",
                "font-medium", "dark:border-white/10", "dark:bg-[#1e1f20]"):
        assert cls in page, cls


def test_the_four_plan_notice_panels_share_one_box_class_constant():
    """Code-review finding 11(a): the four plan notice panels — 确认 (awaiting_con-
    firmation), 通用草案选择 (awaiting_generic_choice), 危险封闭 (blocked_safety),
    恢复草稿 (restore) — each inlined a byte-identical box class. That is a drift
    trap (fix the shadow on three and the fourth disagrees), so the box now has
    ONE home: a module constant every panel references.

    Behaviour-preserving, and there is no frontend runner, so the dedup is pinned
    by the source: the shared box's distinguishing dark-mode tail appears exactly
    once (in the constant, still in the file the CSS-generating scan reads, so
    ``test_the_safety_panel_adds_no_new_tailwind_class`` keeps passing unchanged),
    and all four panels reference the constant rather than re-inlining it.
    """
    page = _ts("src/pages/SuperAgent.tsx")

    assert page.count("dark:bg-[#1e1f20]") == 1, (
        "the shared box class must live in exactly one place"
    )
    assert "const PLAN_NOTICE_PANEL_CLASS" in page
    assert page.count("className={PLAN_NOTICE_PANEL_CLASS}") == 4
