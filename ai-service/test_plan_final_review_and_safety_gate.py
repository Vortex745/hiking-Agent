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
        safety=kwargs.get("safety"),
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
