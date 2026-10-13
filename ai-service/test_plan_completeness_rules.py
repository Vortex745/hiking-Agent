"""Ticket 06 — 计划完整度校验与确定性确认.

Seams under test (the ones notes-05 names for this ticket):

* :func:`agent_v2.plan_workflow.evaluate_plan_completeness` — the deterministic
  completeness verdict: field existence (:func:`missing_plan_fields`), format
  validation and explicit business rules. Zero model calls: the module imports
  no kernel, no provider and no HTTP client, and every turn driven here asserts
  ``kernel_calls == []``;
* :func:`agent_v2.plan_workflow.evaluate_plan_intent` — still the ONLY
  transition decider, now with the rule's verdict as an input, so 「直接生成草案」
  cannot open a summary over a draft that is missing a CORE arrangement;
* :func:`agent_v2.plan_workflow.draft_summary` / :func:`render_minimal_draft` —
  the one renderer, now drawing the eight supplementary arrangements the user
  actually supplied. Presence is still the draft row and the re-read route
  basis: nothing is invented;
* ``PlanArrangementInput`` + ``api.chat_v2.ChatV2Request`` +
  :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` — the wire half.
  A new field arriving from the client must be declared on BOTH request models,
  because ``extra="forbid"`` makes the boundary invisible to flow-level tests;
* ``agent_v2.persistence`` — a NEW table per the storage rule (an added column
  on an existing table silently never exists), for the collected arrangements and
  for the saved rule verdict (AC5: restorable and auditable);
* :func:`route_compare.service.render_report` / ``assess_candidate`` — AC6: the
  rules never produce a route fact and never bypass the safety gate;
* the frontend sources, pinned by reading them (there is no frontend runner).

Acceptance criteria:

* AC1 完整度由字段存在性、格式校验和业务规则确定.
* AC2 安排与路线证据的匹配关系用显式字段和来源状态表示.
* AC3 缺少核心字段进入补充流程，缺少非核心字段进入待确认清单.
* AC4 需要人工确认的原因使用稳定枚举值 + 可读说明.
* AC5 规则结果可被保存、恢复和审计.
* AC6 规则不生成路线事实，也不单独绕过安全放行门.
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
    MINIMAL_ARRANGEMENT_KEYS,
    PLAN_INTENTS,
    REASON_MESSAGE,
    SUPPLEMENT_ARRANGEMENT_KEYS,
    PlanArrangementInput,
    PlanCompleteness,
    PlanDraftInput,
    PlanDraftStage,
    PlanReviewReason,
    RouteBasis,
    RouteFactLine,
    UNCONFIRMED,
    UNVERIFIED,
    arrangement_format_issues,
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

CONVERSATION = "conv-plan-06"
SUBJECT_ID = "sub-plan-06"

_DEMO_ROUTE = "演示林间环线"
_DESTINATION = "四姑娘山长坪沟"
_DATE = "2026-11-08"
_PEOPLE = 12

#: A draft that holds every CORE arrangement in a valid format: the three the
#: entry form has always collected plus the three ticket 06 adds to the core set.
#: Spelled out as literals from the field list, not computed from the module.
_CORE_ARRANGEMENTS = {
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
}

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _cap(field) -> int | None:
    """The max_length constraint declared on a pydantic model field."""
    caps = [m.max_length for m in field.metadata if hasattr(m, "max_length")]
    return caps[0] if caps else None


def _ts(path: str) -> str:
    return (FRONTEND / path).read_text(encoding="utf-8")


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


def _arrangements(**overrides) -> dict[str, str]:
    """The supplementary record, empty by default: an unfilled field stays absent."""
    values = {key: "" for key in SUPPLEMENT_ARRANGEMENT_KEYS}
    values.update(overrides)
    return values


def _submit(request_id: str, *, arrangements: dict[str, str] | None = None,
            **overrides) -> ChatFlowRequest:
    payload = {
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
    }
    payload.update(overrides)
    if arrangements is not None:
        payload["arrangements"] = arrangements
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate(payload),
    )


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
        conversation_id=kwargs.get("conversation_id", "conv-06"),
        route_selection=kwargs.get("route_selection"), plan_type=kwargs.get("plan_type"),
        completeness=kwargs.get("completeness"),
    )


def _verdict(**overrides) -> PlanCompleteness:
    fields = {
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "arrangements": None,
        "route": None,
    }
    fields.update(overrides)
    return evaluate_plan_completeness(**fields)


# ---- The arrangement record: real input, real storage (AC1's precondition) --


def test_the_arrangement_record_is_the_plan_arrangement_record():
    """AC1: the eight collectable arrangements are the PlanDetails fields.

    ``ARRANGEMENT_FIELDS`` is ticket 05's enumeration; ``SUPPLEMENT_...`` is the
    same list minus the three the minimal form has always collected, and every
    one of them is a field of
    :class:`route_compare.models.PlanDetails` with the same length cap. So the
    form cannot grow a private vocabulary, and a field cannot be collected
    without joining the checklist (notes-05 landmine 6).
    """
    enumerated = {key for key, _label in ARRANGEMENT_FIELDS}
    assert set(MINIMAL_ARRANGEMENT_KEYS) | set(SUPPLEMENT_ARRANGEMENT_KEYS) == enumerated
    assert not set(MINIMAL_ARRANGEMENT_KEYS) & set(SUPPLEMENT_ARRANGEMENT_KEYS)

    model_fields = set(PlanArrangementInput.model_fields)
    assert model_fields == set(SUPPLEMENT_ARRANGEMENT_KEYS)
    assert model_fields - {"budget"} <= set(PlanDetails.model_fields)

    # Same caps as the arrangement record, so the draft cannot hold a value the
    # plan document could not carry.
    for key in model_fields - {"budget"}:
        assert (_cap(PlanArrangementInput.model_fields[key])
                == _cap(PlanDetails.model_fields[key])), key


def test_the_supplementary_arrangements_are_optional_and_stay_absent():
    """An unfilled arrangement is an empty string, never a guessed value."""
    parsed = PlanDraftInput.model_validate({
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
    })
    assert parsed.arrangements is not None
    assert all(getattr(parsed.arrangements, key) == "" for key in SUPPLEMENT_ARRANGEMENT_KEYS)

    filled = PlanArrangementInput.model_validate(
        _arrangements(**_CORE_ARRANGEMENTS)
    )
    assert filled.meeting_time == "08:30"
    assert filled.leader == ""


def test_a_route_fact_cannot_be_smuggled_into_the_arrangement_record():
    """``extra="forbid"`` on the new model: the arrangement bag stays an
    arrangement bag. A distance, a weather claim or a ``report_allowed`` flag is
    refused rather than stored as "the user's own arrangement"."""
    for smuggled in ("distance_km", "ascent_m", "weather", "report_allowed", "plan_type"):
        with pytest.raises(ValueError):
            PlanArrangementInput.model_validate(_arrangements() | {smuggled: "8.2"})


def test_the_renderer_reads_only_keys_the_checklist_enumerates():
    """AC4's chokepoint, widened to the new input: the record is a mapping, so the
    only way a value reaches the draft is by being an enumerated arrangement. A
    key outside :data:`ARRANGEMENT_FIELDS` is dropped, not drawn."""
    values = _arrangements(**_CORE_ARRANGEMENTS) | {"distance_km": "8.2", "weather": "晴"}

    rendered = render_minimal_draft(
        destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        arrangements=values,
    )
    assert "08:30" in rendered and "13800000000" in rendered
    assert "8.2" not in rendered and "晴" not in rendered

    gaps = {field.key for field in missing_plan_fields(
        destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        arrangements=values,
    )}
    assert "meeting_time" not in gaps and "emergency_contact" not in gaps
    assert "distance_km" in gaps, "the route fact is still unverified, never filled"


def test_a_supplied_arrangement_leaves_the_missing_list_and_a_blank_one_does_not():
    """AC1's existence half, over a field that can genuinely be filled."""
    empty = {f.key for f in missing_plan_fields(
        destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE)}
    assert {"meeting_time", "leader", "retreat_plan"} <= empty

    supplied = {
        f.key: f.status for f in missing_plan_fields(
            destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
            arrangements=_arrangements(meeting_time="08:30", leader="阿强"),
        )
    }
    assert "meeting_time" not in supplied and "leader" not in supplied
    assert supplied["retreat_plan"] == UNCONFIRMED
    assert supplied["weather"] == UNVERIFIED


def test_the_format_rules_reject_a_value_that_cannot_be_used():
    """AC1's format half, over values that can genuinely be wrong.

    A user CAN type 「明天一早」 as a meeting time: existence alone would call the
    field filled. The rules are explicit per-field formats, deterministic, and
    they never rewrite or normalise the value.
    """
    issues = {i.key: i.message for i in arrangement_format_issues({
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": f"{_PEOPLE} 人",
        **_arrangements(meeting_time="明天一早", budget="看情况吧",
                        emergency_contact="微信联系我", retreat_plan="下撤到沟口"),
    })}
    assert set(issues) == {"meeting_time", "budget", "emergency_contact"}
    assert "HH:MM" in issues["meeting_time"]
    assert "应急联络" in issues["emergency_contact"]
    assert all(message for message in issues.values())

    clean = arrangement_format_issues({
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": f"{_PEOPLE} 人",
        **_arrangements(meeting_time="08:30", budget="300 元/人",
                        emergency_contact="13800000000", retreat_plan="下撤到沟口"),
    })
    assert clean == ()


def test_an_absent_field_is_never_a_format_issue():
    """Presence and format are separate verdicts: 待确认 must not also read as a
    format error, or the checklist would blame the user for not typing something."""
    assert arrangement_format_issues({
        key: "" for key, _label in ARRANGEMENT_FIELDS
    }) == ()


def test_a_bad_activity_date_is_a_format_issue_not_a_type_error():
    """The stored row can hold what an earlier build wrote, so the rules re-check
    the date rather than trusting the column."""
    issues = {i.key for i in arrangement_format_issues({
        "destination_or_route": _DESTINATION, "activity_date": "2026年11月8日",
        "participants": "12 人", **_arrangements(),
    })}
    assert issues == {"activity_date"}


# ---- AC2: arrangements vs evidence, with explicit source status ------------


def _basis(facts: tuple[RouteFactLine, ...]) -> RouteBasis:
    return RouteBasis(
        comparison_id="run-06:1", candidate_id="demo-a", candidate_name=_DEMO_ROUTE,
        role="recommended", data_mode="offline", facts=facts,
    )


def _line(field: str, value: str, source_id: str = "src-weather") -> RouteFactLine:
    label = {f: l for f, l, _u in FACT_DIMENSIONS}[field]
    return RouteFactLine(field=field, label=label, display_value=value, source_id=source_id,
                         source_field=field, source_tool="hiking_domain",
                         observed_at="2026-11-01T00:00:00Z")


def test_the_match_between_an_arrangement_and_the_evidence_is_explicit_data():
    """AC2: the relationship is a field with a status, not a sentence.

    ``activity_date`` is the arrangement the weather and open-status evidence
    speaks to; ``leader`` has no evidence dimension at all. Each match names the
    arrangement, the fact dimensions considered, whether the arrangement was
    supplied, and the SOURCE that would back it — 待核验 when the gate certified
    nothing.
    """
    verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    matches = {m.arrangement_key: m for m in verdict.matches}

    assert matches["activity_date"].fact_fields == ("weather", "trail_status")
    assert matches["activity_date"].status == "unverified"
    assert matches["activity_date"].source_ids == ()
    assert matches["leader"].status == "not_applicable", "no dimension backs a leader's name"
    assert matches["meeting_point"].status == "unverified"

    certified = _basis((_line("weather", "晴", "src-w"), _line("trail_status", "open", "src-t")))
    with_basis = _verdict(
        arrangements=_arrangements(**_CORE_ARRANGEMENTS), route=certified,
    )
    matched = {m.arrangement_key: m for m in with_basis.matches}
    assert matched["activity_date"].status == "matched"
    assert matched["activity_date"].source_ids == ("src-w", "src-t")
    assert matched["meeting_point"].status == "partly_matched"


def test_a_blank_arrangement_matches_nothing():
    """An absent arrangement has no claim to check, so it is never reported as a
    match — and it stays in the 待确认 half (presence is its own verdict)."""
    verdict = _verdict()
    absent = {m.arrangement_key: m.status for m in verdict.matches}
    assert absent["activity_date"] == "arrangement_absent"
    assert "matched" not in absent.values()


# ---- AC1/AC3/AC4: the completeness verdict -------------------------------


def test_the_verdict_is_a_function_of_existence_format_and_business_rules():
    """AC1: three inputs, one deterministic answer, no score and no model.

    A 3-field draft is not ``complete`` and not merely "pending": 集合时间 / 集合地点 /
    应急联络 are core, so the plan cannot move forward. Filling the core in a valid
    format moves it to the 待确认清单 branch; the route facts keep the human-
    confirmation reasons alive even then.
    """
    minimal = _verdict()
    assert minimal.verdict == "supplement_required"
    assert minimal.missing_core_keys == ("meeting_time", "meeting_point", "emergency_contact")
    assert minimal.missing_non_core_keys == ("leader", "transport", "accommodation",
                                            "budget", "retreat_plan")
    assert minimal.invalid_format_keys == ()
    assert minimal.reasons == (
        PlanReviewReason.CORE_ARRANGEMENTS_MISSING,
        PlanReviewReason.NON_CORE_ARRANGEMENTS_MISSING,
        PlanReviewReason.ROUTE_FACTS_UNVERIFIED,
    )
    for code in minimal.reasons:
        assert REASON_MESSAGE[code], code

    core_filled = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    assert core_filled.verdict == "confirmation_required"
    assert core_filled.missing_core_keys == ()

    everything = _arrangements(**_CORE_ARRANGEMENTS, leader="阿强", transport="包车",
                               accommodation="沟内营地", budget="300 元/人",
                               retreat_plan="下撤到沟口")
    all_filled = _verdict(arrangements=everything, route=_basis(tuple(
        _line(field, f"v-{i}", f"src-{i}") for i, (field, _l, _u) in enumerate(FACT_DIMENSIONS)
    )))
    assert all_filled.verdict == "complete"
    assert all_filled.reasons == ()
    assert all_filled.missing_core_keys == () and all_filled.missing_non_core_keys == ()


def test_an_invalid_format_value_counts_as_needing_supplement_not_as_filled():
    """AC1/AC3: 「明天一早」 is a filled field by existence and a rejected one by
    format. Treating it as filled would let the plan move forward on a value no
    schedule can use."""
    verdict = _verdict(arrangements=_arrangements(
        **(_CORE_ARRANGEMENTS | {"meeting_time": "明天一早"})))

    assert verdict.verdict == "supplement_required"
    assert verdict.invalid_format_keys == ("meeting_time",)
    assert PlanReviewReason.ARRANGEMENT_FORMAT_INVALID in verdict.reasons
    assert "meeting_time" not in verdict.missing_core_keys


def test_the_core_split_is_one_vocabulary_with_the_checklist():
    """AC3's rule cannot drift from the fields the renderer draws: core ∪ non-core
    is exactly ``ARRANGEMENT_FIELDS``, and a field outside it is never core."""
    core = {key for key, _label in ARRANGEMENT_FIELDS if key in _core_keys()}
    assert core <= {key for key, _label in ARRANGEMENT_FIELDS}
    assert set(MINIMAL_ARRANGEMENT_KEYS) <= core
    assert "meeting_time" in core and "leader" not in core

    verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    kinds = {f.key: f for f in verdict.fields}
    assert {f.key for f in verdict.fields if f.core and f.kind == "arrangement"} == core
    assert kinds["weather"].kind == "route_fact" and kinds["weather"].core is True
    assert kinds["transport"].kind == "arrangement" and kinds["transport"].core is False


def _core_keys() -> set[str]:
    from agent_v2.plan_workflow import CORE_ARRANGEMENT_KEYS

    return set(CORE_ARRANGEMENT_KEYS)


def test_the_reason_vocabulary_is_stable_and_readable():
    """AC4: stable machine values, readable copy beside them. Never reworded,
    only added — the same contract ``PlanBlockReason`` + ``BLOCK_MESSAGE`` runs on,
    and ``arrangements_not_pending`` stays where it is (a stage guard, not a
    completeness reason)."""
    codes = {
        PlanReviewReason.CORE_ARRANGEMENTS_MISSING,
        PlanReviewReason.ARRANGEMENT_FORMAT_INVALID,
        PlanReviewReason.NON_CORE_ARRANGEMENTS_MISSING,
        PlanReviewReason.ROUTE_FACTS_UNVERIFIED,
    }
    assert set(REASON_MESSAGE) == codes
    assert codes == {
        "core_arrangements_missing", "arrangement_format_invalid",
        "non_core_arrangements_missing", "route_facts_unverified",
    }
    for code, message in REASON_MESSAGE.items():
        assert message and message.endswith("。"), code
        assert code not in BLOCK_MESSAGE, "the two vocabularies stay distinct"


# ---- AC3: the routing decision stays with the state machine ---------------


def test_a_draft_missing_core_arrangements_goes_to_the_supplement_flow():
    """AC3: 「直接生成草案」 over a core-incomplete draft lands on the form, not on
    the summary — and the move is still decided by ``evaluate_plan_intent``."""
    blocked_verdict = _verdict()
    transition = _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT,
                           completeness=blocked_verdict)

    assert transition.status == "ok"
    assert transition.stage is PlanDraftStage.AWAITING_INPUT
    assert transition.completeness is blocked_verdict


def test_a_draft_with_only_non_core_gaps_reaches_the_summary_checklist():
    ok_verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    transition = _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT,
                           completeness=ok_verdict)

    assert transition.stage is PlanDraftStage.AWAITING_FIRST_REVIEW


def test_the_review_gate_fails_closed_without_a_verdict():
    """A missing rule input is never treated as a passing rule: an intent that
    needs the completeness verdict cannot be honoured without one."""
    transition = _evaluate("plan_first_review", PlanDraftStage.MINIMAL_DRAFT)

    assert transition.blocked
    assert transition.reason == "first_review_without_draft"
    assert BLOCK_MESSAGE[transition.reason]


# ---- AC6: the rules never make a route fact and never open the gate --------


def test_the_rules_cannot_produce_a_route_fact():
    """AC6: every value in the verdict is either the user's own arrangement text
    or a source id the gate issued. No number, no weather word, no derived time."""
    verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    dumped = json.dumps([
        {"key": f.key, "label": f.label, "presence": f.presence, "format": f.format_status,
         "value": f.value} for f in verdict.fields
    ], ensure_ascii=False)
    assert "待核验" not in dumped.replace(UNVERIFIED, "")

    without_basis = _verdict(route=None)
    for match in without_basis.matches:
        assert match.source_ids == (), "no evidence, no source claim"
    for field in verdict.fields:
        if field.kind == "route_fact":
            assert field.value == "", field.key


def test_plan_workflow_imports_no_model_and_no_provider():
    """AC1's "不依赖外部模型" as a structural fact: the pure layer cannot reach a
    kernel, an HTTP client or a decision provider."""
    import agent_v2.plan_workflow as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+(openai|httpx|requests|langchain)", source,
                         flags=re.M)
    assert "run_chat_turn" not in source
    assert "DecisionProvider" not in source
    assert "compare_routes" not in source


def test_a_complete_verdict_still_leaves_report_eligibility_with_the_gate(plan_repo):
    """AC6: the rules say nothing about a PDF. A verdict of ``complete`` over a
    comparison the gate refuses must not make ``render_report`` pass, and the plan
    path must not have asked it to."""
    from route_compare.models import ComparisonResult
    from route_compare.service import assess_candidate, render_report

    blocked = _seed_comparison(plan_repo, CONVERSATION, scenario="severe_weather")
    result = ComparisonResult.model_validate(blocked)

    with pytest.raises(ValueError):
        render_report(result)
    assert result.report_allowed is False
    assert any(a.risk_status == "no_go" for a in result.candidates)

    verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
    assert verdict.verdict == "confirmation_required"
    assert not hasattr(verdict, "report_allowed")
    assert assess_candidate(result.candidates[0].candidate).risk_status == "no_go"


# ---- AC5: the verdict is saved, restorable and auditable ------------------


def test_the_verdict_row_round_trips_and_survives_a_reopen(tmp_path):
    """AC5: the rule result is a row, not a render. A fresh repository over the
    same database restores the same verdict, reasons and gaps."""
    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        draft = repo.save_plan_draft(
            conversation_id=CONVERSATION, stage="awaiting_first_review",
            destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        )
        verdict = _verdict(arrangements=_arrangements(**_CORE_ARRANGEMENTS))
        repo.save_completeness_check(
            conversation_id=CONVERSATION, draft_id=draft.draft_id,
            verdict=verdict, checked_by_run_id="run-06",
        )
    finally:
        repo.close()

    reopened = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        stored = reopened.get_completeness_check(CONVERSATION)
        assert stored is not None
        assert stored.draft_id == draft.draft_id
        assert stored.verdict == "confirmation_required"
        assert list(stored.reasons) == list(verdict.reasons)
        assert list(stored.missing_non_core_keys) == list(verdict.missing_non_core_keys)
        assert stored.checked_by_run_id == "run-06"
    finally:
        reopened.close()


def test_the_collected_arrangements_round_trip_and_a_cancel_clears_them(tmp_path):
    """The eight new inputs have real storage (notes-05's "Still not done"), in a
    NEW table per the storage rule — and cancel clears the unfinished plan state
    including this row (ticket 02 AC4)."""
    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    try:
        draft = repo.save_plan_draft(
            conversation_id=CONVERSATION, stage="minimal_draft",
            destination_or_route=_DESTINATION, activity_date=_DATE, participants=_PEOPLE,
        )
        empty = repo.get_plan_arrangements(CONVERSATION)
        assert empty is None, "no row until the user submits one"

        saved = repo.save_plan_arrangements(
            conversation_id=CONVERSATION, draft_id=draft.draft_id,
            arrangements=_arrangements(**_CORE_ARRANGEMENTS), updated_by_run_id="run-06",
        )
        assert saved.meeting_time == "08:30"
        assert saved.draft_id == draft.draft_id

        replaced = repo.save_plan_arrangements(
            conversation_id=CONVERSATION, draft_id=draft.draft_id,
            arrangements=_arrangements(meeting_time="09:00"), updated_by_run_id="run-07",
        )
        assert replaced.meeting_time == "09:00" and replaced.emergency_contact == ""
        assert repo.clear_plan_arrangements(CONVERSATION) == 1
        assert repo.get_plan_arrangements(CONVERSATION) is None
    finally:
        repo.close()


# ---- The flow: one deterministic pass over the stored draft ----------------


def test_the_flow_saves_the_verdict_and_shows_it_on_the_wire(plan_repo):
    """AC1/AC3/AC4/AC5 through the seam: submit → the rules run on the stored row
    and the answer travels. No model, and the checklist and the verdict are the
    same render."""
    result = _drive(plan_repo, _submit("req-submit", arrangements=_arrangements(
        meeting_time="明天一早", leader="阿强")))
    payload = _draft_payload(result["streams"][0])

    assert payload["draft"]["arrangements"] == _arrangements(meeting_time="明天一早",
                                                             leader="阿强")
    rules = payload["rules"]
    assert rules["verdict"] == "supplement_required"
    assert [r["code"] for r in rules["reasons"]] == [
        "core_arrangements_missing", "arrangement_format_invalid",
        "non_core_arrangements_missing", "route_facts_unverified",
    ]
    assert {i["key"] for i in rules["format_issues"]} == {"meeting_time"}
    assert "明天一早" in payload["summary"]
    assert "08:30" not in payload["summary"]
    assert result["kernel_calls"] == []
    assert plan_repo.get_plan_arrangements(CONVERSATION).meeting_time == "明天一早"
    assert plan_repo.get_completeness_check(CONVERSATION).verdict == "supplement_required"


def test_generating_the_draft_now_supplements_instead_of_reviewing(plan_repo):
    """AC3 driven: the click that means 「直接生成草案」 bounces to the form while a
    core arrangement is missing, and reaches the summary once it is filled."""
    result = _drive(plan_repo,
                    _submit("req-submit"),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"))
    payload = _draft_payload(result["streams"][1])

    assert payload["stage"] == "awaiting_input"
    assert "core_arrangements_missing" in payload["summary"]
    assert result["kernel_calls"] == []

    filled = _drive(plan_repo,
                    _submit("req-submit-full", arrangements=_arrangements(**_CORE_ARRANGEMENTS)),
                    _req("req-review-full", message="直接生成草案", intent="plan_first_review"))
    assert _draft_payload(filled["streams"][1])["stage"] == "awaiting_first_review"
    assert filled["kernel_calls"] == []


def test_a_review_turn_keeps_the_supplied_arrangements_after_a_refresh(plan_repo):
    """AC5's restore half: the row, the checklist and the verdict all come back
    from storage on a later turn, without the client resending anything."""
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_arrangements(**_CORE_ARRANGEMENTS)),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"))
    events = result["streams"][1]
    payload = _draft_payload(events)

    assert payload["draft"]["arrangements"]["meeting_time"] == "08:30"
    assert "meeting_time" not in {f["key"] for f in payload["missing_fields"]}
    assert payload["rules"]["verdict"] == "confirmation_required"
    assert plan_repo.get_completeness_check(CONVERSATION).draft_id == payload["draft_id"]


def test_the_rules_never_call_the_kernel_and_emit_only_plan_events(plan_repo):
    """AC1's "不依赖外部模型" at the flow seam, on every 06 turn."""
    PLAN_FLOW_EVENT_TYPES = {
        "run_started", "plan.draft_state", "text", "error", "done", "cancelled",
    }
    result = _drive(plan_repo,
                    _submit("req-submit", arrangements=_arrangements(**_CORE_ARRANGEMENTS)),
                    _req("req-review", message="直接生成草案", intent="plan_first_review"),
                    _req("req-confirm", message="确认安排",
                         intent="plan_confirm_arrangements"))
    for events in result["streams"]:
        assert {e.type for e in events} <= PLAN_FLOW_EVENT_TYPES, [e.type for e in events]
    assert result["kernel_calls"] == []
    assert [e for e in sum(result["streams"], []) if e.type == "tool_result"] == []


# ---- HTTP boundary: the new field on both sides of the wire ----------------


def _seed_comparison(repo: RunRepository, conversation_id: str, *,
                     scenario: str = "complete",
                     request_id: str = "req-compare") -> dict:
    """A comparison persisted as its own run's ``tool_result`` (notes-05 landmine
    7: the offline fixtures only know 演示林间环线)."""
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="route_plan",
        user_message="比较演示路线",
    )
    session = RouteSession(settings=_settings(scenario), run_id=run_id, audit_sink=None)
    result = asyncio.run(session.compare([
        {"name": _DEMO_ROUTE, "city": "演示城市", "origin": "起点", "destination": "终点"},
    ]))
    repo.persist_event(
        run_id=run_id, sequence=1, event_type="tool_result",
        request_id=request_id, conversation_id=conversation_id,
        payload={"route_comparison": result},
    )
    return result


import api.chat_v2 as chat_api  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_v2 import kernel  # noqa: E402
from agent_v2.memory_service import MemoryService  # noqa: E402
from agent_v2.repositories.memory import MemoryStore  # noqa: E402
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router  # noqa: E402

HTTP_CONVERSATION = "conv-plan-06-http"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    settings = _settings()
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


def test_the_collected_arrangements_and_the_verdict_cross_the_http_boundary(flow):
    """The new field must be declared on ``ChatV2Request`` AND
    ``ChatFlowRequest``: a wire model that accepts it while the flow model lacks
    it is a feature that is dead with a green suite (notes-01 landmine 2)."""
    frames = _http_send(flow, "req-http-submit", intent="plan_create", plan_input={
        "destination_or_route": _DESTINATION,
        "activity_date": _DATE,
        "participants": _PEOPLE,
        "arrangements": _arrangements(**_CORE_ARRANGEMENTS),
    })
    payload = [f for f in frames if f["type"] == "plan.draft_state"][-1]["payload"]

    assert payload["draft"]["arrangements"]["emergency_contact"] == "13800000000"
    assert payload["rules"]["verdict"] == "confirmation_required"
    assert flow.model_calls == []

    review = _http_send(flow, "req-http-review", message="直接生成草案",
                        intent="plan_first_review")
    assert [f for f in review if f["type"] == "plan.draft_state"][-1]["payload"][
        "stage"] == "awaiting_first_review"


def test_a_malformed_or_smuggled_arrangement_is_rejected_before_streaming(flow):
    """``extra="forbid"`` at both depths, before a run exists."""
    smuggled = _post(flow.app, flow.subject, {
        "message": "提交草稿", "request_id": "req-smuggle-06",
        "conversation_id": HTTP_CONVERSATION, "intent": "plan_create",
        "plan_input": {
            "destination_or_route": _DESTINATION, "activity_date": _DATE,
            "participants": _PEOPLE,
            "arrangements": _arrangements() | {"distance_km": "8.2"},
        },
    })
    assert smuggled.status_code == 422
    assert flow.repo.resolve_run_id("req-smuggle-06") is None

    top_level = _post(flow.app, flow.subject, {
        "message": "提交草稿", "request_id": "req-top-06",
        "conversation_id": HTTP_CONVERSATION, "intent": "plan_create",
        "meeting_time": "08:30",
    })
    assert top_level.status_code == 422
    assert flow.repo.resolve_run_id("req-top-06") is None

    too_long = _post(flow.app, flow.subject, {
        "message": "提交草稿", "request_id": "req-long-06",
        "conversation_id": HTTP_CONVERSATION, "intent": "plan_create",
        "plan_input": {
            "destination_or_route": _DESTINATION, "activity_date": _DATE,
            "participants": _PEOPLE,
            "arrangements": _arrangements(retreat_plan="下撤" * 300),
        },
    })
    assert too_long.status_code == 422
    assert flow.repo.resolve_run_id("req-long-06") is None


def test_a_review_decision_still_carries_no_arrangement_over_http(flow):
    """Notes-05 landmine 6's other side: the decision click has nowhere to put a
    value, so the only way an arrangement reaches the draft is the form."""
    frames = _http_send(flow, "req-review-smuggle", message="直接生成草案",
                        intent="plan_first_review", plan_input={
        "destination_or_route": _DESTINATION, "activity_date": _DATE,
        "participants": _PEOPLE,
    })
    assert frames[-1]["type"] == "error"
    assert frames[-1]["payload"]["reason"] == "plan_input_not_allowed"
    assert [f for f in frames if f["type"] == "plan.draft_state"] == []


# ---- Frontend wiring, pinned from the sources (no test runner exists) -----


def test_the_collected_arrangements_have_all_four_frontend_homes():
    """Notes-05 landmine 5 generalised: every new draft payload field needs the
    type, the single reader, the equality check and the panel. Miss the reader
    and a hand-edited localStorage entry smuggles a value into the form; miss
    the comparison and a just-filled arrangement looks like "no change"."""
    wire = _ts("src/api/chatV2.ts")
    reducer = _ts("src/api/v2Reducer.ts")
    store = _ts("src/api/planDraftStore.ts")
    dialog = _ts("src/components/PlanDraftDialog.tsx")

    assert "export interface V2PlanArrangementFields" in wire
    for key in SUPPLEMENT_ARRANGEMENT_KEYS:
        assert f"{key}: string" in wire, key
    assert "arrangements: readPlanArrangements(draft['arrangements'])" in reducer
    assert "isSameArrangements(a.draft.arrangements, b.draft.arrangements)" in store
    assert "arrangements: V2PlanArrangementFields" in wire

    for key in SUPPLEMENT_ARRANGEMENT_KEYS:
        assert f'name="{key}"' in dialog, key

    rules = _ts("src/pages/SuperAgent.tsx")
    assert "planDraft.rules" in rules


def test_the_six_plan_wire_readers_share_one_object_guard():
    """Code-review finding 11(b): the six wire readers (``readRouteSelection``,
    ``readMissingPlanFields`` per-entry, ``readPlanRules`` top + four per-entry
    loops, ``readPlanSafety``, ``readPlanArrangements``, ``readObject``) each
    spelled the same `!value || typeof value !== 'object' || Array.isArray(value)`
    shape check inline. That is a drift trap TypeScript cannot see: a reader
    that drops the array check would accept a hostile localStorage mirror's
    ``[]`` as a record, and a reader that inverts the null check would accept
    ``null`` — both would then index a non-object as if it were a payload.

    All six now delegate to one ``asRecord`` helper, so the predicate is
    spelled once and every reader agrees on what counts as an object.

    Pinned by the source, since there is no frontend runner: ``!== 'object'``
    appears exactly once (inside the helper), and each of the six reader
    bodies calls ``asRecord(``.
    """
    reducer = _ts("src/api/v2Reducer.ts")

    assert "function asRecord(value: unknown)" in reducer, "the shared helper exists"
    helper_start = reducer.index("function asRecord(value: unknown)")
    helper_end = reducer.index("\n}", helper_start) + 2
    helper = reducer[helper_start:helper_end]
    assert "!== 'object'" in helper and "Array.isArray(value)" in helper

    assert reducer.count("!== 'object'") == 1, (
        "only asRecord re-spells the negative guard inline"
    )
    for reader in ("readRouteSelection", "readMissingPlanFields",
                   "readPlanArrangements", "readPlanRules",
                   "readPlanSafety", "readObject"):
        start = reducer.index(f"function {reader}(")
        # Slice to the next top-level function boundary.
        nxt = re.search(r"\n(?:export )?function ", reducer[start + 1:])
        end = start + 1 + (nxt.start() if nxt else len(reducer) - start - 1)
        assert "asRecord(" in reducer[start:end], f"{reader} must delegate to asRecord"


def test_the_dialog_still_parks_on_dismiss_and_keeps_its_props_contract():
    """The dialog is ticket 06's to extend, but not to restructure: the read-only
    ``summary`` slot and the park/cancel split stay exactly as tickets 01/02 left
    them, or 关闭草稿 would start cancelling drafts."""
    dialog = _ts("src/components/PlanDraftDialog.tsx")

    for prop in ("open: boolean", "onDismiss: () => void", "onCancel?: () => void",
                 "onSubmit: (fields: V2PlanDraftFields) => void", "summary?: string"):
        assert prop in dialog, prop
    assert "if (event.key === 'Escape') onDismiss()" in dialog
    assert "plan_cancel" not in dialog
