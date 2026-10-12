"""Ticket 01 — plan-book entry: the deterministic plan domain layer.

Seam under test: :mod:`agent_v2.plan_workflow` — a pure, constructor-free
module (the ``route_compare/service.py`` style: explicit enums, no
invented facts). Every function here is called with plain data and
returns plain data, so these tests need no SQLite, no ASGI and no model.

Acceptance criteria proven here:

* AC4 自然语言只经过确定性规则归一化 — :func:`classify_plan_request`
  returns one of three outcomes and never a fourth; nothing is inferred.
* AC6 目的地或路线、活动日期、参与人数完成最小字段校验 —
  :class:`PlanDraftInput` accepts a well-formed draft and rejects
  malformed values.
* AC8 服务端能识别计划书草稿状态 — :class:`PlanDraftStage` plus the state
  machine in :func:`evaluate_plan_intent`, which rejects an intent that
  does not fit the current stage with a stable reason.
"""

from __future__ import annotations

import pytest

from agent_v2.plan_workflow import (
    PlanDraftInput,
    PlanDraftStage,
    classify_plan_request,
    evaluate_plan_intent,
)


def _valid_payload() -> dict:
    return {
        "destination_or_route": "四姑娘山长坪沟",
        "activity_date": "2026-11-08",
        "participants": 12,
    }


def test_minimal_draft_accepts_the_three_collected_fields():
    """AC6: destination/route + activity date + participant count validate."""
    draft = PlanDraftInput.model_validate(_valid_payload())

    assert draft.destination_or_route == "四姑娘山长坪沟"
    assert draft.activity_date == "2026-11-08"
    assert draft.participants == 12


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("destination_or_route", ""),
        ("destination_or_route", "   "),
        ("destination_or_route", "香" * 121),
        ("activity_date", "2026-11-8"),
        ("activity_date", "11/08/2026"),
        ("activity_date", "2026-02-30"),
        ("activity_date", "下周六"),
        ("participants", 0),
        ("participants", 10001),
        ("participants", "twelve"),
    ],
)
def test_minimal_draft_rejects_each_malformed_field(field, bad_value):
    """AC6: every collected field is validated, not merely stored."""
    payload = _valid_payload() | {field: bad_value}

    with pytest.raises(ValueError):
        PlanDraftInput.model_validate(payload)


def test_minimal_draft_rejects_fields_outside_the_contract():
    """AC6: the draft carries only the three collected fields.

    A client cannot smuggle a "verified" route fact — or anything else —
    through the plan form (content boundary in docs/plan-pdf-format.md).
    """
    with pytest.raises(ValueError):
        PlanDraftInput.model_validate(_valid_payload() | {"distance_km": 12})


def test_classifier_returns_three_outcomes_only():
    """AC4: deterministic rules classify free text into exactly three states."""
    assert classify_plan_request("帮我生成一份徒步计划书").outcome == "clear_plan_intent"
    assert classify_plan_request("周末想去爬山，还没定").outcome == "ambiguous"
    assert classify_plan_request("今天天气怎么样").outcome == "not_plan_intent"

    outcomes = {classify_plan_request(text).outcome for text in (
        "生成行程", "把我这个月的开销整理一下", "路线", "", "帮我规划一下行程并导出计划书",
    )}
    assert outcomes <= {"clear_plan_intent", "ambiguous", "not_plan_intent"}


def test_classifier_does_not_hijack_ordinary_chat():
    """AC4: planning-adjacent chatter is not a plan-book request."""
    for text in (
        "这条路线爬升多少？",
        "帮我查一下北京的天气",
        "推荐一条附近的徒步路线",
        "把上面的装备清单整理成表格",
    ):
        assert classify_plan_request(text).outcome == "not_plan_intent", text


def test_classifier_requires_an_explicit_plan_book_request():
    """AC4: only an unambiguous plan-book ask opens the flow directly."""
    for text in (
        "生成行程",
        "帮我们做一份行程计划",
        "生成这次活动的计划书",
        "给这次徒步做份路书",
    ):
        assert classify_plan_request(text).outcome == "clear_plan_intent", text


def test_draft_stage_values_are_the_shared_vocabulary():
    """AC8: the stage enum is the vocabulary both sides speak.

    Ticket 04 added ``awaiting_generic_choice`` — the parked answer to a
    research turn whose material did not pass the gate. Ticket 05 added
    ``awaiting_first_review`` (the missing-field decision made, first summary
    not yet confirmed) and ``first_reviewed`` (the user confirmed the
    arrangements they supplied). Members are only ever ADDED here, never
    renamed, and every added member has to show up in the store, the wire
    union and the summary renderer at once (see ``test_plan_draft_restore.py``
    for the pinned cross-check).
    """
    assert {stage.value for stage in PlanDraftStage} == {
        "awaiting_confirmation",
        "awaiting_input",
        "minimal_draft",
        "awaiting_generic_choice",
        "awaiting_first_review",
        "first_reviewed",
        "cancelled",
    }


# ---- State machine ---------------------------------------------------------

STAGES = PlanDraftStage


def _evaluate(intent, stage=None, *, plan_input=None, conversation_id="conv-01"):
    return evaluate_plan_intent(
        intent=intent,
        current_stage=stage,
        plan_input=plan_input,
        conversation_id=conversation_id,
    )


def test_plan_create_opens_the_form_when_no_fields_have_been_collected():
    """AC1: an explicit create intent moves the flow to the dialog stage."""
    transition = _evaluate("plan_create")

    assert transition.status == "ok"
    assert transition.stage is STAGES.AWAITING_INPUT


def test_plan_create_with_valid_fields_lands_on_a_minimal_draft():
    """AC6 → AC8: a submitted form becomes a previewable minimal draft."""
    transition = _evaluate("plan_create", plan_input=PlanDraftInput.model_validate(_valid_payload()))

    assert transition.status == "ok"
    assert transition.stage is STAGES.MINIMAL_DRAFT
    assert transition.draft_input is not None


@pytest.mark.parametrize(
    "current_stage",
    [None, STAGES.AWAITING_INPUT, STAGES.MINIMAL_DRAFT, STAGES.CANCELLED],
)
def test_confirm_is_rejected_without_a_pending_confirmation(current_stage):
    """AC5: ``plan_confirm`` only continues an ambiguous request."""
    transition = _evaluate("plan_confirm", current_stage)

    assert transition.status == "blocked"
    assert transition.reason == "confirm_without_pending_confirmation"


def test_confirm_advances_a_pending_confirmation_to_the_form():
    transition = _evaluate("plan_confirm", STAGES.AWAITING_CONFIRMATION)

    assert transition.status == "ok"
    assert transition.stage is STAGES.AWAITING_INPUT


@pytest.mark.parametrize(
    "current_stage",
    [None, STAGES.AWAITING_CONFIRMATION, STAGES.AWAITING_INPUT, STAGES.CANCELLED],
)
def test_regenerate_is_rejected_before_a_draft_exists(current_stage):
    """AC5: there is nothing to regenerate until a draft exists."""
    transition = _evaluate("plan_regenerate", current_stage)

    assert transition.status == "blocked"
    assert transition.reason == "regenerate_without_draft"


def test_regenerate_reopens_an_existing_minimal_draft():
    transition = _evaluate("plan_regenerate", STAGES.MINIMAL_DRAFT)

    assert transition.status == "ok"
    assert transition.stage is STAGES.MINIMAL_DRAFT


@pytest.mark.parametrize(
    "current_stage",
    [STAGES.AWAITING_CONFIRMATION, STAGES.AWAITING_INPUT, STAGES.MINIMAL_DRAFT],
)
def test_cancel_closes_whatever_plan_state_is_open(current_stage):
    transition = _evaluate("plan_cancel", current_stage)

    assert transition.status == "ok"
    assert transition.stage is STAGES.CANCELLED


def test_cancel_is_rejected_when_no_plan_flow_is_open():
    """AC5: 与当前计划状态不匹配的 intent 带稳定 reason 被拒绝."""
    transition = _evaluate("plan_cancel", None)

    assert transition.status == "blocked"
    assert transition.reason == "cancel_without_active_draft"


def test_plan_state_needs_a_conversation_to_key_the_draft_on():
    """AC5: 缺少必要上下文的 intent 被拒绝."""
    for intent in ("plan_create", "plan_confirm", "plan_cancel", "plan_regenerate"):
        transition = _evaluate(intent, None, conversation_id=None)
        assert transition.status == "blocked", intent
        assert transition.reason == "plan_context_required", intent


def test_intent_outside_the_whitelist_is_rejected_with_a_stable_reason():
    """AC5: the flow defends in depth even if a caller bypassed the model."""
    transition = _evaluate("plan_teleport")

    assert transition.status == "blocked"
    assert transition.reason == "unknown_intent"


def test_structured_fields_are_only_accepted_with_a_field_collecting_intent():
    """AC5: ``plan_cancel`` must not silently carry draft data."""
    transition = _evaluate(
        "plan_cancel", STAGES.MINIMAL_DRAFT, plan_input=_valid_payload()
    )

    assert transition.status == "blocked"
    assert transition.reason == "plan_input_not_allowed"


def test_unvalidated_draft_payload_is_rejected_as_invalid_fields():
    """AC6: the flow never trusts a dict that skipped field validation."""
    transition = _evaluate("plan_create", plan_input={"destination_or_route": "香" * 500})

    assert transition.status == "blocked"
    assert transition.reason == "invalid_draft_fields"


# ---- Draft persistence (real temp SQLite, no ASGI) -------------------------

import sqlite3  # noqa: E402

from agent_v2.persistence import CheckpointsStore, ControlStore  # noqa: E402
from agent_v2.runs import RunRepository  # noqa: E402


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def test_conversation_has_no_draft_until_the_plan_flow_opens(tmp_path):
    """AC8: absence of a draft is a queryable state, not an error."""
    repo = _repo(tmp_path)
    try:
        assert repo.get_plan_draft("conv-empty") is None
    finally:
        repo.close()


def test_minimal_draft_round_trips_its_validated_fields(tmp_path):
    """AC8: the draft the user submitted is readable back on a later run."""
    repo = _repo(tmp_path)
    draft = PlanDraftInput.model_validate(_valid_payload())
    try:
        saved = repo.save_plan_draft(
            conversation_id="conv-01",
            stage=PlanDraftStage.MINIMAL_DRAFT.value,
            destination_or_route=draft.destination_or_route,
            activity_date=draft.activity_date,
            participants=draft.participants,
            updated_by_run_id="run-1",
        )
        loaded = repo.get_plan_draft("conv-01")

        assert loaded is not None
        assert loaded.draft_id == saved.draft_id
        assert loaded.stage == "minimal_draft"
        assert (loaded.destination_or_route, loaded.activity_date, loaded.participants) == (
            "四姑娘山长坪沟",
            "2026-11-08",
            12,
        )
    finally:
        repo.close()


def test_plan_draft_is_upserted_per_conversation_not_per_run(tmp_path):
    """AC8: one draft per conversation; a later stage overwrites in place."""
    repo = _repo(tmp_path)
    try:
        first = repo.save_plan_draft(
            conversation_id="conv-01", stage=PlanDraftStage.AWAITING_INPUT.value
        )
        second = repo.save_plan_draft(
            conversation_id="conv-01",
            stage=PlanDraftStage.MINIMAL_DRAFT.value,
            destination_or_route="武功山",
            activity_date="2026-12-01",
            participants=8,
        )

        assert second.draft_id == first.draft_id
        assert second.stage == "minimal_draft"
        assert repo.get_plan_draft("conv-01").destination_or_route == "武功山"
    finally:
        repo.close()


@pytest.mark.parametrize("stage", list(PlanDraftStage))
def test_every_stage_in_the_shared_vocabulary_is_storable(tmp_path, stage):
    """AC8: the enum and the storage vocabulary cannot drift apart."""
    repo = _repo(tmp_path)
    try:
        assert repo.save_plan_draft(
            conversation_id="conv-01", stage=stage.value
        ).stage == stage.value
    finally:
        repo.close()


def test_unknown_stage_is_rejected_at_the_store_boundary(tmp_path):
    repo = _repo(tmp_path)
    try:
        with pytest.raises(ValueError):
            repo.save_plan_draft(conversation_id="conv-01", stage="fully_formed")
    finally:
        repo.close()


def test_plan_draft_is_not_run_scoped(tmp_path):
    """The draft must survive run cleanup, so it has no FK to runs."""
    repo = _repo(tmp_path)
    try:
        repo.save_plan_draft(conversation_id="conv-01", stage=PlanDraftStage.MINIMAL_DRAFT.value)
        with sqlite3.connect(str(tmp_path / "control.sqlite")) as conn:
            fk_rows = conn.execute("PRAGMA foreign_key_list('plan_drafts')").fetchall()
        assert fk_rows == []
    finally:
        repo.close()


