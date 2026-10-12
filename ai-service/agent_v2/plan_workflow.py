"""Ticket 01 — plan-book entry routing and the minimal plan draft.

Plan-domain logic lives here so it stays a pure, constructor-free layer:
no HTTP types, no SQLite, no model call. The precedent for this style is
:mod:`route_compare.service` — deterministic functions over explicit
enums, where code owns the decisions and nothing is invented.

The contract this module owns:

* **Intent whitelist** (:data:`PLAN_INTENTS`). Entry routing is structured
  intent + deterministic rules only. An intent outside the whitelist is
  rejected by the request model before any streaming starts.
* **Deterministic classifier** (:func:`classify_plan_request`). Free text
  maps to exactly one of ``clear_plan_intent`` / ``ambiguous`` /
  ``not_plan_intent``. ``ambiguous`` never opens the plan flow — the
  server asks the user to confirm first.
* **Draft stages** (:class:`PlanDraftStage`) and the state machine
  (:func:`evaluate_plan_intent`) that rejects an intent which does not fit
  the current draft state with a stable machine-readable reason.
* **Minimal draft validation** (:class:`PlanDraftInput`) — destination or
  route, activity date, participant count.
* **Draft preview rendering** (:func:`render_minimal_draft`) — user
  arrangements only; anything the user has not given stays 待确认, and no
  route fact is ever produced here (docs/plan-pdf-format.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- Intent whitelist ------------------------------------------------------

PlanIntent = Literal["plan_create", "plan_confirm", "plan_cancel", "plan_regenerate"]

#: The only intents a client may send, derived from :data:`PlanIntent` so the
#: wire contract and the domain contract cannot drift apart.
#: ``api.chat_v2.ChatV2Request`` annotates ``intent`` with ``PlanIntent``:
#: an unknown value is rejected by Pydantic (422) before any streaming.
PLAN_INTENTS: frozenset[str] = frozenset(get_args(PlanIntent))


# ---- Draft stages ----------------------------------------------------------


class PlanDraftStage(str, Enum):
    """Lifecycle of a conversation's plan draft (spec AC8).

    The same values travel on the wire (``plan.draft_state`` payload) and
    are stored in ``plan_drafts.stage``, so the server and the frontend
    recognise the draft state from one vocabulary.

    ``awaiting_confirmation`` is the *ambiguous-intent* parking lot: the
    user said something planning-shaped, the server asked "要不要生成
    计划书？" and NOTHING else has happened. The dialog does not open from
    here; only a subsequent ``plan_confirm`` moves to ``awaiting_input``.
    """

    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_INPUT = "awaiting_input"
    MINIMAL_DRAFT = "minimal_draft"
    CANCELLED = "cancelled"


# ---- Minimal draft fields --------------------------------------------------

_DESTINATION_MAX = 120
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PlanDraftInput(BaseModel):
    """The three fields the entry dialog collects (spec AC6).

    Deliberately narrow: ``extra="forbid"`` means the form cannot smuggle
    a route fact, a price, or a model-config blob into the draft. Unknown
    arrangements stay absent — they are never filled in here
    (docs/plan-pdf-format.md: 未知安排留空，显示"待确认").

    ``participants`` is strict so ``"12"``/``True`` are refused rather
    than silently coerced; the dialog sends a JSON number.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    destination_or_route: str = Field(min_length=1, max_length=_DESTINATION_MAX)
    activity_date: str = Field(min_length=1, max_length=10)
    participants: int = Field(ge=1, le=10000, strict=True)

    @field_validator("activity_date")
    @classmethod
    def _activity_date_is_a_real_calendar_day(cls, value: str) -> str:
        """Exact ``YYYY-MM-DD`` only — no relative wording, no 2026-02-30."""
        if not _ISO_DATE_PATTERN.match(value):
            raise ValueError("activity_date must be YYYY-MM-DD")
        try:
            date.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - fromisoformat guards above
            raise ValueError("activity_date is not a real calendar day") from exc
        return value


# ---- Deterministic natural-language classifier -----------------------------

PlanIntentOutcome = Literal["clear_plan_intent", "ambiguous", "not_plan_intent"]

#: A plan book was named outright. These are unambiguous by themselves.
_PLAN_BOOK_NOUNS: tuple[str, ...] = (
    "计划书", "路书", "行程计划", "行程单", "行程表", "计划手册",
)
#: Producing something as a request, not asking a question about a route.
_PLAN_VERBS: tuple[str, ...] = ("生成", "制作", "制定", "做", "写", "出", "整理成")
#: Planning vocabulary that shows intent without asking for a document.
#: "路线" is deliberately absent: asking about a route is a question, not
#: a plan-book request, and must fall through to ordinary chat.
_PLANNING_HINTS: tuple[str, ...] = ("行程", "规划", "安排", "计划")
#: Aspiring-but-uncommitted phrasings ("想去…还没定").
_ASPIRATION_HINTS: tuple[str, ...] = ("想去", "打算去", "还没定", "还没决定", "考虑去")


@dataclass(frozen=True, slots=True)
class PlanIntentClassification:
    """Classifier result: the outcome plus the terms that produced it.

    ``evidence`` keeps the decision auditable — the rules are the whole
    story, so a reviewer can see exactly which term fired without
    re-running a model.
    """

    outcome: PlanIntentOutcome
    evidence: tuple[str, ...] = ()


def classify_plan_request(text: str) -> PlanIntentClassification:
    """Map free text to one of three outcomes with rules only (spec AC4).

    No model is consulted: ticket 06 forbids external models for plan
    judgement, and the entry path must be reproducible. The ordering is
    the safety property — an explicit plan-book ask wins, otherwise
    planning vocabulary parks the request in ``ambiguous`` (which must NOT
    open the flow), otherwise it is ordinary chat.
    """
    message = (text or "").strip()
    if not message:
        return PlanIntentClassification("not_plan_intent")

    nouns = tuple(term for term in _PLAN_BOOK_NOUNS if term in message)
    if nouns:
        return PlanIntentClassification("clear_plan_intent", nouns)

    verbs = tuple(term for term in _PLAN_VERBS if term in message)
    if verbs and "行程" in message:
        return PlanIntentClassification("clear_plan_intent", verbs + ("行程",))

    hints = tuple(term for term in _PLANNING_HINTS if term in message)
    aspirations = tuple(term for term in _ASPIRATION_HINTS if term in message)
    if hints or aspirations:
        return PlanIntentClassification("ambiguous", hints + aspirations)

    return PlanIntentClassification("not_plan_intent")


# ---- Draft state machine ---------------------------------------------------

#: Stable, machine-readable rejection reasons. The flow puts the value in
#: the terminal ``error`` payload's ``reason`` field so tickets 05-07 can
#: branch on it; never reword a member, only add. Precedent:
#: ``RouteSession.prepare_report`` returning ``{"status":"blocked","reason":…}``.
class PlanBlockReason:
    UNKNOWN_INTENT = "unknown_intent"
    PLAN_CONTEXT_REQUIRED = "plan_context_required"
    CONFIRM_WITHOUT_PENDING_CONFIRMATION = "confirm_without_pending_confirmation"
    REGENERATE_WITHOUT_DRAFT = "regenerate_without_draft"
    CANCEL_WITHOUT_ACTIVE_DRAFT = "cancel_without_active_draft"
    PLAN_INPUT_NOT_ALLOWED = "plan_input_not_allowed"
    INVALID_DRAFT_FIELDS = "invalid_draft_fields"


@dataclass(frozen=True, slots=True)
class PlanTransition:
    """Result of one intent against the current draft state.

    ``status`` is ``ok`` (with the resulting ``stage``) or ``blocked``
    (with the stable ``reason``). Nothing is mutated: the caller persists
    the returned stage, keeping this module free of storage.
    """

    status: Literal["ok", "blocked"]
    stage: PlanDraftStage | None = None
    reason: str | None = None
    draft_input: PlanDraftInput | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"


_INTENTS_THAT_COLLECT_FIELDS: frozenset[str] = frozenset({"plan_create", "plan_regenerate"})

#: User-facing text per stable rejection reason. The machine-readable
#: ``reason`` is what clients branch on; this is only what the user reads.
BLOCK_MESSAGE: dict[str, str] = {
    PlanBlockReason.UNKNOWN_INTENT: "无法识别的计划书操作，请重新发起。",
    PlanBlockReason.PLAN_CONTEXT_REQUIRED: "缺少会话上下文，暂时无法保存计划书草稿。",
    PlanBlockReason.CONFIRM_WITHOUT_PENDING_CONFIRMATION: (
        "当前没有待确认的计划书请求。需要生成计划书请直接点「生成行程」。"
    ),
    PlanBlockReason.REGENERATE_WITHOUT_DRAFT: "还没有可重新生成的草稿，请先创建计划书。",
    PlanBlockReason.CANCEL_WITHOUT_ACTIVE_DRAFT: "当前没有进行中的计划书草稿。",
    PlanBlockReason.PLAN_INPUT_NOT_ALLOWED: "该计划书操作不接受草稿字段。",
    PlanBlockReason.INVALID_DRAFT_FIELDS: (
        "目的地或路线、活动日期、参与人数校验未通过，请修正后重新提交。"
    ),
}


def evaluate_plan_intent(
    *,
    intent: str | None,
    current_stage: PlanDraftStage | None,
    plan_input: PlanDraftInput | dict | None,
    conversation_id: str | None,
) -> PlanTransition:
    """Decide what a structured intent may do, given the current stage.

    Pure and total: every path returns either the next stage or a stable
    reason, so the flow can never guess. Three rejections are the point of
    the exercise (spec AC5): an intent outside the whitelist, an intent
    with no conversation to key the draft on, and an intent that does not
    fit the stage the plan flow is in.
    """
    if intent not in PLAN_INTENTS:
        return PlanTransition("blocked", reason=PlanBlockReason.UNKNOWN_INTENT)

    if not (conversation_id or "").strip():
        # Drafts live on the conversation axis, not the run axis: without
        # a conversation there is nothing to persist or restore later.
        return PlanTransition("blocked", reason=PlanBlockReason.PLAN_CONTEXT_REQUIRED)

    if plan_input is not None and intent not in _INTENTS_THAT_COLLECT_FIELDS:
        return PlanTransition("blocked", reason=PlanBlockReason.PLAN_INPUT_NOT_ALLOWED)

    validated: PlanDraftInput | None = None
    if plan_input is not None:
        if isinstance(plan_input, dict):
            try:
                validated = PlanDraftInput.model_validate(plan_input)
            except ValueError:
                return PlanTransition("blocked", reason=PlanBlockReason.INVALID_DRAFT_FIELDS)
        elif isinstance(plan_input, PlanDraftInput):
            validated = plan_input
        else:
            return PlanTransition("blocked", reason=PlanBlockReason.INVALID_DRAFT_FIELDS)

    if intent == "plan_create":
        # Re-entry from any stage is legal: the dialog is idempotent and a
        # user may want to start the arrangements over.
        return PlanTransition(
            "ok",
            stage=PlanDraftStage.MINIMAL_DRAFT if validated else PlanDraftStage.AWAITING_INPUT,
            draft_input=validated,
        )

    if intent == "plan_confirm":
        if current_stage is not PlanDraftStage.AWAITING_CONFIRMATION:
            return PlanTransition(
                "blocked", reason=PlanBlockReason.CONFIRM_WITHOUT_PENDING_CONFIRMATION
            )
        return PlanTransition("ok", stage=PlanDraftStage.AWAITING_INPUT)

    if intent == "plan_regenerate":
        if current_stage is not PlanDraftStage.MINIMAL_DRAFT:
            return PlanTransition("blocked", reason=PlanBlockReason.REGENERATE_WITHOUT_DRAFT)
        # Ticket 01 keeps regenerate as "re-render the draft I already have";
        # ticket 09 owns content regeneration.
        return PlanTransition("ok", stage=PlanDraftStage.MINIMAL_DRAFT, draft_input=validated)

    # plan_cancel
    if current_stage is None or current_stage is PlanDraftStage.CANCELLED:
        return PlanTransition("blocked", reason=PlanBlockReason.CANCEL_WITHOUT_ACTIVE_DRAFT)
    return PlanTransition("ok", stage=PlanDraftStage.CANCELLED)


# ---- User-facing stage copy + minimal draft preview ------------------------

#: Unknown user arrangement — the field was never filled in. Unknown route
#: fact — no verified evidence exists. The distinction is the content
#: boundary in docs/plan-pdf-format.md; they are never interchangeable.
UNCONFIRMED = "待确认"
UNVERIFIED = "待核验"

#: What the flow says per stage. Deterministic strings, so the entry path
#: contains no model-generated claims (ticket 06 keeps it that way).
STAGE_MESSAGE: dict[PlanDraftStage, str] = {
    PlanDraftStage.AWAITING_CONFIRMATION: (
        "你可能想生成一份徒步计划书草稿。是否需要我现在打开填写弹窗？"
        "回复确认后我再开始，未确认前不会生成任何计划内容。"
    ),
    PlanDraftStage.AWAITING_INPUT: (
        "已打开计划书填写弹窗：请填写目的地或路线、活动日期和参与人数。"
    ),
    PlanDraftStage.CANCELLED: "已取消本次计划书草稿，需要时再点「生成行程」重新开始。",
}


def draft_summary(
    *,
    stage: PlanDraftStage,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
) -> str:
    """The draft summary for a stage and the fields stored with it (Ticket 02 AC3).

    One deterministic renderer for both consumers — the ``plan.draft_state``
    payload the restore UX reads back after a refresh, and the chat bubble the
    flow streams — so they cannot drift. Every stage must be covered here:
    a stage the summary cannot render is a crash on restore, not a silent
    miss (``MINIMAL_DRAFT`` is the one stage whose copy is the preview, which
    is why :data:`STAGE_MESSAGE` has no entry for it).

    Nothing is invented: the summary is a pure function of the stored fields,
    so editing a field and re-rendering is the same call.
    """
    if stage is PlanDraftStage.MINIMAL_DRAFT:
        return render_minimal_draft(
            destination_or_route=destination_or_route,
            activity_date=activity_date,
            participants=participants,
        )
    return STAGE_MESSAGE[stage]


def render_minimal_draft(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
) -> str:
    """Render the previewable minimal draft (spec AC1 "可预览的最小草稿").

    Only the three user-supplied arrangements are filled in. Everything the
    user has not given is 待确认 and every route fact is 待核验: the draft
    is explicitly a 草案, never a verified route plan, and this function
    cannot produce a distance, an ascent, a weather claim or a price even
    if asked (docs/plan-pdf-format.md 内容边界).
    """
    def arrangement(value: str) -> str:
        return value.strip() or UNCONFIRMED

    people = f"{participants} 人" if participants else UNCONFIRMED
    return "\n".join([
        "**最小计划草稿（草案，未经路线核验）**",
        f"- 目的地或路线：{arrangement(destination_or_route)}",
        f"- 活动日期：{arrangement(activity_date)}",
        f"- 参与人数：{people}",
        f"- 集合时间 / 集合地点：{UNCONFIRMED}",
        f"- 领队 / 交通 / 住宿：{UNCONFIRMED}",
        f"- 费用预算 / 应急联络 / 撤退方案：{UNCONFIRMED}",
        f"- 距离 / 爬升 / 天气 / 路况 / 风险：{UNVERIFIED}（需要服务器证据）",
        "以上只是用户填写的安排草案，不是已核验的路线计划。",
    ])

