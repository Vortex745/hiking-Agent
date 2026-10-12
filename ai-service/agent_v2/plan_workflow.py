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
* **Route selection reference** (:class:`RouteSelectionInput`) — a draft
  points at an existing route-comparison result by ``comparison_id`` +
  ``candidate_id`` and never stores a copy of its facts, so every render
  re-reads them through the comparison leaf that re-runs the deterministic
  gate (:mod:`agent_v2.route_tools`).
* **Draft preview rendering** (:func:`render_minimal_draft`) — user
  arrangements only; anything the user has not given stays 待确认, and no
  route fact is ever produced here (docs/plan-pdf-format.md).
* **Route basis rendering** (:func:`render_route_basis`) — the verified side
  of the same draft, in its own section and always with its source
  provenance. Facts the gate cannot certify stay absent, which renders as
  待核验: nothing is ever invented to fill a gap.
* **Draft type** (:data:`PlanType`) — ``route_plan`` only when a gate-passed
  route basis backs the draft, otherwise ``generic_draft``. The marker is
  derived from that evidence and rendered by :func:`draft_summary`, so an
  unverified draft is never presented as a formal route plan (spec 04 AC3/AC5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- Intent whitelist ------------------------------------------------------

PlanIntent = Literal[
    "plan_create",
    "plan_confirm",
    "plan_cancel",
    "plan_regenerate",
    "plan_select_route",
    # Ticket 04: attempt route research for the draft's own free-text
    # destination, and answer the "research found nothing verifiable"
    # question the two ways the user is allowed to answer it.
    "plan_research_route",
    "plan_generic_draft",
]

#: The only intents a client may send, derived from :data:`PlanIntent` so the
#: wire contract and the domain contract cannot drift apart.
#: ``api.chat_v2.ChatV2Request`` annotates ``intent`` with ``PlanIntent``:
#: an unknown value is rejected by Pydantic (422) before any streaming.
PLAN_INTENTS: frozenset[str] = frozenset(get_args(PlanIntent))


# ---- Draft type (ticket 04) ------------------------------------------------

PlanType = Literal["route_plan", "generic_draft"]

#: What a draft is, in the user's terms and in the plan data. Derived by the
#: flow from one question with a server-side answer: does the draft point at a
#: route whose facts still pass the deterministic gate?
#:
#: * ``route_plan`` — a gate-passed route basis exists, so 距离/爬升/天气… can
#:   be shown with provenance.
#: * ``generic_draft`` — no verified route basis. Every unknown stays unknown
#:   (待确认 for arrangements, 待核验 for facts) and the summary says outright
#:   that this is not a route plan (spec AC3/AC5).
#:
#: A client may only ever *ask* for ``generic_draft``: claiming ``route_plan``
#: is refused by :func:`evaluate_plan_intent`, because a plan type is a claim
#: about evidence the client does not hold.
PLAN_TYPES: frozenset[str] = frozenset(get_args(PlanType))


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
    #: Ticket 04 AC3/AC4: route research ran for the draft's free destination
    #: and nothing passed the deterministic gate. The draft is parked HERE,
    #: waiting for the user to choose between a generic draft and cancelling —
    #: insufficient material never produces a plan on its own. Like
    #: ``awaiting_confirmation`` this stage opens no dialog: it is answered by
    #: a structured intent.
    AWAITING_GENERIC_CHOICE = "awaiting_generic_choice"
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


# ---- Route selection reference (ticket 03) ---------------------------------

#: The caps mirror the producing contracts — ``route_tools.ReportParams`` for
#: ``comparison_id`` and ``route_compare.models.RouteCandidate`` for
#: ``candidate_id`` — so a reference the draft can hold is exactly a reference
#: a comparison result could have issued.
_COMPARISON_ID_MAX = 150
_CANDIDATE_ID_MAX = 80


class RouteSelectionInput(BaseModel):
    """A pointer into an existing route-comparison result (spec AC1).

    This is the whole of what a selection contributes to a draft: two ids.
    ``extra="forbid"`` is the invariant made executable — a client cannot
    attach a distance, an ascent, a weather string or a
    ``report_allowed`` flag alongside the reference, because verified facts
    are only ever read back through
    :meth:`agent_v2.route_tools.RouteSession.load_result`, which re-runs the
    deterministic gate on every use.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    comparison_id: str = Field(min_length=1, max_length=_COMPARISON_ID_MAX)
    candidate_id: str = Field(min_length=1, max_length=_CANDIDATE_ID_MAX)


@dataclass(frozen=True, slots=True)
class PlanRouteReference:
    """The stored half of a selection: the draft's pointer, nothing else.

    ``draft_id`` ties the reference to the plan draft it belongs to, so a
    draft that was replaced cannot be confused with the current one.
    """

    conversation_id: str
    draft_id: str
    comparison_id: str
    candidate_id: str


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
    # Ticket 03 — a route selection is its own structured input, with its own
    # stable reasons. Following the ``prepare_report`` precedent, a selection
    # the server cannot honour is a *reason*, never a best-effort draft that
    # quietly carries a made-up route.
    ROUTE_SELECTION_NOT_ALLOWED = "route_selection_not_allowed"
    INVALID_ROUTE_SELECTION = "invalid_route_selection"
    ROUTE_SELECTION_WITHOUT_DRAFT = "route_selection_without_draft"
    COMPARISON_NOT_FOUND = "comparison_not_found"
    ROUTE_NOT_SELECTABLE = "route_not_selectable"
    # Ticket 04 — research and the generic-draft fallback are structured
    # operations too: each refusal is a stable reason, and the fallback is
    # only ever reachable from the stage that asks for it.
    RESEARCH_WITHOUT_DRAFT = "route_research_without_draft"
    RESEARCH_UNAVAILABLE = "route_research_unavailable"
    PLAN_TYPE_NOT_ALLOWED = "plan_type_not_allowed"
    INVALID_PLAN_TYPE = "invalid_plan_type"
    GENERIC_DRAFT_WITHOUT_PENDING = "generic_draft_without_pending"


@dataclass(frozen=True, slots=True)
class PlanTransition:
    """Result of one intent against the current draft state.

    ``status`` is ``ok`` (with the resulting ``stage``) or ``blocked``
    (with the stable ``reason``). Nothing is mutated: the caller persists
    the returned stage, keeping this module free of storage.

    ``fallback_stage`` is ticket 04's only concession to something outside
    this layer: a research turn's outcome is a storage-and-gate question the
    pure rules cannot see, so the rules name BOTH stages the turn may land on
    and the flow picks between them with the leaf's own answer. The decision
    stays here — the flow never invents a stage.
    """

    status: Literal["ok", "blocked"]
    stage: PlanDraftStage | None = None
    reason: str | None = None
    draft_input: PlanDraftInput | None = None
    route_selection: RouteSelectionInput | None = None
    fallback_stage: PlanDraftStage | None = None
    plan_type: str | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"


_INTENTS_THAT_COLLECT_FIELDS: frozenset[str] = frozenset({"plan_create", "plan_regenerate"})
#: ``plan_select_route`` is the only intent that may carry a route reference,
#: and the reference is only meaningful on a draft the user has already
#: submitted — the stage that owns the arrangements the facts join.
_INTENTS_THAT_SELECT_ROUTES: frozenset[str] = frozenset({"plan_select_route"})
#: ``plan_generic_draft`` is the only intent that may declare a draft type,
#: and it declares exactly one: the client cannot claim ``route_plan``.
_INTENTS_THAT_DECLARE_PLAN_TYPE: frozenset[str] = frozenset({"plan_generic_draft"})

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
    PlanBlockReason.ROUTE_SELECTION_NOT_ALLOWED: "该计划书操作不接受路线选择。",
    PlanBlockReason.INVALID_ROUTE_SELECTION: "路线选择参数无效，请从比较结果中重新选择。",
    PlanBlockReason.ROUTE_SELECTION_WITHOUT_DRAFT: (
        "还没有可带入路线事实的草稿，请先填写目的地或路线、活动日期和参与人数。"
    ),
    PlanBlockReason.COMPARISON_NOT_FOUND: (
        "没有找到可复用的路线比较结果，请先完成一次路线比较。草稿保留，路线事实仍为待核验。"
    ),
    PlanBlockReason.ROUTE_NOT_SELECTABLE: (
        "该路线不在可比选的推荐或备选范围内，或资料未通过核验。草稿保留，路线事实仍为待核验。"
    ),
    PlanBlockReason.RESEARCH_WITHOUT_DRAFT: (
        "还没有可研究的目的地或路线，请先填写并提交这三项安排。"
    ),
    PlanBlockReason.RESEARCH_UNAVAILABLE: (
        "当前没有可用的路线资料来源，本次未能开展研究。草稿保持原状，路线事实仍为待核验。"
    ),
    PlanBlockReason.PLAN_TYPE_NOT_ALLOWED: "该计划书操作不接受草案类型标记。",
    PlanBlockReason.INVALID_PLAN_TYPE: (
        "只能显式选择「通用草案」：正式路线计划的类型由服务器依据已核验资料判定。"
    ),
    PlanBlockReason.GENERIC_DRAFT_WITHOUT_PENDING: (
        "当前没有等待决定的资料不足情形，无法直接生成通用草案。"
    ),
}


def evaluate_plan_intent(
    *,
    intent: str | None,
    current_stage: PlanDraftStage | None,
    plan_input: PlanDraftInput | dict | None,
    conversation_id: str | None,
    route_selection: RouteSelectionInput | dict | None = None,
    plan_type: str | None = None,
) -> PlanTransition:
    """Decide what a structured intent may do, given the current stage.

    Pure and total: every path returns either the next stage or a stable
    reason, so the flow can never guess. Three rejections are the point of
    the exercise (spec AC5): an intent outside the whitelist, an intent
    with no conversation to key the draft on, and an intent that does not
    fit the stage the plan flow is in. Ticket 03 adds a fourth structured
    input, validated the same way: a route selection is accepted only by the
    intent that means it, and only on a draft that already holds arrangements.
    """
    if intent not in PLAN_INTENTS:
        return PlanTransition("blocked", reason=PlanBlockReason.UNKNOWN_INTENT)

    if not (conversation_id or "").strip():
        # Drafts live on the conversation axis, not the run axis: without
        # a conversation there is nothing to persist or restore later.
        return PlanTransition("blocked", reason=PlanBlockReason.PLAN_CONTEXT_REQUIRED)

    if plan_input is not None and intent not in _INTENTS_THAT_COLLECT_FIELDS:
        return PlanTransition("blocked", reason=PlanBlockReason.PLAN_INPUT_NOT_ALLOWED)

    if route_selection is not None and intent not in _INTENTS_THAT_SELECT_ROUTES:
        return PlanTransition("blocked", reason=PlanBlockReason.ROUTE_SELECTION_NOT_ALLOWED)

    if plan_type is not None and intent not in _INTENTS_THAT_DECLARE_PLAN_TYPE:
        return PlanTransition("blocked", reason=PlanBlockReason.PLAN_TYPE_NOT_ALLOWED)

    if intent in _INTENTS_THAT_DECLARE_PLAN_TYPE and plan_type != "generic_draft":
        # AC5 in its strictest form: a draft is never *declared* a route plan.
        # Only the server, reading a gate-passed basis, can say that.
        return PlanTransition("blocked", reason=PlanBlockReason.INVALID_PLAN_TYPE)

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

    validated_selection: RouteSelectionInput | None = None
    if route_selection is not None:
        if isinstance(route_selection, dict):
            try:
                validated_selection = RouteSelectionInput.model_validate(route_selection)
            except ValueError:
                return PlanTransition("blocked", reason=PlanBlockReason.INVALID_ROUTE_SELECTION)
        elif isinstance(route_selection, RouteSelectionInput):
            validated_selection = route_selection
        else:
            return PlanTransition("blocked", reason=PlanBlockReason.INVALID_ROUTE_SELECTION)

    if intent == "plan_select_route":
        # AC1/AC3: verified route facts join a draft that already holds the
        # user's arrangements. Selecting into ``awaiting_input`` (nothing
        # submitted yet) or into a cancelled draft would put a route card in
        # front of an empty form, so the stage gate refuses it with a stable
        # reason and the flow stores nothing. Whether the referenced comparison
        # exists, and whether the candidate still passes the deterministic
        # gate, are storage-and-gate questions this pure layer cannot see: the
        # flow resolves them through
        # :meth:`agent_v2.route_tools.RouteSession.select_candidate`.
        if validated_selection is None:
            return PlanTransition("blocked", reason=PlanBlockReason.INVALID_ROUTE_SELECTION)
        if current_stage is not PlanDraftStage.MINIMAL_DRAFT:
            return PlanTransition("blocked", reason=PlanBlockReason.ROUTE_SELECTION_WITHOUT_DRAFT)
        # A selection does not move the draft stage: the reference rides on the
        # draft that ticket 05's summary confirmation reads (AC5).
        return PlanTransition(
            "ok", stage=PlanDraftStage.MINIMAL_DRAFT, route_selection=validated_selection
        )

    if intent == "plan_research_route":
        # AC1: research reads the draft's own free-text destination, so it only
        # makes sense once the form was submitted — ``minimal_draft`` is the
        # stage that guarantees a non-empty 目的地或路线. The turn does not move
        # the stage by itself: ``stage`` is what a gate-passed research result
        # leaves it at, ``fallback_stage`` where insufficient material parks it
        # (AC3 — parked, never promoted). Whether the material was enough is
        # the leaf's answer, not this layer's.
        if current_stage is not PlanDraftStage.MINIMAL_DRAFT:
            return PlanTransition("blocked", reason=PlanBlockReason.RESEARCH_WITHOUT_DRAFT)
        return PlanTransition(
            "ok",
            stage=PlanDraftStage.MINIMAL_DRAFT,
            fallback_stage=PlanDraftStage.AWAITING_GENERIC_CHOICE,
        )

    if intent == "plan_generic_draft":
        # AC4: the way out of a parked research turn. Chosen by the user, on
        # purpose — the server does not fall through to a draft on its own.
        if current_stage is not PlanDraftStage.AWAITING_GENERIC_CHOICE:
            return PlanTransition("blocked", reason=PlanBlockReason.GENERIC_DRAFT_WITHOUT_PENDING)
        return PlanTransition("ok", stage=PlanDraftStage.MINIMAL_DRAFT, plan_type="generic_draft")

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
    PlanDraftStage.AWAITING_GENERIC_CHOICE: (
        "已针对该目的地/路线发起路线研究，但没有资料通过确定性核验（资料不足、时效无法确认或存在风险），"
        "因此不会据此生成正式路线计划。你可以继续生成「通用草案」，或取消本次计划书；"
        "未通过核验的路线事实仍为待核验。"
    ),
}


def draft_summary(
    *,
    stage: PlanDraftStage,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    route: RouteBasis | None = None,
) -> str:
    """The draft summary for a stage and the fields stored with it (Ticket 02 AC3).

    One deterministic renderer for both consumers — the ``plan.draft_state``
    payload the restore UX reads back after a refresh, and the chat bubble the
    flow streams — so they cannot drift. Every stage must be covered here:
    a stage the summary cannot render is a crash on restore, not a silent
    miss (``MINIMAL_DRAFT`` is the one stage whose copy is the preview, which
    is why :data:`STAGE_MESSAGE` has no entry for it).

    ``route`` is ticket 03's addition: the verified route basis the draft
    points at, re-read from the comparison result by the caller. It is
    rendered as its own section under its own heading so the user's
    arrangements and the server's verified facts stay visibly separate (AC3),
    and it is ``None`` whenever no usable comparison exists — which is what
    keeps AC4 honest, because this function has no way to invent a number.

    Ticket 04 reads that ``None`` as exactly one thing: a ``generic_draft``.
    The "this is not a route plan" disclaimer therefore lives in this renderer
    rather than in any component, so the streamed text, the ``plan.draft_state``
    payload and a draft restored after a refresh cannot disagree about what
    kind of draft the user is looking at (AC5) — and a draft with no basis
    cannot be dressed up as a formal route plan even by a caller that asked for
    that label, because the marker is derived from ``route``, never passed in.

    Nothing is invented: the summary is a pure function of the stored fields
    plus the caller's re-read basis, so editing a field and re-rendering is
    the same call.
    """
    if stage is PlanDraftStage.MINIMAL_DRAFT:
        base = render_minimal_draft(
            destination_or_route=destination_or_route,
            activity_date=activity_date,
            participants=participants,
            route_attached=route is not None,
        )
    else:
        base = STAGE_MESSAGE[stage]
    if route is None:
        return base
    return f"{base}\n\n{render_route_basis(route)}"


def render_minimal_draft(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    route_attached: bool = False,
) -> str:
    """Render the previewable minimal draft (spec AC1 "可预览的最小草稿").

    Only the three user-supplied arrangements are filled in. Everything the
    user has not given is 待确认 and every route fact is 待核验: the draft
    is explicitly a 草案, never a verified route plan, and this function
    cannot produce a distance, an ascent, a weather claim or a price even
    if asked (docs/plan-pdf-format.md 内容边界).

    ``route_attached`` is ticket 03's only exception, and it is a pointer
    rather than a value: when the caller has re-read a certified route basis
    it renders that basis as a separate section below, so the blanket
    "facts 待核验" line would otherwise contradict it. The arrangements stay
    待确认 and the closing disclaimer still says the draft is unreviewed —
    ``route_attached`` never promotes a 草案 to a 路书.

    Ticket 04 AC5: that flag is also what the draft's *type* is marked from.
    ``route_attached=False`` IS the ``generic_draft`` case — no verified route
    basis — so the label is derived from the evidence the caller re-read, never
    from a claim, and the same line appears in the streamed text, the payload's
    summary and a restored draft because this is the only place it is written.
    """
    def arrangement(value: str) -> str:
        return value.strip() or UNCONFIRMED

    people = f"{participants} 人" if participants else UNCONFIRMED
    facts_line = (
        f"- 距离 / 爬升 / 天气 / 路况 / 风险：见下方「路线核验事实」（未列出者仍为{UNVERIFIED}）"
        if route_attached
        else f"- 距离 / 爬升 / 天气 / 路况 / 风险：{UNVERIFIED}（需要服务器证据）"
    )
    closing = (
        "以上用户安排仍为待确认；路线事实由服务器读取并保留来源，两者分开保存，"
        "整份草稿尚未通过计划审核。"
        if route_attached
        else "以上只是用户填写的安排草案，不是已核验的路线计划。"
    )
    return "\n".join([
        ("**最小计划草稿（路线草案 · 依据已核验路线事实）**" if route_attached
         else "**最小计划草稿（通用草案 · 无已核验路线依据）**"),
        ("- 草案类型：route_plan（正式路线计划草案，仍需计划审核）" if route_attached
         else "- 草案类型：generic_draft（通用草案，不是路线计划）"),
        f"- 目的地或路线：{arrangement(destination_or_route)}",
        f"- 活动日期：{arrangement(activity_date)}",
        f"- 参与人数：{people}",
        f"- 集合时间 / 集合地点：{UNCONFIRMED}",
        f"- 领队 / 交通 / 住宿：{UNCONFIRMED}",
        f"- 费用预算 / 应急联络 / 撤退方案：{UNCONFIRMED}",
        facts_line,
        closing,
    ])


# ---- Verified route basis (ticket 03) --------------------------------------

#: The dimensions a route fact may answer, in display order, with the labels
#: the route-comparison card uses — one vocabulary for the card and the draft.
FACT_DIMENSIONS: tuple[tuple[str, str, str], ...] = (
    ("distance_km", "距离", "km"),
    ("ascent_m", "爬升", "m"),
    ("duration_hours", "耗时", "小时"),
    ("weather", "天气", ""),
    ("map", "地图", ""),
    ("trail_status", "开放状态", ""),
    ("risk", "风险资料", ""),
)

#: Same value wording the comparison card renders, so the draft and the card
#: the user selected from say the same thing about the same fact.
_FACT_VALUE_LABELS: dict[str, str] = {
    "open": "开放", "closed": "封闭", "low": "低风险",
    "moderate": "中等风险", "high": "高风险",
}


@dataclass(frozen=True, slots=True)
class RouteFactLine:
    """One verified fact and the provenance that makes it verified (AC2).

    ``source_id`` + ``source_field`` is the candidate's own claim about which
    source backs it, and ``source_tool`` / ``observed_at`` are that source's
    origin and timestamp. The line exists only for a fact the gate supported:
    an unsupported fact is absent from ``RouteBasis.facts`` and renders
    :data:`UNVERIFIED`, never a value with an empty provenance slot.
    """

    field: str
    label: str
    display_value: str
    source_id: str
    source_field: str
    source_tool: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class RouteBasis:
    """What a draft's route reference resolved to, right now (AC1/AC5).

    Built fresh from :class:`agent_v2.route_tools.RouteSession.load_result` on
    every render, so it is a *view* and not a second copy of the comparison:
    the stored draft keeps only the two ids.
    """

    comparison_id: str
    candidate_id: str
    candidate_name: str
    role: Literal["recommended", "alternative"]
    data_mode: str
    facts: tuple[RouteFactLine, ...] = ()
    missing_labels: tuple[str, ...] = ()


def format_fact_value(field: str, value: str | float) -> str:
    """Display a verified value without rounding, re-unitting or re-deriving it."""
    if isinstance(value, bool):  # bool is an int subclass; never a route fact
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:g}"
    text = str(value).strip()
    return _FACT_VALUE_LABELS.get(text.lower(), text)


def render_route_basis(route: RouteBasis) -> str:
    """Render the verified half of the draft, with provenance (AC2/AC3/AC4).

    A separate section under its own heading: these lines are the server's
    evidence, not the user's arrangements, and the two must never read as one
    list. Facts the gate did not certify are listed as 待核验 — this function
    has no parameter that accepts a value, so it cannot fabricate one.
    """
    lines = [
        "**路线核验事实（来自已完成的路线比较，非用户填写）**",
        f"- 已选路线：{route.candidate_name}（"
        f"{'推荐路线' if route.role == 'recommended' else '备选路线'}）",
        f"- 引用：comparison_id={route.comparison_id} · candidate_id={route.candidate_id}",
    ]
    if route.data_mode == "offline":
        lines.append("- 资料模式：离线演示 · 合成数据，不可用于真实出行。")
    by_field = {line.field: line for line in route.facts}
    for field, label, unit in FACT_DIMENSIONS:
        line = by_field.get(field)
        if line is None:
            lines.append(f"- {label}：{UNVERIFIED}（无服务器证据）")
            continue
        suffix = f" {unit}" if unit else ""
        lines.append(
            f"- {label}：{line.display_value}{suffix}"
            f"（来源 [{line.source_id}] {line.source_tool} · "
            f"字段 {line.source_field} · 观测 {line.observed_at}）"
        )
    lines.append("路线事实由服务器复核，用户安排仍为待确认，两者分开保存。")
    return "\n".join(lines)

