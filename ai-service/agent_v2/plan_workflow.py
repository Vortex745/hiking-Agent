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
  route fact is ever produced here (docs/plan-pdf-format.md). The same
  function renders ticket 05's first summary, split into what the user
  supplied and the 缺失字段清单 (:func:`missing_plan_fields`) that is still
  blank — the first confirmation covers the supplied half only, and the
  unfilled half never gets a value from the system.
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
from typing import Any, Literal, get_args

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
    # Ticket 05: the two answers to "these fields are still missing" — keep
    # collecting, or generate the draft now — plus the confirmation of the
    # first summary that only ever covers what the user supplied.
    "plan_first_review",
    "plan_more_input",
    "plan_confirm_arrangements",
    # Ticket 07: the SECOND summary — 「查看最终摘要」 shows the verified route
    # results, the user's own arrangements, the missing items and the plan type;
    # 「确认最终摘要」 is the explicit confirmation that opens PDF approval. Both
    # are plain conversational turns and both carry no fields at all.
    "plan_final_review",
    "plan_confirm_final",
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
    #: Ticket 05 AC1/AC2: the user has seen which fields are still missing and
    #: chose 直接生成草案, so the FIRST summary is on screen and awaiting their
    #: confirmation. Like the two parked stages above it, this stage opens no
    #: dialog — it is answered by a structured intent (继续补充 / 确认摘要).
    AWAITING_FIRST_REVIEW = "awaiting_first_review"
    #: Ticket 05 AC3/AC5: the first summary was confirmed. The record says the
    #: user confirmed *the arrangements they themselves supplied* — it is not a
    #: claim about route facts, about completeness, or about any field the
    #: system left 待确认 (that would be invention, docs/plan-pdf-format.md).
    FIRST_REVIEWED = "first_reviewed"
    #: Ticket 07 AC1/AC4: the FIRST summary was confirmed and the user asked for
    #: the second one, so the final review is on screen and awaiting an explicit
    #: confirmation. It is the only stage 进入 PDF 审批 may be reached from — and
    #: like every review stage it opens no dialog: it is answered by a structured
    #: intent (查看最终摘要 / 确认最终摘要).
    AWAITING_FINAL_REVIEW = "awaiting_final_review"
    #: Ticket 07 AC4: the user explicitly confirmed the SECOND summary. This is
    #: the record that the confirmation happened, on which draft, on which turn —
    #: and it is the door PDF approval (ticket 09) walks through, never a claim
    #: about safety clearance, completeness, or report eligibility.
    FINAL_REVIEWED = "final_reviewed"
    #: Ticket 07 AC2: the deterministic gate refused the route outright (no_go:
    #: 严重天气、封闭路线、高风险). A hard stop, NOT a choice: ticket 04's
    #: 资料不足 branch offers 「继续生成通用草案」 and this one does not, because
    #: laundering a dangerous route into any advancing plan is the failure this
    #: stage exists to prevent. The only exits are 取消 and a re-submitted form,
    #: which the gate judges again from scratch.
    BLOCKED_SAFETY = "blocked_safety"
    CANCELLED = "cancelled"


# ---- Minimal draft fields --------------------------------------------------

_DESTINATION_MAX = 120
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HH_MM_PATTERN = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
_HEADCOUNT_PATTERN = re.compile(r"^\d{1,5} 人$")
_PHONE_DIGITS_PATTERN = re.compile(r"\d{7,}")
_AMOUNT_PATTERN = re.compile(r"\d")


class PlanArrangementInput(BaseModel):
    """The supplementary arrangements the dialog can collect (ticket 06 AC1).

    Completeness rules are only worth writing over fields that can genuinely be
    filled in, so the eight arrangements ticket 05 listed as permanently 待确认
    now have real input and real storage. Every key here is a key of
    :data:`ARRANGEMENT_FIELDS` (and, but for ``budget``, a field of
    :class:`route_compare.models.PlanDetails` with the same length cap), so this
    is the arrangement record and not a second vocabulary.

    ``extra="forbid"`` is the AC6 boundary in the shape of the input: a distance,
    an ascent, a weather claim, a ``report_allowed`` flag or a draft type cannot
    travel inside the user's own arrangements, because those are the server's
    evidence and never the user's claim. Every field defaults to the empty
    string, so "not filled in" stays "not filled in" — nothing is prefilled and
    nothing is derived (docs/plan-pdf-format.md: 未知安排留空，显示"待确认").

    ``budget`` is the user's own free-text 费用预算 note. The itemised
    ``PlanDetails.budget`` entries are ticket 08's PDF record; no total is ever
    computed here.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    meeting_time: str = Field(default="", max_length=100)
    meeting_point: str = Field(default="", max_length=300)
    leader: str = Field(default="", max_length=100)
    transport: str = Field(default="", max_length=500)
    accommodation: str = Field(default="", max_length=500)
    budget: str = Field(default="", max_length=500)
    emergency_contact: str = Field(default="", max_length=300)
    retreat_plan: str = Field(default="", max_length=500)

    def as_values(self) -> dict[str, str]:
        """The record as the draft stores and renders it: key → text."""
        return {key: getattr(self, key) for key in SUPPLEMENT_ARRANGEMENT_KEYS}


class PlanDraftInput(BaseModel):
    """The fields the entry dialog collects (spec AC6, widened by ticket 06).

    Deliberately narrow: ``extra="forbid"`` means the form cannot smuggle
    a route fact, a price, or a model-config blob into the draft. Unknown
    arrangements stay absent — they are never filled in here
    (docs/plan-pdf-format.md: 未知安排留空，显示"待确认").

    ``participants`` is strict so ``"12"``/``True`` are refused rather
    than silently coerced; the dialog sends a JSON number.

    ``arrangements`` is ticket 06's addition and is optional, so a draft that
    only holds the three minimal fields stays a valid submission: the rule
    verdict is what changes, not what the form demands.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    destination_or_route: str = Field(min_length=1, max_length=_DESTINATION_MAX)
    activity_date: str = Field(min_length=1, max_length=10)
    participants: int = Field(ge=1, le=10000, strict=True)
    arrangements: PlanArrangementInput = Field(default_factory=PlanArrangementInput)

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
    # Ticket 05 — the missing-field decisions and the first confirmation are
    # structured steps too. ``ARRANGEMENTS_NOT_PENDING`` is the AC3/AC5 guard:
    # a confirmation only exists for a summary the user was actually shown, so
    # the flow can never record approval of arrangements that were never
    # reviewed.
    FIRST_REVIEW_WITHOUT_DRAFT = "first_review_without_draft"
    SUPPLEMENT_WITHOUT_DRAFT = "supplement_without_draft"
    ARRANGEMENTS_NOT_PENDING = "arrangements_not_pending"
    # Ticket 07 — the second summary, its explicit confirmation, and the safety
    # hard stop. ``SAFETY_BLOCKED`` is the AC5 carrier: the same stable code the
    # event record holds is what the panel explains to the user.
    FIRST_REVIEW_NOT_CONFIRMED = "first_review_not_confirmed"
    FINAL_REVIEW_WITHOUT_VERDICT = "final_review_without_verdict"
    FINAL_REVIEW_NOT_PENDING = "final_review_not_pending"
    SAFETY_BLOCKED = "safety_blocked"


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

    ``blocked_stage`` is ticket 07 extending that same mechanism to the leaf's
    *third* answer: when the deterministic gate refused the route outright
    (``no_go``), the turn does not land on the 资料不足 choice ticket 04 parks on,
    it lands on the hard stop. Naming the stage here is what keeps
    :func:`evaluate_plan_intent` the only transition decider — the flow reads the
    leaf's status and picks among stages this layer already named.
    """

    status: Literal["ok", "blocked"]
    stage: PlanDraftStage | None = None
    reason: str | None = None
    draft_input: PlanDraftInput | None = None
    route_selection: RouteSelectionInput | None = None
    fallback_stage: PlanDraftStage | None = None
    blocked_stage: PlanDraftStage | None = None
    plan_type: str | None = None
    completeness: "PlanCompleteness | None" = None

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
#: Ticket 05: the stages from which the user is allowed to ask for the first
#: summary. Only a submitted draft has arrangements to summarise, so
#: ``minimal_draft`` is the single source (``awaiting_generic_choice`` is a
#: parked research turn answered by ticket 04's two exits, and promoting a
#: draft the user has not settled would reopen that question).
_FIRST_REVIEW_SOURCE_STAGES: frozenset[PlanDraftStage] = frozenset(
    {PlanDraftStage.MINIMAL_DRAFT}
)
#: Ticket 05: the stages 「继续补充」 is offered on — the ones that show the
#: missing-field list. The draft is still unfinished in both, which is exactly
#: why reopening the form is legal from either.
_SUPPLEMENT_SOURCE_STAGES: frozenset[PlanDraftStage] = frozenset(
    {PlanDraftStage.MINIMAL_DRAFT, PlanDraftStage.AWAITING_FIRST_REVIEW}
)
#: Ticket 07: the second summary is asked for only once the FIRST one is
#: confirmed. ``first_reviewed`` is ticket 05's record that the user read and
#: confirmed their own arrangements, and the stored rule verdict (see
#: :class:`RecordedCompleteness`) has to be on record beside it — so a final
#: review can never be built on a draft nobody reviewed, and can never be
#: honoured from a stale row the fields have since superseded.
_FINAL_REVIEW_SOURCE_STAGES: frozenset[PlanDraftStage] = frozenset(
    {PlanDraftStage.FIRST_REVIEWED}
)
#: Ticket 07 AC4: the confirmation exists only while that summary is pending.
#: A repeat click from ``final_reviewed`` is refused — re-confirmation has to go
#: through a NEW summary (ticket 05's rule, applied one stage later).
_FINAL_CONFIRM_SOURCE_STAGES: frozenset[PlanDraftStage] = frozenset(
    {PlanDraftStage.AWAITING_FINAL_REVIEW}
)
#: Ticket 07 AC2: the stage the deterministic gate's 危险封闭 answer parks on. No
#: advancing intent is honoured from it, whatever else its stage set says, so the
#: hard stop cannot be walked through by a click that means 资料不足.
_DANGEROUS_STAGES: frozenset[PlanDraftStage] = frozenset({PlanDraftStage.BLOCKED_SAFETY})
#: The intents 危险封闭 refuses. 取消, 重新填写 and 研究路线资料 are deliberately
#: absent: the first two are the user's exits, and re-researching a destination
#: is how the gate gets a chance to say otherwise.
_ADVANCING_INTENTS: frozenset[str] = frozenset({
    "plan_first_review", "plan_generic_draft", "plan_final_review", "plan_confirm_final",
})

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
    PlanBlockReason.FIRST_REVIEW_WITHOUT_DRAFT: (
        "还没有提交任何安排，无法生成第一次摘要。请先填写并提交目的地或路线、活动日期和参与人数。"
    ),
    PlanBlockReason.SUPPLEMENT_WITHOUT_DRAFT: (
        "当前没有可继续补充的草稿安排，请先创建计划书。"
    ),
    PlanBlockReason.ARRANGEMENTS_NOT_PENDING: (
        "当前没有等待确认的第一次摘要。生成草案后才会请你确认本人填写的安排，"
        "未经摘要确认不会记录任何同意。"
    ),
    PlanBlockReason.FIRST_REVIEW_NOT_CONFIRMED: (
        "还没有确认过第一次摘要。第二次摘要只在它之后出现："
        "先确认本人提供的安排，再查看并确认最终摘要。"
    ),
    PlanBlockReason.FINAL_REVIEW_WITHOUT_VERDICT: (
        "本次会话还没有可用的完整度规则结论，无法生成或确认第二次摘要。"
        "规则结论缺失时一律按未通过处理，不会放行。"
    ),
    PlanBlockReason.FINAL_REVIEW_NOT_PENDING: (
        "当前没有等待确认的第二次摘要。进入 PDF 审批必须先查看并最终确认摘要，"
        "未经确认不会进入任何审批环节。"
    ),
    PlanBlockReason.SAFETY_BLOCKED: (
        "确定性安全门判定该路线处于危险封闭状态（严重天气、封闭路线或高风险），"
        "已阻止正式路线计划。你只能取消本次计划书，或修改目的地/路线后重新核验。"
    ),
}


@dataclass(frozen=True, slots=True)
class RecordedCompleteness:
    """The rules' answer as it is ON RECORD (ticket 07 AC4's precondition).

    Ticket 06 stores one verdict per conversation in ``plan_completeness_checks``,
    scoped to the draft it judged and the run that computed it. This is that row
    seen from the domain: *which* draft it judges and *what* it concluded — no
    field value, no route fact, no report flag (06 AC6 survives the trip).

    Deliberately NOT the thing the decision is made from: :func:`evaluate_plan_intent`
    requires a record to exist (the audit chain: the rules really ran on this
    draft) and then decides on the *recomputed* verdict, because a stored row is
    superseded the moment the fields change. Trusting the row would let a draft
    that lost 集合时间 since that run read as review-ready.
    """

    draft_id: str
    verdict: str
    reasons: tuple[str, ...] = ()
    checked_by_run_id: str = ""


class PlanSafetyReason:
    """Why the deterministic gate refused (ticket 07 AC2/AC5).

    Stable, machine-readable values — the :class:`PlanBlockReason` discipline, so
    a client branches on the code and shows :data:`SAFETY_MESSAGE` beside it, and
    an event record explains itself long after the turn. They are NOT a second
    decider: ``no_go`` / ``severe_weather`` / ``route_closed`` name the gate's own
    ``risk_status == "no_go"`` answer (split only by which word in
    :data:`route_compare.service.SEVERE` fired, so the *explanation* is the gate's
    vocabulary too), and ``evidence_gate_failed`` names its
    ``evidence_status != "complete"`` / non-clear answer. Nothing here re-derives a
    threshold, and no member implies ``report_allowed`` — that stays
    :func:`route_compare.service.render_report`'s authority.
    """

    NO_GO = "no_go"
    SEVERE_WEATHER = "severe_weather"
    ROUTE_CLOSED = "route_closed"
    EVIDENCE_GATE_FAILED = "evidence_gate_failed"


#: The readable copy beside each stable code, never reworded — only added.
SAFETY_MESSAGE: dict[str, str] = {
    PlanSafetyReason.NO_GO: (
        "确定性安全门判定该路线危险封闭（严重天气、封闭路线或高风险条件），"
        "已阻止生成正式路线计划。"
    ),
    PlanSafetyReason.SEVERE_WEATHER: (
        "该日期存在雷暴、台风、暴雨、暴雪或大风预警等严重天气，安全门禁止据此出计划。"
    ),
    PlanSafetyReason.ROUTE_CLOSED: (
        "官方或资料来源显示路线封闭，安全门禁止据此出计划。"
    ),
    PlanSafetyReason.EVIDENCE_GATE_FAILED: (
        "天气、开放状态或风险资料不足或未通过时效核验，正式路线计划被阻止；"
        "缺口仍为待核验，可继续通用草案。"
    ),
}


@dataclass(frozen=True, slots=True)
class PlanSafety:
    """The gate's own refusal, carried as data (ticket 07 AC2/AC5).

    Built by the route leaf from the :class:`route_compare.models.ComparisonResult`
    it already produced: ``codes`` are the stable names, ``gate_reasons`` and
    ``required_action`` are the gate's own strings copied verbatim (this module has
    no parameter by which a caller could write a safety sentence), and
    ``comparison_status`` is the comparison's status untouched. ``hard_stop`` is the
    AC2/AC3 split in one boolean: a dangerous route refuses *everything* advancing
    (``no_go``), while a missing-evidence answer refuses only the formal route plan
    and leaves ticket 04's generic-draft choice standing.
    """

    hard_stop: bool
    codes: tuple[str, ...] = ()
    comparison_id: str = ""
    comparison_status: str = ""
    gate_reasons: tuple[str, ...] = ()
    required_action: str = ""

    @property
    def blocks_formal_route_plan(self) -> bool:
        """AC2: no formal route plan may be built on this answer.

        True for the hard stop AND for an evidence-gate failure — which is exactly
        the case the derived ``plan_type`` already refuses, stated here so the
        refusal is *named* in the record instead of only implied by an absent basis.
        """
        return self.hard_stop or PlanSafetyReason.EVIDENCE_GATE_FAILED in self.codes

    def to_payload(self) -> dict[str, Any]:
        """The wire shape (AC5): codes, the gate's own copy, and its own reasons."""
        return {
            "hard_stop": self.hard_stop,
            "codes": list(self.codes),
            "formal_route_plan_allowed": not self.blocks_formal_route_plan,
            "message": "；".join(
                SAFETY_MESSAGE[code] for code in self.codes if code in SAFETY_MESSAGE
            ),
            "comparison_id": self.comparison_id,
            "comparison_status": self.comparison_status,
            "gate_reasons": list(self.gate_reasons),
            "required_action": self.required_action,
        }


def safety_lines(safety: PlanSafety) -> str:
    """The refusal as deterministic copy, in one place (ticket 07 AC5).

    Rendered from the record the gate produced, so the streamed text, the
    ``plan.draft_state`` payload and the stored event cannot describe different
    refusals. The only sentences here are :data:`SAFETY_MESSAGE` and the gate's own
    reason lines — never a paraphrase that could soften 禁止推荐 into 建议谨慎.
    """
    lines = ["**安全门结论（由 route_compare 的确定性核验给出，本模块只转述）**"]
    if safety.codes:
        lines.append(
            "- 阻止原因："
            + "、".join(f"{code}（{SAFETY_MESSAGE.get(code, code)}）" for code in safety.codes)
        )
    if safety.comparison_status:
        lines.append(
            f"- 比较结果：{safety.comparison_status}"
            + (f"（comparison_id={safety.comparison_id}）" if safety.comparison_id else "")
        )
    for reason in safety.gate_reasons:
        lines.append(f"- 核验给出的原因：{reason}")
    if safety.required_action:
        lines.append(f"- 需要采取的行动：{safety.required_action}")
    lines.append(
        "- 结论：危险封闭已阻止正式路线计划，只能取消本次计划书或修改目的地/路线后重新核验。"
        if safety.hard_stop
        else (
            "- 结论：证据/安全门未通过，正式路线计划被阻止；未通过核验的路线事实仍为待核验，"
            "你可以继续生成通用草案或取消。"
            if safety.blocks_formal_route_plan
            else "- 结论：路线资料已通过确定性核验。"
        )
    )
    return "\n".join(lines)


def pdf_approval_block_reason(
    *,
    stage: PlanDraftStage | None,
    recorded_completeness: RecordedCompleteness | None,
    safety: PlanSafety | None,
) -> str | None:
    """AC4's door, as one pure question: may this draft enter PDF approval?

    ``None`` means yes. Ticket 09 owns the approval itself (the existing artifact
    lifecycle and the tool-approval machinery), and calls this instead of
    re-deciding anything: the answer is three facts already owned elsewhere — the
    stage :func:`evaluate_plan_intent` wrote, the verdict ticket 06 stored, and the
    refusal the deterministic gate produced. There is deliberately no
    ``report_allowed`` parameter: that claim belongs to
    :func:`route_compare.service.render_report` and nothing here may accept it from
    a client.
    """
    if stage is PlanDraftStage.BLOCKED_SAFETY:
        return PlanBlockReason.SAFETY_BLOCKED
    if safety is not None and safety.hard_stop and safety.codes:
        return safety.codes[0]
    if stage is not PlanDraftStage.FINAL_REVIEWED:
        return PlanBlockReason.FINAL_REVIEW_NOT_PENDING
    if recorded_completeness is None:
        return PlanBlockReason.FINAL_REVIEW_WITHOUT_VERDICT
    return None


def evaluate_plan_intent(
    *,
    intent: str | None,
    current_stage: PlanDraftStage | None,
    plan_input: PlanDraftInput | dict | None,
    conversation_id: str | None,
    route_selection: RouteSelectionInput | dict | None = None,
    plan_type: str | None = None,
    completeness: "PlanCompleteness | None" = None,
    recorded_completeness: "RecordedCompleteness | None" = None,
    safety: "PlanSafety | None" = None,
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

    if intent in _ADVANCING_INTENTS and current_stage in _DANGEROUS_STAGES:
        # Ticket 07 AC2: 危险封闭 is a hard stop, checked before any stage set, so
        # no advancing click walks through it. The refusal is a stable reason the
        # event record and the panel both explain, and the shape of the input is
        # judged first on purpose — a smuggled field is still plan_input_not_allowed.
        return PlanTransition("blocked", reason=PlanBlockReason.SAFETY_BLOCKED)

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
            # Ticket 07 AC2: the leaf's third answer. Material that is merely thin
            # parks on the choice; material the gate refused outright parks on the
            # hard stop. Both stages are named HERE, so the flow still only picks.
            blocked_stage=PlanDraftStage.BLOCKED_SAFETY,
        )

    if intent == "plan_generic_draft":
        # AC4: the way out of a parked research turn. Chosen by the user, on
        # purpose — the server does not fall through to a draft on its own.
        if current_stage is not PlanDraftStage.AWAITING_GENERIC_CHOICE:
            return PlanTransition("blocked", reason=PlanBlockReason.GENERIC_DRAFT_WITHOUT_PENDING)
        return PlanTransition("ok", stage=PlanDraftStage.MINIMAL_DRAFT, plan_type="generic_draft")

    if intent == "plan_first_review":
        # AC1/AC2: 「直接生成草案」 — the user has read the missing-field list and
        # decided to stop collecting. This is the only move that produces the
        # FIRST summary, and it is a stage move, never a claim: the summary it
        # lands on is rendered from the stored arrangements plus the re-read
        # route basis, so nothing here (or on the wire) can add a field the user
        # did not supply (AC4).
        if current_stage not in _FIRST_REVIEW_SOURCE_STAGES:
            return PlanTransition("blocked", reason=PlanBlockReason.FIRST_REVIEW_WITHOUT_DRAFT)
        # Ticket 06 AC3: the click cannot open a summary over a draft that is
        # missing a CORE arrangement — it bounces to the supplement flow instead,
        # and the move is still decided here, with the rule's verdict as an
        # input. The gate also fails CLOSED (AC1): no verdict, no review — a
        # missing rule input is never treated as a passing one, so a flow that
        # forgot to run the rules cannot silently honour the click.
        if completeness is None:
            return PlanTransition("blocked", reason=PlanBlockReason.FIRST_REVIEW_WITHOUT_DRAFT)
        if completeness.supplement_required:
            return PlanTransition(
                "ok", stage=PlanDraftStage.AWAITING_INPUT, completeness=completeness
            )
        return PlanTransition(
            "ok", stage=PlanDraftStage.AWAITING_FIRST_REVIEW, completeness=completeness
        )

    if intent == "plan_more_input":
        # AC2: 「继续补充」 — the other answer to the missing-field list. It reopens
        # the form (ticket 01's ``awaiting_input``) and collects nothing itself,
        # so a user who wants to fill more never has to cancel and start over.
        if current_stage not in _SUPPLEMENT_SOURCE_STAGES:
            return PlanTransition("blocked", reason=PlanBlockReason.SUPPLEMENT_WITHOUT_DRAFT)
        return PlanTransition("ok", stage=PlanDraftStage.AWAITING_INPUT)

    if intent == "plan_confirm_arrangements":
        # AC3/AC5: the first summary is confirmed, and only the first summary.
        # Requiring ``awaiting_first_review`` is what makes the stored record
        # honest: the draft can only say "the user confirmed the arrangements we
        # showed them", never "the user approved a plan" (that is ticket 07's
        # final review), and never anything about a field still marked 待确认.
        if current_stage is not PlanDraftStage.AWAITING_FIRST_REVIEW:
            return PlanTransition("blocked", reason=PlanBlockReason.ARRANGEMENTS_NOT_PENDING)
        return PlanTransition("ok", stage=PlanDraftStage.FIRST_REVIEWED)

    if intent == "plan_final_review":
        # Ticket 07 AC1/AC4: 「查看最终摘要」 — the SECOND summary, the one that
        # shows the verified route results beside the user's own arrangements, the
        # missing items and the plan type. It is reachable only from the stage that
        # proves the first summary was confirmed, and only with the rules' answer
        # both recomputed AND on record; the guard order is the safety property,
        # because a final review built on an unreviewed first summary would be a
        # second confirmation of nothing.
        if current_stage not in _FINAL_REVIEW_SOURCE_STAGES:
            return PlanTransition(
                "blocked", reason=PlanBlockReason.FIRST_REVIEW_NOT_CONFIRMED
            )
        # AC4 fails CLOSED, exactly like ticket 06's review gate: no fresh verdict,
        # or no stored row for this draft, means the rules never decided anything
        # here, and a missing rule input is never a passing one.
        if completeness is None or recorded_completeness is None:
            return PlanTransition(
                "blocked", reason=PlanBlockReason.FINAL_REVIEW_WITHOUT_VERDICT
            )
        # AC3 vs AC2 kept apart: an ordinary missing arrangement is ticket 06's
        # bounce (补充流程 + the stable reason codes), NOT a safety block, and NOT
        # a second decider. The dangerous case never reaches this branch — the
        # stage guard above refused it with ``safety_blocked``.
        if completeness.supplement_required:
            return PlanTransition(
                "ok", stage=PlanDraftStage.AWAITING_INPUT, completeness=completeness
            )
        return PlanTransition(
            "ok", stage=PlanDraftStage.AWAITING_FINAL_REVIEW, completeness=completeness
        )

    if intent == "plan_confirm_final":
        # Ticket 07 AC4: the explicit confirmation, and the only move that reaches
        # ``final_reviewed`` — the stage 进入 PDF 审批 is read from. Requiring
        # ``awaiting_final_review`` is what makes the record honest: a click cannot
        # confirm a summary the user was never shown, and a second click on an
        # already-confirmed plan cannot re-open the door (05's rule, one stage on).
        if current_stage not in _FINAL_CONFIRM_SOURCE_STAGES:
            return PlanTransition("blocked", reason=PlanBlockReason.FINAL_REVIEW_NOT_PENDING)
        if completeness is None or recorded_completeness is None:
            return PlanTransition(
                "blocked", reason=PlanBlockReason.FINAL_REVIEW_WITHOUT_VERDICT
            )
        return PlanTransition("ok", stage=PlanDraftStage.FINAL_REVIEWED)

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

#: The user-arrangement record, in display order and with the labels the plan
#: document uses (docs/plan-pdf-format.md: 日期、集合、人数、领队、交通住宿、
#: 应急联系、撤退安排…). Every key but ``destination_or_route`` is a field of
#: :class:`route_compare.models.PlanDetails`, which is the arrangement record
#: the seam contract names — and ``test_plan_missing_fields_first_review.py``
#: pins that, so this cannot become a parallel list of made-up fields.
#:
#: The tuple is the SINGLE source for both halves of ticket 05: the lines
#: :func:`render_minimal_draft` draws and the 缺失字段清单
#: :func:`missing_plan_fields` enumerates. A field added here appears in the
#: draft and in the missing list in the same render, and a field removed
#: disappears from both. The three minimal fields are what
#: :class:`PlanDraftInput` has always required; the other eight are collected by
#: ticket 06's :class:`PlanArrangementInput` and stored in their own table. A
#: field the user left empty still renders 待确认 — an absent arrangement is
#: reported as absent and never filled (spec AC4).
ARRANGEMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("destination_or_route", "目的地或路线"),
    ("activity_date", "活动日期"),
    ("participants", "参与人数"),
    ("meeting_time", "集合时间"),
    ("meeting_point", "集合地点"),
    ("leader", "领队"),
    ("transport", "交通"),
    ("accommodation", "住宿"),
    ("budget", "费用预算"),
    ("emergency_contact", "应急联络"),
    ("retreat_plan", "撤退方案"),
)

#: The three arrangements the minimal entry form has always made mandatory
#: (:class:`PlanDraftInput`'s own fields). Named here so the completeness rules
#: and the renderer read one vocabulary instead of two lists that can drift.
MINIMAL_ARRANGEMENT_KEYS: tuple[str, ...] = (
    "destination_or_route", "activity_date", "participants",
)
#: The supplementary arrangements ticket 06 gave real input and real storage to.
#: Together with :data:`MINIMAL_ARRANGEMENT_KEYS` these are exactly
#: :data:`ARRANGEMENT_FIELDS` — pinned, so a field cannot be collected without
#: joining the checklist, nor listed without being collectable.
SUPPLEMENT_ARRANGEMENT_KEYS: tuple[str, ...] = (
    "meeting_time", "meeting_point", "leader", "transport", "accommodation",
    "budget", "emergency_contact", "retreat_plan",
)

#: The business rule behind ticket 06 AC3, stated once: the arrangements an
#: outing cannot run without. 集合时间/集合地点 decide whether the group can
#: meet at all, and 应急联络 is the contact the plan is unsafe without — so a
#: missing one of those sends the draft back to the form (补充流程) instead of
#: forward to a summary. 领队 / 交通 / 住宿 / 费用预算 / 撤退方案 are needed for a
#: complete plan book but do not block a draft the user can still review, so a
#: missing one joins the 待确认清单. Route-fact dimensions are always treated as
#: core: the user cannot supply a certified fact by typing one (AC6).
CORE_ARRANGEMENT_KEYS: frozenset[str] = frozenset({
    *MINIMAL_ARRANGEMENT_KEYS, "meeting_time", "meeting_point", "emergency_contact",
})
#: Every other arrangement — the 待确认清单 half of ticket 06 AC3. Derived from
#: the enumeration rather than written out, so the two halves cannot overlap and
#: a new field defaults to non-core until someone argues for more.
NON_CORE_ARRANGEMENT_KEYS: frozenset[str] = frozenset(
    key for key, _label in ARRANGEMENT_FIELDS if key not in CORE_ARRANGEMENT_KEYS
)

#: Which route-fact dimensions an arrangement can be checked against, expressed
#: as the two vocabularies that already exist (ticket 06 AC2). A fact the gate
#: never certified cannot confirm an arrangement, and an arrangement with no
#: mapped dimension is stated as such rather than silently matched: these lines
#: are a *relationship* record, and they never carry a value of their own.
ARRANGEMENT_EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "destination_or_route": ("map",),
    "activity_date": ("weather", "trail_status"),
    "participants": (),
    "meeting_time": ("duration_hours",),
    "meeting_point": ("map", "trail_status"),
    "leader": (),
    "transport": ("distance_km",),
    "accommodation": (),
    "budget": (),
    "emergency_contact": (),
    "retreat_plan": ("risk", "trail_status"),
}

#: Explicit per-field format rules (ticket 06 AC1's 格式校验). Field existence
#: alone would call 「明天一早」 a filled 集合时间, which is the difference between
#: a draft and a usable plan. Each entry is (pattern, what the user must change);
#: the rules never rewrite, normalise or complete a value — an unfilled field is
#: a presence question, never a format one.
ARRANGEMENT_FORMATS: dict[str, tuple[re.Pattern[str], str]] = {
    "destination_or_route": (re.compile(r"^.{1,120}$"), "目的地或路线需为 1–120 字"),
    "activity_date": (_ISO_DATE_PATTERN, "活动日期需为 YYYY-MM-DD"),
    "participants": (_HEADCOUNT_PATTERN, "参与人数需为「数字 + 人」"),
    "meeting_time": (_HH_MM_PATTERN, "集合时间需为 HH:MM（24 小时制）"),
    "emergency_contact": (_PHONE_DIGITS_PATTERN, "应急联络需包含至少 7 位可拨打的电话号码"),
    "budget": (_AMOUNT_PATTERN, "费用预算需包含金额数字（系统不会替你估算）"),
}


@dataclass(frozen=True, slots=True)
class PlanFormatIssue:
    """One field that is filled in a form the plan cannot use (AC1)."""

    key: str
    label: str
    message: str


def arrangement_format_issues(values: dict[str, str]) -> tuple[PlanFormatIssue, ...]:
    """Run the explicit format rules over the arrangement record (AC1).

    Deterministic, total, and reads only the record it is given: an empty value
    is a missing field and therefore nobody's format problem, and a value with no
    rule for its key (领队, 交通, 住宿, 集合地点, 撤退方案) is free text the user owns.
    Nothing is corrected — the issue is reported with the copy the user needs to
    fix it, and the value stays as written until they change it.
    """
    label_of = dict(ARRANGEMENT_FIELDS)
    issues: list[PlanFormatIssue] = []
    for key, (pattern, message) in ARRANGEMENT_FORMATS.items():
        value = (values.get(key) or "").strip()
        if not value:
            continue
        if not pattern.search(value):
            issues.append(PlanFormatIssue(key=key, label=label_of[key], message=message))
    return tuple(issues)


# ---- Deterministic completeness verdict (ticket 06) ------------------------


class PlanReviewReason:
    """Why a human still has to confirm this draft (ticket 06 AC4).

    Stable, machine-readable values, never reworded — only added — following the
    :class:`PlanBlockReason` + :data:`BLOCK_MESSAGE` precedent. They are distinct
    from the block reasons: ``arrangements_not_pending`` stays what ticket 05
    made it (a stage guard: no summary is on screen to confirm) and is reused,
    never re-named, while these describe the *content* a person must decide on.
    """

    CORE_ARRANGEMENTS_MISSING = "core_arrangements_missing"
    ARRANGEMENT_FORMAT_INVALID = "arrangement_format_invalid"
    NON_CORE_ARRANGEMENTS_MISSING = "non_core_arrangements_missing"
    ROUTE_FACTS_UNVERIFIED = "route_facts_unverified"


#: The readable copy beside each stable value. A client branches on the code and
#: shows this text; neither is derived from the other at render time.
REASON_MESSAGE: dict[str, str] = {
    PlanReviewReason.CORE_ARRANGEMENTS_MISSING: (
        "缺少核心安排（集合时间、集合地点、应急联络等），请先在表单里补齐再生成草案；"
        "系统不会替你填这些值。"
    ),
    PlanReviewReason.ARRANGEMENT_FORMAT_INVALID: (
        "已填写的安排里有格式不通过的字段，请按提示修正后再继续。"
    ),
    PlanReviewReason.NON_CORE_ARRANGEMENTS_MISSING: (
        "部分非核心安排仍为待确认，会随草案一并列出，等你补充。"
    ),
    PlanReviewReason.ROUTE_FACTS_UNVERIFIED: (
        "路线事实仍为待核验：没有服务器证据的事实不会因草案生成而变成已核验。"
    ),
}


@dataclass(frozen=True, slots=True)
class PlanFieldVerdict:
    """One field's answer: is it there, and is it in a usable form (AC1).

    ``kind`` keeps the two vocabularies distinguishable in the record itself —
    an ``arrangement`` is the user's claim, a ``route_fact`` is the gate's. That
    is why ``value`` is always empty for a route fact: the verdict reports the
    *status* of a dimension and never carries a number, a weather word or an
    open status (AC6). ``core`` is the business rule that decides which list a
    gap lands in (AC3).
    """

    key: str
    label: str
    kind: Literal["arrangement", "route_fact"]
    core: bool
    presence: Literal["filled", "absent"]
    format_status: Literal["valid", "invalid", "not_applicable"]
    value: str


@dataclass(frozen=True, slots=True)
class PlanEvidenceMatch:
    """How one arrangement stands against the server's evidence (AC2).

    Explicit fields and a source status, never a sentence: ``fact_fields`` are
    the dimensions that could speak to this arrangement, ``status`` says whether
    the arrangement exists and whether those dimensions were certified, and
    ``source_ids`` names the sources that did. An uncertified dimension
    contributes no source id, so a match can never imply evidence the gate did
    not produce — and ``not_applicable`` is the honest answer for an arrangement
    no route fact bears on (a leader's name, a budget note).
    """

    arrangement_key: str
    arrangement_label: str
    fact_fields: tuple[str, ...]
    status: Literal[
        "matched", "partly_matched", "unverified", "arrangement_absent", "not_applicable"
    ]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanCompleteness:
    """The deterministic verdict on one draft (ticket 06 AC1/AC3/AC4).

    ``verdict`` is the business rule's answer and only three things can produce
    it: a core arrangement missing or a field in an unusable format
    (``supplement_required`` — go back to the form), any remaining gap
    (``confirmation_required`` — the 待确认清单 plus the reasons a human must
    confirm), or nothing missing at all (``complete``). There is no score: a
    weighted number would let a draft be 70% of a plan, which is not a decision
    anyone can act on.
    """

    verdict: Literal["complete", "confirmation_required", "supplement_required"]
    fields: tuple[PlanFieldVerdict, ...]
    missing_core_keys: tuple[str, ...]
    missing_non_core_keys: tuple[str, ...]
    format_issues: tuple[PlanFormatIssue, ...]
    matches: tuple[PlanEvidenceMatch, ...]
    reasons: tuple[str, ...]

    @property
    def invalid_format_keys(self) -> tuple[str, ...]:
        return tuple(issue.key for issue in self.format_issues)

    @property
    def supplement_required(self) -> bool:
        return self.verdict == "supplement_required"

    def to_payload(self) -> dict[str, Any]:
        """The wire shape of the verdict (AC5): codes, keys and labels only.

        Emitted verbatim from this record so the streamed text, the payload and
        the stored row are one computation, and a client can show the reasons
        after a refresh without re-deciding anything.
        """
        labels = {field.key: field.label for field in self.fields}
        return {
            "verdict": self.verdict,
            "reasons": [
                {"code": code, "message": REASON_MESSAGE[code]} for code in self.reasons
            ],
            "missing_core": [
                {"key": key, "label": labels.get(key, key)} for key in self.missing_core_keys
            ],
            "missing_non_core": [
                {"key": key, "label": labels.get(key, key)} for key in self.missing_non_core_keys
            ],
            "format_issues": [
                {
                    "key": issue.key,
                    "label": issue.label,
                    "message": issue.message,
                }
                for issue in self.format_issues
            ],
            "matches": [
                {
                    "arrangement_key": match.arrangement_key,
                    "arrangement_label": match.arrangement_label,
                    "fact_fields": list(match.fact_fields),
                    "status": match.status,
                    "source_ids": list(match.source_ids),
                }
                for match in self.matches
            ],
        }


def evaluate_plan_completeness(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    arrangements: dict[str, str] | None = None,
    route: RouteBasis | None = None,
) -> PlanCompleteness:
    """Decide how complete a draft is — field existence, format, business rules.

    A separate pure function over the same two tuples ticket 05's
    :func:`missing_plan_fields` enumerates (existence is already computed there
    and travels as ``plan.draft_state.missing_fields``), so ticket 05's two
    markers stay exactly two: this adds a *verdict* beside them, never a third
    status, never a score, and never re-reads the summary text. No model is
    involved — this module has no kernel, no provider and no HTTP client, and the
    only external answer it consumes is the route basis the caller already
    re-read through the comparison leaf.

    Args: the stored arrangement fields, the supplementary arrangement record,
    and the caller's re-read :class:`RouteBasis` (``None`` for a generic draft).
    """
    values = _arrangement_values(
        destination_or_route=destination_or_route,
        activity_date=activity_date,
        participants=participants,
        arrangements=arrangements,
    )
    issues = arrangement_format_issues(values)
    invalid = tuple(issue.key for issue in issues)
    labels = dict(ARRANGEMENT_FIELDS)

    missing_core = tuple(
        key for key, _label in ARRANGEMENT_FIELDS
        if key in CORE_ARRANGEMENT_KEYS and not values[key]
    )
    missing_non_core = tuple(
        key for key, _label in ARRANGEMENT_FIELDS
        if key in NON_CORE_ARRANGEMENT_KEYS and not values[key]
    )

    certified = {line.field: line for line in route.facts} if route is not None else {}
    fields = tuple(
        PlanFieldVerdict(
            key=key,
            label=label,
            kind="arrangement",
            core=key in CORE_ARRANGEMENT_KEYS,
            presence="filled" if values[key] else "absent",
            format_status=(
                "invalid" if key in invalid
                else "valid" if (key in ARRANGEMENT_FORMATS and values[key])
                else "not_applicable"
            ),
            value=values[key],
        )
        for key, label in ARRANGEMENT_FIELDS
    ) + tuple(
        PlanFieldVerdict(
            key=field,
            label=label,
            kind="route_fact",
            core=True,
            presence="filled" if field in certified else "absent",
            format_status="not_applicable",
            # AC6: the verdict never carries a route value. Only ``missing_plan_
            # fields`` and ``render_route_basis`` report a certified fact, and
            # both read the caller's re-read basis rather than this record.
            value="",
        )
        for field, label, _unit in FACT_DIMENSIONS
    )

    matches: list[PlanEvidenceMatch] = []
    # AC2: a match compares a *claim* against the server's evidence. Without a
    # submitted arrangement record there is no claim to check at all — even the
    # three entry-form fields are presence answers here, not evidence claims —
    # so every dimension-mapped arrangement reports ``arrangement_absent``.
    record_submitted = arrangements is not None
    for key, label in ARRANGEMENT_FIELDS:
        fact_fields = ARRANGEMENT_EVIDENCE_FIELDS.get(key, ())
        if not fact_fields:
            matches.append(PlanEvidenceMatch(key, label, (), "not_applicable", ()))
            continue
        if not record_submitted or not values[key]:
            matches.append(PlanEvidenceMatch(key, label, fact_fields, "arrangement_absent", ()))
            continue
        sources = tuple(
            certified[field].source_id for field in fact_fields if field in certified
        )
        status = (
            "matched" if len(sources) == len(fact_fields)
            else "partly_matched" if sources
            else "unverified"
        )
        matches.append(PlanEvidenceMatch(key, label, fact_fields, status, sources))

    unverified_facts = tuple(
        field for field, _label, _unit in FACT_DIMENSIONS if field not in certified
    )

    reasons: list[str] = []
    if missing_core:
        reasons.append(PlanReviewReason.CORE_ARRANGEMENTS_MISSING)
    if invalid:
        reasons.append(PlanReviewReason.ARRANGEMENT_FORMAT_INVALID)
    if missing_non_core:
        reasons.append(PlanReviewReason.NON_CORE_ARRANGEMENTS_MISSING)
    if unverified_facts:
        reasons.append(PlanReviewReason.ROUTE_FACTS_UNVERIFIED)

    if missing_core or invalid:
        verdict = "supplement_required"
    elif missing_non_core or unverified_facts:
        verdict = "confirmation_required"
    else:
        verdict = "complete"

    return PlanCompleteness(
        verdict=verdict,
        fields=fields,
        missing_core_keys=missing_core,
        missing_non_core_keys=missing_non_core,
        format_issues=issues,
        matches=tuple(matches),
        reasons=tuple(reasons),
    )


def completeness_lines(completeness: PlanCompleteness) -> str:
    """The verdict as deterministic copy (ticket 06 AC3/AC4).

    Rendered from the record, in one place, so the streamed text, the payload and
    the stored row cannot describe different verdicts. It names the stable reason
    codes beside their readable copy — a reviewer can grep the value out of the
    conversation history — and it lists gaps, never values: the only text here is
    a field label, a code, or the message that tells the user how to fix a field
    they filled in themselves.
    """
    labels = {key: label for key, label in ARRANGEMENT_FIELDS}
    lines = ["**完整度规则结论（字段存在性 + 格式校验 + 业务规则，未使用任何模型）**"]
    lines.append(f"- 结论：{completeness.verdict}")
    if completeness.missing_core_keys:
        lines.append(
            "- 待补充（核心安排）："
            + "、".join(labels[key] for key in completeness.missing_core_keys)
        )
    if completeness.format_issues:
        lines.append(
            "- 格式未通过："
            + "；".join(f"{issue.label}（{issue.message}）" for issue in completeness.format_issues)
        )
    if completeness.missing_non_core_keys:
        lines.append(
            "- 待确认（非核心安排）："
            + "、".join(labels[key] for key in completeness.missing_non_core_keys)
        )
    if completeness.reasons:
        lines.append(
            "- 需人工确认原因："
            + "、".join(f"{code}（{REASON_MESSAGE[code]}）" for code in completeness.reasons)
        )
    else:
        lines.append("- 需人工确认原因：无")
    return "\n".join(lines)


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
    # Ticket 07 AC2/AC5: the hard stop says what was refused and what the user's
    # two exits are. It never names 通用草案 (that is 04's answer to thin material,
    # not to danger) and it never prints a value: the explanation beside it comes
    # from :func:`safety_lines`, i.e. from the gate's own strings.
    PlanDraftStage.BLOCKED_SAFETY: (
        "确定性安全门判定该目的地/路线处于危险封闭状态（严重天气、封闭路线或高风险），"
        "已阻止生成正式路线计划，本次不提供通用草案。你可以取消本次计划书，"
        "或修改目的地/路线后重新核验；阻止结论与原因由服务器给出并记录在事件里。"
    ),
}


def draft_summary(
    *,
    stage: PlanDraftStage,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    route: RouteBasis | None = None,
    arrangements: dict[str, str] | None = None,
    completeness: PlanCompleteness | None = None,
    safety: PlanSafety | None = None,
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

    Ticket 05 adds the two review stages, and they are rendered by the SAME
    function as the draft preview rather than by a second renderer: the first
    summary is the draft preview re-organised into "what you supplied" and
    "what is still missing", so the 待确认 / 待核验 boundary, the 草案类型 marker
    and the missing-field list cannot disagree with the draft they summarise.
    ``awaiting_first_review`` asks the user to confirm; ``first_reviewed``
    records that they did — and only for the arrangements they themselves
    supplied (AC3), which is why the confirmed render still lists every
    unfilled field instead of quietly dropping it.

    Nothing is invented: the summary is a pure function of the stored fields
    plus the caller's re-read basis, so editing a field and re-rendering is
    the same call.
    """
    if stage is PlanDraftStage.MINIMAL_DRAFT:
        base = render_minimal_draft(
            destination_or_route=destination_or_route,
            activity_date=activity_date,
            participants=participants,
            route=route,
            arrangements=arrangements,
        )
    elif stage in (
        # Ticket 05's two stages, and ticket 07's two, are the SAME render with a
        # different ``review`` scope: the second summary is the draft re-organised
        # for a final look at 路线核验结果 + 用户安排 + 缺失项 + 计划类型, so the
        # 待确认 / 待核验 boundary and the 草案类型 marker cannot drift between the
        # two confirmations (AC1) — and there is still no parameter that could
        # carry a value the draft does not hold.
        PlanDraftStage.AWAITING_FIRST_REVIEW, PlanDraftStage.FIRST_REVIEWED,
        PlanDraftStage.AWAITING_FINAL_REVIEW, PlanDraftStage.FINAL_REVIEWED,
    ):
        review_mode = {
            PlanDraftStage.AWAITING_FIRST_REVIEW: "pending",
            PlanDraftStage.FIRST_REVIEWED: "confirmed",
            PlanDraftStage.AWAITING_FINAL_REVIEW: "final_pending",
            PlanDraftStage.FINAL_REVIEWED: "final_confirmed",
        }[stage]
        base = render_minimal_draft(
            destination_or_route=destination_or_route,
            activity_date=activity_date,
            participants=participants,
            route=route,
            arrangements=arrangements,
            review=review_mode,
        )
        # Ticket 06 AC4: the summary the user confirms also carries the rule's
        # answer beside it, so 待确认 (arrangement), 待核验 (fact) and "these are
        # the reasons a person still has to decide" are one render rather than
        # three vocabularies that can disagree.
        if completeness is not None:
            base = f"{base}\n\n{completeness_lines(completeness)}"
        # Ticket 07 AC2/AC5: what the gate refused is said in the same text the
        # user confirms, in the gate's own words — a final summary that hid a
        # refusal would otherwise read as an approved plan.
        if safety is not None:
            base = f"{base}\n\n{safety_lines(safety)}"
    elif stage is PlanDraftStage.BLOCKED_SAFETY:
        # AC5: the hard stop explains itself. Without the refusal lines the stage
        # copy says 危险封闭 without the gate's reason, its required action, or the
        # comparison it came from — and the event record would be the only place a
        # reviewer could find out why no plan exists.
        base = STAGE_MESSAGE[stage]
        if safety is not None:
            base = f"{base}\n\n{safety_lines(safety)}"
    elif stage is PlanDraftStage.AWAITING_INPUT and completeness is not None:
        # Ticket 06 AC3: 直接生成草案 over a draft missing a core arrangement
        # lands HERE — on the supplement flow — and the copy has to say why, or
        # the click looks like it did nothing.
        base = (
            f"{STAGE_MESSAGE[stage]}\n\n{completeness_lines(completeness)}"
        )
    else:
        base = STAGE_MESSAGE[stage]
    if route is None:
        return base
    return f"{base}\n\n{render_route_basis(route)}"


def _arrangement_values(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    arrangements: dict[str, str] | None = None,
) -> dict[str, str]:
    """The draft's arrangement record as stored: a value, or empty.

    Every key of :data:`ARRANGEMENT_FIELDS` is present, and a value can only
    come from one of two places: the three minimal fields the entry form has
    always collected, or ``arrangements`` — the eight supplementary arrangements
    the user actually submitted (ticket 06). ``arrangements`` is a mapping keyed
    by those keys, and only keys :data:`ARRANGEMENT_FIELDS` enumerates are read
    from it, so this function still has no path by which a route fact, a derived
    segment time or a computed cost becomes a value here: a key the checklist
    does not name is dropped, not drawn (AC4/AC6).
    """
    values = {key: "" for key, _label in ARRANGEMENT_FIELDS}
    values["destination_or_route"] = (destination_or_route or "").strip()
    values["activity_date"] = (activity_date or "").strip()
    values["participants"] = f"{participants} 人" if participants else ""
    supplied = arrangements or {}
    for key in SUPPLEMENT_ARRANGEMENT_KEYS:
        value = supplied.get(key)
        values[key] = value.strip() if isinstance(value, str) else ""
    return values


@dataclass(frozen=True, slots=True)
class MissingPlanField:
    """One field the draft does not hold (ticket 05 AC1).

    ``status`` is one of the two markers and nothing else: :data:`UNCONFIRMED`
    for an arrangement the user has not supplied, :data:`UNVERIFIED` for a route
    fact the gate cannot certify. There is no third state — an "inferred",
    "defaulted" or "assumed" field would be the invention the spec forbids
    (docs/plan-pdf-format.md: 未知安排留空显示"待确认"，未知事实显示"待核验").
    """

    key: str
    label: str
    status: str


def missing_plan_fields(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    route: RouteBasis | None = None,
    arrangements: dict[str, str] | None = None,
) -> tuple[MissingPlanField, ...]:
    """The 缺失字段清单, derived from the two existing vocabularies (AC1).

    Nothing is counted, scored or weighted here — ticket 05 only reports what is
    absent, and :func:`evaluate_plan_completeness` (ticket 06) turns that same
    existence answer into a verdict. The list is built from
    :data:`ARRANGEMENT_FIELDS` (the user's own record) and
    :data:`FACT_DIMENSIONS` (the dimensions a route fact can answer), the same
    two tuples the renderers already draw from, so the list cannot drift away
    from the draft it describes:

    * an arrangement is missing when the stored value is empty;
    * a fact dimension is missing when the caller's re-read basis holds no
      certified line for it — which is all seven of them when there is no basis
      at all, and never includes one the gate just certified.
    """
    values = _arrangement_values(
        destination_or_route=destination_or_route,
        activity_date=activity_date,
        participants=participants,
        arrangements=arrangements,
    )
    missing = [
        MissingPlanField(key, label, UNCONFIRMED)
        for key, label in ARRANGEMENT_FIELDS
        if not values[key]
    ]
    certified = {line.field for line in route.facts} if route is not None else set()
    missing += [
        MissingPlanField(field, label, UNVERIFIED)
        for field, label, _unit in FACT_DIMENSIONS
        if field not in certified
    ]
    return tuple(missing)


def render_minimal_draft(
    *,
    destination_or_route: str,
    activity_date: str,
    participants: int | None,
    route: RouteBasis | None = None,
    review: Literal["pending", "confirmed", "final_pending", "final_confirmed"] | None = None,
    arrangements: dict[str, str] | None = None,
) -> str:
    """Render the previewable minimal draft (spec AC1 "可预览的最小草稿").

    Only the three user-supplied arrangements are filled in. Everything the
    user has not given is 待确认 and every route fact is 待核验: the draft
    is explicitly a 草案, never a verified route plan, and this function
    cannot produce a distance, an ascent, a weather claim or a price even
    if asked (docs/plan-pdf-format.md 内容边界).

    ``route`` is ticket 03's exception and ticket 04's type marker, read for
    what it is: when the caller has re-read a certified route basis the blanket
    "facts 待核验" line becomes a pointer to the section rendered below, and
    ``route is None`` IS the ``generic_draft`` case. The flag is derived from
    the evidence the caller re-read, never from a claim, and the same line
    appears in the streamed text, the payload's summary and a restored draft
    because this is the only place it is written.

    ``review`` is ticket 05: the same draft drawn as the FIRST summary, split
    into what the user supplied and what is still missing. AC3 is the reason
    for the split — the first summary confirms only 用户填写的安排和输入, so the
    supplied block is the whole of what a confirmation can cover, and the
    missing block stays on screen (as an explicit 缺失字段清单 heading in the
    draft render, and as the 尚未填写 section in the review render) so nobody
    can read a blank as an agreement. AC4 is enforced by construction rather
    than by copy: the only values this function can print come from
    :func:`_arrangement_values`, which returns an empty string for every field
    the draft does not hold, and from the caller's re-read basis.

    ``review="confirmed"`` is the render the draft carries after
    ``plan_confirm_arrangements``: the same two sections, restated as what the
    user did and did not approve, so the stored stage and the streamed text
    agree about the scope of the confirmation (AC5).
    """
    values = _arrangement_values(
        destination_or_route=destination_or_route,
        activity_date=activity_date,
        participants=participants,
        arrangements=arrangements,
    )
    route_attached = route is not None
    missing = missing_plan_fields(
        destination_or_route=destination_or_route,
        activity_date=activity_date,
        participants=participants,
        route=route,
        arrangements=arrangements,
    )
    arrangement_lines = [
        f"- {label}：{values[key] or UNCONFIRMED}" for key, label in ARRANGEMENT_FIELDS
    ]
    # Ticket 05 AC1: the route-fact dimensions are itemised with the labels from
    # :data:`FACT_DIMENSIONS` — the vocabulary the comparison card and
    # :func:`render_route_basis` already use — instead of a paraphrased blanket
    # line that named 路况 but never 地图 / 开放状态. One vocabulary means the
    # 缺失字段清单 and the draft can never disagree about which dimensions are
    # still unverified, and a certified one is never listed as missing.
    def gap_line(field: MissingPlanField) -> str:
        marker = (
            f"{UNVERIFIED}（无服务器证据）" if field.status == UNVERIFIED else UNCONFIRMED
        )
        return f"- {field.label}：{marker}"

    missing_lines = [gap_line(field) for field in missing]
    missing_fact_lines = [gap_line(field) for field in missing if field.status == UNVERIFIED]
    facts_pointer_line = (
        "- " + " / ".join(label for _field, label, _unit in FACT_DIMENSIONS)
        + f"：见下方「路线核验事实」（未列出者仍为{UNVERIFIED}）"
    )
    type_line = (
        "- 草案类型：route_plan（正式路线计划草案，仍需计划审核）" if route_attached
        else "- 草案类型：generic_draft（通用草案，不是路线计划）"
    )

    if review is None:
        closing = (
            "以上用户安排仍为待确认；路线事实由服务器读取并保留来源，两者分开保存，"
            "整份草稿尚未通过计划审核。"
            if route_attached
            else "以上只是用户填写的安排草案，不是已核验的路线计划。"
        )
        return "\n".join([
            ("**最小计划草稿（路线草案 · 依据已核验路线事实）**" if route_attached
             else "**最小计划草稿（通用草案 · 无已核验路线依据）**"),
            type_line,
            *arrangement_lines,
            *(
                [facts_pointer_line] if route_attached
                else missing_fact_lines or [f"- 路线事实：{UNVERIFIED}（无服务器证据）"]
            ),
            closing,
        ])

    supplied_lines = [line for line in arrangement_lines
                      if not line.endswith(f"：{UNCONFIRMED}")]
    is_final = review in ("final_pending", "final_confirmed")
    if review == "confirmed":
        title = "**第一次摘要（已确认：仅确认由你本人提供的安排）**"
    elif review == "pending":
        title = "**第一次摘要确认（只确认由你本人提供的安排）**"
    elif review == "final_confirmed":
        title = "**最终摘要（已确认：可进入 PDF 审批）**"
    else:
        title = "**最终摘要确认（PDF 生成前的第二次摘要）**"
    supplied_heading = (
        "**由你提供的安排（本人填写，逐项列出）**" if is_final
        else "**由你提供的安排（本摘要只确认这些）**"
    )
    missing_heading = (
        f"**尚未填写的字段（待确认 / {UNVERIFIED}清单 · 进入 PDF 审批前请核对）**" if is_final
        else "**尚未填写的字段（缺失清单 · 等你补充，系统不会代填）**"
    )
    sections = [
        title,
        type_line,
        supplied_heading,
        *(supplied_lines or ["- （尚无填写：以下缺失清单即全部字段）"]),
        missing_heading,
        *(missing_lines or ["- 无缺失字段"]),
    ]
    if route_attached:
        sections.append(facts_pointer_line)
    if is_final:
        final_closing = (
            "你已经确认第一次摘要中的本人安排，以上是第二次（最终）摘要："
            "路线核验结果由服务器复核并保持来源，未填写的字段仍为待确认，"
            "未通过核验的路线事实仍为待核验。"
            + (
                "你已明确确认本次最终摘要，计划书可进入 PDF 审批环节；"
                "未经这份确认不会进入审批，且本确认不代表计划完整、安排已补齐或已通过安全审核。"
                if review == "final_confirmed"
                else "确认以上内容后才会进入 PDF 审批；不确认可以继续补充或取消本次计划书，"
                "系统不会替你生成 PDF，也不会替你判断风险。"
            )
        )
        sections.append(final_closing)
    else:
        sections.append(
            "你已确认以上由你本人提供的安排。未填写的字段仍为待确认，路线事实仍以服务器核验结果为准："
            "本次确认不代表这些安排已补齐，也不代表计划完整或已通过安全审核。"
            if review == "confirmed"
            else "以上未填写的字段一律留空：系统不会替你补充日期、费用、联系电话或撤退地点。"
            "你可以选择「继续补充」，或确认本摘要只反映你本人提供的安排与输入。"
        )
    return "\n".join(sections)


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

