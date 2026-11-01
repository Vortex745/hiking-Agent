"""Ticket 11 — 全流程回归与发布门禁: whole plan-book journeys through the real app.

Tickets 01–10 each proved their own behaviour at their own seam. This module does
not re-prove any of them: it drives **whole journeys** — one conversation, from
the first user message to a downloadable versioned PDF — through the real ASGI
app, the real ``LocalNativeComputer``, a real ``CapacityGuard`` and real temp
SQLite, and it fails when a branch of the plan book is not exercised anywhere.

Seam: ``POST /api/v2/chat/sse`` + ``POST /api/v2/approvals/{id}/decision`` +
``POST /api/v2/chat/resume`` + ``GET /api/v2/artifacts/{id}`` — the public surface
only; ``_stream_plan_flow`` is never called directly. The plan-book driver is the
one ticket 10 built
(:func:`test_plan_recovery_cancel_idempotency._pause_plan_book`), imported rather
than copied so this repo keeps exactly one harness that knows how a plan book
reaches an approval pause.

Acceptance criteria and the journey that proves each (the machine-readable form of
this table is :data:`CRITERIA`, enforced by the four gate tests at the bottom):

* AC1 结构化 intent 的明确入口流程通过 — ``test_journey_structured_intent_entry_publishes_the_book``
* AC2 模糊意图确认流程通过 — ``test_journey_ambiguous_intent_confirms_before_it_opens_the_form``
* AC3 缺失字段草案流程通过 — ``test_journey_missing_fields_supplement_review_and_publish``
* AC4 已有路线比较和自由目的地流程通过 — the ``…existing_route_comparison…``,
  ``…free_destination…`` and ``…research_success_surfaces_candidates_and_the_user_selects_one``
  journeys (the last one is ticket 04 AC2's success arm: research passes the gate,
  the user's own click attaches the route)
* AC5 计划完整度和确认路径完全由确定性规则驱动 — ``test_journey_the_whole_deterministic_path_never_consults_a_model``
  and ``test_journey_completeness_and_confirmation_are_recomputed_not_trusted``:
  every plan click runs against a kernel that RAISES, and the route-decision seam is
  guarded so a provider other than ``deterministic`` — or any live/MCP data call —
  explodes. "Deterministic" is proven by explosion, not by an absence-of-log
  bookkeeping claim
* AC6 no_go 阻止流程通过 — ``test_journey_no_go_blocks_the_plan_from_entry_to_pdf``
* AC7 PDF 失败、取消、恢复和重生成流程通过 — the ``…render_failure…``, ``…cancel…``,
  ``…duplicate_resume…``, ``…refused_approval…``, ``…disconnected…`` and ``…expires…``
  journeys
* AC8 视觉验收覆盖封面、摘要、时间线、风险区和跨页表格 —
  ``test_journey_published_book_renders_the_five_visual_zones_and_a_split_table``

No live model, no network, no gateway. The offline research fixtures speak only for
演示林间环线 (notes-05 landmine 7): journeys that need route evidence ask for that
destination, and the free-destination journey deliberately uses one the fixtures
cannot evidence — that is the branch it is testing.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_api
from agent_v2 import events, kernel
from agent_v2.computer import (
    PLAN_SECTIONS,
    ComputerError,
    LocalNativeComputer,
)
from agent_v2.memory_service import MemoryService
from agent_v2.observability import CapacityGuard
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import (
    CORE_ARRANGEMENT_KEYS,
    MINIMAL_ARRANGEMENT_KEYS,
    PLAN_INTENTS,
    PlanDraftStage,
    SUPPLEMENT_ARRANGEMENT_KEYS,
)
from agent_v2.repositories.memory import MemoryStore
from agent_v2.runs import RunRepository
from api.memory_v2 import SUBJECT_COOKIE_NAME, memory_v2_router

# ---------------------------------------------------------------------------
# Ticket 10's plan-book driver, reused rather than reimplemented.
#
# ``_pause_plan_book`` and friends are the only in-repo harness that drives draft
# turn → route_plan kernel turn (route_compare + report_export) → approval pause →
# decision → resume → ``artifact.available`` → version row over ASGI. They are
# keyed on that module's conversation id and its ``SUBMIT_PAYLOAD``, so every
# journey below runs in that same conversation — which is why :data:`CONV` is
# imported instead of being re-declared.
# ---------------------------------------------------------------------------
import test_plan_recovery_cancel_idempotency as book

CONV = book.CONV
_DEMO_ROUTE = "演示林间环线"
_UNKNOWN_ROUTE = "雨崩上村"
_ALT_ROUTE = "演示山脊路线"
_DATE = "2026-11-08"
_PEOPLE = 12
_CORE = {
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
}
_BLANK = {key: "" for key in SUPPLEMENT_ARRANGEMENT_KEYS}
_CORE_ONLY = _BLANK | _CORE
_ALL_ARRANGEMENTS = _BLANK | {
    **_CORE,
    "leader": "阿强",
    "transport": "包车往返",
    "accommodation": "当日往返，无住宿",
    "budget": "300 元/人",
    "retreat_plan": "体力不支者由领队安排下撤至沟口",
}

#: The stages :class:`agent_v2.plan_workflow.PlanDraftStage` can hold, member name
#: next to wire value. Derived from the enum, so a stage added later cannot be left
#: unvisited silently (gate 3 explains why the pairing matters).
STAGES = {stage.name: stage.value for stage in PlanDraftStage}

#: The plan transitions a client may click — the product whitelist, not a copy.
INTENTS = frozenset(PLAN_INTENTS)

#: Run-terminal event types, derived from the product's own stream-terminal set
#: minus the interrupt types that do NOT end a run (landmine 1: ``run.paused`` is
#: a stream terminal but not a run terminal).
RUN_TERMINAL_EVENTS = events.TERMINAL_EVENT_TYPES - events.RUN_LIFECYCLE_NON_TERMINAL

#: The run-row terminal statuses, from the storage layer's own enumeration.
TERMINAL_RUN_STATUSES = frozenset({"done", "error", "cancelled"})


# ---- harness ---------------------------------------------------------------

def _settings(tmp_path, scenario: str = "complete") -> SimpleNamespace:
    """Ticket 10's settings, with the demo scenario left adjustable per journey."""
    return SimpleNamespace(
        agent_v2_enabled=True, openai_api_key="test", openai_model="test",
        agent_v2_cowork_policy="disabled", agent_v2_state_dir=str(tmp_path),
        route_compare_data_mode="offline", route_decision_provider="deterministic",
        route_compare_demo_scenario=scenario,
    )


class _ExplodingKernel:
    """A kernel that refuses to be the reason a plan click worked.

    Ticket 06's determinism claim is "no model on this path". The strongest
    end-to-end form of that is a model call that fails the test: a journey that
    quietly fell through to the kernel cannot pass because a polite fake returned
    ``"好的。"``.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, **kwargs):
        self.calls.append(str(kwargs.get("message")))
        raise AssertionError(f"the plan path consulted a model: {kwargs.get('message')!r}")


class _DeterminismSentinel:
    """The other half of AC5: not only no model — no decision provider either.

    :class:`_ExplodingKernel` closes the kernel door. The route-decision seam is a
    second, independent door: ``agent_v2.route_tools.RouteSession`` builds its
    provider through ``select_provider(provider_config(settings))``, which can be
    ``llm`` (a chat model) or ``external`` (the Typesafe/Jev adapter), and its
    ``LiveTools.call`` is the network/MCP data door. A plan path that quietly asked
    for either would still be "deterministic" as long as the config happened to be
    right in this process — a config claim, not a behaviour claim.

    So both doors are guarded at runtime: the sentinel records every provider the
    path asks for and refuses anything but ``deterministic``, and it refuses any
    live tool call outright. What a journey then asserts is that the whole
    deterministic path passed WITH the guard installed.
    """

    def __init__(self) -> None:
        self.providers: list[tuple[str, str]] = []
        self.live_calls: list[str] = []

    def install(self, monkeypatch) -> "_DeterminismSentinel":
        import agent_v2.route_tools as route_tools

        real_select = route_tools.select_provider

        def select(config):
            self.providers.append((config.provider, config.data_mode))
            if config.provider != "deterministic":
                raise AssertionError(
                    f"the plan path asked for the {config.provider!r} decision provider")
            return real_select(config)

        async def live_call(self_tools, server, tool, args):
            self.live_calls.append(f"{server}:{tool}")
            raise AssertionError(
                f"the plan path consulted a live data source: {server}:{tool}")

        monkeypatch.setattr(route_tools, "select_provider", select)
        monkeypatch.setattr(route_tools.LiveTools, "call", live_call)
        return self


@pytest.fixture
def flow(tmp_path, monkeypatch):
    """One rebuildable v2 runtime per journey: real SQLite, real capacity guard.

    Mirrors ticket 10's fixture — every ``build()`` makes a fresh app over the SAME
    database files, so what a journey asserts afterwards is durable state, and the
    capacity guard is a real one because AC7's claims are about the slot. ``settings``
    is exposed because the no-go journey flips ``route_compare_demo_scenario``.
    """
    settings = _settings(tmp_path)
    monkeypatch.setattr(chat_api, "settings", settings)

    repos: list[RunRepository] = []
    model_calls: list[str] = []

    def build() -> FastAPI:
        repo = RunRepository(
            control=ControlStore(str(tmp_path / "control.sqlite")),
            checkpoints=CheckpointsStore(str(tmp_path / "checkpoints.sqlite")),
        )
        repos.append(repo)
        memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
        app = FastAPI()
        app.state.agent_v2_runtime = SimpleNamespace(
            runs=repo, memory=memory,
            capacity=CapacityGuard(max_runs=5, max_sandboxes=2),
        )
        app.include_router(chat_api.chat_v2_router, prefix="/api/v2")
        app.include_router(chat_api.approvals_v2_router, prefix="/api/v2")
        app.include_router(chat_api.artifacts_v2_router, prefix="/api/v2")
        app.include_router(memory_v2_router, prefix="/api/v2")
        return app

    app = build()
    memory = app.state.agent_v2_runtime.memory
    subject = memory.resolve_subject(None)
    yield SimpleNamespace(app=app, build=build, repo=repos[-1], memory=memory,
                          subject=subject, settings=settings, path=tmp_path,
                          model_calls=model_calls)
    for repo in repos:
        repo.close()
    memory.close()


def _plan(request_id: str, *, message: str = "生成行程", intent: str | None = None,
          **extra) -> dict[str, Any]:
    """One plan click on the wire: an ordinary conversational turn carrying an intent.

    Never the tool-approval pause (seam contract ruling): confirmations ride
    ``intent`` + fields on ``profile_id="chat"``, the only profile that may carry
    them.
    """
    body: dict[str, Any] = {"message": message, "request_id": request_id,
                            "conversation_id": CONV, "profile_id": "chat"}
    if intent is not None:
        body["intent"] = intent
    body.update(extra)
    return body


def _submit_fields(*, destination: str = _DEMO_ROUTE,
                   arrangements: dict[str, str] | None = None) -> dict[str, Any]:
    """The ``plan_create`` kwargs for a submission, for :func:`_send`.

    Only the two fields the entry form owns: a journey never sends a route fact,
    and ``extra="forbid"`` on the request model is what keeps that true.
    """
    return {"intent": "plan_create", "plan_input": {
        "destination_or_route": destination, "activity_date": _DATE,
        "participants": _PEOPLE, "arrangements": arrangements}}


def _send(flow, request_id: str, **kwargs) -> list[dict[str, Any]]:
    return book._post_sse(flow.app, flow.subject, _plan(request_id, **kwargs))


def _stage(frames: list[dict[str, Any]]) -> str:
    return _payload(frames)["stage"]


def _payload(frames: list[dict[str, Any]]) -> dict[str, Any]:
    drafts = book._drafts(frames)
    assert drafts, (
        f"no plan.draft_state in {[f['type'] for f in frames]} — "
        f"refused with {[f['payload'] for f in frames if f['type'] == 'error']}"
    )
    return drafts[-1]["payload"]


def _blocked(frames: list[dict[str, Any]]) -> str:
    """The stable reason a refused plan click came back with."""
    error = next((f for f in frames if f["type"] == "error"), None)
    assert error is not None, f"expected a refused turn, got {[f['type'] for f in frames]}"
    assert error["payload"]["code"] == "plan_intent_blocked", error["payload"]
    return error["payload"]["reason"]


def _terminal(frames: list[dict[str, Any]]) -> str:
    return frames[-1]["type"]


#: The single advancing click for each source stage, so a draft can be walked to
#: the ``final_reviewed`` stage the PDF door reads (ticket 07 AC4). Every turn is
#: a model-free plan-flow intent (the seam contract's ruling), so this never
#: touches the kernel — the AC5 determinism journeys still see ``calls == []``.
_ADVANCE = {
    PlanDraftStage.MINIMAL_DRAFT.value: "plan_first_review",
    PlanDraftStage.AWAITING_FIRST_REVIEW.value: "plan_confirm_arrangements",
    PlanDraftStage.FIRST_REVIEWED.value: "plan_final_review",
    PlanDraftStage.AWAITING_FINAL_REVIEW.value: "plan_confirm_final",
}


def _advance_to_final(flow, *, tag: str) -> None:
    """Walk a submitted draft through the review intents to ``final_reviewed``.

    Idempotent: a draft already past any of these stages only performs the steps
    it has not, and a draft already at ``final_reviewed`` does nothing. Used by
    the publish helpers so every journey that expects a PDF has genuinely
    confirmed the final review — the confirmation the door now enforces.
    """
    for step in range(5):
        stage = flow.repo.get_plan_draft(CONV).stage
        if stage == PlanDraftStage.FINAL_REVIEWED.value:
            return
        intent = _ADVANCE.get(stage)
        assert intent is not None, f"cannot advance a plan at stage {stage!r}"
        frames = _send(flow, f"{tag}-advance-{step}", message="确认", intent=intent)
        assert _terminal(frames) == "done", (intent, frames[-1])
    raise AssertionError("plan did not reach final_reviewed")


def _install_export_model(
    flow, monkeypatch, *, meeting_point: str = _CORE["meeting_point"],
    stages: list[dict[str, str]] | None = None, export: bool = True,
    route_names: tuple[str, ...] = (_DEMO_ROUTE,),
) -> None:
    """A route_plan turn that runs the real comparison and (optionally) the export.

    Keeps ticket 10's separation: the turn carries NO plan intent, and the book is
    queued by the ``report_export`` tool call — never by ``plan_create`` merged onto
    ``profile_id="route_plan"``, which would route into the model-free plan flow and
    publish nothing at all (landmine 5). With ``export=False`` the turn stops at
    ``route_compare``: that is how a journey obtains an *existing* comparison.
    """

    async def model(**kwargs):
        flow.model_calls.append(str(kwargs.get("message")))
        compare = next(t for t in kwargs["extra_tools"] if t.name == "route_compare")
        result = json.loads((await compare(candidates=[
            {"name": name, "city": "演示城市", "origin": "起点", "destination": "终点"}
            for name in route_names])).output)
        if export:
            report = next(t for t in kwargs["extra_tools"] if t.name == "report_export")
            plan_details: dict[str, Any] = {
                "activity_date": _DATE, "meeting_time": _CORE["meeting_time"],
                "meeting_point": meeting_point, "participants": _PEOPLE,
                "emergency_contact": _CORE["emergency_contact"],
            }
            if stages is not None:
                plan_details["itinerary"] = stages
            await report(comparison_id=result["comparison_id"], plan_details=plan_details)
        return "模型正文不应进入计划书", []

    monkeypatch.setattr(kernel, "run_chat_turn", model)


def _pause_book(flow, monkeypatch, request_id: str, *, produced: list[str],
                meeting_point: str = _CORE["meeting_point"],
                stages: list[dict[str, str]] | None = None,
                computer: Any | None = None,
                approve: bool = True):
    """Drive the export turn to its approval pause (and record the decision).

    Ticket 10's ``_pause_and_approve`` is the reference, but it installs ITS model
    and ITS computer, so it cannot carry this gate's knobs: a long itinerary that
    forces a table across a page (AC8), a computer whose render fails (AC7), or a
    refusal instead of an approval. The wire sequence is identical — an
    ``_export_payload`` turn, ``approval.required``, the decision POST, no resume —
    and the same module's ``_resume`` / ``_recording_computer`` still do the work.

    Ticket 07 AC4: publishing now requires a confirmed final review, so the helper
    walks the conversation's draft to ``final_reviewed`` (model-free) before the
    export turn — every lifecycle journey that expects a PDF gets one, and the
    refusal journey is the deliberate exception that drives the export itself
    without confirming.
    """
    _advance_to_final(flow, tag=request_id)
    _install_export_model(flow, monkeypatch, meeting_point=meeting_point, stages=stages)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: computer or book._recording_computer(produced))

    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=flow.app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: flow.subject.token},
        ) as client:
            paused = book._frames(await client.post(
                "/api/v2/chat/sse", json=book._export_payload(request_id)))
            assert paused[-1]["type"] == "run.paused", paused
            approval = next(e["payload"] for e in paused if e["type"] == "approval.required")
            if approve:
                decided = await client.post(
                    f"/api/v2/approvals/{approval['approval_id']}/decision",
                    json={"decision": "approve", "decision_request_id": f"dec-{request_id}"})
                assert decided.status_code == 200
            return paused, approval

    paused, approval = asyncio.run(run())
    return paused, paused[-1]["run_id"], paused[-1]["sequence"], approval


def _publish(flow, monkeypatch, request_id: str, *, meeting_point: str = _CORE["meeting_point"],
             stages: list[dict[str, str]] | None = None,
             computer: Any | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """Drive the export turn to a pause, approve it, resume it: the whole book."""
    produced: list[str] = []
    paused, run_id, last_seq, _approval = _pause_book(
        flow, monkeypatch, request_id, produced=produced, meeting_point=meeting_point,
        stages=stages, computer=computer)
    return paused, book._resume(flow.app, flow.subject, run_id, last_seq), produced


def _decide(app, subject, approval_id: str, key: str, decision: str = "approve"):
    """Record an approval decision through the public endpoint."""
    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: subject.token},
        ) as client:
            return await client.post(
                f"/api/v2/approvals/{approval_id}/decision",
                json={"decision": decision, "decision_request_id": key})

    return asyncio.run(run())


def _download(flow, artifact_id: str):
    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=flow.app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: flow.subject.token},
        ) as client:
            return await client.get(f"/api/v2/artifacts/{artifact_id}")

    return asyncio.run(run())


def _published(flow, frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The one ``artifact.available`` payload a journey published."""
    available = book._available(frames)
    assert len(available) == 1, [f["payload"] for f in available]
    return available[0]["payload"]


def _run_rows(repo: RunRepository) -> dict[str, str]:
    return {row[0]: row[1]
            for row in repo.control._conn.execute("SELECT run_id, status FROM runs")}


def _assert_terminal_ledger(flow) -> dict[str, str]:
    """The release-gate invariant, applied to EVERY run the journey produced.

    **A run row in a terminal state must hold a run-terminal event in its log —
    exactly one, and it must name the same terminal.**

    Ticket 10 made this true for pause-then-cancel and pause-then-expire, the two
    paths that used to break it; a gate that never sweeps it can let a third path
    re-open the hole, and the visible symptom is a reconnecting client being told
    by ``GET /chat/events`` that a run still awaits a decision it no longer has.

    ``paused`` is the asserted mirror case: a terminal *stream* event with a
    non-terminal run row is legitimate while the run waits for a decision — which
    is exactly the distinction landmine 1 warns about, so the sweep checks it
    instead of "repairing" every paused run.

    Returns ``{terminal_status: run_id}`` for the caller to assert on, so a journey
    that silently stopped reaching one terminal fails rather than sweeping a
    smaller set and reporting green.
    """
    repo: RunRepository = flow.repo
    rows = _run_rows(repo)
    assert TERMINAL_RUN_STATUSES == RUN_TERMINAL_EVENTS, (
        "the run-status vocabulary and the event-name vocabulary drifted apart; "
        "this sweep's mapping is only valid while they agree"
    )
    seen: dict[str, str] = {}
    for run_id, status in rows.items():
        logged = [e["type"] for e in repo.get_events_since(run_id=run_id, from_seq=0)]
        terminals = [name for name in logged if name in RUN_TERMINAL_EVENTS]
        if status in TERMINAL_RUN_STATUSES:
            assert terminals == [status], (
                f"run {run_id} is {status!r} but its log holds run-terminal events "
                f"{terminals} out of {logged}"
            )
            seen[status] = run_id
        else:
            assert terminals == [], (
                f"run {run_id} is {status!r} (non-terminal) yet its log already holds "
                f"{terminals}: the stream closed a run the row says is still alive"
            )
            assert status == "paused" and logged and logged[-1] == "run.paused", (
                f"run {run_id} is {status!r} with log {logged}"
            )
    return seen


# ---- AC1: the structured-intent entry journey, end to end ------------------

def test_journey_structured_intent_entry_publishes_the_book(flow, monkeypatch):
    """AC1: a ``plan_create`` intent opens the draft, and the same conversation
    ends holding a downloadable, versioned PDF.

    What makes this a journey and not a unit proof: the entry turn is model-free
    (the installed kernel RAISES), the stage on the wire and the stage in
    ``plan_drafts`` are one fact in two places, and the published book is the one
    the export turn queued — with real bytes behind the download endpoint.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)

    frames = _send(flow, "e11-create", **{
        "intent": "plan_create", "plan_input": {
            "destination_or_route": _DEMO_ROUTE, "activity_date": _DATE,
            "participants": _PEOPLE, "arrangements": _ALL_ARRANGEMENTS}})
    assert _terminal(frames) == "done"
    assert _stage(frames) == PlanDraftStage.MINIMAL_DRAFT.value
    payload = _payload(frames)
    assert payload["intent"] == "plan_create"
    assert payload["source"] == "structured_intent"
    assert payload["draft"]["arrangements"] == _ALL_ARRANGEMENTS
    assert payload["rules"]["verdict"] == "confirmation_required"
    assert [r["code"] for r in payload["rules"]["reasons"]] == ["route_facts_unverified"]
    assert boom.calls == []

    stored = flow.repo.get_plan_draft(CONV)
    assert stored is not None and stored.stage == payload["stage"]

    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-book")

    assert _paused[-1]["type"] == "run.paused"
    required = next(e for e in _paused if e["type"] == "approval.required")
    assert required["payload"]["tool"] == "generate_pdf"
    assert required["payload"]["risk_level"] == "high"
    assert required["payload"]["args"] == {"filename": "hiking-plan.pdf", "plan_revision": 1}

    artifact = _published(flow, resumed)
    assert artifact["plan_revision"] == 1
    assert artifact["previous_artifact_id"] == ""
    assert len(produced) == 1 and produced[0].lstrip().startswith("{")
    document = json.loads(produced[0])
    assert document["kind"] == "route_comparison_report"

    downloaded = _download(flow, artifact["artifact_id"])
    assert downloaded.status_code == 200
    assert downloaded.content.startswith(b"%PDF-")
    assert flow.repo.get_current_plan_artifact_version(CONV).artifact_id == artifact["artifact_id"]
    assert _terminal(resumed) == "done"
    assert flow.model_calls == ["请导出带来源的计划书 PDF"]
    assert _assert_terminal_ledger(flow) == {"done": resumed[-1]["run_id"]}


def test_journey_an_unconfirmed_draft_cannot_reach_a_pdf(flow, monkeypatch):
    """Ticket 07 AC4, the door itself: no final-review confirmation → no PDF.

    A ``plan_create`` leaves the draft at ``minimal_draft``. The export turn's
    ``report_export`` PASSES ``render_report`` (eligibility is intact — the route
    is a real gate-passed comparison), yet the book must still not be queued: PDF
    approval requires the draft to have reached ``final_reviewed`` and a verdict
    on record, which is exactly what :func:`pdf_approval_block_reason` reads. The
    turn refuses before the tool-approval pause, publishes nothing, and leaves no
    version row — while a draft that HAS confirmed the final review publishes (the
    journey above). render_report stays the eligibility authority; this is the
    additional confirmation door AC4 asked for.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)
    _send(flow, "e11-unconf-create", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    assert flow.repo.get_plan_draft(CONV).stage == PlanDraftStage.MINIMAL_DRAFT.value

    produced: list[str] = []
    _install_export_model(flow, monkeypatch)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: book._recording_computer(produced))

    export = book._post_sse(flow.app, flow.subject, book._export_payload("e11-unconf-book"))
    # Refused before any approval is requested, and nothing rendered.
    assert [f for f in export if f["type"] == "approval.required"] == []
    assert book._available(export) == []
    assert produced == [], "an unconfirmed draft published a PDF"
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert flow.repo.get_current_plan_artifact_version(CONV) is None
    # The refusal is the stable AC4 reason, not a crash and not eligibility.
    refused = next(f for f in export if f["type"] == "error")
    assert refused["payload"]["code"] == "plan_pdf_approval_required", refused["payload"]
    assert refused["payload"]["reason"] == "final_review_not_pending", refused["payload"]
    # The draft is untouched — the refusal stores nothing.
    assert flow.repo.get_plan_draft(CONV).stage == PlanDraftStage.MINIMAL_DRAFT.value
    _assert_terminal_ledger(flow)


def test_a_confirmed_draft_clears_the_pdf_door_the_unconfirmed_one_could_not(flow, monkeypatch):
    """The other arm of the same door: confirming the final review lets it publish.

    Pairs with the refusal journey above so the gate is shown to be about the
    stage the confirmation writes, not about eligibility or a fluke.
    """
    _send(flow, "e11-conf-create", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    _advance_to_final(flow, tag="e11-conf")
    assert flow.repo.get_plan_draft(CONV).stage == PlanDraftStage.FINAL_REVIEWED.value
    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-conf-book")
    assert _published(flow, resumed)["plan_revision"] == 1
    assert len(produced) == 1
    _assert_terminal_ledger(flow)


# ---- AC2: the ambiguous-intent confirmation journey ------------------------

def test_journey_ambiguous_intent_confirms_before_it_opens_the_form(flow, monkeypatch):
    """AC2: planning-shaped-but-ambiguous text parks the draft, and the user's
    ``plan_confirm`` is what opens the form — and that confirmation is not an
    approval.

    The parked stage never reaches a PDF either: the journey checks that the only
    pause in the whole conversation is the ``generate_pdf`` one the export turn
    queues once the draft is real.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)

    ambiguous = _send(flow, "e11-ambiguous", message="周末想去爬山，还没定")
    assert _terminal(ambiguous) == "done"
    assert _stage(ambiguous) == PlanDraftStage.AWAITING_CONFIRMATION.value
    assert _payload(ambiguous)["source"] == "natural_language"
    assert _payload(ambiguous)["draft"]["destination_or_route"] == ""
    assert [f for f in ambiguous if f["type"] == "approval.required"] == []

    # The ask explains itself in text instead of opening a dialog on the user's
    # behalf, and no model was consulted to decide that.
    spoken = "".join(f["payload"].get("content", "") for f in ambiguous
                     if f["type"] == "text")
    assert "计划书" in spoken and ("是否" in spoken or "确认" in spoken)

    # Only the user's own confirmation opens the form.
    confirmed = _send(flow, "e11-confirm", message="好的，生成计划书", intent="plan_confirm")
    assert _stage(confirmed) == PlanDraftStage.AWAITING_INPUT.value

    submitted = _send(flow, "e11-form", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    assert _stage(submitted) == PlanDraftStage.MINIMAL_DRAFT.value

    # The confirmation is spent: a second one cannot re-open a door already walked
    # through, and it cannot move a draft that has a stage of its own.
    again = _send(flow, "e11-confirm-again", message="再确认一次", intent="plan_confirm")
    assert _blocked(again) == "confirm_without_pending_confirmation"
    assert flow.repo.get_plan_draft(CONV).stage == "minimal_draft"
    assert boom.calls == []

    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-ambiguous-book")
    assert _published(flow, resumed)["plan_revision"] == 1
    assert boom.calls == [], "the confirmation path fell through to the kernel"
    _assert_terminal_ledger(flow)


# ---- AC3: the missing-field draft journey ----------------------------------

def test_journey_missing_fields_supplement_review_and_publish(flow, monkeypatch):
    """AC3: missing core arrangements bounce 直接生成草案 back to the form; 继续补充
    keeps collecting; the 待确认清单 is what the first summary shows; and the journey
    still ends in a published book.

    Ticket 05's and 06's branches in ONE conversation instead of one test each:
    缺失清单 → 继续补充 → 直接生成草案 → 确认安排 → 查看最终摘要 → 确认最终摘要 → PDF.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)

    thin = _send(flow, "e11-thin", **_submit_fields(arrangements=_BLANK))
    assert _stage(thin) == PlanDraftStage.MINIMAL_DRAFT.value
    checklist = {entry["key"] for entry in _payload(thin)["missing_fields"]}
    assert set(CORE_ARRANGEMENT_KEYS) - set(MINIMAL_ARRANGEMENT_KEYS) <= checklist
    assert _payload(thin)["rules"]["verdict"] == "supplement_required"

    # 直接生成草案 over a draft missing a CORE arrangement: bounced back to the
    # form, with the reason spelled out in the same summary the client shows.
    bounced = _send(flow, "e11-review-too-early", message="直接生成草案",
                    intent="plan_first_review")
    assert _stage(bounced) == PlanDraftStage.AWAITING_INPUT.value
    assert "core_arrangements_missing" in _payload(bounced)["summary"]

    # 继续补充: the other answer to the missing-field list, offered on a draft —
    # so it is issued from ``minimal_draft`` again after the form is refilled,
    # and it reopens the form without cancelling anything.
    _send(flow, "e11-half", **_submit_fields(arrangements=_BLANK | {"leader": "阿强"}))
    supplement = _send(flow, "e11-more-input", message="继续补充", intent="plan_more_input")
    assert _stage(supplement) == PlanDraftStage.AWAITING_INPUT.value
    # …and from a stage that shows no list at all, it is a stable refusal.
    nowhere = _send(flow, "e11-more-input-none", message="继续补充", intent="plan_more_input")
    assert _blocked(nowhere) == "supplement_without_draft"

    filled = _send(flow, "e11-refill", **_submit_fields(arrangements=_CORE_ONLY))
    assert _payload(filled)["rules"]["verdict"] == "confirmation_required"

    review = _send(flow, "e11-review", message="直接生成草案", intent="plan_first_review")
    assert _stage(review) == PlanDraftStage.AWAITING_FIRST_REVIEW.value
    # The three core ones are gone; the non-core ones are the 待确认清单.
    assert [entry["key"] for entry in _payload(review)["missing_fields"]][:5] == [
        "leader", "transport", "accommodation", "budget", "retreat_plan"]
    assert "由你提供的安排" in _payload(review)["summary"]

    # A confirmation for a summary that was never shown is refused — then, is kept.
    early = _send(flow, "e11-confirm-early", message="确认最终摘要", intent="plan_confirm_final")
    assert _blocked(early) == "final_review_not_pending"

    confirmed = _send(flow, "e11-confirm-arrangements", message="确认我提供的安排",
                      intent="plan_confirm_arrangements")
    assert _stage(confirmed) == PlanDraftStage.FIRST_REVIEWED.value

    final = _send(flow, "e11-final-review", message="查看最终摘要", intent="plan_final_review")
    assert _stage(final) == PlanDraftStage.AWAITING_FINAL_REVIEW.value
    assert _payload(final)["plan_type"] == "generic_draft"
    assert "待确认" in _payload(final)["summary"]

    accepted = _send(flow, "e11-confirm-final", message="确认最终摘要", intent="plan_confirm_final")
    assert _stage(accepted) == PlanDraftStage.FINAL_REVIEWED.value
    assert boom.calls == []

    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-missing-fields-book")
    artifact = _published(flow, resumed)
    assert _download(flow, artifact["artifact_id"]).content.startswith(b"%PDF-")
    # The published book records which draft it came from: the audit trail the
    # 待确认清单 keeps all the way to the bytes that were handed out.
    version = flow.repo.list_plan_artifact_versions(CONV)[0]
    assert version.draft_id == flow.repo.get_plan_draft(CONV).draft_id
    # The first summary confirmed only the user's own arrangements, so the missing
    # non-core ones are still 待确认 in the published document.
    assert "待确认" in produced[0]
    _assert_terminal_ledger(flow)


# ---- AC4a: the existing-comparison journey ---------------------------------

def test_journey_existing_route_comparison_selects_a_route_and_publishes(flow, monkeypatch):
    """AC4 first half: a comparison the app already produced becomes the draft's
    route basis, its facts keep their sources, and the book publishes.

    The comparison is produced by a real ``route_plan`` turn that calls
    ``route_compare`` and stops there — that is what 已有路线比较 means on the wire —
    so ``plan_select_route`` resolves a reference the server signed rather than a
    payload the journey prepared.
    """
    _install_export_model(flow, monkeypatch, export=False,
                          route_names=(_DEMO_ROUTE, _ALT_ROUTE))
    compared = book._post_sse(flow.app, flow.subject, {
        "message": "比较一下演示路线", "request_id": "e11-compare",
        "conversation_id": CONV, "profile_id": "route_plan"})
    result = next(f["payload"] for f in compared if "route_comparison" in f["payload"])
    comparison_id = result["route_comparison"]["comparison_id"]
    assert result["route_comparison"]["status"] == "recommended"
    # What the server offered is what the journey may choose: the offline demo
    # comparison recommends one route and (in the complete scenario) lists no
    # alternative, so the journey takes the recommendation rather than inventing a
    # second candidate to click. A candidate outside this list is refused below.
    offered = ([result["route_comparison"]["recommended_id"]]
               + list(result["route_comparison"]["alternatives"]))
    assert offered, result["route_comparison"]
    chosen = offered[-1]
    chosen_name = next(
        entry["candidate"]["name"] for entry in result["route_comparison"]["candidates"]
        if entry["candidate"]["id"] == chosen)

    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)
    _send(flow, "e11-select-submit", **_submit_fields(arrangements=_CORE_ONLY))

    selected = _send(flow, "e11-select", message="选用推荐路线", intent="plan_select_route",
                     route_selection={"comparison_id": comparison_id,
                                      "candidate_id": chosen})
    payload = _payload(selected)
    assert payload["stage"] == PlanDraftStage.MINIMAL_DRAFT.value
    assert payload["route_selection"] == {"comparison_id": comparison_id,
                                          "candidate_id": chosen}
    assert payload["plan_type"] == "route_plan", (
        "a gate-passed basis is the only thing that makes a route plan")
    basis = flow.repo.get_route_selection(CONV)
    assert basis is not None and basis.candidate_id == chosen

    # A candidate the comparison did not offer is refused with a stable reason —
    # the gate's list, not the client's guess (ticket 03 AC4).
    refused = _send(flow, "e11-select-outside", message="选另一条", intent="plan_select_route",
                    route_selection={"comparison_id": comparison_id,
                                     "candidate_id": "demo-not-offered"})
    refusal = next(f for f in refused if f["type"] == "error")["payload"]
    # A selection the comparison cannot honour is its own stable code with its own
    # stable reason — and the draft keeps no route basis it did not earn.
    assert refusal["code"] == "route_selection_unavailable", refusal
    assert refusal["reason"] == "route_not_selectable", refusal
    assert flow.repo.get_route_selection(CONV).candidate_id == chosen

    review = _send(flow, "e11-select-review", message="直接生成草案", intent="plan_first_review")
    summary = _payload(review)["summary"]
    assert _stage(review) == PlanDraftStage.AWAITING_FIRST_REVIEW.value
    # The route the draft points at is named from the server's own re-read, not
    # from anything the client sent; and a dimension the comparison could not
    # certify keeps its 待核验 marker (seam contract invariant 1).
    assert chosen_name in summary
    assert boom.calls == []

    # A client cannot claim the opposite: 声明 route_plan is refused by the rules.
    claim = _send(flow, "e11-select-claim", message="继续生成通用草案",
                  intent="plan_generic_draft", plan_type="route_plan")
    assert _blocked(claim) == "invalid_plan_type"

    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-selected-book")
    assert _published(flow, resumed)["plan_revision"] == 1
    document = json.loads(produced[0])
    assert document["rows"], "the book carries the compared candidates"
    assert document["sources"], "and the sources behind them"
    assert document["recommended"], "and which one the gate recommended"
    for fact in document["route_overview"]:
        if fact["value"] != "待核验":
            assert fact["label"] in {"距离", "累计爬升", "预计耗时", "天气", "开放状态",
                                     "风险状态", "资料状态", "计划路线", "路线地图"}
    _assert_terminal_ledger(flow)


# ---- AC4b: the free-destination journey ------------------------------------

def test_journey_free_destination_research_falls_back_to_a_generic_draft(flow, monkeypatch):
    """AC4 second half: an unknown destination researches, refuses to become a
    formal route plan, and the USER's choice of a generic draft is what continues.

    Honest scope: the offline fixtures can only ever speak for 演示林间环线, so
    "research found nothing verifiable" is a property of the fixture set as much as
    of 雨崩上村 — but the refusal runs through the real deterministic gate
    (:func:`route_compare.service.compare_routes` → ``assess_candidate``), which is
    the branch the journey is about. The fixture is complicit in exactly one way:
    it cannot show a destination whose material IS thin for real reasons.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)

    _send(flow, "e11-free-submit", **_submit_fields(destination=_UNKNOWN_ROUTE,
                                                  arrangements=_CORE_ONLY))

    researched = _send(flow, "e11-free-research", message="研究这条路线",
                       intent="plan_research_route")
    assert _stage(researched) == PlanDraftStage.AWAITING_GENERIC_CHOICE.value
    comparison = next(f for f in researched if f["type"] == "tool_result")["payload"]
    assert comparison["route_comparison"]["status"] != "recommended"
    assert comparison["route_comparison"]["status"] in ("insufficient_evidence",
                                                        "confirmation_required")
    # AC3 of ticket 04: thin material never becomes a formal route plan, and it is
    # NOT the hard stop ticket 06/07 reserve for a refused route.
    safety = flow.repo.get_plan_safety(CONV)
    assert safety is not None and safety.hard_stop is False
    assert "evidence_gate_failed" in safety.codes

    # Refusing to choose keeps the park: an advancing click cannot launder it.
    advanced = _send(flow, "e11-free-review", message="直接生成草案", intent="plan_first_review")
    assert _blocked(advanced) == "first_review_without_draft"

    generic = _send(flow, "e11-generic", message="继续生成通用草案",
                    intent="plan_generic_draft", plan_type="generic_draft")
    assert _stage(generic) == PlanDraftStage.MINIMAL_DRAFT.value
    assert _payload(generic)["plan_type"] == "generic_draft"
    assert "通用草案" in _payload(generic)["summary"]
    assert [f for f in generic if f["type"] == "approval.required"] == []

    review = _send(flow, "e11-generic-review", message="直接生成草案",
                   intent="plan_first_review")
    assert _stage(review) == PlanDraftStage.AWAITING_FIRST_REVIEW.value
    assert "待核验" in _payload(review)["summary"]

    # And the OTHER free-destination exit is honoured: cancel clears the park.
    cancelled = _send(flow, "e11-free-cancel", message="取消计划", intent="plan_cancel")
    assert _stage(cancelled) == PlanDraftStage.CANCELLED.value
    assert boom.calls == [] and flow.model_calls == []
    _assert_terminal_ledger(flow)


def test_journey_research_success_surfaces_candidates_and_the_user_selects_one(flow, monkeypatch):
    """AC4's SUCCESS arm (ticket 04 AC2): research that clears the gate hands the
    user selectable candidates, and the user's own click is what attaches one.

    The reviewed gap was that this branch was never walked: a passing research turn
    stores no route reference, so nothing proved the 04→03 handover. The decision
    recorded here is that this is CORRECT and the missing thing was the proof:
    ticket 03's 「用户可以选择推荐路线或备选路线」 makes selection an explicit user act,
    and notes-09's no-auto-select precedent forbids the server choosing for the
    user. So a research ``ok`` turn SURFACES (the comparison rides the event, and
    its ``comparison_id`` is loadable for the selection leaf) while ``plan_type``
    stays DERIVED from ``route_basis is not None`` — generic until the click,
    ``route_plan`` only after the leaf re-reads a gate-passed candidate.

    Honest scope: the offline fixtures can only ever recommend 演示林间环线
    (notes-05 landmine 7), so that is the destination this journey researches —
    the counterpart 资料不足 arm is the journey above.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)
    sentinel = _DeterminismSentinel().install(monkeypatch)

    _send(flow, "e11-ok-submit", **_submit_fields(arrangements=_CORE_ONLY))

    # --- the success turn: evidence is shown, nothing is attached -------------
    researched = _send(flow, "e11-ok-research", message="研究这条路线",
                       intent="plan_research_route")
    comparison = next(f for f in researched if f["type"] == "tool_result")["payload"][
        "route_comparison"]
    assert comparison["status"] == "recommended", comparison["status"]
    payload = _payload(researched)
    assert payload["stage"] == PlanDraftStage.MINIMAL_DRAFT.value, (
        "a passing research result leaves the draft where the user submitted it")
    # AC2's surface, not a shortcut: the candidates and their sources travel on the
    # event, and no fact is copied into the draft (seam contract invariant 1).
    offered = {entry["candidate"]["id"]: entry for entry in comparison["candidates"]}
    selectable = [comparison["recommended_id"], *comparison["alternatives"]]
    assert selectable and all(cid in offered for cid in selectable)
    assert offered[comparison["recommended_id"]]["evidence_status"] == "complete"
    assert offered[comparison["recommended_id"]]["candidate"]["sources"], (
        "a candidate the user may pick carries its own sources")
    # The no-auto-select pin: with no user act there is no basis, so the draft is
    # still a generic draft and holds no route reference — in the payload, in the
    # row, and in the summary the user is reading.
    assert payload["plan_type"] == "generic_draft"
    assert payload["route_selection"] is None
    assert flow.repo.get_route_selection(CONV) is None
    assert "路线核验事实" not in payload["summary"]

    # Ticket 07's pass record is stored verbatim on the success arm, which is what
    # the PDF cover's 安全门 line reads later instead of staying 待核验.
    gate = flow.repo.get_plan_safety(CONV)
    assert gate is not None and gate.hard_stop is False and gate.codes == ()

    # --- the user's act: selecting is what makes this a route plan ------------
    chosen = comparison["recommended_id"]
    selected = _send(flow, "e11-ok-select", message="选用推荐路线", intent="plan_select_route",
                     route_selection={"comparison_id": comparison["comparison_id"],
                                      "candidate_id": chosen})
    picked = _payload(selected)
    assert picked["stage"] == PlanDraftStage.MINIMAL_DRAFT.value
    assert picked["route_selection"] == {"comparison_id": comparison["comparison_id"],
                                         "candidate_id": chosen}
    assert picked["plan_type"] == "route_plan", (
        "the DERIVED marker: a gate-passed basis the leaf re-read, never a claim")
    assert flow.repo.get_route_selection(CONV).candidate_id == chosen
    # Facts render WITH their provenance — the source id the comparison carried.
    source_id = offered[chosen]["candidate"]["sources"][0]["id"]
    assert f"来源 [{source_id}]" in picked["summary"], picked["summary"]
    assert "演示林间环线" in picked["summary"]

    # A first review over the same draft keeps the sourced facts: the summary the
    # user confirms is the review-grade render of the very same re-read.
    review = _send(flow, "e11-ok-review", message="直接生成草案", intent="plan_first_review")
    assert _stage(review) == PlanDraftStage.AWAITING_FIRST_REVIEW.value
    assert f"来源 [{source_id}]" in _payload(review)["summary"]

    # --- and the book publishes as a formal route plan -----------------------
    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-ok-book")
    assert _published(flow, resumed)["plan_revision"] == 1
    document = json.loads(produced[0])
    cover = "\n".join(f"{line['label']}：{line['value']}"
                      for line in document["cover"]["status_lines"])
    assert "正式路线计划" in cover
    assert "待核验" not in document["cover"]["status_lines"][2]["value"], (
        "the researched book's 安全门 reads the stored gate answer")
    assert document["sources"], "the published book carries the source list"
    assert any(source_id in {entry["id"] for entry in group["entries"]}
               for group in document["sources"]), document["sources"]
    certified = [fact for fact in document["route_overview"] if fact["value"] != "待核验"]
    assert {fact["label"] for fact in certified} >= {"距离", "累计爬升", "预计耗时"}

    # Every plan click in this journey ran against a kernel that RAISES, so no
    # transition, research result or fact render was model-decided. The one model
    # turn below is the export turn the harness itself installs — the book is
    # queued by ``report_export``, never by a plan intent (landmine 5).
    assert set(sentinel.providers) == {("deterministic", "offline")}, sentinel.providers
    assert sentinel.live_calls == []
    assert len(flow.model_calls) == 1, flow.model_calls
    assert boom.calls == [], "a plan click consulted the model"
    _assert_terminal_ledger(flow)


# ---- AC5: the deterministic path, proven by explosion ----------------------

def test_journey_the_whole_deterministic_path_never_consults_a_model(flow, monkeypatch):
    """AC5: entry, completeness, both reviews, selection, research, regenerate and
    cancel all run against a kernel that raises on contact.

    The deterministic contract is "same input, same transition", so the refusal
    arms are clicked twice and must answer with the same stable enum reason, and
    regeneration must publish nothing: the plan path has no side effect at all.

    "No model" is only half of "fully deterministic", so the journey also installs
    :class:`_DeterminismSentinel`: the route-decision provider the path asks for is
    recorded and must be ``deterministic``, and any live/MCP data call explodes. A
    passing journey therefore proves the transition was decided by rules at runtime,
    not by a config value nobody re-reads.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)
    sentinel = _DeterminismSentinel().install(monkeypatch)

    _send(flow, "e11-det-open", message="周末想去爬山，还没定")  # natural language, no intent
    assert _stage(_send(flow, "e11-det-confirm", message="确认", intent="plan_confirm")) \
        == PlanDraftStage.AWAITING_INPUT.value

    submitted = _send(flow, "e11-det-submit", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    verdict = _payload(submitted)["rules"]["verdict"]
    assert verdict == "confirmation_required"

    # A malformed format is a stable, readable reason — and it bounces the review.
    _send(flow, "e11-det-badtime", **_submit_fields(arrangements=_ALL_ARRANGEMENTS | {"meeting_time": "明天一早"}))
    bad = _send(flow, "e11-det-bad-review", message="直接生成草案", intent="plan_first_review")
    assert {issue["key"] for issue in _payload(bad)["rules"]["format_issues"]} == {"meeting_time"}
    assert _stage(bad) == PlanDraftStage.AWAITING_INPUT.value
    assert "arrangement_format_invalid" in json.dumps(_payload(bad)["rules"], ensure_ascii=False)

    # Research on the demo route: the gate, not a model, decides the stage.
    _send(flow, "e11-det-fix", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    research = _send(flow, "e11-det-research", message="研究路线", intent="plan_research_route")
    assert _stage(research) in (PlanDraftStage.MINIMAL_DRAFT.value,
                                PlanDraftStage.AWAITING_GENERIC_CHOICE.value,
                                PlanDraftStage.BLOCKED_SAFETY.value)

    # Regenerating the draft is idempotent: same stage, same verdict, no effect.
    again = _send(flow, "e11-det-regenerate", message="重新生成草稿", intent="plan_regenerate",
                  plan_input={"destination_or_route": _DEMO_ROUTE, "activity_date": _DATE,
                              "participants": _PEOPLE, "arrangements": _ALL_ARRANGEMENTS})
    assert _stage(again) == PlanDraftStage.MINIMAL_DRAFT.value
    assert _payload(again)["rules"]["verdict"] == verdict
    assert [f for f in again if f["type"] in ("approval.required", "artifact.available")] == []

    # Refusals are stable: the same illegal click twice answers identically.
    first_try = _send(flow, "e11-det-refuse-1", message="确认最终摘要", intent="plan_confirm_final")
    second_try = _send(flow, "e11-det-refuse-2", message="确认最终摘要", intent="plan_confirm_final")
    assert _blocked(first_try) == _blocked(second_try) == "final_review_not_pending"

    # …and the legal path to PDF approval runs the same way, with no model.
    review = _send(flow, "e11-det-review", message="直接生成草案", intent="plan_first_review")
    assert _stage(review) == PlanDraftStage.AWAITING_FIRST_REVIEW.value
    assert _stage(_send(flow, "e11-det-confirm-arr", message="确认安排",
                        intent="plan_confirm_arrangements")) == PlanDraftStage.FIRST_REVIEWED.value
    assert _stage(_send(flow, "e11-det-final", message="查看最终摘要",
                        intent="plan_final_review")) == PlanDraftStage.AWAITING_FINAL_REVIEW.value
    assert _stage(_send(flow, "e11-det-confirm-final", message="确认最终摘要",
                        intent="plan_confirm_final")) == PlanDraftStage.FINAL_REVIEWED.value
    assert boom.calls == [] and flow.model_calls == []

    cancelled = _send(flow, "e11-det-cancel", message="取消计划", intent="plan_cancel")
    assert _stage(cancelled) == PlanDraftStage.CANCELLED.value
    assert flow.repo.get_plan_draft(CONV).stage == "cancelled"
    # Every provider this journey's turns reached for was the rule engine over
    # offline fixtures — and the guard really was live, because the research turn
    # built a session and the sentinel saw it.
    assert sentinel.providers, "the journey never reached the decision-provider seam"
    assert set(sentinel.providers) == {("deterministic", "offline")}, sentinel.providers
    assert sentinel.live_calls == []
    assert flow.settings.route_decision_provider == "deterministic"
    assert flow.settings.route_compare_data_mode == "offline"
    _assert_terminal_ledger(flow)


def test_journey_completeness_and_confirmation_are_recomputed_not_trusted(flow, monkeypatch):
    """AC5's other half: the stored verdict is a record, never an authority.

    A draft that LOSES a core arrangement must not stay review-ready on the strength
    of the row written when it still had one — the flow re-runs
    :func:`evaluate_plan_completeness` against the stored fields on every turn.
    """
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)
    _send(flow, "e11-recompute-full", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    assert flow.repo.get_completeness_check(CONV).verdict == "confirmation_required"

    stripped = _send(flow, "e11-recompute-strip", **_submit_fields(arrangements={**_ALL_ARRANGEMENTS, "emergency_contact": ""}))
    assert flow.repo.get_completeness_check(CONV).verdict == "supplement_required"
    assert _stage(stripped) == PlanDraftStage.MINIMAL_DRAFT.value

    review = _send(flow, "e11-recompute-review", message="直接生成草案",
                   intent="plan_first_review")
    assert _stage(review) == PlanDraftStage.AWAITING_INPUT.value, (
        "the review opened on a stale verdict row instead of the stored fields"
    )
    assert boom.calls == []
    _assert_terminal_ledger(flow)


# ---- AC6: no_go blocks the journey ----------------------------------------

def test_journey_no_go_blocks_the_plan_from_entry_to_pdf(flow, monkeypatch):
    """AC6: a gate refusal (严重天气 → ``no_go``) stops the plan at every door.

    Three doors, in the order a user would reach them: the advancing clicks in the
    state machine, 继续生成通用草案 (ticket 04's exit for thin material, deliberately
    NOT offered here), and the export turn that would otherwise queue the PDF. All
    three refuse, nothing is published, and the only ways out stay 取消 and a
    re-submitted form the gate judges again from scratch.
    """
    flow.settings.route_compare_demo_scenario = "severe_weather"
    boom = _ExplodingKernel()
    monkeypatch.setattr(kernel, "run_chat_turn", boom)

    _send(flow, "e11-nogo-submit", **_submit_fields(arrangements=_CORE_ONLY))

    researched = _send(flow, "e11-nogo-research", message="研究这条线",
                       intent="plan_research_route")
    assert _stage(researched) == PlanDraftStage.BLOCKED_SAFETY.value
    safety = _payload(researched)["safety"]
    assert safety["hard_stop"] is True
    assert "severe_weather" in safety["codes"]
    assert safety["comparison_status"] == "no_go"
    assert safety["formal_route_plan_allowed"] is False
    assert flow.repo.get_plan_safety(CONV).hard_stop is True
    # The refusal rides the event record, so the panel can explain it (AC5 of 07).
    logged = flow.repo.get_events_since(
        run_id=researched[-1]["run_id"], from_seq=0)
    draft_event = next(e for e in logged if e["type"] == "plan.draft_state")
    assert draft_event["payload"]["safety"]["codes"] == list(safety["codes"])

    # Every door a plan can advance through, with the guard that owns it: the four
    # that would move the draft answer ``safety_blocked``, and the confirmation of
    # a summary that was never shown is refused one layer earlier by its own
    # source-stage guard. Neither reaches PDF approval.
    doors = {
        "plan_first_review": "safety_blocked",
        "plan_final_review": "safety_blocked",
        "plan_confirm_final": "safety_blocked",
        "plan_generic_draft": "safety_blocked",
        "plan_confirm_arrangements": "arrangements_not_pending",
    }
    for index, (intent, expected) in enumerate(sorted(doors.items())):
        extra: dict[str, Any] = {}
        if intent == "plan_generic_draft":
            # The well-formed claim ticket 04's exit needs, so the refusal this
            # journey measures is the safety one, not a malformed body.
            extra["plan_type"] = "generic_draft"
        if intent == "plan_select_route":
            # The selection is the one intent allowed to carry the field, so the
            # hard stop has to refuse it as a transition, not as a malformed body.
            extra["route_selection"] = {"comparison_id": "run-x:cmp",
                                         "candidate_id": "demo-a"}
        refused = _send(flow, f"e11-nogo-refuse-{index}", message="继续", intent=intent, **extra)
        assert _blocked(refused) == expected, intent
    assert flow.repo.get_plan_draft(CONV).stage == "blocked_safety"
    assert flow.repo.list_plan_artifact_versions(CONV) == []

    # The export turn: the gate re-runs on the stored result and refuses, so
    # report_export never yields a report and no approval is ever requested.
    produced: list[str] = []
    _install_export_model(flow, monkeypatch)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: book._recording_computer(produced))
    export = book._post_sse(flow.app, flow.subject, book._export_payload("e11-nogo-book"))
    assert _terminal(export) == "done", export[-1]
    assert [f for f in export if f["type"] == "approval.required"] == []
    assert book._available(export) == []
    assert produced == [], "a no-go route rendered a PDF"
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert flow.repo.get_current_plan_artifact_version(CONV) is None
    assert flow.repo.get_completeness_check(CONV) is not None

    # Cancel is still available from the hard stop…
    assert _stage(_send(flow, "e11-nogo-cancel", message="取消计划", intent="plan_cancel")) \
        == PlanDraftStage.CANCELLED.value
    # …and a re-submitted form is judged again: the gate has no memory of the
    # refusal, only of the route's evidence.
    flow.settings.route_compare_demo_scenario = "complete"
    reopened = _send(flow, "e11-nogo-resubmit", **_submit_fields(arrangements=_CORE_ONLY))
    assert _stage(reopened) == PlanDraftStage.MINIMAL_DRAFT.value
    assert _send(flow, "e11-nogo-safe-research", message="再研究一次",
                 intent="plan_research_route")[-1]["type"] == "done"
    assert flow.repo.get_plan_safety(CONV).hard_stop is False
    assert boom.calls == []
    seen = _assert_terminal_ledger(flow)
    assert "error" in seen, "the refused advancing clicks are the error arms of this journey"


# ---- AC7: PDF failure, cancel, recovery and regeneration ------------------

def test_journey_render_failure_publishes_nothing_and_the_retry_recovers(
    flow, monkeypatch
):
    """AC7 arms 1+2: a render that fails publishes no half-book, and the retry
    really runs again and lands.

    The failure arm uses the real effect ledger and the real sandbox: the computer
    raises :class:`ComputerError` inside the approved resume, the run ends
    ``error``, and neither an ``artifacts`` row nor a version row exists. The retry
    is a NEW export turn (new run, new approval) — and ``plan_regenerate`` is checked
    in between, so the recovery is not an accident of the retry being a different
    code path.
    """
    _send(flow, "e11-fail-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []

    class _FailingComputer(LocalNativeComputer):
        def generate_pdf(self, workspace_path, content, filename):
            raise ComputerError("渲染失败（演练）")

    _install_export_model(flow, monkeypatch)
    paused, run_id, last_seq, _approval = _pause_book(
        flow, monkeypatch, "e11-render-fail",
        meeting_point=_CORE["meeting_point"], produced=produced,
        computer=_FailingComputer())

    failed = book._resume(flow.app, flow.subject, run_id, last_seq)
    assert _terminal(failed) == "error", failed[-1]
    assert flow.repo.get_run(run_id).status == "error"
    assert book._available(failed) == []
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert produced == []
    # Nothing half-published means the draft and the "which version is current"
    # pointer are untouched. The draft is at ``final_reviewed`` because the publish
    # it was attempting required the confirmed final review (ticket 07 AC4); the
    # failed render leaves that stage exactly as it found it.
    assert flow.repo.get_plan_draft(CONV).stage == PlanDraftStage.FINAL_REVIEWED.value
    assert flow.repo.get_current_plan_artifact_version(CONV) is None

    # 重新生成草稿 is a plan intent, not a render request: still nothing published.
    # A fresh submit re-opens the form at ``minimal_draft``, which is the only
    # stage ``plan_regenerate`` acts on, so the recovery arm proves regenerate
    # renders no PDF without touching the door's confirmation requirement.
    _send(flow, "e11-refill", **_submit_fields(arrangements=_CORE_ONLY))
    regen = _send(flow, "e11-plan-regenerate", message="重新生成草稿", intent="plan_regenerate",
                  plan_input={"destination_or_route": _DEMO_ROUTE, "activity_date": _DATE,
                              "participants": _PEOPLE, "arrangements": _CORE_ONLY})
    assert _stage(regen) == PlanDraftStage.MINIMAL_DRAFT.value
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    assert len(paused) >= 1

    # Recovery: the same conversation, a working computer, a real PDF.
    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-render-retry")
    artifact = _published(flow, resumed)
    assert artifact["plan_revision"] == 1, "the failed render still consumed a revision"
    assert len(produced) == 1
    assert _download(flow, artifact["artifact_id"]).content.startswith(b"%PDF-")
    assert _assert_terminal_ledger(flow) == {"done": resumed[-1]["run_id"], "error": run_id}


def test_journey_cancelling_the_plan_book_run_leaves_nothing_published(flow, monkeypatch):
    """AC7 arm 3: cancelling the paused plan-book run ends it, cleans up, and the
    conversation can still publish afterwards.

    Pause and cancel use the SAME app instance (landmine 4: ``CapacityGuard`` is
    process-local and per-app, so a cancel against a rebuilt app would observe
    nothing). The cancel is idempotent, and the mid-body variant of the same
    interruption is checked in the same journey.
    """
    _send(flow, "e11-cancel-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []
    paused, run_id, _last_seq, _approval = _pause_book(
        flow, monkeypatch, "e11-cancel-book",
        meeting_point=_CORE["meeting_point"], produced=produced, approve=False)
    assert paused[-1]["type"] == "run.paused"
    assert flow.repo.get_run(run_id).status == "paused"

    cancelled = book._cancel(flow.app, flow.subject, run_id, "e11-cancel-1")
    assert cancelled.json()["accepted"] is True
    replay = book._cancel(flow.app, flow.subject, run_id, "e11-cancel-1")
    assert replay.json()["accepted"] is True, "the duplicate cancel is not idempotent"

    assert flow.repo.get_run(run_id).status == "cancelled"
    assert produced == []
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    capacity = flow.app.state.agent_v2_runtime.capacity
    assert capacity.snapshot()["active_sandboxes"] == 0
    tail = book._get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}")
    assert tail.json()["events"][-1]["type"] == "cancelled"

    # A cancel that lands mid-body, on the plan turn itself: same promise.
    types = book._drive_plan_turn_with_cancel(flow, "e11-midbody", "plan.draft_state")
    assert types[-1] == "cancelled" and "done" not in types and "text" not in types
    midbody = flow.repo.resolve_run_id("e11-midbody")
    assert flow.repo.get_run(midbody).status == "cancelled"
    assert flow.repo.list_plan_artifact_versions(CONV) == []

    # Neither cancel poisons the conversation: the next attempt publishes.
    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-after-cancel")
    assert _published(flow, resumed)["plan_revision"] == 1
    seen = _assert_terminal_ledger(flow)
    assert set(seen) == {"cancelled", "done"}, seen
    assert seen["cancelled"] in {run_id, midbody}
    assert seen["done"] == resumed[-1]["run_id"], (
        "the two cancelled runs must still agree with their logs"
    )
    for cancelled_run in (run_id, midbody):
        assert flow.repo.get_run(cancelled_run).status == "cancelled"
        tail = book._get(flow.app, flow.subject, f"/api/v2/chat/events/{cancelled_run}")
        assert tail.json()["events"][-1]["type"] == "cancelled"


def test_journey_duplicate_resume_and_regeneration_keep_one_chain(flow, monkeypatch):
    """AC7 arms 4+5: resuming twice renders once, and a new plan revision renders
    again — the two halves of "recovery and regeneration" that must not be traded
    off against each other.

    Ends with the chain the release actually leaves behind: revisions 1 and 2 as
    different bytes, both downloadable, generation 1 untouched by generation 2.
    """
    _send(flow, "e11-dup-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []
    paused, run_id, last_seq, _approval = _pause_book(
        flow, monkeypatch, "e11-dup-resume",
        meeting_point=_CORE["meeting_point"], produced=produced)

    first = book._resume(flow.app, flow.subject, run_id, last_seq)
    second = book._resume(flow.build(), flow.subject, run_id, last_seq)
    assert _terminal(first) == _terminal(second) == "done"
    assert len(produced) == 1, "a duplicate resume re-rendered the plan book"
    assert len(flow.repo.list_artifacts_for_run(run_id)) == 1
    assert [v.revision for v in flow.repo.list_plan_artifact_versions(CONV)] == [1]

    # A re-POST of the whole request is a replay of the stored stream.
    again = book._post_sse(flow.build(), flow.subject, book._export_payload("e11-dup-resume"))
    assert again == paused + first, "the duplicate request stopped being a replay"

    _next, next_frames, produced = _publish(
        flow, monkeypatch, "e11-regenerated", meeting_point="长坪沟景区南门")
    # ``_publish`` installs its own recording computer, so the renders it counts
    # start again from zero: generation 2 is the one this call produced.
    assert len(produced) == 1
    newer = _published(flow, next_frames)
    older = _published(flow, first)
    assert newer["plan_revision"] == 2
    assert newer["previous_artifact_id"] == older["artifact_id"]
    assert newer["artifact_id"] != older["artifact_id"]

    versions = flow.repo.list_plan_artifact_versions(CONV)
    assert [v.revision for v in versions] == [1, 2]
    for version in versions:
        stored = flow.repo.get_artifact(version.artifact_id)
        assert os.path.isfile(stored.storage_path)
        assert _download(flow, version.artifact_id).status_code == 200
    assert (flow.repo.get_artifact(versions[0].artifact_id).sha256
            != flow.repo.get_artifact(versions[1].artifact_id).sha256)
    assert flow.repo.get_current_plan_artifact_version(CONV).artifact_id == \
        newer["artifact_id"]
    _assert_terminal_ledger(flow)


def test_journey_a_refused_approval_publishes_nothing_and_cleans_up(flow, monkeypatch):
    """AC7's rejection arm: saying no to the PDF ends the run, destroys the
    preserved sandbox, and publishes nothing.

    The asymmetry ticket 09/10 chose is re-pinned here as a journey fact: a
    refusal clears nothing about the plan (the draft and its rows stay) but must not
    leak the workspace or the capacity slot — the run ends ``done``, so the 24 h
    sweep would never look at it.
    """
    _send(flow, "e11-reject-submit", **_submit_fields(arrangements=_CORE_ONLY))
    # The publish that reaches the approval pause requires the confirmed final
    # review (ticket 07 AC4), so the refusal is a refusal of the APPROVAL, not of
    # an unconfirmed draft (that arm has its own journey above).
    _advance_to_final(flow, tag="e11-reject")
    produced: list[str] = []
    _install_export_model(flow, monkeypatch)
    monkeypatch.setattr(chat_api, "_build_artifact_computer",
                        lambda *a, **k: book._recording_computer(produced))

    async def run():
        async with AsyncClient(
            transport=ASGITransport(app=flow.app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: flow.subject.token},
        ) as client:
            paused = book._frames(await client.post(
                "/api/v2/chat/sse", json=book._export_payload("e11-reject")))
            approval = next(e["payload"] for e in paused
                            if e["type"] == "approval.required")
            decided = await client.post(
                f"/api/v2/approvals/{approval['approval_id']}/decision",
                json={"decision": "reject", "decision_request_id": "e11-reject-decision"})
            resumed = book._frames(await client.post(
                f"/api/v2/chat/resume/{paused[-1]['run_id']}",
                params={"from_seq": paused[-1]["sequence"]}))
            return decided, resumed

    decided, resumed = asyncio.run(run())
    assert decided.status_code == 200
    assert _terminal(resumed) == "done"
    assert produced == []
    run_id = resumed[-1]["run_id"]
    assert flow.repo.list_artifacts_for_run(run_id) == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    lease = flow.repo.get_sandbox_lease(run_id)
    assert lease is not None and lease.status == "destroyed"
    assert not os.path.isdir(lease.workspace_path)
    assert flow.app.state.agent_v2_runtime.capacity.snapshot()["active_sandboxes"] == 0
    # The plan itself survives a refusal: it is the run that ended, not the draft.
    # The draft is at ``final_reviewed`` (the confirmation the door requires), and
    # the refusal clears nothing about it — only the sandbox and the slot.
    assert flow.repo.get_plan_draft(CONV).stage == PlanDraftStage.FINAL_REVIEWED.value
    _assert_terminal_ledger(flow)


def test_journey_a_disconnected_book_stream_replays_from_the_log(flow, monkeypatch):
    """AC7's recovery arm at the wire level: the pause is replayable, so the client
    that lost the stream still finds the approval it has to answer.

    Recovery is by ``GET /chat/events`` + ``POST /chat/resume`` only — the run's own
    log is the source, and the frames a reconnecting client gets are the frames that
    were originally sent.
    """
    _send(flow, "e11-disc-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []
    paused, run_id, last_seq, approval = _pause_book(
        flow, monkeypatch, "e11-disconnect",
        meeting_point=_CORE["meeting_point"], produced=produced, approve=False)

    replayed = book._get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}",
                         {"from_seq": 0}).json()["events"]
    assert [e["type"] for e in replayed] == [f["type"] for f in paused]
    assert replayed[-1]["type"] == "run.paused"
    assert flow.repo.get_run(run_id).status == "paused"
    assert produced == [], "the PDF rendered before the decision"

    # The client that only kept its request_id still finds the run.
    by_request = book._get(flow.app, flow.subject, "/api/v2/chat/snapshot/-",
                          {"request_id": "e11-disconnect"})
    assert by_request.json()["run_id"] == run_id

    # It answers the approval from the replayed payload and resumes against a
    # rebuilt runtime — the reconnect after a process restart.
    _decide(flow.build(), flow.subject, approval["approval_id"], "dec-e11-disconnect")
    resumed = book._resume(flow.build(), flow.subject, run_id, last_seq)
    assert _terminal(resumed) == "done"
    artifact = _published(flow, resumed)
    assert _download(flow, artifact["artifact_id"]).content.startswith(b"%PDF-")
    assert len(produced) == 1
    assert flow.repo.get_run(run_id).status == "done"
    _assert_terminal_ledger(flow)


def test_journey_paused_book_expires_and_the_sweep_reconciles_it(flow, monkeypatch):
    """AC7's last recovery arm: an unanswered pause is not left hanging.

    Past the 24 h window the sweep ends the run, frees the capacity slot on the SAME
    app instance the pause used (landmine 4), and reconciles the log — so a
    reconnecting client is never told the run still awaits its decision. The
    workspace directory is deliberately NOT removed: that is the janitor's documented
    semantics, not a bug this gate should "fix".
    """
    from datetime import datetime, timedelta, timezone

    from agent_v2.run_orchestration import sweep_expired_paused_runs

    _send(flow, "e11-expire-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []
    paused, run_id, _seq, _approval = _pause_book(
        flow, monkeypatch, "e11-expire",
        meeting_point=_CORE["meeting_point"], produced=produced, approve=False)
    lease = flow.repo.get_sandbox_lease(run_id)
    capacity = flow.app.state.agent_v2_runtime.capacity
    assert lease.status == "active" and os.path.isdir(lease.workspace_path)
    # Landmine 4 stated honestly: the ``route_plan`` export turn holds a LEASE at
    # pause but takes no sandbox capacity slot — only ``artifact``/``cowork`` do at
    # POST, and a route_plan resume does. So this journey checks the lease and the
    # log; the slot arithmetic is ticket 10's artifact-profile arm's job.
    assert capacity.snapshot()["active_sandboxes"] == 0
    assert flow.repo.get_run(run_id).status == "paused"

    runtime = flow.app.state.agent_v2_runtime
    assert sweep_expired_paused_runs(runtime) == 0, "the sweep fired inside its TTL"

    backdated = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    runtime.runs.control._conn.execute(
        "UPDATE runs SET updated_at = ? WHERE run_id = ?", (backdated, run_id))
    runtime.runs.control._conn.commit()
    assert sweep_expired_paused_runs(runtime) == 1

    assert flow.repo.get_run(run_id).status == "error"
    assert flow.repo.get_sandbox_lease(run_id).status == "failed"
    assert capacity.snapshot()["active_sandboxes"] == 0
    assert produced == []
    assert flow.repo.list_plan_artifact_versions(CONV) == []
    tail = book._get(flow.app, flow.subject, f"/api/v2/chat/events/{run_id}")
    assert tail.json()["events"][-1]["type"] == "error"
    seen = _assert_terminal_ledger(flow)
    assert seen["error"] == run_id, seen
    assert set(seen) == {"error", "done"}, seen


# ---- AC8: visual acceptance of the published bytes ------------------------

_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "gui-test-screenshots"
_TIMELINE_HEADER = ["时间", "阶段", "执行重点", "完成检查"]


def _header_pages(pages: list[str], header: list[str]) -> list[str]:
    """Pages carrying a whole header ROW: its cells as consecutive lines.

    A single word is worthless here (notes-08 landmine 4) — 完成检查 is the one
    header cell that exists in no other section, and adjacency is what separates a
    repeated header row from a page that merely mentions the word.
    """
    wanted = "\n".join(header)
    return [page for page in pages if wanted in page]


def test_journey_published_book_renders_the_five_visual_zones_and_a_split_table(
    flow, monkeypatch
):
    """AC8: the five visual zones exist in the PDF the journey actually published.

    Not a fixture document: the bytes are the artifact the app stored
    (``artifacts.storage_path``), opened page by page. The plan carries 16 user
    stages so the 行程时间线 table crosses a page boundary — pdf-baseline-08 flagged
    exactly that as implemented-but-undemonstrated — and every continuation page
    must open on the repeated header row. Page PNGs are written beside the other
    rendered evidence so a human can look at what the gate asserted.
    """
    fitz = pytest.importorskip("fitz")
    _send(flow, "e11-visual-submit", **_submit_fields(arrangements=_ALL_ARRANGEMENTS))
    stages = [{"time": f"{hour:02d}:00",
               "activity": f"第 {index} 段：结伴徒步、补给、体感与天气复核、装备检查"}
              for index, hour in enumerate(range(6, 22), 1)]
    _paused, resumed, produced = _publish(flow, monkeypatch, "e11-visual", stages=stages)
    artifact = _published(flow, resumed)

    stored = flow.repo.get_artifact(artifact["artifact_id"])
    _EVIDENCE_DIR.mkdir(exist_ok=True)
    pdf_path = _EVIDENCE_DIR / "plan-book-end-to-end.pdf"
    pdf_path.write_bytes(Path(stored.storage_path).read_bytes())

    with fitz.open(pdf_path) as pdf:
        pages = [page.get_text() for page in pdf]
        for index, page in enumerate(pdf, 1):
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(
                _EVIDENCE_DIR / f"plan-book-end-to-end-page-{index}.png")
        assert len(pages) >= 4, len(pages)

        # 1 封面: a page of its own, carrying the plan status.
        cover = pages[0]
        assert "AI Hiking 徒步计划书" in cover
        for line in ("草案类型", "审阅结论", "安全门", "路线核验", "资料模式", "计划编号"):
            assert line in cover, line
        # The running header belongs to pages > 1 (``_pdf_footer`` guards it on
        # ``getPageNumber()``, and the page-number footer is on every page).
        assert "活动计划 / 资料附录" not in cover
        assert any("活动计划 / 资料附录" in page for page in pages[1:])
        assert "计划摘要" not in cover, "the cover is a page, not a band at the top"
        assert "待核验" in cover or "已核验" in cover, "the cover states a status"

        # 2 摘要 / 3 时间线 / 4 风险区 / 出发前检查 / 候选与来源, by their numbered
        # headings — the numbers and labels come from PLAN_SECTIONS, the single
        # vocabulary both stories build from, so a renamed section cannot slip past.
        body = "\n".join(pages[1:])
        labels = {number: label for number, label, _aliases in PLAN_SECTIONS}
        for number in ("01", "04", "09", "10", "11", "13"):
            assert f"{number} {labels[number]}" in body, number
        summary_page = next(page for page in pages if "01 " + labels["01"] in page)
        for label in ("路线", "活动日期", "参与人数"):
            assert label in summary_page, label
        assert f"{_PEOPLE} 人" in summary_page
        assert (_DEMO_ROUTE in summary_page or "演示山脊路线" in summary_page)

        # 5 跨页表格: the split happened AND the header row repeated on every page
        # the table occupies.
        timeline_pages = [page for page in pages if "完成检查" in page]
        assert len(timeline_pages) >= 2, "the timeline never crossed a page boundary"
        assert len(_header_pages(timeline_pages, _TIMELINE_HEADER)) == len(timeline_pages), (
            "a continuation page opened mid-table without repeating the header row"
        )

        # 风险区 is visually its own block, not the default table style.
        spans = []
        with fitz.open(pdf_path) as again:
            for page in again:
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        spans.extend(line["spans"])
        risk = [s for s in spans if "触发条件" in s["text"] or "立即处置" in s["text"]]
        ordinary = [s for s in spans if "检查项" in s["text"] or "负责人" in s["text"]]
        assert risk and ordinary, "neither the risk block nor a plain block rendered"
        assert {s["color"] for s in risk} != {s["color"] for s in ordinary}, (
            "the risk zone renders in the same style as an ordinary table"
        )

        # Nothing was invented for the book: no decorative imagery (ticket 08 AC6).
        for page in pdf:
            assert page.get_images() == []
        assert b"/Subtype /Image" not in pdf_path.read_bytes()

    # The risk content itself is the gate's, not a paraphrase.
    document = json.loads(produced[0])
    assert document["risk"]["rows"], "the published book carries no risk rows"
    _assert_terminal_ledger(flow)


# ---- The gate: nothing on this list may be silently uncovered -------------

#: One entry per acceptance criterion of the ticket, mapped to the journey tests in
#: this module that must exist. The per-ticket seam modules are named too, so the
#: gate checks that the release is not ONLY journeys: the behaviour-level proofs
#: have to remain collectable, and a deleted module would otherwise leave the
#: journeys as the sole — and therefore unexplained — evidence.
CRITERIA: dict[str, dict[str, Any]] = {
    "AC1": {"criterion": "结构化 intent 的明确入口流程通过。", "journeys": (
        "test_journey_structured_intent_entry_publishes_the_book",),
        "seams": ("test_plan_entry_contract.py", "test_plan_entry_flow.py")},
    "AC2": {"criterion": "模糊意图确认流程通过。", "journeys": (
        "test_journey_ambiguous_intent_confirms_before_it_opens_the_form",),
        "seams": ("test_plan_entry_contract.py", "test_plan_entry_flow.py")},
    "AC3": {"criterion": "缺失字段草案流程通过。", "journeys": (
        "test_journey_missing_fields_supplement_review_and_publish",),
        "seams": ("test_plan_missing_fields_first_review.py",
                  "test_plan_completeness_rules.py")},
    "AC4": {"criterion": "已有路线比较和自由目的地流程通过。", "journeys": (
        "test_journey_existing_route_comparison_selects_a_route_and_publishes",
        "test_journey_free_destination_research_falls_back_to_a_generic_draft",
        "test_journey_research_success_surfaces_candidates_and_the_user_selects_one",),
        "seams": ("test_plan_route_selection.py", "test_plan_research_branch.py")},
    "AC5": {"criterion": "计划完整度和确认路径完全由确定性规则驱动。", "journeys": (
        "test_journey_the_whole_deterministic_path_never_consults_a_model",
        "test_journey_completeness_and_confirmation_are_recomputed_not_trusted",
        # Ticket 07 AC4: the confirmation door at the PDF edge is the same
        # deterministic stage the review intents write — an unconfirmed draft is
        # refused with a stable reason, no model and no invented eligibility.
        "test_journey_an_unconfirmed_draft_cannot_reach_a_pdf",),
        "seams": ("test_plan_completeness_rules.py",)},
    "AC6": {"criterion": "no_go 阻止流程通过。", "journeys": (
        "test_journey_no_go_blocks_the_plan_from_entry_to_pdf",),
        "seams": ("test_plan_final_review_and_safety_gate.py",)},
    "AC7": {"criterion": "PDF 失败、取消、恢复和重生成流程通过。", "journeys": (
        "test_journey_render_failure_publishes_nothing_and_the_retry_recovers",
        "test_journey_cancelling_the_plan_book_run_leaves_nothing_published",
        "test_journey_duplicate_resume_and_regeneration_keep_one_chain",
        "test_journey_a_refused_approval_publishes_nothing_and_cleans_up",
        "test_journey_a_disconnected_book_stream_replays_from_the_log",
        "test_journey_paused_book_expires_and_the_sweep_reconciles_it",),
        "seams": ("test_artifact_versioned_generation.py",
                  "test_plan_recovery_cancel_idempotency.py")},
    "AC8": {"criterion": "视觉验收覆盖封面、摘要、时间线、风险区和跨页表格。", "journeys": (
        "test_journey_published_book_renders_the_five_visual_zones_and_a_split_table",),
        "seams": ("test_plan_pdf_layout_book.py",)},
}

#: The failure and recovery branches the ticket names ("covering every failure and
#: recovery branch"), each attributed to the journey that drives it. The marker is
#: the thing a reader can grep for to confirm the branch is really exercised, so a
#: branch nobody drives any more turns the gate red instead of vanishing quietly.
FAILURE_BRANCHES: dict[str, tuple[str, str]] = {
    "insufficient_evidence": ("test_journey_free_destination_research_falls_back_to_a_generic_draft",
                              "insufficient_evidence"),
    "safety_refusal": ("test_journey_no_go_blocks_the_plan_from_entry_to_pdf", "no_go"),
    "render_failure": ("test_journey_render_failure_publishes_nothing_and_the_retry_recovers",
                       "渲染失败"),
    "mid_body_cancel": ("test_journey_cancelling_the_plan_book_run_leaves_nothing_published",
                        "_drive_plan_turn_with_cancel"),
    "run_cancel": ("test_journey_cancelling_the_plan_book_run_leaves_nothing_published",
                   "_cancel("),
    "duplicate_resume": ("test_journey_duplicate_resume_and_regeneration_keep_one_chain",
                         "duplicate resume"),
    "regeneration": ("test_journey_duplicate_resume_and_regeneration_keep_one_chain",
                     "plan_revision"),
    "approval_rejection": ("test_journey_a_refused_approval_publishes_nothing_and_cleans_up",
                           '"decision": "reject"'),
    "run_expiry": ("test_journey_paused_book_expires_and_the_sweep_reconciles_it", "sweep"),
    "disconnect_recovery": ("test_journey_a_disconnected_book_stream_replays_from_the_log",
                            "run.paused"),
}


def _test_source(name: str) -> str:
    """The source text of a test function in this module, for the marker checks."""
    function = globals().get(name)
    assert function is not None and callable(function), f"{name} is not defined here"
    return inspect.getsource(function)


def _journey_names() -> set[str]:
    return {name for name, value in globals().items()
            if name.startswith("test_") and callable(value)}


def test_every_criterion_has_a_journey_and_a_still_collectable_seam():
    """Gate 1: all eight criteria map to ≥1 journey that exists, and to per-ticket
    seam modules that still hold tests.

    Fails if a criterion loses its journey, if a journey is renamed without the
    table, or if a ticket's proof module is deleted while the table keeps claiming
    it.
    """
    assert set(CRITERIA) == {f"AC{index}" for index in range(1, 9)}, sorted(CRITERIA)
    module_dir = Path(__file__).resolve().parent
    journeys = _journey_names()
    for key, entry in CRITERIA.items():
        assert entry["journeys"], key
        assert entry["criterion"].endswith("。"), key
        for journey in entry["journeys"]:
            assert journey in journeys, f"{key}: {journey} does not exist"
        for seam in entry["seams"]:
            path = module_dir / seam
            assert path.is_file(), f"{key}: seam module {seam} is gone"
            collected = [node.name for node in ast.walk(ast.parse(path.read_text("utf-8")))
                         if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
            assert collected, f"{key}: {seam} holds no tests any more"


def test_no_journey_is_unclaimed_by_a_criterion():
    """Gate 2: every ``test_journey_*`` here belongs to exactly one criterion.

    A new journey added without a mapping fails the release: that is how a branch
    would otherwise enter the gate attributed to nothing.
    """
    claimed = [journey for entry in CRITERIA.values() for journey in entry["journeys"]]
    journeys = sorted(name for name in _journey_names() if name.startswith("test_journey_"))
    assert sorted(claimed) == journeys, (
        f"unclaimed journeys: {sorted(set(journeys) - set(claimed))}, "
        f"claimed but missing: {sorted(set(claimed) - set(journeys))}")
    assert len(claimed) == len(set(claimed)), "a journey is claimed twice"


def test_every_plan_intent_and_every_draft_stage_is_walked_by_a_journey():
    """Gate 3: the two checklists come from the product, not from this file.

    ``PLAN_INTENTS`` and ``PlanDraftStage`` are the vocabularies a later ticket can
    grow. Each intent must be SENT by some journey and each stage must be REACHED by
    one, read out of the journeys' own source — so adding an intent or a stage turns
    this gate red until somebody drives it, which is the whole point of a gate.
    """
    text = "\n".join(_test_source(name) for name in _journey_names())

    unclaimed_intents = {intent for intent in INTENTS
                         if f'"{intent}"' not in text and f"'{intent}'" not in text}
    assert unclaimed_intents == set(), f"no journey sends: {sorted(unclaimed_intents)}"

    unvisited = {member for member, value in STAGES.items()
                 if f'"{value}"' not in text and f"'{value}'" not in text
                 and f"PlanDraftStage.{member}" not in text}
    assert unvisited == set(), f"no journey reaches: {sorted(unvisited)}"


def test_every_failure_branch_is_driven_by_some_journey():
    """Gate 4: each failure/recovery arm is attributed to a journey that exists and
    really carries the branch's own marker.
    """
    journeys = _journey_names()
    for branch, (journey, marker) in FAILURE_BRANCHES.items():
        assert journey in journeys, f"{branch}: {journey} is gone"
        assert marker in _test_source(journey), f"{branch} is not marked in {journey}"


def test_the_terminal_ledger_invariant_holds_for_every_terminal_the_gate_can_reach(
    flow, monkeypatch
):
    """Gate 5, the release invariant: swept over the whole database, not sampled.

    Every run row in a terminal state must hold its terminal event. The four arms
    below cover all three run terminals plus the two paths ticket 10 reconciled
    (pause-then-cancel, pause-then-expire), and the assertion that the sweep SAW each
    of them is what makes this a gate rather than a helper.
    """
    from datetime import datetime, timedelta, timezone

    from agent_v2.run_orchestration import sweep_expired_paused_runs

    # The predicate this sweep relies on must stay the product's own: run-terminal
    # names are stream-terminals minus the interrupt types (landmine 1).
    for name in RUN_TERMINAL_EVENTS:
        assert events.is_terminal(events.EventEnvelope(
            version="2", sequence=1, type=name, payload={})) is True
    assert events.is_terminal(events.EventEnvelope(
        version="2", sequence=1, type="run.paused", payload={})) is False

    _send(flow, "e11-ledger-submit", **_submit_fields(arrangements=_CORE_ONLY))
    produced: list[str] = []
    _paused, resumed, _produced = _publish(flow, monkeypatch, "e11-ledger-done")
    assert _terminal(resumed) == "done"

    # cancelled: paused, then cancelled through the API (the reconciled arm).
    paused, cancel_run, _seq, _approval = _pause_book(
        flow, monkeypatch, "e11-ledger-cancel",
        meeting_point="长坪沟景区南门", produced=produced, approve=False)
    assert paused[-1]["type"] == "run.paused"
    assert flow.repo.get_run(cancel_run).status == "paused"
    assert book._cancel(flow.app, flow.subject, cancel_run, "e11-ledger-cancel").json()["accepted"]

    # error: a plan click the state machine refuses, and the expired pause.
    refused = _send(flow, "e11-ledger-error", message="确认最终摘要", intent="plan_confirm_final")
    assert _blocked(refused) == "final_review_not_pending"
    assert flow.repo.get_run(refused[-1]["run_id"]).status == "error"

    expiring, expiring_run, _seq2, _ap2 = _pause_book(
        flow, monkeypatch, "e11-ledger-expire",
        meeting_point="长坪沟景区东门", produced=produced, approve=False)
    assert expiring[-1]["type"] == "run.paused"
    backdated = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    flow.repo.control._conn.execute(
        "UPDATE runs SET updated_at = ? WHERE run_id = ?", (backdated, expiring_run))
    flow.repo.control._conn.commit()
    assert sweep_expired_paused_runs(flow.app.state.agent_v2_runtime) == 1
    assert flow.repo.get_run(expiring_run).status == "error"

    seen = _assert_terminal_ledger(flow)
    assert set(seen) == TERMINAL_RUN_STATUSES, seen
    assert "running" not in set(_run_rows(flow.repo).values()), "a journey left a live run"

    # …and what a reconnecting client is told agrees with the row, for every run.
    for run_id, status in _run_rows(flow.repo).items():
        snapshot = book._get(flow.app, flow.subject, f"/api/v2/chat/snapshot/{run_id}")
        assert snapshot.status_code == 200
        assert snapshot.json()["status"] == status
        if status in TERMINAL_RUN_STATUSES:
            assert snapshot.json()["terminal_type"] == status
