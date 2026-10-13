"""Ticket 02 — 草稿保存、恢复与取消.

Seams under test (the same two the context pack names for this ticket):

* :meth:`agent_v2.run_orchestration.RunOrchestrator.stream_chat` driven
  in-process with a fake kernel over real temp SQLite — every stage
  transition, the stored draft row, and the ``plan.draft_state`` payload;
* :func:`agent_v2.plan_workflow.draft_summary` — the single deterministic
  source of the draft summary both the stream and the restore UX read;
* the frontend sources, pinned the way ticket 01 pinned the event
  vocabulary (there is no frontend test runner, and a missing restore
  wiring is otherwise a *silent* bug: ``readPlanDraftState`` returns null
  for an unknown stage and the reducer keeps the previous state).

Acceptance criteria:

* AC1 草稿保存到当前会话 — one draft row per conversation, keyed by
  ``conversation_id``, stable ``draft_id`` across turns.
* AC2 刷新或重新进入当前会话后可以恢复草稿 — the durable restore half.
* AC3 修改字段后草稿摘要同步更新 — the summary travels with the draft
  state and is re-rendered from the stored fields on every accepted turn.
* AC4 取消流程会清理未完成的计划状态.
* AC5 取消不会启动路线比较、计划语义判断或 PDF 生成.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.plan_workflow import (
    STAGE_MESSAGE,
    PlanDraftInput,
    PlanDraftStage,
    draft_summary,
)
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository

TERMINALS = {"error", "done", "cancelled"}
CONVERSATION = "conv-plan-02"
OTHER_CONVERSATION = "conv-plan-02-other"
SUBJECT_ID = "sub-plan-02"

#: Events a plan turn may legally emit. Anything outside this set — a
#: ``tool_call``, an ``artifact_candidate``, a ``run.paused`` — would mean the
#: draft flow started doing work it does not own (spec AC5).
PLAN_FLOW_EVENT_TYPES = {
    "run_started",
    "plan.draft_state",
    "text",
    "error",
    "done",
    "cancelled",
}


class _FakeMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True


@pytest.fixture
def plan_repo(tmp_path):
    """Repository shared by a whole test, closed at teardown."""
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
        settings=SimpleNamespace(
            agent_v2_enabled=True,
            openai_api_key="sk-test",
            openai_model="test-model",
            agent_v2_cowork_policy="disabled",
            agent_v2_state_dir="/tmp/unused",
        ),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )

    async def drive() -> list[list[Any]]:
        return [
            [e async for e in orchestrator.stream_chat(req, subject_id=SUBJECT_ID)]
            for req in requests
        ]

    return {"streams": asyncio.run(drive()), "kernel_calls": kernel_calls, "repo": repo}


def _req(request_id: str, *, message: str = "生成行程", intent: str | None = None,
         plan_input: Any | None = None,
         conversation_id: str | None = CONVERSATION) -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message,
        request_id=request_id,
        conversation_id=conversation_id,
        intent=intent,
        plan_input=plan_input,
    )


def _draft_events(events):
    return [e for e in events if e.type == "plan.draft_state"]


def _payload(**overrides) -> dict:
    base = {
        "destination_or_route": "四姑娘山长坪沟",
        "activity_date": "2026-11-08",
        "participants": 12,
    }
    base.update(overrides)
    return base


def _submit(request_id: str, **field_overrides) -> ChatFlowRequest:
    return _req(
        request_id,
        message="提交草稿",
        intent="plan_create",
        plan_input=PlanDraftInput.model_validate(_payload(**field_overrides)),
    )


# ---- AC1: the draft is saved to the current conversation -------------------


def test_draft_is_saved_to_the_conversation_that_created_it(plan_repo):
    """AC1: one row per conversation, keyed by conversation_id, stable draft_id."""
    result = _drive(plan_repo, _req("req-open", intent="plan_create"), _submit("req-submit"))
    stored = result["repo"].get_plan_draft(CONVERSATION)

    assert stored is not None
    assert stored.stage == "minimal_draft"
    assert (stored.destination_or_route, stored.activity_date, stored.participants) == (
        "四姑娘山长坪沟", "2026-11-08", 12,
    )

    # Both turns moved the SAME draft: the entry click and the dialog submit
    # are one draft, not two rows, and the id the frontend reopens from is
    # therefore stable.
    opened = _draft_events(result["streams"][0])[-1].payload
    submitted = _draft_events(result["streams"][1])[-1].payload
    assert opened["draft_id"] == submitted["draft_id"] == stored.draft_id
    assert opened["conversation_id"] == submitted["conversation_id"] == CONVERSATION

    # Keyed by conversation, not global state: another conversation is empty.
    assert result["repo"].get_plan_draft(OTHER_CONVERSATION) is None


def test_draft_row_survives_the_run_that_wrote_it(plan_repo):
    """AC1/AC2 precondition: the run terminalled, the draft did not.

    This is what makes refresh-and-restore possible at all: ``plan_drafts``
    is on the conversation axis, while run events/leases/artifacts are
    run-scoped.
    """
    result = _drive(plan_repo, _submit("req-submit"))
    events = result["streams"][0]
    assert events[-1].type == "done"

    run_id = events[-1].run_id
    assert result["repo"].get_run(run_id).is_terminal
    stored = result["repo"].get_plan_draft(CONVERSATION)
    assert stored.stage == "minimal_draft"
    assert stored.updated_by_run_id == run_id


# ---- AC3: field edits keep the draft summary in sync -----------------------


def test_editing_fields_updates_the_stored_draft_and_its_summary(plan_repo):
    """AC3: 修改字段后草稿摘要同步更新 — no stale value survives an edit."""
    result = _drive(plan_repo,
        _submit("req-first"),
        _submit("req-edit", destination_or_route="贡嘎环线", activity_date="2026-12-01",
                participants=8),
    )
    edited = _draft_events(result["streams"][1])[-1].payload

    assert edited["stage"] == "minimal_draft"
    assert edited["draft"] == {
        "destination_or_route": "贡嘎环线",
        "activity_date": "2026-12-01",
        "participants": 8,
    }
    summary = edited["summary"]
    # Independent of the renderer: the summary must name the new arrangements
    # and stop naming the replaced ones, and stay a 草案 with the
    # 待确认 / 待核验 boundary intact (docs/plan-pdf-format.md).
    assert "贡嘎环线" in summary and "2026-12-01" in summary and "8 人" in summary
    assert "四姑娘山长坪沟" not in summary and "2026-11-08" not in summary
    assert "待确认" in summary and "待核验" in summary

    stored = result["repo"].get_plan_draft(CONVERSATION)
    assert (stored.destination_or_route, stored.activity_date, stored.participants) == (
        "贡嘎环线", "2026-12-01", 8,
    )


def test_the_stream_text_and_the_draft_state_summary_are_the_same_draft(plan_repo):
    """AC3: the restore UX and the chat bubble read one summary, not two."""
    result = _drive(plan_repo, _submit("req-submit"))
    events = result["streams"][0]

    summary = _draft_events(events)[-1].payload["summary"]
    text = "".join(e.payload.get("content", "") for e in events if e.type == "text")
    assert text == summary


def test_every_stage_renders_a_summary():
    """AC3 + landmine 4: a stage the summary function cannot render is a crash.

    ``draft_summary`` is what the plan flow emits and what a later restore
    re-reads, so every member of the stage vocabulary has to be covered —
    either by a stage copy in :data:`STAGE_MESSAGE` or, for the stages whose
    text IS the draft (ticket 02's ``minimal_draft``, ticket 05's two review
    stages), by the render of the stored fields.
    """
    field_rendered = {
        PlanDraftStage.MINIMAL_DRAFT,
        PlanDraftStage.AWAITING_FIRST_REVIEW,
        PlanDraftStage.FIRST_REVIEWED,
        # Ticket 07: the second summary is the same field render with a final
        # ``review=`` scope, so it names the stored arrangements too.
        PlanDraftStage.AWAITING_FINAL_REVIEW,
        PlanDraftStage.FINAL_REVIEWED,
    }
    for stage in PlanDraftStage:
        rendered = draft_summary(
            stage=stage,
            destination_or_route="四姑娘山长坪沟",
            activity_date="2026-11-08",
            participants=12,
        )
        assert rendered.strip(), f"stage {stage.value} renders an empty summary"
        if stage in field_rendered:
            assert "四姑娘山长坪沟" in rendered and "2026-11-08" in rendered
        else:
            assert STAGE_MESSAGE[stage] == rendered


# ---- AC4: cancel clears the unfinished plan state --------------------------


def test_cancel_leaves_no_unfinished_plan_state_behind(plan_repo):
    """AC4: 取消流程会清理未完成的计划状态.

    ``cancelled`` is terminal and inert: the intents that need an open draft
    are all rejected again afterwards, so nothing half-finished survives the
    cancel for a later turn to act on.
    """
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-cancel", message="先不用了", intent="plan_cancel"),
    )
    events = result["streams"][1]
    stored = result["repo"].get_plan_draft(CONVERSATION)

    assert _draft_events(events)[-1].payload["stage"] == "cancelled"
    assert stored.stage == "cancelled"

    blocked = _drive(plan_repo,
        _req("req-regen-after-cancel", message="重新生成", intent="plan_regenerate"),
        _req("req-confirm-after-cancel", message="确认", intent="plan_confirm"),
    )
    assert [s[-1].payload["reason"] for s in blocked["streams"]] == [
        "regenerate_without_draft", "confirm_without_pending_confirmation",
    ]
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "cancelled"


def test_a_fresh_draft_can_be_started_after_a_cancel(plan_repo):
    """AC4: cancelling clears the state without killing the conversation."""
    result = _drive(plan_repo,
        _req("req-open", intent="plan_create"),
        _req("req-cancel", message="先不用了", intent="plan_cancel"),
        _req("req-reopen", intent="plan_create"),
    )
    reopened = _draft_events(result["streams"][2])[-1].payload
    opened = _draft_events(result["streams"][0])[-1].payload

    assert reopened["stage"] == "awaiting_input"
    # Same conversation, same draft row: the re-opened form is not a second
    # draft the user has to reconcile.
    assert reopened["draft_id"] == opened["draft_id"]
    assert result["repo"].get_plan_draft(CONVERSATION).stage == "awaiting_input"


# ---- AC5: cancel is inert --------------------------------------------------


def test_cancel_starts_no_route_comparison_semantic_judgement_or_pdf(plan_repo):
    """AC5: 取消不会启动路线比较、计划语义判断或 PDF 生成.

    Asserted as effects, not as absent code paths: no model turn (no plan
    semantic judgement), no tool event (no route comparison), no artifact
    row (no PDF), no approval row and no sandbox lease (the plan flow takes
    no lease, so only the ``max_runs`` semaphore is held and the generator
    exit releases it).
    """
    result = _drive(plan_repo,
        _submit("req-submit"),
        _req("req-cancel", message="先不用了", intent="plan_cancel"),
    )
    cancel_events = result["streams"][1]
    run_id = cancel_events[-1].run_id

    assert [e.type for e in cancel_events if e.type in TERMINALS] == ["done"]
    assert set(e.type for e in cancel_events) <= PLAN_FLOW_EVENT_TYPES
    assert "tool_call" not in [e.type for e in cancel_events]
    assert result["kernel_calls"] == []
    assert result["repo"].list_artifacts_for_run(run_id) == []
    assert result["repo"].list_approvals_for_run(run_id) == []
    assert result["repo"].get_sandbox_lease(run_id) is None

    # A cancelled run leaves no lease behind either: the whole conversation
    # only ever produced plan events.
    for stream in result["streams"]:
        assert result["repo"].get_sandbox_lease(stream[-1].run_id) is None


def test_a_rejected_cancel_is_equally_inert(plan_repo):
    """AC5: the rejection path fires nothing either."""
    result = _drive(plan_repo, _req("req-cancel-cold", message="取消", intent="plan_cancel"))
    events = result["streams"][0]

    assert events[-1].type == "error"
    assert events[-1].payload["reason"] == "cancel_without_active_draft"
    assert set(e.type for e in events) <= PLAN_FLOW_EVENT_TYPES
    assert result["kernel_calls"] == []
    assert result["repo"].get_plan_draft(CONVERSATION) is None


# ---- AC2 / AC8: the stage vocabulary and the restore wiring ----------------

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def _ts_string_set_literal(source: str, const_name: str) -> set[str]:
    block = re.search(rf"{const_name}[^=]*=\s*new Set\(\[(.*?)\]\)", source, re.S)
    assert block, f"{const_name} not found in the frontend sources"
    return set(re.findall(r"'([^']+)'", block.group(1)))


def _ts_union_members(source: str, type_name: str) -> set[str]:
    block = re.search(rf"export type {type_name} =(.*?)\n\n", source, re.S)
    assert block, f"{type_name} not found in the frontend sources"
    return set(re.findall(r"'([^']+)'", block.group(1)))


@pytest.mark.parametrize("stage", [s.value for s in PlanDraftStage])
def test_a_stage_value_is_pinned_in_the_store_and_on_the_wire(stage):
    """Landmine 4: enum ↔ SQLite store ↔ frontend, for every stage value.

    Drift here is silent on both sides — the store would accept an unknown
    stage into a row the frontend then drops, and ``readPlanDraftState``
    returns null for an unknown stage so the reducer keeps the previous
    state and the dialog simply never appears.
    """
    from agent_v2 import persistence

    assert stage in persistence._VALID_PLAN_DRAFT_STAGES, "store boundary"

    wire = (FRONTEND / "src/api/chatV2.ts").read_text(encoding="utf-8")
    assert stage in _ts_string_set_literal(wire, "V2_PLAN_DRAFT_STAGES"), "runtime set"
    assert stage in _ts_union_members(wire, "V2PlanDraftStage"), "compile-time union"

    # And the backend can render it, or a restore would crash on the summary.
    draft_summary(
        stage=PlanDraftStage(stage),
        destination_or_route="四姑娘山长坪沟",
        activity_date="2026-11-08",
        participants=12,
    )


def test_no_extra_stage_exists_on_one_side_only():
    """The pinned sets are equal in both directions, not just supersets."""
    from agent_v2 import persistence

    backend = {s.value for s in PlanDraftStage}
    wire = (FRONTEND / "src/api/chatV2.ts").read_text(encoding="utf-8")

    assert backend == set(persistence._VALID_PLAN_DRAFT_STAGES)
    assert backend == _ts_string_set_literal(wire, "V2_PLAN_DRAFT_STAGES")
    assert backend == _ts_union_members(wire, "V2PlanDraftStage")


def _ts_callback(source: str, name: str) -> str:
    """Body of one ``const name = useCallback(...)`` handler, for pinning."""
    pattern = "const " + name + r" = useCallback\(.*?\n  \}, \[[^\]]*\]\)"
    block = re.search(pattern, source, re.S)
    assert block, f"{name} not found in the page sources"
    return block.group(0)


def test_closing_the_dialog_parks_the_draft_and_cancelling_is_a_separate_action():
    """AC2 + AC4 are only both true if closing is not the same as cancelling.

    02's headline is "close the dialog, restore it later, edit it, or cancel
    it". Ticket 01 wired every dismissal to ``plan_cancel``, which would
    destroy the draft the moment the user stepped away. So the dialog has two
    exits now: backdrop / Escape / X park the draft (no request leaves the
    page), and 取消 is the only one that sends ``plan_cancel``.
    """
    page = (FRONTEND / "src/pages/SuperAgent.tsx").read_text(encoding="utf-8")
    dialog = (FRONTEND / "src/components/PlanDraftDialog.tsx").read_text(encoding="utf-8")

    dismiss = _ts_callback(page, "handlePlanDialogDismiss")
    assert "setPlanDialogOpen(false)" in dismiss, "closing must actually close"
    assert "plan_cancel" not in dismiss, "closing must not cancel the draft"
    assert "clearPlanDraft" not in dismiss, "closing must keep the stored draft"

    cancel = _ts_callback(page, "handlePlanDialogCancel")
    assert "plan_cancel" in cancel, "the 取消 action is the one that cancels"
    assert 'onCancel={handlePlanDialogCancel}' in page
    assert 'onCancel' in dialog and 'onDismiss' in dialog

    # Backdrop, Escape and the X all park; only the footer button cancels.
    assert "onClick={onDismiss}" in dialog
    assert "onDismiss()" in dialog, "the Escape listener parks too"
    assert "onClick={onCancel ?? onDismiss}" in dialog


def test_restore_wiring_is_present_and_conversation_keyed():
    """AC2 + AC4 on the frontend: the durable store is conversation-keyed.

    ``V2PersistedRun`` is nulled at every run terminal, so the run record's
    ``plan_draft`` mirror cannot be the restore source. This pins the
    replacement: its own localStorage key, read on mount / session switch,
    written from an effect (never from a ``setMessages`` updater), re-read
    through the same strict wire reader, and cleared by a cancel.
    """
    store_source = (FRONTEND / "src/api/planDraftStore.ts").read_text(encoding="utf-8")
    page = (FRONTEND / "src/pages/SuperAgent.tsx").read_text(encoding="utf-8")
    dialog = (FRONTEND / "src/components/PlanDraftDialog.tsx").read_text(encoding="utf-8")

    # Conversation-keyed store, validated through the wire reader (an unknown
    # or cancelled stage must never reopen a dialog).
    assert "ai-hiking-agent-plan-drafts" in store_source
    assert "readPlanDraftState" in store_source
    assert "export function loadPlanDraft" in store_source
    assert "export function savePlanDraft" in store_source
    assert "export function clearPlanDraft" in store_source

    # Restore on entry + durable mirror, wired into the page.
    assert "loadPlanDraft(" in page
    assert "savePlanDraft(" in page
    assert "clearPlanDraft(" in page
    assert "from '../api/planDraftStore'" in page
    assert "useEffect" in page

    # Reopen is local: the restore action itself sends no intent, so it can
    # neither start a comparison nor generate a PDF (spec AC5 + 02's brief).
    assert "handleRestorePlanDraft" in page
    assert "setPlanDialogOpen(true)" in page

    # AC3: the summary reaches the dialog through the read-only slot ticket 01
    # left, without restructuring it.
    assert "summary" in dialog
    assert "summary={planDraft?.summary}" in page


def test_the_dialog_editing_an_existing_draft_sends_plan_regenerate():
    """Code-review finding 8: ``plan_regenerate`` was whitelisted + labelled but no
    client path ever sent it — a dead intent is a trap for the next reader.

    It is wired to the one affordance its own contract already describes: the
    reopen affordance (02's 「恢复草稿」 / 「修改安排」, ticket 02 AC2) re-seeds the
    form from the stored draft, so submitting it again is "re-render the draft I
    already have with edited arrangements" — which is exactly the server-side
    ``plan_regenerate`` guard (``evaluate_plan_intent`` re-enters ``minimal_draft``
    only from a draft that is already there, and refuses it everywhere else with
    ``regenerate_without_draft``). A first submission (no draft, or the form-open
    stage) still sends ``plan_create``.

    Deliberately a source pin — there is no frontend runner — but it pins the
    behaviour, not incidental copy: the branch is the existing ``minimal_draft``
    stage, both exits collect-and-store the dialog's fields (``planInput``), and
    reopening itself stays local so no regenerate can fire without a click.
    """
    page = (FRONTEND / "src/pages/SuperAgent.tsx").read_text(encoding="utf-8")
    submit = _ts_callback(page, "handlePlanDialogSubmit")

    # Editing a draft that already sits at minimal_draft regenerates; anything
    # else (the fresh open, a review stage where the server refuses regenerate)
    # creates — the same two intents ticket 01 whitelists for field collection.
    assert "planDraft?.stage === 'minimal_draft'" in submit
    assert "'plan_regenerate'" in submit
    assert "'plan_create'" in submit
    # Both travel the dialog's collected fields, so the rewrite happens on the
    # stored record exactly like a create (no new wire field is introduced).
    assert "planInput: fields" in submit

    # Reaching that edit stays the local reopen (no request leaves the page on
    # reopen itself), so a regenerate is only ever a user act, never automatic.
    restore = _ts_callback(page, "handleRestorePlanDraft")
    assert "handleRestorePlanDraft" in page
    assert "plan_regenerate" not in restore and "handleSend" not in restore
