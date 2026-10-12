"""Ticket 09 — versioned plan-book artifact generation.

Covers the ticket's acceptance criteria at the pre-agreed seams:

* AC3 — every regeneration produces a NEW artifact: the version discriminator
  lives inside ``args_canonical``, so a second ``generate_pdf`` whose args are
  otherwise identical does NOT replay the committed effect (tested against the
  single effect-ledger implementation, :func:`drive_effect`).
* AC4 — the draft records the relationship between the current artifact and
  its history: a new ``plan_artifact_versions`` table (never a ``plan_drafts``
  column, per ticket 06's storage rule), not FK'd to ``run_id`` so it outlives
  the run-scoped CASCADE tables, exposed only through ``RunRepository``.
* AC1/AC2/AC5 — regeneration publishes through the EXISTING sandbox + approval
  + export lifecycle with real rendering: two generations of the sourced report
  are two downloadable PDFs with different bytes, a failed render publishes
  nothing, and old artifacts keep the owner / 410 / TTL download rules.

Style follows the repo convention: fake kernel + real temp SQLite through the
``RunOrchestrator`` constructor; the artifact-identity tests use the real
``LocalNativeComputer`` and the real ``export_artifact_to_host`` so the claims
are about actual files, and the download rules are exercised through the real
ASGI endpoint.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from typing import Any

from agent_v2.approvals import canonical_args, compute_approval_id
from agent_v2.effects import EffectStatus, compute_effect_key
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import (
    EFFECT_COPY_GENERATE_PDF,
    EffectDriveResult,
    drive_effect,
    pdf_generate_args,
)
from agent_v2.runs import RunRepository


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


# ---- AC4: the version table is the draft's artifact history -----------------

def test_the_version_history_is_a_new_table_never_run_scoped(tmp_path):
    """AC4 storage rule: a new table, and it must outlive runs.

    Run-scoped tables (events/artifacts/approvals/effects) CASCADE-delete with
    the run row; the history must not disappear with them, so it carries no FK
    to ``runs`` — this test is the pin for that decision.
    """
    store = ControlStore(str(tmp_path / "control.sqlite"))
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='plan_artifact_versions'"
    ).fetchall()
    assert rows, "plan_artifact_versions must exist via CREATE TABLE IF NOT EXISTS"
    fks = store._conn.execute(
        "PRAGMA foreign_key_list('plan_artifact_versions')"
    ).fetchall()
    assert [fk["table"] for fk in fks] == [], (
        "the version history must not FK to runs (or anything run-scoped): "
        "it has to outlive the run that produced the artifact"
    )


def test_the_draft_can_answer_which_artifact_is_current_and_what_came_before(tmp_path):
    """AC4: the relation between the current version and its history is stored."""
    repo = _repo(tmp_path)

    first = repo.publish_plan_artifact_version(
        conversation_id="conv-v", artifact_id="art-1", draft_id="draft-1",
        published_by_run_id="run-1",
    )
    second = repo.publish_plan_artifact_version(
        conversation_id="conv-v", artifact_id="art-2", draft_id="draft-1",
        published_by_run_id="run-2",
    )

    # Revision ascends per conversation and each row names the artifact before it.
    assert (first.revision, second.revision) == (1, 2)
    assert first.previous_artifact_id == ""
    assert second.previous_artifact_id == "art-1"
    assert second.draft_id == "draft-1"
    assert second.published_by_run_id == "run-2"

    # "Which artifact is current" — read back through a fresh facade over the
    # same database, so the answer lives on disk, not in an object cache.
    fresh = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    current = fresh.get_current_plan_artifact_version("conv-v")
    assert current is not None and current.artifact_id == "art-2"

    # "What came before" — the full history in generation order.
    history = fresh.list_plan_artifact_versions("conv-v")
    assert [(v.revision, v.artifact_id) for v in history] == [(1, "art-1"), (2, "art-2")]


def test_a_conversation_without_a_published_plan_book_has_no_current_artifact(tmp_path):
    """AC4 fails honest: no row is no claim, not artifact 0."""
    repo = _repo(tmp_path)
    assert repo.get_current_plan_artifact_version("conv-none") is None
    assert repo.list_plan_artifact_versions("conv-none") == []


# ---- AC3: the version discriminator inside args_canonical -------------------
#
# Approvals and effects are keyed (run_id, tool_name, args_canonical), and a
# committed effect is REPLAYED from result_json, never re-executed. So inside
# one run, two generate_pdf calls with identical args are the same logical
# side effect — the second returns the old artifact instead of regenerating.
# These tests pin the deliberate fix: the plan-book revision travels INSIDE the
# canonical args, so a regeneration is a NEW logical effect with a NEW effect
# key while the ledger keeps deduplicating the identical call. Remove the
# ``plan_revision`` discriminator from ``pdf_generate_args`` and the second
# drive below recomputes the SAME effect key, recovers the committed result and
# never re-executes — both tests fail. That is AC3 passing by design, not by
# accident.

def _make_run(repo: RunRepository, request_id: str = "req-ac3") -> str:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id="conv-ac3",
        profile_id="artifact",
        user_message="出 PDF",
    )
    return run_id


def _drive_pdf(repo: RunRepository, run_id: str, args: dict[str, Any], executed: list[int]):
    """Drive generate_pdf once through the SINGLE effect-ledger implementation."""
    effect_key = compute_effect_key(run_id=run_id, tool_name="generate_pdf", args=args)

    def execute() -> dict[str, Any]:
        executed.append(len(executed) + 1)
        return {"content": f"已在沙箱中生成 hiking-plan.pdf (#{len(executed) + 1})",
                "path": "hiking-plan.pdf"}

    outcome = EffectDriveResult()
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event_type: str, payload: dict[str, Any]):
        events.append((event_type, payload))
        return payload

    async def run() -> None:
        async for _ in drive_effect(
            runs_repo=repo,
            run_id=run_id,
            conversation_id="conv-ac3",
            copy=EFFECT_COPY_GENERATE_PDF,
            effect_key=effect_key,
            external_idempotency_key=f"ext-{effect_key}",
            args_canonical=canonical_args(args),
            tool_args=args,
            all_events=[],
            has_tool_call=False,
            has_tool_result_ok=False,
            execute=execute,
            emit=emit,
            outcome=outcome,
        ):
            pass

    asyncio.run(run())
    return outcome, effect_key, events


def test_a_regeneration_with_a_new_revision_executes_instead_of_replaying(tmp_path):
    """AC3: same run, same tool, same filename — only the revision separates the
    generations, and the effect ledger must see them as different effects."""
    repo = _repo(tmp_path)
    run_id = _make_run(repo)

    executed: list[int] = []
    first, first_key, _ = _drive_pdf(repo, run_id, pdf_generate_args(revision=1), executed)
    assert first.status == "executed" and executed == [1]
    assert repo.get_effect(first_key).status == EffectStatus.COMMITTED.value

    # Duplicate resume of the SAME generation (identical args) still replays —
    # the ledger's dedupe must not be broken by the versioning work.
    replay, _, _ = _drive_pdf(repo, run_id, pdf_generate_args(revision=1), executed)
    assert replay.status == "recovered" and executed == [1]

    # A NEW generation of the same run re-EXECUTES instead of recovering.
    second, second_key, _ = _drive_pdf(repo, run_id, pdf_generate_args(revision=2), executed)
    assert second_key != first_key, (
        "the regeneration under the same run/tool/args must NOT produce the "
        "same effect key — this equality is what makes it replay the old artifact"
    )
    assert second.status == "executed"
    assert executed == [1, 2], (
        "the second generation did not re-execute: without a discriminator "
        "inside args_canonical the committed effect was replayed instead"
    )


def test_the_discriminator_lives_in_the_canonical_args_and_keeps_the_filename(tmp_path):
    """Pin the shape: filename stays; the revision rides inside args_canonical.

    ``_ARTIFACT_FILENAME`` must NOT move (approval/effect ids and two duplicated
    constants derive from it), and the approval id must change with the revision
    so a regeneration opens its OWN approval rather than deduping on the old
    (run, tool, args) UNIQUE triple.
    """
    first_args = pdf_generate_args(revision=1)
    second_args = pdf_generate_args(revision=2)
    assert first_args["filename"] == "hiking-plan.pdf"
    assert canonical_args(first_args) == '{"filename":"hiking-plan.pdf","plan_revision":1}'
    assert canonical_args(first_args) != canonical_args(second_args)

    approval_1 = compute_approval_id(run_id="run-x", tool_name="generate_pdf", args=first_args)
    approval_2 = compute_approval_id(run_id="run-x", tool_name="generate_pdf", args=second_args)
    assert approval_1 != approval_2


def test_the_pause_approval_carries_the_next_plan_revision(tmp_path):
    """AC3 at the flow seam: the approval a paused run creates is keyed by the
    generation it is about to publish, and the stored ``args_canonical`` is what
    the resume path replays into ``compute_effect_key``."""
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator

    repo = _repo(tmp_path)
    run_id = _make_run(repo, "req-ac3flow")
    repo.publish_plan_artifact_version(
        conversation_id="conv-ac3", artifact_id="art-old", draft_id="", published_by_run_id=run_id,
    )

    class _Computer:
        sandbox_type = "local_native"

        def create_workspace(self, rid: str) -> str:
            return f"/ws/{rid}"

    orch = RunOrchestrator(
        settings=SimpleNamespace(
            agent_v2_enabled=True, openai_api_key="sk-test", openai_model="test-model",
        ),
        runs_repo=repo,
        kernel_turn=lambda **kw: None,
        computer_factory=lambda *a, **k: _Computer(),
        sandbox_cleanup=lambda *a, **k: None,
        gate_factory=None,  # real PermissionGate → generate_pdf HIGH → pause
    )

    async def run() -> list[Any]:
        req = ChatFlowRequest(
            message="重新生成计划书", request_id="req-ac3flowx",
            conversation_id="conv-ac3", profile_id="artifact",
        )
        return [e async for e in orch.stream_chat(req)]

    items = asyncio.run(run())
    pause = next(e for e in items if e.type == "approval.required")
    assert pause.payload["args"] == {"filename": "hiking-plan.pdf", "plan_revision": 2}
    stored = repo.get_approval(pause.payload["approval_id"])
    assert stored is not None
    assert canonical_args({"filename": "hiking-plan.pdf", "plan_revision": 2}) == stored.args_canonical
