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

    def emit(event_type: str, payload: dict[str, Any]):
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
            emit_required=emit,
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


# ---- AC1/AC2/AC3/AC5: two real generations through the existing lifecycle ---
#
# These run the REAL LocalNativeComputer (mkdtemp workspace, real ReportLab
# render, real host-side verified export into ``agent_v2_state_dir/artifacts``)
# and the REAL download endpoint, so the claims are about actual files and
# actual bytes — not a mock's bookkeeping.

from datetime import datetime, timedelta, timezone

from agent_v2.artifacts import ARTIFACT_TTL_DAYS
from agent_v2.computer import LocalNativeComputer
from agent_v2.memory_service import MemoryService
from agent_v2.repositories.memory import MemoryStore
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from api.chat_v2 import artifacts_v2_router, _cleanup_sandbox
from api.memory_v2 import SUBJECT_COOKIE_NAME
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import hashlib


def _e2e_orchestrator(tmp_path, repo, computer, texts: list[str]):
    calls: list[int] = []

    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        text = texts[min(len(calls), len(texts) - 1)]
        calls.append(1)
        return text, []

    settings = SimpleNamespace(
        agent_v2_enabled=True, openai_api_key="sk-test", openai_model="test-model",
        agent_v2_state_dir=str(tmp_path / "state"),
    )
    return RunOrchestrator(
        settings=settings,
        runs_repo=repo,
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=_cleanup_sandbox,
    )


def _pause_approve_resume(orch: RunOrchestrator, repo: RunRepository, request_id: str,
                          conversation_id: str) -> tuple[list[Any], str]:
    req = ChatFlowRequest(
        message="重新生成计划书 PDF", request_id=request_id,
        conversation_id=conversation_id, profile_id="artifact",
    )

    async def pause() -> list[Any]:
        return [e async for e in orch.stream_chat(req)]

    events = asyncio.run(pause())
    assert events[-1].type == "run.paused"
    run_id = events[-1].run_id
    approval = repo.list_pending_approvals_for_run(run_id)[0]
    decision = orch.resolve_approval_decision(
        approval.approval_id, decision="approve",
        decision_request_id=f"dec-{request_id}",
    )
    assert decision.status in ("recorded", "idempotent")

    async def resume() -> list[Any]:
        return [e async for e in orch.stream_resume(run_id=run_id)]

    resumed = asyncio.run(resume())
    return resumed, run_id


def test_a_changed_plan_regenerates_a_new_artifact_without_touching_the_old_one(tmp_path):
    """AC1+AC3+AC4+AC5 end to end on real bytes: generation 2 is a NEW artifact
    rendered through the SAME physical sandbox lifecycle; generation 1 keeps its
    exact bytes, its own file, and its own download/expiry rules."""
    repo = _repo(tmp_path)
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
    owner = memory.resolve_subject(None)
    stranger = memory.resolve_subject(None)
    memory.claim_conversation(owner.subject_id, "conv-regen")

    orch = _e2e_orchestrator(
        tmp_path, repo, LocalNativeComputer(),
        ["# 徒步计划 第一版\n集合时间上午八点", "# 徒步计划 第二版\n集合时间改到清晨六点"],
    )

    first, run1 = _pause_approve_resume(orch, repo, "req-regen-1", "conv-regen")
    avail1 = next(e for e in first if e.type == "artifact.available")
    assert avail1.payload["plan_revision"] == 1
    assert avail1.payload["previous_artifact_id"] == ""
    lease1 = repo.get_sandbox_lease(run1)
    assert lease1.status == "destroyed"  # AC1: cleanup ran, no host fallback

    second, run2 = _pause_approve_resume(orch, repo, "req-regen-2", "conv-regen")
    avail2 = next(e for e in second if e.type == "artifact.available")
    assert avail2.payload["plan_revision"] == 2
    assert avail2.payload["previous_artifact_id"] == avail1.payload["artifact_id"]

    art1 = repo.get_artifact(avail1.payload["artifact_id"])
    art2 = repo.get_artifact(avail2.payload["artifact_id"])
    assert art1 is not None and art2 is not None
    # AC3: a NEW artifact — new opaque id, new host directory, different bytes.
    assert art1.artifact_id != art2.artifact_id
    assert art1.storage_path != art2.storage_path
    import os
    assert os.path.isfile(art1.storage_path) and os.path.isfile(art2.storage_path)
    assert art1.sha256 != art2.sha256
    # AC4: the draft's relation rows — current is generation 2, history is ordered.
    current = repo.get_current_plan_artifact_version("conv-regen")
    assert current is not None and current.artifact_id == art2.artifact_id
    assert [v.revision for v in repo.list_plan_artifact_versions("conv-regen")] == [1, 2]

    # AC5: BOTH generations still obey the download rules — the old file was
    # not overwritten, so it still hashes to its own recorded digest.
    app = FastAPI()
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
    app.include_router(artifacts_v2_router, prefix="/api/v2")

    async def download() -> tuple[Any, Any, Any]:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
            cookies={SUBJECT_COOKIE_NAME: owner.token},
        ) as client:
            got1 = await client.get(f"/api/v2/artifacts/{art1.artifact_id}")
            got2 = await client.get(f"/api/v2/artifacts/{art2.artifact_id}")
            client.cookies.set(SUBJECT_COOKIE_NAME, stranger.token)
            denied = await client.get(f"/api/v2/artifacts/{art1.artifact_id}")
            return got1, got2, denied

    got1, got2, denied = asyncio.run(download())
    assert got1.status_code == 200 and hashlib.sha256(got1.content).hexdigest() == art1.sha256
    assert got2.status_code == 200 and hashlib.sha256(got2.content).hexdigest() == art2.sha256
    assert got1.content != got2.content and got1.content.startswith(b"%PDF-")
    # AC1 physical sandbox: the bytes were rendered by the real reportlab path
    # inside the mkdtemp workspace and exported — never a host-side fallback.
    assert b"v2-sandbox-" not in got1.content  # no path leak in the payload
    assert "storage_path" not in avail1.payload and "workspace_path" not in avail1.payload
    # Cross-subject access stays denied for the OLD artifact too.
    assert denied.status_code in (403, 404)


def test_expiry_still_bites_old_generations_after_regeneration(tmp_path):
    """AC5: TTL + expired status keep their meaning for superseded generations.

    There is no cron sweeper in this repo — expiry is enforced at download
    time (runtime fallback) — so this test does NOT claim a background sweep;
    it pins the 7-day TTL on the new row and the 410 once the row is past it.
    """
    repo = _repo(tmp_path)
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
    owner = memory.resolve_subject(None)
    memory.claim_conversation(owner.subject_id, "conv-expire")
    orch = _e2e_orchestrator(tmp_path, repo, LocalNativeComputer(), ["# 计划"])
    first, _run1 = _pause_approve_resume(orch, repo, "req-expire-1", "conv-expire")
    avail = next(e for e in first if e.type == "artifact.available")
    art = repo.get_artifact(avail.payload["artifact_id"])
    created = datetime.fromisoformat(art.created_at)
    expires = datetime.fromisoformat(art.expires_at)
    assert (expires - created) == timedelta(days=ARTIFACT_TTL_DAYS)

    repo.mark_artifact_expired(art.artifact_id)
    app = FastAPI()
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
    app.include_router(artifacts_v2_router, prefix="/api/v2")

    async def download():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                               cookies={SUBJECT_COOKIE_NAME: owner.token}) as c:
            return await c.get(f"/api/v2/artifacts/{art.artifact_id}")

    gone = asyncio.run(download())
    assert gone.status_code == 410


def test_a_sandbox_creation_failure_on_a_regeneration_attempt_stays_fail_closed(tmp_path):
    """AC1: the regeneration attempt provisions the sandbox the same way and
    fails CLOSED — a ComputerError becomes a terminal error event and the run
    NEVER falls back to a host-side render. (LocalNative injects the failure at
    create_workspace; the flow must stop before any approval or tool_call.)"""
    repo = _repo(tmp_path)
    orch = _e2e_orchestrator(tmp_path, repo, LocalNativeComputer(failure_mode="creation"),
                             ["# 计划"])
    req = ChatFlowRequest(
        message="重新生成计划书", request_id="req-nodevice",
        conversation_id="conv-nodevice", profile_id="artifact",
    )
    items = asyncio.run(_stream(orch, req))
    types = [e.type for e in items]
    assert types[-1] == "error"
    assert items[-1].payload["code"] == "sandbox_creation_failed"
    assert "tool_call" not in types and "approval.required" not in types
    assert repo.list_artifacts_for_run(items[0].run_id) == []


def test_a_failed_render_publishes_nothing(tmp_path):
    """AC2: a failed render publishes no half-artifact.

    The render fails INSIDE the sandbox (injected timeout in the real
    computer); the run terminates with an error, the effect is ``failed``,
    NO artifact row exists, NO ``artifact.available`` was emitted, and the
    version history stayed empty — nothing was published to supersede the
    draft's current artifact.
    """
    repo = _repo(tmp_path)
    orch = _e2e_orchestrator(tmp_path, repo, LocalNativeComputer(failure_mode="timeout"),
                             ["# 会失败的计划"])
    resumed, run_id = _pause_approve_resume(orch, repo, "req-failrender", "conv-fail")
    types = [e.type for e in resumed]
    assert "artifact.available" not in types
    assert types[-1] == "error"
    assert repo.list_artifacts_for_run(run_id) == []
    assert repo.list_plan_artifact_versions("conv-fail") == []
    run = repo.get_run(run_id)
    assert run is not None and run.status == "error"


def test_a_failed_export_persists_evidence_before_cleanup_and_downloads_410(tmp_path):
    """AC2+AC5: the export-side failure path keeps its contract.

    A sandbox whose manifest lies about the checksum gets NO published
    artifact: ``export_artifact_to_host`` persists the ``failed`` evidence row
    BEFORE cleanup (proven: the row exists and its artifact id appears in no
    available state) and the REAL download endpoint answers 410 for it."""
    class LyingComputer(LocalNativeComputer):
        def build_manifest(self, workspace_path, relative_path, declared_mime):
            from agent_v2.artifacts import ArtifactManifest
            real = super().build_manifest(workspace_path, relative_path, declared_mime)
            return ArtifactManifest(
                relative_path=real.relative_path,
                public_filename=real.public_filename,
                declared_mime=real.declared_mime,
                size_bytes=real.size_bytes,
                sha256="0" * 64,  # sandbox tampering: the host must reject
            )

    repo = _repo(tmp_path)
    memory = MemoryService(MemoryStore(str(tmp_path / "memory.sqlite")))
    owner = memory.resolve_subject(None)
    memory.claim_conversation(owner.subject_id, "conv-lying")

    class _AllowGate:
        def evaluate(self, profile, tool_name, args=None):
            from agent_v2.permissions import (
                PermissionDecision, PermissionVerdict, RiskLevel,
            )
            return PermissionVerdict(
                decision=PermissionDecision.ALLOW, risk=RiskLevel.LOW, reason="allow",
            )

    orch = _e2e_orchestrator(tmp_path, repo, LyingComputer(), ["# 篡改校验和的计划"])
    orch._gate_factory = _AllowGate

    req = ChatFlowRequest(
        message="出 PDF", request_id="req-lying-export",
        conversation_id="conv-lying", profile_id="artifact",
    )
    items = asyncio.run(_stream(orch, req))
    types = [e.type for e in items]
    assert "artifact.available" not in types
    assert types[-1] == "error"
    failed_rows = [a for a in repo.list_artifacts_for_run(items[0].run_id)
                   if a.status == "failed"]
    assert len(failed_rows) == 1

    app = FastAPI()
    app.state.agent_v2_runtime = SimpleNamespace(runs=repo, memory=memory)
    app.include_router(artifacts_v2_router, prefix="/api/v2")

    async def download():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                               cookies={SUBJECT_COOKIE_NAME: owner.token}) as c:
            return await c.get(f"/api/v2/artifacts/{failed_rows[0].artifact_id}")

    gone = asyncio.run(download())
    assert gone.status_code == 410


async def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    return [e async for e in orch.stream_chat(req)]


async def _pause_plan_kernel(**kwargs):
    return "# 已准备的计划书", []


def test_the_pause_approval_carries_the_next_plan_revision(tmp_path):
    """AC3 at the flow seam: the approval a paused run creates is keyed by the
    generation it is about to publish, and the stored ``args_canonical`` is what
    the resume path replays into ``compute_effect_key``."""
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
        kernel_turn=_pause_plan_kernel,
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


def test_the_discriminator_change_never_resurrects_the_replay_for_identical_calls(tmp_path):
    """Control half of AC3: the ledger still dedupes the SAME generation.

    Without this, "make regeneration execute" could be cheated by disabling
    effect replay entirely. Two drives with the same run + same args (same
    revision) must recover the committed result without re-executing.
    """
    repo = _repo(tmp_path)
    run_id = _make_run(repo, "req-dedup")
    executed: list[int] = []
    first, _k1, _e1 = _drive_pdf(repo, run_id, pdf_generate_args(revision=7), executed)
    assert first.status == "executed" and executed == [1]
    again, k2, _e2 = _drive_pdf(repo, run_id, pdf_generate_args(revision=7), executed)
    assert k2 == _k1
    assert again.status == "recovered"
    assert executed == [1]


# ---- The cover's 安全门 line: ticket 07's safety record, now a row ------------
#
# Ticket 08 left this gap: the cover read 待核验 in real exports because the
# safety answer only rode ``plan.draft_state.safety``. These tests drive the
# REAL plan flow (offline fixture + deterministic provider, no model — the
# ``kernel_calls == []`` invariant is asserted) and then read the rows that the
# two export sites consume via ``_pdf_plan_status_inputs``.

from agent_v2.plan_workflow import (
    SAFETY_MESSAGE,
    PlanSafetyReason,
    PlanDraftInput,
    SUPPLEMENT_ARRANGEMENT_KEYS,
)


_PLAN_CONV = "conv-safety-09"


class _SilentMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True


def _plan_settings(scenario: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True, openai_api_key="sk-test", openai_model="test-model",
        agent_v2_cowork_policy="disabled", agent_v2_state_dir="/tmp/unused",
        route_compare_data_mode="offline", route_decision_provider="deterministic",
        route_compare_demo_scenario=scenario,
    )


def _plan_drive(repo: RunRepository, *requests: ChatFlowRequest, scenario: str):
    kernel_calls: list[str] = []

    async def fake_kernel(**kwargs):
        kernel_calls.append(kwargs["message"])
        return "普通回答。", []

    orch = RunOrchestrator(
        settings=_plan_settings(scenario), runs_repo=repo, memory=_SilentMemory(),
        kernel_turn=fake_kernel,
    )

    async def drive():
        return [[e async for e in orch.stream_chat(req, subject_id="subj-09")]
                for req in requests]

    streams = asyncio.run(drive())
    return streams, kernel_calls


def _plan_req(request_id: str, *, message="生成行程", intent=None, plan_input=None,
              conversation_id=_PLAN_CONV) -> ChatFlowRequest:
    return ChatFlowRequest(
        message=message, request_id=request_id, conversation_id=conversation_id,
        profile_id="chat", intent=intent, plan_input=plan_input,
    )


_CORE_ARRANGEMENTS = {
    "meeting_time": "08:30",
    "meeting_point": "长坪沟景区北门停车场",
    "emergency_contact": "13800000000",
}


def _plan_submit(request_id: str) -> ChatFlowRequest:
    arrangements = {key: "" for key in SUPPLEMENT_ARRANGEMENT_KEYS}
    arrangements.update(_CORE_ARRANGEMENTS)
    return _plan_req(
        request_id, message="提交草稿", intent="plan_create",
        plan_input=PlanDraftInput.model_validate({
            # notes-05 landmine 7: the offline fixtures only speak for the demo
            # route; any other place name researches as insufficient_evidence.
            "destination_or_route": "演示林间环线",
            "activity_date": "2026-11-08",
            "participants": 12,
            "arrangements": arrangements,
        }),
    )


def _draft_payload_of(events: list[Any]) -> dict:
    return [e for e in events if e.type == "plan.draft_state"][-1].payload


def test_a_dangerous_research_turn_stores_the_gate_refusal_as_a_row(tmp_path):
    """The refusal the event carried is now ALSO a row, verbatim, on the draft."""
    repo = _repo(tmp_path)
    (submit_stream, research_stream), kernel_calls = _plan_drive(
        repo, _plan_submit("req-safety-submit"),
        _plan_req("req-safety-research", intent="plan_research_route"),
        scenario="severe_weather",
    )
    payload = _draft_payload_of(research_stream)
    assert payload["safety"]["hard_stop"] is True
    assert PlanSafetyReason.SEVERE_WEATHER in payload["safety"]["codes"]

    draft = repo.get_plan_draft(_PLAN_CONV)
    row = repo.get_plan_safety(_PLAN_CONV)
    assert row is not None
    assert row.draft_id == draft.draft_id
    # Verbatim: the row replays exactly the payload the event carried.
    assert row.to_payload()["message"] == payload["safety"]["message"]
    assert row.codes == tuple(payload["safety"]["codes"])
    assert row.hard_stop is True
    assert row.formal_route_plan_allowed == payload["safety"]["formal_route_plan_allowed"]
    assert kernel_calls == []  # plan turns stay model-free


def test_a_research_turn_that_the_gate_recommended_stores_a_no_block_row(tmp_path):
    """The fix for the real-export gap: the gate's PASS is also on record.

    Every field comes from the leaf's own answer (status ``recommended`` from
    the deterministic gate) — no sentence, no threshold re-derived here. The
    cover line report_pdf draws from this row is its pre-existing
    未拦截 wording, chosen by the stored booleans, not new copy.
    """
    repo = _repo(tmp_path)
    _streams, _calls = _plan_drive(
        repo, _plan_submit("req-pass-submit"),
        _plan_req("req-pass-research", intent="plan_research_route"),
        scenario="complete",
    )
    row = repo.get_plan_safety(_PLAN_CONV)
    assert row is not None
    assert row.hard_stop is False
    assert row.formal_route_plan_allowed is True
    assert row.message == ""
    assert row.comparison_status == "recommended"
    assert row.codes == ()


def test_a_cancelled_plan_clears_the_safety_row(tmp_path):
    """Ticket 06 landmine 7's rule, widened: unfinished plan state goes together."""
    repo = _repo(tmp_path)
    _streams, _calls = _plan_drive(
        repo, _plan_submit("req-cancel-submit"),
        _plan_req("req-cancel-research", intent="plan_research_route"),
        _plan_req("req-cancel-go", message="取消计划", intent="plan_cancel"),
        scenario="severe_weather",
    )
    assert repo.get_plan_safety(_PLAN_CONV) is None


def test_the_pdf_cover_inputs_read_the_stored_safety_row(tmp_path):
    """``_pdf_plan_status_inputs`` now passes ``safety=`` — the payload shape
    ``build_report_document`` already expects — and the cover renders the gate's
    message instead of 待核验."""
    from agent_v2.run_orchestration import _pdf_plan_status_inputs
    from route_compare.models import ComparisonResult
    from route_compare.report_pdf import build_report_document

    repo = _repo(tmp_path)
    (submit_stream, research_stream), kernel_calls = _plan_drive(
        repo, _plan_submit("req-cover-submit"),
        _plan_req("req-cover-research", intent="plan_research_route"),
        scenario="severe_weather",
    )
    assert kernel_calls == []
    inputs = _pdf_plan_status_inputs(repo, _PLAN_CONV)
    assert "safety" in inputs
    assert inputs["safety"]["hard_stop"] is True
    assert SAFETY_MESSAGE[PlanSafetyReason.SEVERE_WEATHER] in inputs["safety"]["message"]

    # And the document builder that BOTH export sites call draws the cover line
    # from that record, over the very comparison the research turn surfaced —
    # the 安全门 row is the stored message, not 待核验.
    comparison = next(
        e.payload["route_comparison"] for e in research_stream
        if e.type == "tool_result" and "route_comparison" in e.payload
    )
    result = ComparisonResult.model_validate(comparison)
    document = build_report_document(result, **inputs)
    safety_line = next(
        line for line in document["cover"]["status_lines"] if line["label"] == "安全门")
    assert safety_line["value"] == inputs["safety"]["message"]
    assert safety_line["value"] != "待核验"


def test_a_safety_row_from_an_older_draft_never_describes_the_current_book(tmp_path):
    """Ticket 07's staleness discipline, applied to the new row: the safety row
    counts only while it belongs to the draft on record."""
    from agent_v2.run_orchestration import _pdf_plan_status_inputs

    repo = _repo(tmp_path)
    draft = repo.save_plan_draft(
        conversation_id="conv-stale", stage="minimal_draft",
        destination_or_route="某条线", activity_date="2026-10-18", participants=4,
    )
    repo.save_plan_safety(
        conversation_id="conv-stale", draft_id="draft-of-a-previous-plan",
        payload={"hard_stop": True, "codes": ["no_go"],
                 "formal_route_plan_allowed": False,
                 "message": SAFETY_MESSAGE[PlanSafetyReason.NO_GO]},
        recorded_by_run_id="run-old",
    )
    inputs = _pdf_plan_status_inputs(repo, "conv-stale")
    assert "safety" not in inputs  # stays 待核验 rather than inheriting a stale refusal
    assert draft.stage == "minimal_draft"

    # The row written for THIS draft does count.
    repo.save_plan_safety(
        conversation_id="conv-stale", draft_id=draft.draft_id,
        payload={"hard_stop": False, "codes": [], "formal_route_plan_allowed": True,
                 "message": ""},
        recorded_by_run_id="run-new",
    )
    inputs = _pdf_plan_status_inputs(repo, "conv-stale")
    assert inputs["safety"]["formal_route_plan_allowed"] is True
