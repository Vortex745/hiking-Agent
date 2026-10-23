"""BCDEF closeout ticket 02 — review item S3: persisted terminal state and
the resume entry, as behavioral evidence.

S3 said the shared terminal vocabulary only existed *in code*: comparing
two constant sets proves nothing about what a restarted process reads from
disk, what the persistence guard accepts, or what a resume actually does.
This suite therefore drives the two seams the spec picked for S3.

* the real durable repository (:class:`agent_v2.runs.RunRepository` over
  two temp SQLite files, re-opened through NEW store objects to simulate a
  process restart) and its state-transition guard
  (:meth:`agent_v2.persistence.ControlStore.update_run_status`,
  :meth:`RunRepository.resume_run`,
  :meth:`ControlStore.finalize_run_status`,
  :meth:`ControlStore.mark_run_running`);
* the existing orchestrator resume entry
  (:meth:`agent_v2.run_orchestration.RunOrchestrator.stream_resume`).

Evidence is external only: the stored run row, the run snapshot, the event
stream's stable error code, the row counts of ``runs``/``effects``, and the
model / effect / sandbox CREATION CALL COUNTS. There is no new public API,
no test-only entry point, no extra status enum, and no assertion on
internal method call order.

Acceptance criteria (review item S3) → test nodes:

1. a durably persisted ``outcome_unknown`` row still reads as
   non-continuable
   -> ``test_durable_outcome_unknown_row_reads_as_non_continuable_after_reopen``
2. no generic CAS / resume edge can flip it to ``running``
   -> ``test_generic_resume_and_cas_cannot_flip_outcome_unknown_to_running``
3. a rejected resume creates zero model / effect / sandbox work
   -> ``test_resume_of_persisted_outcome_unknown_creates_no_model_effect_or_sandbox_work``
4. after a restart, resuming again is still rejected and creates no run
   -> ``test_restarted_store_rejects_unknown_outcome_resume_again_without_new_run``
5. ``paused`` is the positive control and keeps run / lease / effect /
   policy identity
   -> ``test_paused_positive_resume_continues_on_the_original_run_lease_effect_and_policy``
6. ``admission_failed`` is not generic-resumable, while the dedicated
   admission repair still completes the run
   -> ``test_admission_failed_is_not_generically_resumable_but_dedicated_repair_completes_run``
7. ``done`` / ``error`` / ``cancelled`` stay replay-only
   -> ``test_done_error_cancelled_resume_stays_replay_only_after_reopen``
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.approvals import compute_approval_id, compute_expiry
from agent_v2.effects import EffectStatus
from agent_v2.execution_contract import (
    ContractViolation,
    ReasonCode,
    assert_transition,
    policy_hash,
)
from agent_v2.execution_policy import build_execution_policy
from agent_v2.persistence import (
    CheckpointsStore,
    ControlStore,
    PolicySnapshotWrite,
    RunAlreadyTerminal,
)
from agent_v2.profiles import resolve_profile
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from agent_v2.sandbox_lifecycle import run_sandbox_cleanup


# ---- seams / fixtures -------------------------------------------------------


CONTROL_DB = "control.sqlite"
CHECKPOINT_DB = "ckpt.sqlite"


def _open_repo(tmp_path) -> RunRepository:
    """A real repository over two temp SQLite files.

    Calling it twice on the same ``tmp_path`` hands back a repository built
    from brand-new store objects — the restart proof: nothing but the
    on-disk rows carries over.
    """
    return RunRepository(
        control=ControlStore(str(tmp_path / CONTROL_DB)),
        checkpoints=CheckpointsStore(str(tmp_path / CHECKPOINT_DB)),
    )


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "trusted",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_frozen_policy(repo: RunRepository, run_id: str, profile_id: str) -> str:
    """Freeze the execution policy of a hand-assembled run; return its hash."""
    policy = build_execution_policy(resolve_profile(profile_id), _settings())
    write = PolicySnapshotWrite(
        policy_json=json.dumps(policy, sort_keys=True, separators=(",", ":")),
        policy_hash=policy_hash(policy),
        schema_version=policy["schema_version"],
        source_version=policy["source_version"],
        compatibility_json=json.dumps(policy["compatibility"]),
        created_at=_now_iso(),
    )
    repo.control.insert_policy_snapshot(run_id=run_id, policy=write)
    return write.policy_hash


class _FakeComputer:
    """Sandbox seam: records every CREATION and every execution call."""

    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.workspaces: list[str] = []
        self.pdfs: list[tuple[str, str, str]] = []
        self.bash_calls: list[tuple[str, str]] = []
        self.cleaned: list[str] = []

    def create_workspace(self, run_id: str) -> str:
        self.workspaces.append(run_id)
        return f"/ws/{run_id}"

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.pdfs.append((workspace_path, content, filename))
        return filename

    def run_bash(self, workspace_path, command, **_kwargs):
        self.bash_calls.append((workspace_path, command))
        return SimpleNamespace(
            exit_code=0, stdout="hi", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


def _fake_export(**kwargs: Any) -> dict[str, Any]:
    runs_repo = kwargs.get("runs_repo")
    run_id = kwargs.get("run_id")
    artifact_id = f"art-{uuid.uuid4().hex[:12]}"
    if runs_repo is not None and run_id is not None:
        runs_repo.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=kwargs.get("conversation_id") or "",
            public_filename="hiking-plan.pdf",
            mime_type="application/pdf",
            size_bytes=10,
            sha256="deadbeef",
            storage_path=f"/host/{artifact_id}.pdf",
            status="available",
            failure_reason=None,
            created_at=_now_iso(),
            expires_at="2026-11-01T00:00:00+00:00",
        )
    return {
        "artifact_id": artifact_id,
        "public_filename": "hiking-plan.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 10,
        "sha256": "deadbeef",
        "expires_at": "2026-11-01T00:00:00+00:00",
    }


def _orchestrator(
    repo: RunRepository,
    computer: _FakeComputer,
    *,
    kernel: Any = None,
    factory_calls: list[int] | None = None,
    export_calls: list[dict[str, Any]] | None = None,
):
    """The resume entry under test, with counting stubs for its seams.

    The counters observe CREATION CALLS at the injected boundaries (model
    turn, computer factory, host export). The effect ledger is counted
    where it is actually created — see :func:`_count_effect_creations`.
    """

    async def default_kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return ("# 计划正文", [])

    def factory(*_a: Any, **_k: Any) -> _FakeComputer:
        if factory_calls is not None:
            factory_calls.append(1)
        return computer

    def export(**kwargs: Any) -> dict[str, Any]:
        if export_calls is not None:
            export_calls.append(kwargs)
        return _fake_export(**kwargs)

    def cleanup(comp, workspace_path, runs, run_id):
        # ALR ticket 03: explicit typed result via the production helper —
        # the lifecycle settlement destroys the lease (the hook does not).
        return run_sandbox_cleanup(comp, workspace_path)

    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        kernel_turn=kernel or default_kernel,
        artifact_export=export,
        computer_factory=factory,
        sandbox_cleanup=cleanup,
        gate_factory=None,
    )


def _count_effect_creations(repo: RunRepository) -> list[str]:
    """Count effect-row creations without replacing the real insert.

    The wrapper still delegates to the repository, so the durable row is
    written for real; only the number of creations is observed.
    """
    creations: list[str] = []
    original = repo.insert_effect

    def counting(**kwargs: Any) -> Any:
        creations.append(kwargs.get("effect_key") or kwargs.get("effect_id") or "")
        return original(**kwargs)

    repo.insert_effect = counting  # type: ignore[method-assign]
    return creations


def _resume(orch: RunOrchestrator, run_id: str, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [
            item async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)
        ]

    return asyncio.run(run())


def _stream_chat(orch: RunOrchestrator, request_id: str, profile: str) -> list[Any]:
    req = ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id=request_id,
        profile_id=profile,
    )

    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req)]

    return asyncio.run(run())


def _count_rows(repo: RunRepository, table: str, where: str = "1=1") -> int:
    row = repo.control._conn.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE {where}"
    ).fetchone()
    return int(row["n"])


# ---- durable seeding through the repository's own guard ----------------------


def _persist_outcome_unknown_run(repo: RunRepository, request_id: str) -> str:
    """Persist an ``outcome_unknown`` RUN row through the guarded writes.

    The row is written by the repository itself (``create_run_idempotent``
    then :meth:`ControlStore.update_run_status`, the same guard every
    caller goes through) — never by raw SQL — so what a later process reads
    is exactly what a real transition stored.
    """
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    _seed_frozen_policy(repo, run_id, "artifact")
    repo.control.update_run_status(run_id=run_id, status="outcome_unknown")
    return run_id


def _paused_pdf_run(
    repo: RunRepository, request_id: str, *, prepared_content: str | None = None
) -> tuple[str, str]:
    """Hand-assemble the durable state after an artifact approval pause.

    Everything a continuation would need is in place — active sandbox lease,
    resolved approval, frozen policy — so a resume that is NOT stopped by
    the terminal guard would really do the work.
    """
    run_id, _created = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    _seed_frozen_policy(repo, run_id, "artifact")
    repo.create_sandbox_lease(
        run_id=run_id,
        sandbox_type="local_native",
        workspace_path=f"/ws/{run_id}",
    )
    approval_id = compute_approval_id(
        run_id=run_id, tool_name="generate_pdf", args={"filename": "hiking-plan.pdf"}
    )
    repo.insert_approval(
        approval_id=approval_id,
        run_id=run_id,
        conversation_id=f"conv-{request_id}",
        tool_name="generate_pdf",
        args_canonical='{"filename": "hiking-plan.pdf"}',
        risk_level="high",
        impact="生成徒步计划 PDF 并导出到宿主存储",
        created_at=_now_iso(),
        expires_at=compute_expiry(_now_iso()),
        prepared_content=prepared_content,
    )
    repo.resolve_approval(
        approval_id=approval_id,
        status="approved",
        decision_request_id=f"dec-{request_id}",
    )
    return run_id, approval_id


def _persist_outcome_unknown_over_resumable_state(
    repo: RunRepository, request_id: str
) -> str:
    """A run that LOOKS resumable (lease + resolved approval) but is terminal.

    ``outcome_unknown`` is written through the repository's own
    state-transition guard, exactly as a reconciliation writer would.
    """
    run_id, _approval = _paused_pdf_run(repo, request_id)
    repo.control.update_run_status(run_id=run_id, status="outcome_unknown")
    return run_id


# ---- AC1: the stored row itself reads as non-continuable --------------------


def test_durable_outcome_unknown_row_reads_as_non_continuable_after_reopen(tmp_path):
    repo = _open_repo(tmp_path)
    try:
        run_id = _persist_outcome_unknown_run(repo, "req-s3-unknown")
        assert repo.get_run(run_id).status == "outcome_unknown"
    finally:
        repo.close()

    # Restart: brand-new store objects over the same files.
    reopened = _open_repo(tmp_path)
    try:
        run = reopened.get_run(run_id)
        assert run is not None
        # The verdict comes from the row that survived the restart, not
        # from a set literal in this file: the repository reports the run
        # as terminal, i.e. its execution is over.
        assert run.status == "outcome_unknown"
        assert run.is_terminal is True

        # The product-facing snapshot explains the same thing with a stable
        # code, so a client cannot read the run as "still running".
        snapshot = reopened.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "outcome_unknown"
        assert snapshot.error_code == "effect_outcome_unknown"
        assert snapshot.recoverable is False

        # ...and the persistence guard agrees: the transition that a resume
        # would need is refused for THIS stored row.
        with pytest.raises(RunAlreadyTerminal):
            reopened.control.update_run_status(run_id=run_id, status="running")
        assert reopened.get_run(run_id).status == "outcome_unknown"
    finally:
        reopened.close()


# ---- AC2: no generic CAS / resume edge reaches ``running`` ------------------


def test_generic_resume_and_cas_cannot_flip_outcome_unknown_to_running(tmp_path):
    repo = _open_repo(tmp_path)
    try:
        run_id = _persist_outcome_unknown_run(repo, "req-s3-cas")
        stored_before = repo.get_run(run_id)

        attempts: list[tuple[str, Any, type[Exception] | None]] = [
            # the exact call the approval-decision endpoint makes to move a
            # paused run back into execution
            ("resume_run", lambda: repo.resume_run(run_id), RunAlreadyTerminal),
            ("update_run_status(running)",
             lambda: repo.control.update_run_status(run_id=run_id, status="running"),
             RunAlreadyTerminal),
            # a terminal CAS on an already-terminal run is refused too —
            # the guard never reopens a finished run
            ("finalize_run_status(done)",
             lambda: repo.control.finalize_run_status(
                 run_id=run_id, terminal_type="done"
             ),
             RunAlreadyTerminal),
            ("mark_run_paused", lambda: repo.mark_run_paused(run_id), RunAlreadyTerminal),
            # the admission-style conditional CAS simply matches no row
            ("mark_run_running", lambda: repo.control.mark_run_running(run_id), None),
        ]

        for name, attempt, expected_error in attempts:
            if expected_error is None:
                assert attempt() is False, name
            else:
                with pytest.raises(expected_error):
                    attempt()
            current = repo.get_run(run_id)
            assert current is not None
            assert current.status == "outcome_unknown", name
            assert current.is_terminal is True, name

        # The lifecycle contract itself has no outgoing edge from the
        # status that was read back from disk.
        with pytest.raises(ContractViolation) as violation:
            assert_transition(repo.get_run(run_id).status, "running")
        assert violation.value.code is ReasonCode.INVALID_TRANSITION
    finally:
        repo.close()

    # Nothing was written: a second process reads the same terminal row.
    reopened = _open_repo(tmp_path)
    try:
        after = reopened.get_run(run_id)
        assert after is not None
        assert after.status == "outcome_unknown"
        assert after.updated_at == stored_before.updated_at
        assert _count_rows(reopened, "runs", f"run_id = '{run_id}'") == 1
    finally:
        reopened.close()


# ---- AC3: the rejection happens before any model / effect / sandbox work ----


def test_resume_of_persisted_outcome_unknown_creates_no_model_effect_or_sandbox_work(
    tmp_path,
):
    repo = _open_repo(tmp_path)
    try:
        run_id = _persist_outcome_unknown_over_resumable_state(repo, "req-s3-zero-work")
        runs_before = _count_rows(repo, "runs")
        effects_before = _count_rows(repo, "effects")
        artifacts_before = _count_rows(repo, "artifacts")
        events_before = len(repo.get_events_since(run_id=run_id, from_seq=0))
        lease_before = repo.get_sandbox_lease(run_id)
        assert lease_before is not None and lease_before.status == "active"

        kernel_calls: list[int] = []
        factory_calls: list[int] = []
        export_calls: list[dict[str, Any]] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 不应该产生的新答案", [])

        computer = _FakeComputer()
        effect_creations = _count_effect_creations(repo)
        orch = _orchestrator(
            repo,
            computer,
            kernel=kernel,
            factory_calls=factory_calls,
            export_calls=export_calls,
        )

        items = _resume(orch, run_id)
        types = [item.type for item in items]

        # Stable contract: the refusal is the same code the live effect
        # driver and the snapshot use, carried on a terminal frame.
        assert items[-1].type == "error"
        assert items[-1].payload["code"] == "effect_outcome_unknown"
        assert items[-1].payload["run_status"] == "outcome_unknown"
        assert items[-1].payload["recoverable"] is False
        # No continuation step of any kind appears on the wire.
        assert not {
            "run_started", "progress", "heartbeat", "approval.resolved",
            "tool_call", "tool_result", "artifact_candidate",
            "artifact.available", "text", "done", "cancelled",
        } & set(types)

        # CREATION CALL COUNTS — all exactly zero.
        assert len(kernel_calls) == 0
        assert len(factory_calls) == 0
        assert len(export_calls) == 0
        assert len(effect_creations) == 0
        assert computer.workspaces == []
        assert computer.pdfs == []
        assert computer.bash_calls == []
        assert computer.cleaned == []

        # Durable facts behind the counts: nothing was created or written.
        assert _count_rows(repo, "runs") == runs_before
        assert _count_rows(repo, "effects") == effects_before
        assert _count_rows(repo, "artifacts") == artifacts_before
        assert len(repo.get_events_since(run_id=run_id, from_seq=0)) == events_before
        lease_after = repo.get_sandbox_lease(run_id)
        assert lease_after is not None
        assert lease_after.lease_id == lease_before.lease_id
        assert lease_after.fencing_token == lease_before.fencing_token
        assert lease_after.status == lease_before.status

        # Final run state: untouched terminal, never flipped to running.
        run = repo.get_run(run_id)
        assert run is not None
        assert run.status == "outcome_unknown"
        assert run.is_terminal is True
    finally:
        repo.close()


# ---- AC4: a restarted process reads the same refusal -----------------------


def test_restarted_store_rejects_unknown_outcome_resume_again_without_new_run(tmp_path):
    # First process: the resume is refused.
    repo = _open_repo(tmp_path)
    try:
        run_id = _persist_outcome_unknown_over_resumable_state(repo, "req-s3-restart")
        first_items = _resume(
            _orchestrator(repo, _FakeComputer(), factory_calls=[]), run_id
        )
        assert first_items[-1].payload["code"] == "effect_outcome_unknown"
    finally:
        repo.close()

    # Second process: brand-new store objects, brand-new orchestrator and
    # brand-new counters. Only the rows on disk survive the restart, so a
    # refusal that depended on in-process state cannot repeat.
    reopened = _open_repo(tmp_path)
    try:
        kernel_calls: list[int] = []
        factory_calls: list[int] = []
        export_calls: list[dict[str, Any]] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 新答案", [])

        computer = _FakeComputer()
        effect_creations = _count_effect_creations(reopened)
        orch = _orchestrator(
            reopened,
            computer,
            kernel=kernel,
            factory_calls=factory_calls,
            export_calls=export_calls,
        )

        again = _resume(orch, run_id)

        assert again[-1].type == "error"
        assert again[-1].payload["code"] == "effect_outcome_unknown"
        assert len(kernel_calls) == 0
        assert len(factory_calls) == 0
        assert len(export_calls) == 0
        assert len(effect_creations) == 0
        assert computer.workspaces == []

        # Still rejected through the repository's generic resume entry too.
        with pytest.raises(RunAlreadyTerminal):
            reopened.resume_run(run_id)

        # No new run: the same single row, the same request, still terminal.
        assert _count_rows(reopened, "runs") == 1
        assert _count_rows(reopened, "effects") == 0
        stored = reopened.control.get_run_by_request("req-s3-restart")
        assert stored is not None and stored.run_id == run_id
        run = reopened.get_run(run_id)
        assert run is not None and run.status == "outcome_unknown"
        assert run.is_terminal is True
    finally:
        reopened.close()


# ---- AC5: positive control — paused really is the resumable state -----------


def test_paused_positive_resume_continues_on_the_original_run_lease_effect_and_policy(
    tmp_path,
):
    repo = _open_repo(tmp_path)
    try:
        kernel_calls: list[int] = []
        factory_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划正文", [])

        computer = _FakeComputer()
        effect_creations = _count_effect_creations(repo)
        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )

        items = _stream_chat(orch, "req-s3-paused", "artifact")
        assert items[-1].type == "run.paused"
        run = repo.control.get_run_by_request("req-s3-paused")
        assert run is not None and run.status == "paused"
        assert run.is_terminal is False
        run_id = run.run_id

        lease_before = repo.get_sandbox_lease(run_id)
        assert lease_before is not None and lease_before.status == "active"
        snapshot_before = repo.control.get_policy_snapshot(run_id)
        assert snapshot_before is not None
        approval = repo.list_pending_approvals_for_run(run_id)[0]
        model_calls_before_pause = len(kernel_calls)

        # The generic resume entry DOES work for a paused run — this is the
        # control that keeps the S3 restriction honest rather than blanket.
        decision = orch.resolve_approval_decision(
            approval.approval_id, decision="approve", decision_request_id="dec-s3-paused"
        )
        assert decision.status == "recorded"
        assert repo.get_run(run_id).status == "running"

        resumed = _resume(orch, run_id)
        assert resumed[-1].type == "done"

        # Same run, same conversation, no second execution identity.
        assert all(item.run_id == run_id for item in resumed)
        assert _count_rows(repo, "runs", "request_id = 'req-s3-paused'") == 1
        assert _count_rows(repo, "runs") == 1

        # Same sandbox lease: same id, same fencing generation, one row.
        lease_after = repo.get_sandbox_lease(run_id)
        assert lease_after is not None
        assert lease_after.lease_id == lease_before.lease_id
        assert lease_after.fencing_token == lease_before.fencing_token
        assert _count_rows(repo, "sandbox_leases", f"run_id = '{run_id}'") == 1

        # One effect, bound to the ORIGINAL run and the ORIGINAL lease.
        effects = repo.list_effects_for_run(run_id)
        assert len(effects) == 1
        assert len(effect_creations) == 1
        assert effects[0].run_id == run_id
        assert effects[0].lease_id == lease_before.lease_id
        assert effects[0].status == EffectStatus.COMMITTED.value

        # One frozen policy: the resume authorized against the snapshot the
        # admission wrote, not a re-derived one.
        snapshot_after = repo.control.get_policy_snapshot(run_id)
        assert snapshot_after is not None
        assert snapshot_after.policy_hash == snapshot_before.policy_hash
        assert _count_rows(repo, "policy_snapshots", f"run_id = '{run_id}'") == 1
        product_snapshot = repo.get_snapshot(run_id)
        assert product_snapshot is not None
        assert product_snapshot.policy_hash == snapshot_before.policy_hash
        assert product_snapshot.status == "done"

        # The continuation ran the approved tool once and never re-drafted
        # the plan with a second model turn.
        assert len(computer.pdfs) == 1
        assert len(kernel_calls) == model_calls_before_pause
    finally:
        repo.close()


# ---- AC6: admission_failed — generic resume no, dedicated repair yes --------


def _fail_once(monkeypatch: Any, owner: Any, name: str, exc: Exception) -> dict[str, int]:
    """Make the FIRST call of ``owner.name`` raise, then delegate for real."""
    state = {"calls": 0}
    original = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["calls"] == 1:
            raise exc
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return state


def test_admission_failed_is_not_generically_resumable_but_dedicated_repair_completes_run(
    tmp_path, monkeypatch
):
    repo = _open_repo(tmp_path)
    try:
        kernel_calls: list[int] = []
        factory_calls: list[int] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 计划正文", [])

        computer = _FakeComputer()
        effect_creations = _count_effect_creations(repo)
        orch = _orchestrator(
            repo, computer, kernel=kernel, factory_calls=factory_calls
        )

        # A real admission failure: the fence breaks and the run row is
        # persisted in ``admission_failed`` with no model call at all.
        _fail_once(
            monkeypatch,
            repo.checkpoints,
            "ensure_conversation",
            RuntimeError("checkpoint store unavailable"),
        )
        first = _stream_chat(orch, "req-s3-admfail", "artifact")
        assert [item.type for item in first] == ["error"]
        assert first[0].payload["code"] == "admission_failed"
        assert kernel_calls == []
        run = repo.control.get_run_by_request("req-s3-admfail")
        assert run is not None and run.status == "admission_failed"
        assert run.is_terminal is True
        run_id = run.run_id

        # (a) the GENERAL lifecycle resume cannot continue it.
        with pytest.raises(RunAlreadyTerminal):
            repo.resume_run(run_id)
        assert repo.control.mark_run_running(run_id) is False
        with pytest.raises(RunAlreadyTerminal):
            repo.control.update_run_status(run_id=run_id, status="running")

        refused = _resume(orch, run_id)
        assert refused[-1].type == "error"
        assert refused[-1].payload["code"] == "admission_failed"
        assert not {
            "run_started", "progress", "approval.resolved", "tool_call",
            "tool_result", "artifact_candidate", "artifact.available", "text",
            "done",
        } & {item.type for item in refused}
        assert kernel_calls == []
        assert factory_calls == []
        assert effect_creations == []
        assert computer.workspaces == []
        assert computer.pdfs == []
        assert _count_rows(repo, "effects") == 0
        assert repo.get_run(run_id).status == "admission_failed"

        # (b) the DEDICATED admission-repair edge still completes the SAME
        # run: the same request_id backfills the missing admission records,
        # executes exactly one model turn, and never inserts a second run.
        second = _stream_chat(orch, "req-s3-admfail", "artifact")
        assert second[-1].type == "run.paused"
        assert second[0].payload["admission"]["status"] == "repaired"
        assert len(kernel_calls) == 1
        assert _count_rows(repo, "runs", "request_id = 'req-s3-admfail'") == 1
        repaired = repo.control.get_run_by_request("req-s3-admfail")
        assert repaired is not None and repaired.run_id == run_id
        assert repaired.status == "paused"

        approval = repo.list_pending_approvals_for_run(run_id)[0]
        decision = orch.resolve_approval_decision(
            approval.approval_id,
            decision="approve",
            decision_request_id="dec-s3-admfail",
        )
        assert decision.status == "recorded"
        finished = _resume(orch, run_id)
        assert finished[-1].type == "done"
        assert repo.get_run(run_id).status == "done"
        assert len(kernel_calls) == 1
        assert len(effect_creations) == 1
    finally:
        repo.close()


# ---- AC7: done / error / cancelled stay replay-only ------------------------


def _terminate_run(repo: RunRepository, run_id: str, status: str) -> None:
    """Move a running run to ``status`` through the repository's own API."""
    if status == "done":
        repo.mark_run_done(run_id)
    elif status == "error":
        repo.mark_run_error(run_id)
    elif status == "cancelled":
        accepted, reason = repo.control.request_cancel(
            run_id=run_id, idempotency_key=f"cancel-{run_id}"
        )
        assert accepted is True, reason
    else:  # pragma: no cover - guard for the parametrization itself
        raise AssertionError(f"unexpected seed status: {status}")


@pytest.mark.parametrize("status", ["done", "error", "cancelled"])
def test_done_error_cancelled_resume_stays_replay_only_after_reopen(tmp_path, status):
    repo = _open_repo(tmp_path)
    try:
        run_id, _approval = _paused_pdf_run(repo, f"req-s3-{status}")
        _terminate_run(repo, run_id, status)
        events_before = len(repo.get_events_since(run_id=run_id, from_seq=0))
    finally:
        repo.close()

    reopened = _open_repo(tmp_path)
    try:
        stored = reopened.get_run(run_id)
        assert stored is not None
        assert stored.status == status
        assert stored.is_terminal is True
        with pytest.raises(RunAlreadyTerminal):
            reopened.resume_run(run_id)

        kernel_calls: list[int] = []
        factory_calls: list[int] = []
        export_calls: list[dict[str, Any]] = []

        async def kernel(**_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            kernel_calls.append(1)
            return ("# 新答案", [])

        computer = _FakeComputer()
        effect_creations = _count_effect_creations(reopened)
        orch = _orchestrator(
            reopened,
            computer,
            kernel=kernel,
            factory_calls=factory_calls,
            export_calls=export_calls,
        )

        items = _resume(orch, run_id)
        types = [item.type for item in items]

        # Replay-only: exactly the stored frames plus the deterministic
        # synthetic terminal, and no continuation step.
        assert not {
            "run_started", "progress", "approval.resolved", "tool_call",
            "tool_result", "artifact_candidate", "artifact.available", "text",
        } & set(types)
        assert types[-1] in ("done", "error", "cancelled")

        assert kernel_calls == []
        assert factory_calls == []
        assert export_calls == []
        assert effect_creations == []
        assert computer.workspaces == []
        assert computer.pdfs == []

        # Nothing written: the second read is byte-identical to the first.
        repeated = _resume(orch, run_id)
        assert [item.type for item in repeated] == types
        assert [item.sequence for item in repeated] == [item.sequence for item in items]
        assert (
            len(reopened.get_events_since(run_id=run_id, from_seq=0)) == events_before
        )
        assert _count_rows(reopened, "runs") == 1
        assert _count_rows(reopened, "effects") == 0
        assert reopened.get_run(run_id).status == status
    finally:
        reopened.close()
