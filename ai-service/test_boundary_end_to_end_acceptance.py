"""BCDEF ticket 12 — cross-module acceptance from admission to snapshot/replay.

Tickets 01-11 each proved their own slice. This file proves the SLICES FORM
ONE PATH: one HTTP request goes through admission → frozen policy → model →
approval → effect ledger → lease → terminal fence, and comes back out through
snapshot/replay with the same identities, one durable run, one user message,
one terminal and no duplicated work — under injected partial commits.

Seam: the public product surface only — ``POST /api/v2/chat/sse``,
``GET /snapshot``, ``GET /events``, ``POST /resume``,
``POST /approvals/{id}/decision`` and ``POST /effects/{id}/repair`` — driven
in-process over ASGI against a real tmp-SQLite ``RunRepository``. The model
turn, the sandbox computer, the host export, the cleanup hook and the subject
memory are counting fakes. Assertions are on external state (row counts,
durable statuses, SSE event order, idempotency) and on how many times a side
effect actually happened — never on internal call order or class names.

Encoded here is the evidence the main agent took by hand against a live
uvicorn (3 POSTs with one ``request_id`` → 1 run, 2 messages, 8 events, one
terminal, every event row carrying the canonical conversation id).

No model, no network, no real sandbox, no production database.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module
from agent_v2.effects import EffectStatus
from agent_v2.execution_contract import ExecutionErrorCode
from agent_v2.janitor import run_pending_janitor_tasks
from agent_v2.observability import (
    BOUNDARY_METRIC_NAMES,
    get_boundary_metrics,
    reset_boundary_metrics,
)
from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.run_orchestration import RunOrchestrator
from agent_v2.runs import RunRepository

RUN_TERMINALS = {"done", "error", "cancelled"}
ENVELOPE_KEYS = {
    "version",
    "event_id",
    "sequence",
    "run_id",
    "conversation_id",
    "timestamp",
    "type",
    "request_id",
    "payload",
}
SUBJECT_ID = "sub-e2e"


# ---- fakes -------------------------------------------------------------------


class _FakeMemory:
    """Server-resolved subject: owns every conversation in these tests."""

    def memory_enabled(self, subject_id: str) -> bool:
        return False

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        return True


class _FakeComputer:
    """Counting sandbox stand-in: records every host-visible side effect."""

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
            exit_code=0, stdout="ok", stderr="", duration_ms=1, truncated=False
        )

    def cleanup(self, workspace_path: str) -> dict[str, int]:
        self.cleaned.append(workspace_path)
        return {"files_deleted": 0}


# ---- harness -----------------------------------------------------------------


class Harness:
    """One HTTP surface + one orchestrator over one tmp repository."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.repo = RunRepository(
            control=ControlStore(str(tmp_path / "control.sqlite")),
            checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
        )
        self.computer = _FakeComputer()
        self.kernel_calls: list[dict[str, Any]] = []
        self.export_calls: list[dict[str, Any]] = []
        self.cleanup_calls: list[str] = []
        self.cleanup_fails = False
        self.kernel_result: tuple[str, list[dict[str, Any]]] = ("两日行程，注意天气。", [])
        self.kernel_error: BaseException | None = None

        async def kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
            self.kernel_calls.append(kwargs)
            if self.kernel_error is not None:
                raise self.kernel_error
            return self.kernel_result

        def export(**kwargs: Any) -> dict[str, Any]:
            self.export_calls.append(kwargs)
            artifact_id = f"art-{len(self.export_calls)}"
            self.repo.insert_artifact(
                artifact_id=artifact_id,
                run_id=kwargs.get("run_id") or "",
                conversation_id=kwargs.get("conversation_id") or "",
                public_filename="hiking-plan.pdf",
                mime_type="application/pdf",
                size_bytes=12,
                sha256="abc123",
                storage_path=f"/host/{artifact_id}.pdf",
                status="available",
                failure_reason=None,
                created_at=_now_iso(),
                expires_at=_iso(days=7),
            )
            return {
                "artifact_id": artifact_id,
                "public_filename": "hiking-plan.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 12,
                "sha256": "abc123",
                "expires_at": _iso(days=7),
            }

        def cleanup(computer, workspace_path, runs_repo, run_id) -> None:
            self.cleanup_calls.append(workspace_path)
            if self.cleanup_fails:
                raise RuntimeError("sandbox teardown timed out")
            computer.cleanup(workspace_path)
            if runs_repo is not None and run_id is not None:
                runs_repo.destroy_sandbox_lease(run_id=run_id)

        self.orchestrator = RunOrchestrator(
            settings=_settings(),
            runs_repo=self.repo,
            memory=_FakeMemory(),
            kernel_turn=kernel,
            artifact_export=export,
            computer_factory=lambda *a, **k: self.computer,
            sandbox_cleanup=cleanup,
            gate_factory=None,
        )
        self.app = FastAPI()
        self.app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
        self.app.include_router(
            chat_v2_module.approvals_v2_router, prefix="/api/v2"
        )
        self.app.state.agent_v2_runtime = SimpleNamespace(
            runs=self.repo, memory=_FakeMemory()
        )
        monkeypatch.setattr(
            chat_v2_module, "_build_run_orchestrator", lambda _runtime: self.orchestrator
        )
        monkeypatch.setattr(
            chat_v2_module,
            "resolve_request_subject",
            lambda request: (
                _FakeMemory(),
                setattr(request.state, "subject_id", SUBJECT_ID)
                or SimpleNamespace(issued=False, token="tok", subject_id=SUBJECT_ID),
            ),
        )

    def close(self) -> None:
        self.repo.close()

    # -- HTTP calls (the product surface) --

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None):
        async def run():
            transport = ASGITransport(app=self.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, json=body)

        return asyncio.run(run())

    def stream(
        self,
        request_id: str,
        *,
        profile: str = "chat",
        message: str = "带什么装备？",
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "message": message,
            "request_id": request_id,
            "profile_id": profile,
        }
        if conversation_id:
            body["conversation_id"] = conversation_id
        response = self._call("POST", "/api/v2/chat/sse", body)
        assert response.status_code == 200, response.text
        return _frames(response.text)

    def resume(self, run_id: str, *, from_seq: int = 0) -> list[dict[str, Any]]:
        response = self._call(
            "POST", f"/api/v2/chat/resume/{run_id}?from_seq={from_seq}"
        )
        assert response.status_code == 200, response.text
        return _frames(response.text)

    def snapshot(self, identifier: str) -> dict[str, Any]:
        response = self._call("GET", f"/api/v2/chat/snapshot/{identifier}")
        assert response.status_code == 200, response.text
        return response.json()

    def events(self, identifier: str, *, from_seq: int = 0) -> list[dict[str, Any]]:
        response = self._call("GET", f"/api/v2/chat/events/{identifier}?from_seq={from_seq}")
        assert response.status_code == 200, response.text
        return response.json()["events"]

    def decide(self, approval_id: str, decision: str = "approve", *, request_id: str) -> dict:
        response = self._call(
            "POST",
            f"/api/v2/approvals/{approval_id}/decision",
            {"decision": decision, "decision_request_id": request_id},
        )
        assert response.status_code == 200, response.text
        return response.json()

    def repair(
        self,
        effect_id: str,
        *,
        action: str,
        result: dict | None,
        key: str,
        expect_status: int = 200,
    ) -> dict:
        body: dict[str, Any] = {"action": action, "idempotency_key": key}
        if result is not None:
            body["result"] = result
        response = self._call("POST", f"/api/v2/chat/effects/{effect_id}/repair", body)
        assert response.status_code == expect_status, response.text
        return response.json()

    # -- durable reads --

    def rows(self, table: str, where: str = "1=1", *params: Any) -> int:
        return int(
            self.repo.control._conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params
            ).fetchone()["n"]
        )

    def checkpoint_rows(self, table: str, where: str = "1=1", *params: Any) -> int:
        return int(
            self.repo.checkpoints._conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params
            ).fetchone()["n"]
        )

    def run_id_for(self, request_id: str) -> str:
        run = self.repo.control.get_run_by_request(request_id)
        assert run is not None, f"no run row for {request_id}"
        return run.run_id

    def get_run_status(self, run_id: str) -> str:
        run = self.repo.control.get_run(run_id)
        assert run is not None, f"no run row for {run_id}"
        return run.status

    def paused_approval_id(self, request_id: str) -> tuple[str, str]:
        run_id = self.run_id_for(request_id)
        pending = self.repo.list_pending_approvals_for_run(run_id)
        assert len(pending) == 1, pending
        return run_id, pending[0].approval_id

    # -- the acceptance invariants shared by every scenario --

    def assert_single_durable_request(self, request_id: str) -> None:
        run_id = self.run_id_for(request_id)
        assert self.rows("runs", "request_id = ?", request_id) == 1
        assert self.checkpoint_rows(
            "messages", "run_id = ? AND role = 'user'", run_id
        ) == 1
        terminals = [
            event
            for event in self.events(run_id)
            if event["type"] in RUN_TERMINALS
        ]
        assert len(terminals) <= 1

    @property
    def side_effect_counts(self) -> dict[str, int]:
        """How much real work the boundary was allowed to do."""
        return {
            "model_calls": len(self.kernel_calls),
            "workspaces": len(self.computer.workspaces),
            "pdf_generations": len(self.computer.pdfs),
            "bash_executions": len(self.computer.bash_calls),
            "exports": len(self.export_calls),
        }


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_enabled=True,
        openai_api_key="sk-test",
        openai_model="test-model",
        agent_v2_cowork_policy="trusted",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _frames(text: str) -> list[dict[str, Any]]:
    frames = []
    for line in text.splitlines():
        if line.startswith("data: "):
            frames.append(json.loads(line[len("data: ") :]))
    return frames


def _types(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["type"] for frame in frames]


@pytest.fixture()
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_boundary_metrics()
    item = Harness(tmp_path, monkeypatch)
    yield item
    item.close()
    reset_boundary_metrics()


# ---- one full request: one run, one user message, one terminal ---------------


def test_one_request_id_yields_one_run_one_user_message_and_one_terminal(
    harness: Harness,
):
    """The live-uvicorn smoke evidence, automated (spec 07 / ticket 12).

    Three POSTs with the same ``request_id`` (the exact shape of the manual
    smoke) must leave ONE durable run, ONE user message and ONE terminal —
    and only the first request may spend a model call.
    """
    first = harness.stream("bcdef-e2e-1")

    assert _types(first)[0] == "run_started"
    assert _types(first)[-1] == "done"
    assert len([t for t in _types(first) if t in RUN_TERMINALS]) == 1

    # Duplicate requests replay the stored stream instead of executing.
    replayed = harness.stream("bcdef-e2e-1")
    again = harness.stream("bcdef-e2e-1")

    assert harness.rows("runs", "request_id = ?", "bcdef-e2e-1") == 1
    assert harness.side_effect_counts["model_calls"] == 1

    run_id = harness.run_id_for("bcdef-e2e-1")
    assert harness.checkpoint_rows(
        "messages", "run_id = ? AND role = 'user'", run_id
    ) == 1
    assert harness.checkpoint_rows(
        "messages", "run_id = ? AND role = 'assistant'", run_id
    ) == 1

    for frames in (replayed, again):
        assert {frame["run_id"] for frame in frames} == {run_id}
        assert [t for t in _types(frames) if t in RUN_TERMINALS] == ["done"]

    terminal_rows = harness.checkpoint_rows(
        "run_events", "run_id = ? AND event_type IN ('done','error','cancelled')", run_id
    )
    assert terminal_rows == 1

    snapshot = harness.snapshot(run_id)
    assert snapshot["status"] == "done"
    assert snapshot["terminal_persisted"] is True
    assert snapshot["terminal_persist_failed"] is False


def test_first_stream_reconnect_and_snapshot_agree_on_canonical_identifiers(
    harness: Harness,
):
    """One canonical conversation_id + run_id across all three read paths."""
    streamed = harness.stream("bcdef-e2e-ids")
    run_id = streamed[0]["run_id"]
    canonical_conversation = f"conv-bcdef-e2e-ids"

    # The client-supplied ids on the wire, the reconnect read, and the
    # snapshot must be the SAME identities — no derived-vs-stored drift.
    reconnected = harness.events(run_id, from_seq=0)
    snapshot = harness.snapshot(run_id)
    by_request = harness.snapshot("bcdef-e2e-ids")

    # Same run through either identifier (request_id resolves to run_id).
    identity_fields = (
        "run_id",
        "request_id",
        "conversation_id",
        "profile_id",
        "status",
        "terminal_type",
        "sequence_cursor",
        "messages",
        "policy_hash",
    )
    assert {k: snapshot[k] for k in identity_fields} == {
        k: by_request[k] for k in identity_fields
    }

    for frame in streamed:
        assert frame["run_id"] == run_id
        assert frame["conversation_id"] == canonical_conversation
    for frame in reconnected:
        assert frame["run_id"] == run_id
        assert frame["conversation_id"] == canonical_conversation

    assert snapshot["run_id"] == run_id
    assert snapshot["conversation_id"] == canonical_conversation
    assert snapshot["request_id"] == "bcdef-e2e-ids"

    # Every durable event row carries the canonical conversation id: an
    # empty one is what made replay and history disagree after a refresh.
    rows = harness.repo.checkpoints._conn.execute(
        "SELECT conversation_id FROM run_events WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert rows and {row["conversation_id"] for row in rows} == {
        canonical_conversation
    }

    # The reconnect stream is a prefix-preserving replay of the first one.
    assert [(f["sequence"], f["type"]) for f in reconnected][: len(streamed)] == [
        (f["sequence"], f["type"]) for f in streamed
    ]


def test_sse_envelope_shape_is_unchanged_and_terminal_is_last(harness: Harness):
    """Additive fields only: the v2 envelope field set is a hard boundary."""
    frames = harness.stream("bcdef-e2e-envelope")
    for frame in frames:
        assert set(frame.keys()) == ENVELOPE_KEYS
        assert frame["version"] == "2"
    sequences = [frame["sequence"] for frame in frames]
    assert sequences == sorted(sequences) and sequences[0] == 1
    assert _types(frames)[-1] in RUN_TERMINALS


# ---- failure injection: no duplicate model call, no duplicate side effect ----


def _fail_once(target: Callable[..., Any], error: BaseException) -> Callable[..., Any]:
    """Wrap ``target`` so its FIRST call raises, later calls pass through."""
    state = {"raised": False}

    def wrapper(*args: Any, **kwargs: Any):
        if not state["raised"]:
            state["raised"] = True
            raise error
        return target(*args, **kwargs)

    return wrapper


def test_run_row_write_failure_admits_nothing_and_the_retry_creates_one_run(
    harness: Harness,
):
    """Partial commit at the very first write: zero execution, then repair."""
    original = harness.repo.control.admit_run
    harness.repo.control.admit_run = _fail_once(
        original, sqlite3.OperationalError("control db unavailable")
    )
    try:
        frames = harness.stream("bcdef-inject-run")
    finally:
        harness.repo.control.admit_run = original

    assert _types(frames) == ["error"]
    assert frames[-1]["payload"]["code"] == "admission_failed"
    assert harness.rows("runs") == 0
    assert harness.side_effect_counts == {
        "model_calls": 0,
        "workspaces": 0,
        "pdf_generations": 0,
        "bash_executions": 0,
        "exports": 0,
    }

    # Same request_id once the store recovers: exactly one durable request.
    retried = harness.stream("bcdef-inject-run")
    assert _types(retried)[-1] == "done"
    harness.assert_single_durable_request("bcdef-inject-run")
    assert harness.rows("runs") == 1
    assert harness.side_effect_counts["model_calls"] == 1


def test_user_message_write_failure_leaves_no_duplicate_and_the_retry_repairs(
    harness: Harness,
):
    """The two-database split must not produce two user messages."""
    original = harness.repo.checkpoints.ensure_user_message
    harness.repo.checkpoints.ensure_user_message = _fail_once(
        original, sqlite3.OperationalError("checkpoints db unavailable")
    )
    try:
        frames = harness.stream("bcdef-inject-user")
    finally:
        harness.repo.checkpoints.ensure_user_message = original

    assert frames[-1]["payload"]["code"] == "admission_failed"
    run_id = harness.run_id_for("bcdef-inject-user")
    assert harness.rows("runs", "request_id = ?", "bcdef-inject-user") == 1
    assert harness.get_run_status(run_id) == "admission_failed"
    assert harness.checkpoint_rows("messages", "run_id = ?", run_id) == 0
    assert harness.side_effect_counts["model_calls"] == 0

    retried = harness.stream("bcdef-inject-user")
    assert _types(retried)[-1] == "done"
    assert harness.rows("runs", "request_id = ?", "bcdef-inject-user") == 1
    assert harness.checkpoint_rows(
        "messages", "run_id = ? AND role = 'user'", run_id
    ) == 1
    admission_events = [
        event
        for event in harness.events(run_id)
        if event["type"] == "run_started"
    ]
    assert len(admission_events) == 1
    assert harness.side_effect_counts["model_calls"] == 1


def test_assistant_write_failure_publishes_no_done_and_replay_never_reruns_model(
    harness: Harness,
):
    """A lost answer becomes a stable error, never a second model turn."""
    from agent_v2.persistence import AssistantMessagePersistenceError

    original = harness.repo.checkpoints.finalize_terminal
    harness.repo.checkpoints.finalize_terminal = _fail_once(
        original, AssistantMessagePersistenceError("assistant insert failed")
    )
    try:
        frames = harness.stream("bcdef-inject-assistant")
    finally:
        harness.repo.checkpoints.finalize_terminal = original

    terminals = [frame for frame in frames if frame["type"] in RUN_TERMINALS]
    assert [frame["type"] for frame in terminals] == ["error"]
    assert terminals[0]["payload"]["code"] == "assistant_persist_failed"
    run_id = harness.run_id_for("bcdef-inject-assistant")
    assert harness.get_run_status(run_id) == "error"
    assert harness.checkpoint_rows(
        "messages", "run_id = ? AND role = 'assistant'", run_id
    ) == 0

    # A duplicate request replays: the model is not a repair mechanism.
    replayed = harness.stream("bcdef-inject-assistant")
    assert [f["type"] for f in replayed if f["type"] in RUN_TERMINALS] == ["error"]
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.rows("runs", "request_id = ?", "bcdef-inject-assistant") == 1


def test_terminal_event_write_failure_keeps_replay_authoritative(
    harness: Harness,
):
    """The wire may lose the terminal; replay/snapshot still explain it once."""
    from agent_v2.persistence import TerminalEventPersistenceError

    original = harness.repo.checkpoints.finalize_terminal
    harness.repo.checkpoints.finalize_terminal = _fail_once(
        original, TerminalEventPersistenceError("terminal event row lost")
    )
    try:
        frames = harness.stream("bcdef-inject-terminal")
    finally:
        harness.repo.checkpoints.finalize_terminal = original

    # No terminal frame reached the client (the fence closed the stream).
    assert not any(frame["type"] in RUN_TERMINALS for frame in frames)
    run_id = harness.run_id_for("bcdef-inject-terminal")
    assert (
        harness.checkpoint_rows(
            "run_events",
            "run_id = ? AND event_type IN ('done','error','cancelled')",
            run_id,
        )
        == 0
    )

    snapshot = harness.snapshot(run_id)
    assert snapshot["terminal_persisted"] is False
    assert snapshot["terminal_persist_failed"] is True
    assert snapshot["error_code"] == "terminal_persist_failed"

    replayed = harness.events(run_id)
    terminals = [event for event in replayed if event["type"] in RUN_TERMINALS]
    assert [event["type"] for event in terminals] == ["error"]
    assert terminals[0]["payload"]["code"] == "terminal_persist_failed"

    # Reconnecting with the same request_id must not re-execute anything.
    restreamed = harness.stream("bcdef-inject-terminal")
    assert [f["type"] for f in restreamed if f["type"] in RUN_TERMINALS] == ["error"]
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.checkpoint_rows(
        "run_events",
        "run_id = ? AND event_type IN ('done','error','cancelled')",
        run_id,
    ) == 0
    assert get_boundary_metrics().snapshot()["terminal_persist_failed"] >= 1


def test_effect_commit_write_failure_lands_unknown_and_is_repaired_without_reexecution(
    harness: Harness,
):
    """The external PDF ran once; a lost ledger write must not run it again."""
    frames = harness.stream(
        "bcdef-inject-effect",
        profile="artifact",
        message="规划杭州两日徒步并出 PDF 计划",
    )
    assert _types(frames)[-1] == "run.paused"
    run_id, approval_id = harness.paused_approval_id("bcdef-inject-effect")
    harness.decide(approval_id, "approve", request_id="dec-effect")

    original = harness.repo.commit_effect
    harness.repo.commit_effect = _fail_once(
        original, sqlite3.OperationalError("effects table locked")
    )
    try:
        resumed = harness.resume(run_id)
    finally:
        harness.repo.commit_effect = original

    assert _types(resumed)[-1] == "error"
    assert resumed[-1]["payload"]["code"] == "effect_outcome_unknown"
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1
    # The run stops (error terminal, code effect_outcome_unknown) while the
    # EFFECT keeps the unknown outcome: the durable fact that forbids a
    # re-execution. Snapshot/replay must say so, with the next action.
    assert harness.get_run_status(run_id) == "error"
    stalled = harness.snapshot(run_id)
    assert stalled["error_code"] == "effect_outcome_unknown"
    assert stalled["effects"][0]["stored_status"] == "outcome_unknown"
    assert stalled["effects"][0]["repair_next_action"] in (
        "confirm_known_result",
        "manual_confirm",
    )
    assert stalled["effects"][0]["repair_age_seconds"] >= 0

    # A second resume is NOT a retry of the external effect.
    again = harness.resume(run_id)
    assert _types(again)[-1] == "error"
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1
    assert harness.side_effect_counts["model_calls"] == 1  # only the draft

    # BCDEF review S5: nothing was durably recorded, so NO payload — from
    # the conversation owner or anybody else — can be confirmed into an
    # authoritative success. The decision is refused with a stable code and
    # the ledger keeps its unknown outcome.
    effect = harness.repo.list_effects_for_run(run_id)[0]
    forged = harness.repair(
        effect.resolved_effect_id,
        action="confirm",
        result={"path": "hiking-plan.pdf"},
        key="repair-effect-forge",
        expect_status=409,
    )
    assert forged["code"] == "repair_result_not_durable"
    assert [
        event
        for event in harness.events(run_id)
        if event["type"] == "tool_result" and event["payload"].get("status") == "ok"
    ] == []
    assert harness.snapshot(run_id)["effects"][0]["stored_status"] == (
        "outcome_unknown"
    )

    # The fail-closed manual decision still works and stays replayable.
    abandoned = harness.repair(
        effect.resolved_effect_id,
        action="abandon",
        result=None,
        key="repair-effect-abandon",
    )
    assert abandoned["status"] == "abandoned"
    assert harness.snapshot(run_id)["effects"][0]["stored_status"] == (
        "outcome_unknown"
    )
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1


def _fail_when(original, predicate, error):
    """Call wrapper that raises ``error`` for the first matching kwargs set."""

    state = {"raised": False}

    def wrapper(*args, **kwargs):
        if not state["raised"] and predicate(kwargs):
            state["raised"] = True
            raise error
        return original(*args, **kwargs)

    return wrapper


def test_effect_publication_failure_keeps_the_result_durable_and_is_repaired(
    harness: Harness,
):
    """A lost success EVENT keeps a durable result the repair may confirm.

    This is the only shape ``confirm`` can repair (BCDEF review S5): the
    ledger recorded the external result, only its replayable publication
    was lost. Reconciliation publishes THAT result — once — and never
    re-executes the PDF.
    """
    frames = harness.stream(
        "bcdef-inject-publication",
        profile="artifact",
        message="规划杭州两日徒步并出 PDF 计划",
    )
    assert _types(frames)[-1] == "run.paused"
    run_id, approval_id = harness.paused_approval_id("bcdef-inject-publication")
    harness.decide(approval_id, "approve", request_id="dec-publication")

    original = harness.repo.persist_event
    harness.repo.persist_event = _fail_when(
        original,
        lambda kwargs: kwargs.get("event_type") == "tool_result",
        sqlite3.OperationalError("run_events unavailable"),
    )
    try:
        resumed = harness.resume(run_id)
    finally:
        harness.repo.persist_event = original

    assert _types(resumed)[-1] == "error"
    assert resumed[-1]["payload"]["code"] == "effect_outcome_unknown"
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1

    stalled = harness.snapshot(run_id)
    assert stalled["effects"][0]["stored_status"] == "outcome_unknown"
    # The result IS a durable fact now, so the next action is a confirm.
    assert stalled["effects"][0]["repair_next_action"] == "confirm_known_result"

    effect = harness.repo.list_effects_for_run(run_id)[0]
    recorded = json.loads(effect.result_json or "{}")
    assert recorded.get("path")

    # A payload that disagrees with the recorded result is refused…
    wrong = harness.repair(
        effect.resolved_effect_id,
        action="confirm",
        result={"path": "other.pdf"},
        key="repair-wrong",
        expect_status=409,
    )
    assert wrong["code"] == "repair_result_not_durable"

    # …while the honest decision republishes what the ledger recorded.
    repaired = harness.repair(
        effect.resolved_effect_id,
        action="confirm",
        result=None,
        key="repair-honest",
    )
    assert repaired["status"] == "repaired", repaired
    assert repaired["result"] == recorded

    snapshot = harness.snapshot(run_id)
    assert snapshot["effects"][0]["status"] == "committed"
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.side_effect_counts["model_calls"] == 1
    ok_results = [
        event
        for event in harness.events(run_id)
        if event["type"] == "tool_result" and event["payload"].get("status") == "ok"
    ]
    assert len(ok_results) == 1
    assert ok_results[0]["payload"]["result"] == recorded


def test_lease_acquire_failure_cleans_the_workspace_and_a_retry_creates_no_second_sandbox(
    harness: Harness,
):
    """Sandbox created, lease refused: fail closed, clean up, never re-mint."""
    original = harness.repo.acquire_sandbox_lease
    harness.repo.acquire_sandbox_lease = _fail_once(
        original, sqlite3.OperationalError("sandbox_leases unavailable")
    )
    try:
        frames = harness.stream(
            "bcdef-inject-lease",
            profile="artifact",
            message="规划杭州两日徒步并出 PDF 计划",
        )
    finally:
        harness.repo.acquire_sandbox_lease = original

    assert _types(frames)[-1] == "error"
    assert frames[-1]["payload"]["code"] == "sandbox_lease_unavailable"
    assert harness.side_effect_counts["workspaces"] == 1
    assert harness.cleanup_calls == [f"/ws/{harness.run_id_for('bcdef-inject-lease')}"]
    assert harness.side_effect_counts["pdf_generations"] == 0
    assert harness.side_effect_counts["model_calls"] == 0
    assert harness.rows("sandbox_leases") == 0
    assert harness.rows("approvals") == 0

    # Retry: the run is terminal, so nothing executes again — no second
    # sandbox, no second lease, no second model call.
    retried = harness.stream(
        "bcdef-inject-lease",
        profile="artifact",
        message="规划杭州两日徒步并出 PDF 计划",
    )
    assert [f["type"] for f in retried if f["type"] in RUN_TERMINALS] == ["error"]
    assert harness.side_effect_counts["workspaces"] == 1
    assert harness.rows("sandbox_leases") == 0
    assert harness.rows("runs", "request_id = ?", "bcdef-inject-lease") == 1


# ---- every boundary outcome is explainable from snapshot/replay, and
# ---- recoverable without re-running the model --------------------------------


def _paused_artifact_run(harness: Harness, request_id: str) -> tuple[str, str]:
    frames = harness.stream(
        request_id,
        profile="artifact",
        message="规划杭州两日徒步并出 PDF 计划",
    )
    assert _types(frames)[-1] == "run.paused"
    return harness.paused_approval_id(request_id)


def test_effect_success_is_explainable_from_snapshot_and_replay(harness: Harness):
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-success")
    harness.decide(approval_id, "approve", request_id="dec-success")

    resumed = harness.resume(run_id)
    assert _types(resumed)[-1] == "done"

    snapshot = harness.snapshot(run_id)
    assert snapshot["status"] == "done"
    assert snapshot["conversation_id"] == "conv-bcdef-out-success"
    effect = snapshot["effects"][0]
    assert effect["status"] == "committed"
    assert effect["stored_status"] == "committed"
    assert effect["result"]["path"] == "hiking-plan.pdf"
    assert effect["repair_next_action"] == "none"
    assert snapshot["sandbox_lease"]["status"] == "destroyed"
    assert snapshot["sandbox_lease"]["cleanup_status"] == "succeeded"

    replayed = harness.events(run_id)
    types = [event["type"] for event in replayed]
    # The durable effect facts — approval, tool call/result and the terminal
    # — are all reconstructible from the replay alone.
    for required in ("approval.required", "tool_call", "tool_result", "done"):
        assert required in types, types
    assert len([t for t in types if t in RUN_TERMINALS]) == 1
    assert {event["run_id"] for event in replayed} == {run_id}
    assert harness.side_effect_counts == {
        "model_calls": 1,  # the pre-pause draft; resume never re-runs it
        "workspaces": 1,
        "pdf_generations": 1,
        "bash_executions": 0,
        "exports": 1,
    }
    assert harness.rows("artifacts", "run_id = ?", run_id) == 1


def test_outcome_unknown_is_recoverable_by_the_janitor_without_re_running_model(
    harness: Harness,
):
    """Reconciliation (not re-execution) is the only recovery the boundary offers."""
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-unknown")
    harness.decide(approval_id, "approve", request_id="dec-unknown")

    original = harness.repo.commit_effect
    harness.repo.commit_effect = _fail_once(
        original, sqlite3.OperationalError("effects table locked")
    )
    try:
        resumed = harness.resume(run_id)
    finally:
        harness.repo.commit_effect = original
    assert resumed[-1]["payload"]["code"] == "effect_outcome_unknown"
    effect = harness.repo.list_effects_for_run(run_id)[0]

    # The escalation path the janitor runner serves: one durable repair work
    # item for the unknown effect, then a sweep.
    task, created = harness.repo.enqueue_janitor_task(
        kind="repair",
        run_id=run_id,
        effect_id=effect.resolved_effect_id,
        evidence={"run_id": run_id, "effect_id": effect.resolved_effect_id},
        dedupe_key=f"repair:{run_id}",
    )
    assert created is True
    results = run_pending_janitor_tasks(harness.repo, claimer="janitor-e2e")
    assert [result.kind for result in results] == ["repair"]
    assert results[0].status == "done"

    # Zero re-execution: the PDF ran once, the model ran once.
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1

    snapshot = harness.snapshot(run_id)
    repaired = snapshot["effects"][0]
    assert repaired["repair_status"] in ("repaired", "repair_claimed")
    assert repaired["repair_attempts"] >= 0
    # The run is still not continuable: an unknown outcome is terminal for
    # execution, and repair never reopens the model turn.
    after = harness.resume(run_id)
    assert [f["type"] for f in after if f["type"] in RUN_TERMINALS] == ["error"]
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.side_effect_counts["pdf_generations"] == 1


def test_approval_pause_is_explainable_and_resumes_the_same_run(harness: Harness):
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-approval")

    snapshot = harness.snapshot(run_id)
    assert snapshot["status"] == "paused"
    assert snapshot["pending_approvals"], "approval prompt must survive a refresh"
    assert snapshot["pending_approvals"][0]["approval_id"] == approval_id
    assert snapshot["pending_approvals"][0]["tool"] == "generate_pdf"

    # A reconnect while paused replays the prompt and never adds a run terminal.
    reconnect = harness.events(run_id)
    assert "approval.required" in [event["type"] for event in reconnect]
    assert "run.paused" in [event["type"] for event in reconnect]
    assert not any(event["type"] in RUN_TERMINALS for event in reconnect)

    harness.decide(approval_id, "approve", request_id="dec-approval")
    run_status_after_decision = harness.get_run_status(run_id)
    # A repeated decision with the same idempotency key is accepted as the
    # same decision and changes no durable state (200, not a new transition).
    duplicate = harness.decide(approval_id, "approve", request_id="dec-approval")
    assert duplicate["approval_id"] == approval_id
    assert harness.get_run_status(run_id) == run_status_after_decision

    resumed = harness.resume(run_id)
    assert _types(resumed)[-1] == "done"
    assert {frame["run_id"] for frame in resumed} == {run_id}

    # ONE run, approval, lease and effect advanced by the decision.
    assert harness.rows("runs", "request_id = ?", "bcdef-out-approval") == 1
    assert harness.rows("approvals", "run_id = ?", run_id) == 1
    assert harness.rows("sandbox_leases", "run_id = ?", run_id) == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 1
    assert harness.side_effect_counts["model_calls"] == 1


def test_expired_lease_is_explainable_and_refuses_work_without_a_new_generation(
    harness: Harness,
):
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-lease")
    harness.decide(approval_id, "approve", request_id="dec-lease")
    lease_before = harness.repo.get_sandbox_lease(run_id)
    assert lease_before is not None
    harness.repo.control._conn.execute(
        "UPDATE sandbox_leases SET expires_at = ? WHERE run_id = ?",
        ("2000-01-01T00:00:00+00:00", run_id),
    )

    resumed = harness.resume(run_id)
    assert _types(resumed)[-1] == "error"
    assert resumed[-1]["payload"]["code"] == "lease_expired"

    snapshot = harness.snapshot(run_id)
    assert snapshot["sandbox_lease"]["status"] == "expired"
    assert snapshot["error_code"] == "lease_expired"
    # No new fencing generation, no second sandbox, no effect, no model turn.
    assert snapshot["sandbox_lease"]["fencing_token"] == lease_before.fencing_token
    assert harness.side_effect_counts["workspaces"] == 1
    assert harness.side_effect_counts["pdf_generations"] == 0
    assert harness.side_effect_counts["model_calls"] == 1
    assert harness.rows("effects", "run_id = ?", run_id) == 0
    assert harness.rows("sandbox_leases", "run_id = ?", run_id) == 1
    assert get_boundary_metrics().snapshot()["lease_expired"] >= 1

    # The decision seam is fenced the same way: once the lease is gone, a
    # fresh decision is refused outright instead of resuming onto a new
    # sandbox (409, no state change).
    second = harness._call(
        "POST",
        f"/api/v2/approvals/{approval_id}/decision",
        {"decision": "approve", "decision_request_id": "dec-lease-2"},
    )
    assert second.status_code == 409, second.text
    assert harness.rows("effects", "run_id = ?", run_id) == 0


def test_cleanup_pending_is_explainable_and_the_janitor_uses_the_original_workspace(
    harness: Harness,
):
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-cleanup")
    harness.decide(approval_id, "approve", request_id="dec-cleanup")
    harness.cleanup_fails = True

    resumed = harness.resume(run_id)
    # Cleanup bookkeeping never rewrites the terminal: the effect and the
    # answer are durable, so ``done`` still stands.
    assert _types(resumed)[-1] == "done"

    snapshot = harness.snapshot(run_id)
    assert snapshot["sandbox_lease"]["cleanup_status"] == "pending_janitor"
    assert snapshot["sandbox_lease"]["janitor_task_id"]
    assert harness.rows(
        "janitor_tasks", "kind = 'cleanup' AND status = 'pending'"
    ) == 1
    assert get_boundary_metrics().snapshot()["cleanup_pending"] == 1

    # The janitor retries the ORIGINAL workspace and never mints a sandbox.
    workspace = harness.repo.get_sandbox_lease(run_id).workspace_path
    assert workspace == f"/ws/{run_id}"
    results = run_pending_janitor_tasks(
        harness.repo, computer_factory=lambda *_a, **_k: harness.computer
    )
    assert [result.status for result in results] == ["done"]
    assert results[0].kind == "cleanup"
    assert harness.side_effect_counts["workspaces"] == 1
    assert workspace in harness.computer.cleaned
    assert harness.rows("janitor_tasks", "kind = 'cleanup' AND status = 'pending'") == 0

    settled = harness.snapshot(run_id)
    assert settled["sandbox_lease"]["cleanup_status"] == "succeeded"
    # The terminal never moved and no effect ran twice.
    assert settled["status"] == "done"
    assert harness.side_effect_counts["pdf_generations"] == 1
    assert len([e for e in harness.events(run_id) if e["type"] in RUN_TERMINALS]) == 1


def test_incompatible_policy_is_explainable_and_never_falls_back_to_live_config(
    harness: Harness,
):
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-out-policy")
    harness.repo.control._conn.execute(
        "UPDATE policy_snapshots SET schema_version = 99 WHERE run_id = ?", (run_id,)
    )
    harness.decide(approval_id, "approve", request_id="dec-policy")

    resumed = harness.resume(run_id)
    assert _types(resumed)[-1] == "error"
    assert resumed[-1]["payload"]["code"] == "policy_incompatible"
    assert resumed[-1]["payload"]["recoverable"] is False

    snapshot = harness.snapshot(run_id)
    assert snapshot["policy_failure"] is not None
    assert snapshot["policy_failure"]["next_action"]

    # Refused before any work: no effect, no PDF, no extra model turn.
    assert harness.rows("effects", "run_id = ?", run_id) == 0
    assert harness.side_effect_counts["pdf_generations"] == 0
    assert harness.side_effect_counts["model_calls"] == 1

    # Recovery is a NEW admission (this code is not continuable by design):
    # a fresh request with an intact policy completes without touching the
    # broken run.
    new_run_id, new_approval = _paused_artifact_run(harness, "bcdef-out-policy-new")
    assert new_run_id != run_id
    harness.decide(new_approval, "approve", request_id="dec-policy-new")
    completed = harness.resume(new_run_id)
    assert _types(completed)[-1] == "done"
    assert harness.rows("runs", "request_id = ?", "bcdef-out-policy") == 1
    assert harness.side_effect_counts["model_calls"] == 2  # one draft per run


def test_replay_gap_is_explainable_and_never_delays_or_duplicates_the_terminal(
    harness: Harness,
):
    """A dropped progress frame becomes a visible hole, not a silent one."""
    original = harness.repo.persist_event

    def dropping_progress(**kwargs: Any):
        if kwargs.get("event_type") == "progress":
            return None  # the write is lost, the stream continues
        return original(**kwargs)

    harness.repo.persist_event = dropping_progress
    try:
        frames = harness.stream("bcdef-out-gap")
    finally:
        harness.repo.persist_event = original

    assert _types(frames)[-1] == "done", "the live stream is unaffected"
    run_id = harness.run_id_for("bcdef-out-gap")

    replayed = harness.events(run_id)
    types = [event["type"] for event in replayed]
    assert types.count("progress.gap") == 1
    gap = next(event for event in replayed if event["type"] == "progress.gap")
    assert set(gap["payload"]) == {"from_sequence", "to_sequence", "dropped"}
    assert gap["payload"]["dropped"] >= 1
    # The gap lands inside the hole and the terminal stays last, alone.
    assert types[-1] == "done"
    assert len([t for t in types if t in RUN_TERMINALS]) == 1
    assert [event["sequence"] for event in replayed] == sorted(
        event["sequence"] for event in replayed
    )
    assert get_boundary_metrics().snapshot()["replay_gap"] == 1

    # The client's authoritative read is unchanged: the answer is durable.
    snapshot = harness.snapshot(run_id)
    assert snapshot["status"] == "done"
    assert [m["role"] for m in snapshot["messages"]] == ["user", "assistant"]


def test_a_clean_request_moves_no_boundary_counter(harness: Harness):
    """No news is good news only if the counters were really wired."""
    harness.stream("bcdef-metrics-clean")
    run_id, approval_id = _paused_artifact_run(harness, "bcdef-metrics-clean-artifact")
    harness.decide(approval_id, "approve", request_id="dec-metrics-clean")
    harness.resume(run_id)

    snapshot = get_boundary_metrics().snapshot()
    assert {
        name: snapshot[name] for name in BOUNDARY_METRIC_NAMES
    } == {name: 0 for name in BOUNDARY_METRIC_NAMES}
    assert snapshot["repair_age_max_seconds"] == 0.0


