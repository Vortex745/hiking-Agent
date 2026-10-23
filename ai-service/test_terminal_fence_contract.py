"""Ticket 03 — assistant/terminal persistence fence and replay.

Behavior asserted at the execution boundary (admission → stream →
snapshot/events):

* the assistant reply is durable BEFORE any terminal frame leaves the
  stream, and the terminal event shares the checkpoints transaction;
* a failed assistant write stops the run without ``done`` and without a
  model re-run;
* a failed terminal-event write closes the stream and leaves a stable,
  replayable ``terminal_persist_failed`` explanation;
* duplicate finalize calls return the one persisted terminal;
* message sequences are derived per conversation (no hardcoded ``2``);
* first stream and replay agree on assistant content, including a
  cancelled partial answer;
* dropped progress never reorders or duplicates the terminal.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from typing import Any

import pytest

from agent_v2.persistence import (
    CheckpointsStore,
    ControlStore,
)
from agent_v2.runs import (
    AssistantPersistFailure,
    RunRepository,
    TerminalPersistFailure,
)
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from test_run_orchestration_chat import (
    SUBJECT_ID,
    TERMINALS,
    _FakeMemory,
    _repo,
    _settings,
)


EVIDENCE = [{
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "evidence": {"has_chunks": True, "sources": []},
}]


def _admit(
    repo: RunRepository,
    *,
    request_id: str,
    conversation_id: str | None = None,
    user_message: str = "徒步建议",
) -> str:
    result = repo.admit(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="chat",
        user_message=user_message,
    )
    assert result.ok and result.execute, result
    repo.mark_run_running(result.run_id)
    return result.run_id


def _finalize(
    repo: RunRepository,
    run_id: str,
    *,
    terminal: str = "done",
    sequence: int = 2,
    assistant: str | None = "安全第一。",
    request_id: str = "req-fence",
) -> Any:
    return repo.finalize_run(
        run_id=run_id,
        request_id=request_id,
        terminal_type=terminal,
        payload={"code": terminal},
        sequence=sequence,
        assistant_content=assistant,
    )


# ---- repository fence --------------------------------------------------------


def test_finalize_persists_assistant_once_and_returns_one_terminal(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-fence")

        first = _finalize(repo, run_id)

        assert first.created is True
        assert first.terminal["type"] == "done"
        assert first.terminal["run_id"] == run_id
        assert first.terminal["conversation_id"] == "conv-req-fence"
        assert first.assistant_persisted is True

        messages = repo.checkpoints.get_messages(conversation_id="conv-req-fence")
        assert [(m.role, m.sequence) for m in messages] == [("user", 0), ("assistant", 1)]
        assert [m.content for m in messages if m.role == "assistant"] == ["安全第一。"]

        second = _finalize(repo, run_id, sequence=3)

        assert second.created is False
        assert second.terminal["event_id"] == first.terminal["event_id"]
        assert second.terminal["sequence"] == first.terminal["sequence"]

        terminals = [
            e for e in repo.get_events_since(run_id=run_id, from_seq=0)
            if e["type"] in TERMINALS
        ]
        assert len(terminals) == 1
        assistants = [
            m for m in repo.checkpoints.get_messages(conversation_id="conv-req-fence")
            if m.role == "assistant"
        ]
        assert len(assistants) == 1
    finally:
        repo.close()


def test_message_sequences_are_derived_per_conversation_not_hardcoded(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_a = _admit(repo, request_id="req-seq-a", conversation_id="conv-seq")
        _finalize(repo, run_a, sequence=2, assistant="第一轮", request_id="req-seq-a")

        run_b = _admit(repo, request_id="req-seq-b", conversation_id="conv-seq")
        _finalize(repo, run_b, sequence=4, assistant="第二轮", request_id="req-seq-b")

        messages = repo.checkpoints.get_messages(conversation_id="conv-seq")
        assert [(m.role, m.sequence) for m in messages] == [
            ("user", 0),
            ("assistant", 1),
            ("user", 2),
            ("assistant", 3),
        ]
    finally:
        repo.close()


def test_finalize_uses_canonical_conversation_without_client_supplied_id(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-canonical")
        _finalize(repo, run_id, request_id="req-canonical")

        events = repo.get_events_since(run_id=run_id, from_seq=0)
        assert events
        assert {e["conversation_id"] for e in events} == {"conv-req-canonical"}
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.conversation_id == "conv-req-canonical"
        assert all(
            m["run_id"] == run_id for m in snapshot.messages
        )
    finally:
        repo.close()


def test_cancelled_finalize_persists_partial_assistant(tmp_path):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-cancel-partial")
        result = _finalize(
            repo, run_id, terminal="cancelled", sequence=2,
            assistant="已交付", request_id="req-cancel-partial",
        )
        assert result.terminal["type"] == "cancelled"
        repo.request_cancel(run_id=run_id, idempotency_key="cancel-1")

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assistant = [m for m in snapshot.messages if m["role"] == "assistant"]
        assert [m["content"] for m in assistant] == ["已交付"]
        assert snapshot.status == "cancelled"
        assert snapshot.terminal_event_type == "cancelled"
        assert snapshot.terminal_persisted is True
    finally:
        repo.close()


# ---- failure semantics -------------------------------------------------------


def test_assistant_write_failure_writes_no_terminal_and_no_assistant(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-assistant-fail")

        def boom(run_id: str):
            raise RuntimeError("disk full")

        monkeypatch.setattr(repo.checkpoints, "get_assistant_message", boom)
        with pytest.raises(AssistantPersistFailure):
            _finalize(repo, run_id, request_id="req-assistant-fail")

        assert repo.checkpoints.get_terminal_event(run_id) is None
        run = repo.get_run(run_id)
        assert run is not None and run.status == "running"
        messages = repo.checkpoints.get_messages(conversation_id="conv-req-assistant-fail")
        assert [m.role for m in messages] == ["user"]
    finally:
        repo.close()


def test_terminal_write_failure_rolls_back_and_replay_is_stable(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-terminal-fail")

        def boom(**kwargs):
            raise RuntimeError("event table unavailable")

        monkeypatch.setattr(repo.checkpoints, "_insert_terminal_event", boom)
        with pytest.raises(TerminalPersistFailure):
            _finalize(repo, run_id, request_id="req-terminal-fail")

        # The assistant insert rolled back with the terminal event.
        assert repo.checkpoints.get_terminal_event(run_id) is None
        assert repo.checkpoints.get_assistant_message(run_id) is None

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "error"
        assert snapshot.terminal_persisted is False
        assert snapshot.terminal_persist_failed is True
        assert snapshot.error_code == "terminal_persist_failed"
        assert snapshot.recoverable is True

        first = repo.get_replay_events(run_id=run_id)
        second = repo.get_replay_events(run_id=run_id)
        assert first[-1]["type"] == "error"
        assert first[-1]["payload"]["code"] == "terminal_persist_failed"
        assert first[-1]["event_id"] == second[-1]["event_id"]
        assert first[-1]["sequence"] == second[-1]["sequence"]

        # A retry after the store recovers completes the fence and clears
        # the failure marker; replay still has exactly one terminal.
        monkeypatch.undo()
        retried = _finalize(repo, run_id, sequence=3, request_id="req-terminal-fail")
        assert retried.created is True
        assert retried.terminal["type"] == "error"
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.terminal_persist_failed is False
        assert snapshot.status == "error"
        assert snapshot.terminal_persisted is True
        terminals = [
            e for e in repo.get_replay_events(run_id=run_id)
            if e["type"] in TERMINALS
        ]
        assert len(terminals) == 1
    finally:
        repo.close()


def test_status_cas_failure_reconciles_snapshot_without_second_terminal(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-cas-fail")

        def boom(**kwargs):
            raise RuntimeError("control db locked")

        monkeypatch.setattr(repo.control, "finalize_run_status", boom)
        result = _finalize(repo, run_id, request_id="req-cas-fail")

        assert result.created is True
        assert result.reconciled is True
        assert repo.get_run(run_id).status == "running"

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "done"
        assert snapshot.terminal_event_type == "done"
        assert snapshot.terminal_persisted is True
        assert snapshot.state_reconciled is True
        assert snapshot.terminal_persist_failed is False

        replay = repo.get_replay_events(run_id=run_id)
        terminals = [e for e in replay if e["type"] in TERMINALS]
        assert [e["type"] for e in terminals] == ["done"]
        assert terminals[0]["event_id"] == result.terminal["event_id"]
    finally:
        repo.close()


# ---- S2: one durable terminal per run at the repository seam ----------------


def _terminal_rows(path: str, run_id: str) -> list[tuple[str, int, str]]:
    """Read the durable terminals with an independent connection."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        return [
            (row["event_id"], row["sequence"], row["event_type"])
            for row in conn.execute(
                "SELECT event_id, sequence, event_type FROM run_events"
                " WHERE run_id = ? AND event_type IN ('cancelled', 'done', 'error') "
                "ORDER BY sequence",
                (run_id,),
            )
        ]
    finally:
        conn.close()


def _assistant_rows(path: str, run_id: str) -> list[str]:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        return [
            row["content"]
            for row in conn.execute(
                "SELECT content FROM messages WHERE run_id = ? AND role = 'assistant'"
                " ORDER BY rowid",
                (run_id,),
            )
        ]
    finally:
        conn.close()


def _race_repos(tmp_path):
    """TWO repositories = two independent connections over the same files."""
    return [_repo(tmp_path) for _ in range(2)]


def test_two_connections_racing_finalize_produce_one_terminal(tmp_path):
    creator = _repo(tmp_path)
    try:
        run_id = _admit(creator, request_id="req-race")
    finally:
        creator.close()

    checkpoints_path = str(tmp_path / "ckpt.sqlite")
    repos = _race_repos(tmp_path)
    barrier = threading.Barrier(2)
    results: list[Any] = []
    errors: list[BaseException] = []

    def worker(index: int, repo: RunRepository) -> None:
        try:
            barrier.wait(timeout=10)
            results.append(
                repo.finalize_run(
                    run_id=run_id,
                    request_id="req-race",
                    terminal_type="done" if index == 0 else "cancelled",
                    payload={"code": "done" if index == 0 else "cancelled"},
                    sequence=2 if index == 0 else 5,
                    assistant_content=f"答案-{index}",
                )
            )
        except BaseException as exc:  # pragma: no cover - failure evidence
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index, repo))
        for index, repo in enumerate(repos)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not errors, errors
        assert len(results) == 2
        # One terminal row, and both callers were handed THAT row.
        terminals = _terminal_rows(checkpoints_path, run_id)
        assert len(terminals) == 1
        assert {result.terminal["event_id"] for result in results} == {terminals[0][0]}
        assert {result.terminal["sequence"] for result in results} == {terminals[0][1]}
        assert sorted(result.created for result in results) == [False, True]

        # The loser committed nothing: exactly one assistant message, and it
        # is the winner's — never a half of the losing attempt.
        assistants = _assistant_rows(checkpoints_path, run_id)
        winner = next(result for result in results if result.created)
        loser = next(result for result in results if not result.created)
        assert assistants == [winner.assistant_content]
        assert {"答案-0", "答案-1"} - set(assistants) == (
            {"答案-1"} if winner.assistant_content == "答案-0" else {"答案-0"}
        )
        # Both callers report the same durable answer they can show the user.
        assert loser.assistant_content == winner.assistant_content
        assert loser.assistant_persisted is True

        replay = repos[0].get_replay_events(run_id=run_id)
        assert [e["type"] for e in replay if e["type"] in TERMINALS] == [
            terminals[0][2]
        ]
        assert replay[-1]["event_id"] == winner.terminal["event_id"]
    finally:
        for repo in repos:
            repo.close()


def test_a_later_done_never_overrides_an_earlier_durable_error(tmp_path):
    """The canonical is the earliest terminal, whoever wrote it."""
    first = _repo(tmp_path)
    second = _repo(tmp_path)
    try:
        run_id = _admit(first, request_id="req-override")
        written = _finalize(
            first, run_id, terminal="error", sequence=3,
            assistant="部分答案", request_id="req-override",
        )
        assert written.created is True

        repeated = second.finalize_run(
            run_id=run_id,
            request_id="req-override",
            terminal_type="done",
            payload={"code": "done"},
            sequence=9,
            assistant_content="迟到的成功",
        )

        assert repeated.created is False
        assert repeated.terminal["event_id"] == written.terminal["event_id"]
        assert repeated.terminal["type"] == "error"
        assert _terminal_rows(str(tmp_path / "ckpt.sqlite"), run_id) == [
            (written.terminal["event_id"], 3, "error")
        ]
        assert _assistant_rows(str(tmp_path / "ckpt.sqlite"), run_id) == ["部分答案"]
        assert repeated.assistant_content == "部分答案"

        snapshot = second.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.terminal_event_type == "error"
        assert [m["content"] for m in snapshot.messages if m["role"] == "assistant"] == [
            "部分答案"
        ]
        replay = second.get_replay_events(run_id=run_id)
        assert [e["event_id"] for e in replay if e["type"] in TERMINALS] == [
            written.terminal["event_id"]
        ]
    finally:
        first.close()
        second.close()


def test_terminal_write_rejected_without_a_durable_canonical_is_a_failure(
    tmp_path,
):
    """No persistent terminal → never pretend the finalize succeeded."""
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-seq-clash")
        # Something else already owns sequence 2 of this run's stream.
        repo.persist_event(
            run_id=run_id, sequence=2, event_type="tool_result",
            payload={"tool": "x", "ok": True}, request_id="req-seq-clash",
            conversation_id="conv-req-seq-clash",
        )

        with pytest.raises(TerminalPersistFailure):
            _finalize(repo, run_id, sequence=2, request_id="req-seq-clash")

        assert _terminal_rows(str(tmp_path / "ckpt.sqlite"), run_id) == []
        # The assistant half of the failed transaction is gone too.
        assert _assistant_rows(str(tmp_path / "ckpt.sqlite"), run_id) == []
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.terminal_persisted is False
        assert snapshot.terminal_persist_failed is True
    finally:
        repo.close()


def test_two_paused_events_of_one_run_are_both_durable(tmp_path):
    """``run.paused`` is a stream terminal, not a run terminal.

    The one-terminal-per-run constraint must not block the approval flow:
    a run may pause, resume and pause again before it ends.
    """
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-paused-twice")
        for sequence in (2, 3):
            repo.persist_event(
                run_id=run_id,
                sequence=sequence,
                event_type="run.paused",
                payload={"reason": "approval", "approval_id": f"ap-{sequence}"},
                request_id="req-paused-twice",
                conversation_id="conv-req-paused-twice",
            )

        events = repo.get_events_since(run_id=run_id, from_seq=0)
        assert [e["type"] for e in events].count("run.paused") == 2
        assert repo.checkpoints.get_terminal_event(run_id) is None
        assert repo.get_run(run_id).status == "running"

        # ...and the run still finalizes exactly once afterwards.
        result = _finalize(repo, run_id, sequence=4, request_id="req-paused-twice")
        assert result.created is True
        assert _terminal_rows(str(tmp_path / "ckpt.sqlite"), run_id) == [
            (result.terminal["event_id"], 4, "done")
        ]
    finally:
        repo.close()


def test_upgraded_legacy_run_reports_one_canonical_everywhere(tmp_path):
    """Archive the conflicts, then first response, snapshot and replay agree."""
    repo = _repo(tmp_path)
    try:
        run_id = _admit(repo, request_id="req-legacy-canonical")
        canonical = _finalize(
            repo, run_id, terminal="error", sequence=3,
            assistant="部分答案", request_id="req-legacy-canonical",
        )
        path = repo.checkpoints.db_path
        # A pre-S2 process left two more terminals behind.
        conn = sqlite3.connect(path, isolation_level=None)
        try:
            conn.execute("BEGIN")
            conn.execute("DROP INDEX idx_run_events_one_terminal")
            for event_id, sequence, event_type in (
                ("legacy-done", 6, "done"),
                ("legacy-cancelled", 7, "cancelled"),
            ):
                conn.execute(
                    "INSERT INTO run_events (event_id, run_id, sequence, event_type,"
                    " payload_json, request_id, conversation_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (event_id, run_id, sequence, event_type,
                     json.dumps({"code": event_type}), "req-legacy-canonical",
                     "conv-req-legacy-canonical", "2026-01-01T00:00:00Z"),
                )
            conn.execute("COMMIT")
        finally:
            conn.close()
    finally:
        repo.close()

    reopened = _repo(tmp_path)
    try:
        assert _terminal_rows(path, run_id) == [
            (canonical.terminal["event_id"], 3, "error")
        ]
        archived = reopened.checkpoints._conn.execute(
            "SELECT event_id, canonical_event_id FROM run_event_terminal_archive"
            " WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        assert [row["event_id"] for row in archived] == [
            "legacy-done", "legacy-cancelled",
        ]
        assert {row["canonical_event_id"] for row in archived} == {
            canonical.terminal["event_id"]
        }

        snapshot = reopened.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.terminal_event_type == "error"
        assert snapshot.terminal_persisted is True
        assert [m["content"] for m in snapshot.messages if m["role"] == "assistant"] == [
            "部分答案"
        ]

        replay = reopened.get_replay_events(run_id=run_id)
        terminals = [e for e in replay if e["type"] in TERMINALS]
        assert [e["event_id"] for e in terminals] == [canonical.terminal["event_id"]]

        again = reopened.finalize_run(
            run_id=run_id,
            request_id="req-legacy-canonical",
            terminal_type="done",
            payload={"code": "done"},
            sequence=8,
            assistant_content="部分答案",
        )
        assert again.created is False
        assert again.terminal["event_id"] == canonical.terminal["event_id"]
        assert again.terminal["sequence"] == snapshot.terminal_event_sequence
        assert snapshot.terminal_event_id == canonical.terminal["event_id"]
        assert again.assistant_content == "部分答案"
    finally:
        reopened.close()


# ---- orchestration fence -----------------------------------------------------


def _orch(repo: RunRepository, kernel: Any, **kwargs: Any) -> RunOrchestrator:
    return RunOrchestrator(
        settings=_settings(),
        runs_repo=repo,
        memory=_FakeMemory(),
        kernel_turn=kernel,
        **kwargs,
    )


def _stream(repo: RunRepository, orch: RunOrchestrator, request_id: str) -> list[Any]:
    async def run() -> list[Any]:
        return [
            event async for event in orch.stream_chat(
                ChatFlowRequest(message="徒步建议", request_id=request_id),
                subject_id=SUBJECT_ID,
            )
        ]

    return asyncio.run(run())


def test_done_is_released_only_after_assistant_and_terminal_are_durable(tmp_path):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []

    async def kernel(**kwargs):
        calls.append(kwargs)
        await kwargs["on_text"]("安全第一。", EVIDENCE)
        return "安全第一。", EVIDENCE

    orch = _orch(repo, kernel)
    observed: dict[str, Any] = {}

    async def scenario():
        events = []
        async for event in orch.stream_chat(
            ChatFlowRequest(message="徒步建议", request_id="fence-durable"),
            subject_id=SUBJECT_ID,
        ):
            events.append(event)
            if event.type == "done":
                # The consumer has the done frame: both durable writes
                # must already be visible to a fresh read.
                observed["assistant"] = repo.checkpoints.get_assistant_message(
                    event.run_id
                )
                observed["terminal"] = repo.checkpoints.get_terminal_event(
                    event.run_id
                )
        return events

    try:
        events = asyncio.run(scenario())
        assert [e.type for e in events if e.type in TERMINALS] == ["done"]
        assert observed["assistant"] is not None
        assert observed["assistant"].content == "安全第一。"
        assert observed["terminal"] is not None
        assert observed["terminal"].event_type == "done"
        assert observed["terminal"].event_id == events[-1].event_id
        assert len(calls) == 1
    finally:
        repo.close()


def test_assistant_persist_failure_emits_stable_error_and_no_rerun(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []

    async def kernel(**kwargs):
        calls.append(kwargs)
        await kwargs["on_text"]("不应发布的答案", EVIDENCE)
        return "不应发布的答案", EVIDENCE

    orch = _orch(repo, kernel)

    def boom(run_id: str):
        raise RuntimeError("assistant store unavailable")

    try:
        monkeypatch.setattr(repo.checkpoints, "get_assistant_message", boom)
        events = _stream(repo, orch, "fence-assistant-fail")

        types = [e.type for e in events]
        assert "done" not in types
        terminals = [e for e in events if e.type in TERMINALS]
        assert [e.type for e in terminals] == ["error"]
        assert terminals[0].payload["code"] == "assistant_persist_failed"
        assert len(calls) == 1  # no model re-run on persistence failure

        run_id = terminals[0].run_id
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "error"
        assert snapshot.terminal_persisted is True
        assert snapshot.terminal_error_code == "assistant_persist_failed"
        assert snapshot.error_code == "assistant_persist_failed"
        assert snapshot.recoverable is True
        assert [
            m for m in snapshot.messages if m["role"] == "assistant"
        ] == []

        replay = repo.get_replay_events(run_id=run_id)
        assert [e["type"] for e in replay if e["type"] in TERMINALS] == ["error"]
        assert replay[-1]["payload"]["code"] == "assistant_persist_failed"
    finally:
        monkeypatch.undo()
        repo.close()


def test_terminal_persist_failure_closes_stream_and_replay_explains(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    calls: list[dict[str, Any]] = []

    async def kernel(**kwargs):
        calls.append(kwargs)
        await kwargs["on_text"]("完整答案", EVIDENCE)
        return "完整答案", EVIDENCE

    orch = _orch(repo, kernel)

    def boom(**kwargs):
        raise RuntimeError("event store unavailable")

    try:
        monkeypatch.setattr(repo.checkpoints, "_insert_terminal_event", boom)
        events = _stream(repo, orch, "fence-terminal-fail")

        assert not [e for e in events if e.type in TERMINALS]  # closed, no done
        assert len(calls) == 1
        run_id = repo.control.get_run_by_request("fence-terminal-fail").run_id

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.status == "error"
        assert snapshot.terminal_persisted is False
        assert snapshot.terminal_persist_failed is True
        assert snapshot.error_code == "terminal_persist_failed"
        assert snapshot.recoverable is True

        first = orch.load_events(run_id)
        second = orch.load_events(run_id)
        assert first[-1]["type"] == "error"
        assert first[-1]["payload"]["code"] == "terminal_persist_failed"
        assert first[-1]["event_id"] == second[-1]["event_id"]
        assert first[-1]["sequence"] == second[-1]["sequence"]
        assert len([e for e in first if e["type"] in TERMINALS]) == 1
    finally:
        monkeypatch.undo()
        repo.close()


def test_cancelled_partial_assistant_matches_snapshot_and_replay(tmp_path):
    repo = _repo(tmp_path)
    release = asyncio.Event()
    calls: list[dict[str, Any]] = []

    async def kernel(**kwargs):
        calls.append(kwargs)
        await kwargs["on_text"]("已交付", EVIDENCE)
        await release.wait()
        return "已交付取消后不可交付", EVIDENCE

    orch = _orch(repo, kernel)

    async def scenario():
        events = []
        gen = orch.stream_chat(
            ChatFlowRequest(message="徒步建议", request_id="fence-cancel"),
            subject_id=SUBJECT_ID,
        )
        try:
            while not events or events[-1].type != "text":
                events.append(await asyncio.wait_for(anext(gen), 5))
            run_id = events[-1].run_id
            repo.request_cancel(run_id=run_id, idempotency_key="cancel-fence")
            release.set()
            events.extend([event async for event in gen])
        finally:
            release.set()
            await gen.aclose()
        return events

    try:
        events = asyncio.run(scenario())
        terminals = [e for e in events if e.type in TERMINALS]
        assert [e.type for e in terminals] == ["cancelled"]
        partial = "".join(
            e.payload["content"] for e in events if e.type == "text"
        )
        assert partial == "已交付"

        run_id = terminals[0].run_id
        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assistant = [m for m in snapshot.messages if m["role"] == "assistant"]
        assert [m["content"] for m in assistant] == [partial]

        replay = _stream(repo, orch, "fence-cancel")
        assert [e.type for e in replay if e.type in TERMINALS] == ["cancelled"]
        assert "".join(
            e.payload["content"] for e in replay if e.type == "text"
        ) == partial
        assistants = [
            m
            for m in repo.checkpoints.get_messages(conversation_id="conv-fence-cancel")
            if m.role == "assistant"
        ]
        assert len(assistants) == 1
        assert len(calls) == 1
    finally:
        repo.close()


def test_terminal_ordering_survives_dropped_progress(tmp_path, monkeypatch):
    repo = _repo(tmp_path)

    async def kernel(**kwargs):
        return "完整答案", EVIDENCE

    orch = _orch(repo, kernel)
    real_persist = repo.persist_event

    def flaky(**kwargs):
        if kwargs["event_type"] in ("progress", "heartbeat"):
            raise RuntimeError("progress dropped")
        return real_persist(**kwargs)

    try:
        monkeypatch.setattr(repo, "persist_event", flaky)
        events = _stream(repo, orch, "fence-gap")
        assert [e.type for e in events if e.type in TERMINALS] == ["done"]

        run_id = repo.control.get_run_by_request("fence-gap").run_id
        stored = repo.get_replay_events(run_id=run_id)
        assert stored[-1]["type"] == "done"
        assert len([e for e in stored if e["type"] in TERMINALS]) == 1
        sequences = [e["sequence"] for e in stored]
        assert sequences == sorted(sequences)
        assert sequences[-1] == events[-1].sequence
        # Progress events were dropped, so replay has a visible hole.
        assert len(stored) < len(events)
    finally:
        monkeypatch.undo()
        repo.close()


def test_error_path_persists_partial_assistant_before_terminal(tmp_path):
    from agent_v2.kernel import KernelModelFailure

    repo = _repo(tmp_path)

    async def kernel(**kwargs):
        await kwargs["on_text"]("部分答案", EVIDENCE)
        raise KernelModelFailure("model_timeout", "超时")

    orch = _orch(repo, kernel)
    try:
        events = _stream(repo, orch, "fence-error-partial")
        terminals = [e for e in events if e.type in TERMINALS]
        assert [e.type for e in terminals] == ["error"]
        run_id = terminals[0].run_id

        snapshot = repo.get_snapshot(run_id)
        assert snapshot is not None
        assistant = [m for m in snapshot.messages if m["role"] == "assistant"]
        assert [m["content"] for m in assistant] == ["部分答案"]
        assert snapshot.terminal_error_code == "model_timeout"
        assert snapshot.terminal_persisted is True
    finally:
        repo.close()


def test_terminal_run_replay_never_adds_a_second_terminal(tmp_path):
    repo = _repo(tmp_path)

    async def kernel(**kwargs):
        return "答案", EVIDENCE

    orch = _orch(repo, kernel)
    try:
        first = _stream(repo, orch, "fence-replay")
        second = _stream(repo, orch, "fence-replay")
        assert [e.type for e in second] == [e.type for e in first]
        run_id = first[-1].run_id
        terminals = [
            e for e in repo.get_replay_events(run_id=run_id)
            if e["type"] in TERMINALS
        ]
        assert len(terminals) == 1
        assistants = [
            m
            for m in repo.checkpoints.get_messages(conversation_id="conv-fence-replay")
            if m.role == "assistant"
        ]
        assert len(assistants) == 1
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
