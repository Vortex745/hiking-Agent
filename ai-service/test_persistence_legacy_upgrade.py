"""BCDEF upgrade-path guard for the ticket 02 admission constraint.

The one-user-message-per-run UNIQUE index is created by a migration, not by
the schema script, because databases written before ticket 02 legitimately
contain duplicate user messages. If the index lived in the schema script,
``CheckpointsStore.__init__`` would raise on such a file and NO store (and
therefore no admission) could open at all.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_v2.persistence import CheckpointsStore


def _insert(store: CheckpointsStore, *, message_id: str, run_id: str, role: str, sequence: int) -> None:
    store._conn.execute(
        "INSERT INTO messages (message_id, conversation_id, run_id, role,"
        " content, sequence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (message_id, "conv-upgrade", run_id, role, f"{role}-{sequence}", sequence, "2026-01-01T00:00:00Z"),
    )


def _seed_conversation(store: CheckpointsStore) -> None:
    store.ensure_conversation(conversation_id="conv-upgrade", profile_id="chat")


def test_legacy_duplicate_user_messages_do_not_block_store_startup(tmp_path):
    path = str(tmp_path / "ckpt.sqlite")
    store = CheckpointsStore(path)
    try:
        _seed_conversation(store)
        # Emulate a pre-ticket-02 database: drop the constraint and write the
        # duplicate user messages the old code path allowed.
        store._conn.execute("BEGIN")
        store._conn.execute("DROP INDEX idx_messages_run_user")
        _insert(store, message_id="m-1", run_id="run-1", role="user", sequence=0)
        _insert(store, message_id="m-2", run_id="run-1", role="user", sequence=0)
        _insert(store, message_id="m-3", run_id="run-1", role="assistant", sequence=1)
        store._conn.execute("COMMIT")
    finally:
        store.close()

    # Reopening is what the app does on the next boot / in tests.
    reopened = CheckpointsStore(path)
    try:
        rows = reopened._conn.execute(
            "SELECT message_id, role FROM messages WHERE run_id = 'run-1'"
            " ORDER BY role, message_id"
        ).fetchall()
        assert [(row["role"], row["message_id"]) for row in rows] == [
            ("assistant", "m-3"),
            ("user", "m-1"),
        ]
        indexed = reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
            " AND name = 'idx_messages_run_user'"
        ).fetchall()
        assert indexed

        with pytest.raises(sqlite3.IntegrityError):
            _insert(reopened, message_id="m-4", run_id="run-1", role="user", sequence=0)
    finally:
        reopened.close()


def test_assistant_messages_are_not_deduped_by_the_user_index(tmp_path):
    store = CheckpointsStore(str(tmp_path / "ckpt.sqlite"))
    try:
        _seed_conversation(store)
        store._conn.execute("BEGIN")
        _insert(store, message_id="a-1", run_id="run-9", role="assistant", sequence=1)
        _insert(store, message_id="a-2", run_id="run-9", role="assistant", sequence=2)
        store._conn.execute("COMMIT")
    except sqlite3.IntegrityError as exc:  # pragma: no cover - guard
        pytest.fail(f"assistant messages must not share the user uniqueness rule: {exc}")
    finally:
        store.close()

    reopened = CheckpointsStore(str(tmp_path / "ckpt.sqlite"))
    try:
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE run_id = 'run-9'"
        ).fetchone()[0] == 2
    finally:
        reopened.close()
