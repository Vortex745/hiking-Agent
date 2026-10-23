"""BCDEF upgrade-path guard for the ticket 02 + S2 storage constraints.

The one-user-message-per-run UNIQUE index is created by a migration, not by
the schema script, because databases written before ticket 02 legitimately
contain duplicate user messages. If the index lived in the schema script,
``CheckpointsStore.__init__`` would raise on such a file and NO store (and
therefore no admission) could open at all.

S2 adds the sibling case, same shape: one durable ``done``/``error``/
``cancelled`` terminal per run, enforced by the partial unique index
``idx_run_events_one_terminal``. Legacy databases hold contradicting
terminals, so the upgrade must archive the surplus (earliest sequence
wins), and only then install the constraint.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

import agent_v2.persistence as persistence
from agent_v2.persistence import CheckpointsStore


TERMINAL_INDEX = "idx_run_events_one_terminal"

# SQL fragments only the S2 upgrade emits. Watching them on the real
# connection (SQLite's own statement trace) shows whether a boot did the
# historical work or skipped it — without reaching into private helpers and
# without ordering calls. ``PARTITION BY run_id`` is the whole-history
# terminal scan; no per-run read uses it.
_DEDUPE_SCAN_FRAGMENT = "PARTITION BY run_id"
_ARCHIVE_INSERT_FRAGMENT = "INSERT INTO run_event_terminal_archive"
_INDEX_CREATE_FRAGMENT = f"CREATE UNIQUE INDEX IF NOT EXISTS {TERMINAL_INDEX}"


def _statements_containing(statements: list[str], fragment: str) -> list[str]:
    return [sql for sql in statements if fragment in sql]


def _trace_connect(monkeypatch, sink: list[str]) -> None:
    """Let the real store connect, then record every statement it runs."""
    real_connect = persistence._connect

    def connect(path: str) -> sqlite3.Connection:
        conn = real_connect(path)
        conn.set_trace_callback(sink.append)
        return conn

    monkeypatch.setattr(persistence, "_connect", connect)


def _inject_statement_fault(monkeypatch, fragment: str) -> None:
    """Make one SQL statement fail on a REAL connection, once per open.

    The store, its transaction and every other statement stay production
    code; only the injected statement raises, which is how a broken archive
    write or a refused ``CREATE INDEX`` reaches the upgrade in the field.
    """

    class _FaultyConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # noqa: ANN001 - sqlite3 API
            if isinstance(sql, str) and fragment in sql:
                raise RuntimeError(f"injected write failure: {fragment!r}")
            return super().execute(sql, *args, **kwargs)

    def connect(path: str) -> sqlite3.Connection:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None,
                               check_same_thread=False, factory=_FaultyConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    monkeypatch.setattr(persistence, "_connect", connect)


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


# ---- S2: one durable terminal per run --------------------------------------


def _insert_terminal(
    store: CheckpointsStore,
    *,
    event_id: str,
    run_id: str,
    sequence: int,
    event_type: str,
    payload: dict | None = None,
    conversation_id: str = "conv-upgrade",
) -> None:
    """Raw event row — the point of a legacy upgrade is exactly that the
    application layer no longer produced these rows politely."""
    store._conn.execute(
        "INSERT INTO run_events (event_id, run_id, sequence, event_type,"
        " payload_json, request_id, conversation_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            run_id,
            sequence,
            event_type,
            json.dumps(payload or {"code": event_type}, ensure_ascii=False),
            f"req-{run_id}",
            conversation_id,
            "2026-01-01T00:00:00Z",
        ),
    )


def _make_legacy_terminal_db(tmp_path, events, *, run_id: str = "run-1"):
    """Write a pre-S2 database holding ``events`` (duplicate terminals allowed).

    Returns its path. The constraint index is dropped before the duplicate
    rows are written, because a store opened today could never hold them.
    """
    path = str(tmp_path / "ckpt.sqlite")
    store = CheckpointsStore(path)
    try:
        _seed_conversation(store)
        store._conn.execute("BEGIN")
        store._conn.execute(f"DROP INDEX IF EXISTS {TERMINAL_INDEX}")
        for event in events:
            _insert_terminal(store, run_id=run_id, **event)
        store._conn.execute("COMMIT")
    finally:
        store.close()
    return path


def _active_terminals(store: CheckpointsStore, run_id: str) -> list[sqlite3.Row]:
    return store._conn.execute(
        "SELECT * FROM run_events WHERE run_id = ? AND event_type IN"
        " ('cancelled', 'done', 'error') ORDER BY sequence ASC",
        (run_id,),
    ).fetchall()


def _archived_terminals(store: CheckpointsStore, run_id: str) -> list[sqlite3.Row]:
    return store._conn.execute(
        "SELECT * FROM run_event_terminal_archive WHERE run_id = ?"
        " ORDER BY sequence ASC",
        (run_id,),
    ).fetchall()


def _index_names(store: CheckpointsStore) -> set[str]:
    return {
        row["name"]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }


def test_upgrade_archives_duplicate_terminals_and_keeps_the_earliest(tmp_path):
    path = _make_legacy_terminal_db(
        tmp_path,
        [
            {"event_id": "t-late", "sequence": 9, "event_type": "done"},
            {"event_id": "t-early", "sequence": 4, "event_type": "error",
             "payload": {"code": "model_timeout", "message": "超时"}},
            {"event_id": "t-middle", "sequence": 7, "event_type": "error",
             "payload": {"code": "duplicate"}},
        ],
    )

    upgraded = CheckpointsStore(path)
    try:
        active = _active_terminals(upgraded, "run-1")
        # The earliest sequence wins; a later ``done`` never overrides the
        # earlier ``error``, and nothing is renumbered or rewritten.
        assert [(row["event_id"], row["sequence"], row["event_type"]) for row in active] == [
            ("t-early", 4, "error")
        ]
        assert json.loads(active[0]["payload_json"]) == {
            "code": "model_timeout", "message": "超时",
        }

        archived = _archived_terminals(upgraded, "run-1")
        assert [
            (row["event_id"], row["sequence"], row["event_type"]) for row in archived
        ] == [("t-middle", 7, "error"), ("t-late", 9, "done")]
        for row, original in zip(archived, (
            {"sequence": 7, "event_type": "error", "payload_json": json.dumps({"code": "duplicate"})},
            {"sequence": 9, "event_type": "done", "payload_json": json.dumps({"code": "done"})},
        )):
            # Every original field is preserved verbatim.
            assert row["run_id"] == "run-1"
            assert row["request_id"] == "req-run-1"
            assert row["conversation_id"] == "conv-upgrade"
            assert row["created_at"] == "2026-01-01T00:00:00Z"
            assert row["payload_json"] == original["payload_json"]
            assert row["canonical_event_id"] == "t-early"
            assert isinstance(row["archived_at"], str) and row["archived_at"]
            assert isinstance(row["archive_reason"], str) and row["archive_reason"]

        assert TERMINAL_INDEX in _index_names(upgraded)
        # The surviving canonical is what the read seam reports.
        assert upgraded.get_terminal_event("run-1").event_id == "t-early"
    finally:
        upgraded.close()


@pytest.mark.parametrize(
    "events",
    [
        pytest.param([], id="empty-database"),
        pytest.param(
            [{"event_id": "t-1", "sequence": 5, "event_type": "done"}],
            id="healthy-single-terminal",
        ),
        pytest.param(
            [
                {"event_id": "t-1", "sequence": 5, "event_type": "done"},
                {"event_id": "t-2", "sequence": 8, "event_type": "done"},
            ],
            id="same-kind-duplicates",
        ),
        pytest.param(
            [
                {"event_id": "t-1", "sequence": 2, "event_type": "error"},
                {"event_id": "t-2", "sequence": 4, "event_type": "cancelled"},
                {"event_id": "t-3", "sequence": 6, "event_type": "done"},
            ],
            id="mixed-done-error-cancelled",
        ),
        pytest.param(
            [
                {"event_id": "t-1", "sequence": 5, "event_type": "done"},
                {"event_id": "t-2", "sequence": 8, "event_type": "done"},
                {"event_id": "t-3", "sequence": 9, "event_type": "error"},
            ],
            id="three-terminals-one-run",
        ),
    ],
)
def test_store_initialises_for_every_legacy_terminal_shape(tmp_path, events):
    path = _make_legacy_terminal_db(tmp_path, events)

    store = CheckpointsStore(path)
    try:
        assert TERMINAL_INDEX in _index_names(store)
        active = _active_terminals(store, "run-1")
        assert len(active) == (1 if events else 0)
        if events:
            assert active[0]["event_id"] == "t-1"
        else:
            assert active == []
    finally:
        store.close()


def test_non_terminal_events_are_not_limited_by_the_terminal_index(tmp_path):
    store = CheckpointsStore(str(tmp_path / "ckpt.sqlite"))
    try:
        _seed_conversation(store)
        store._conn.execute("BEGIN")
        for sequence, event_type in (
            (1, "run.paused"),
            (2, "run.paused"),
            (3, "tool_result"),
            (4, "tool_result"),
            (5, "progress.gap"),
            (6, "progress.gap"),
        ):
            _insert_terminal(
                store, event_id=f"e-{sequence}", run_id="run-paused",
                sequence=sequence, event_type=event_type,
            )
        store._conn.execute("COMMIT")

        assert store.get_terminal_event("run-paused") is None
        # The first terminal of the run is accepted ...
        _insert_terminal(
            store, event_id="e-terminal", run_id="run-paused",
            sequence=7, event_type="done",
        )
        # ... and the second is rejected by the database itself (autocommit:
        # the rejected insert leaves no transaction behind).
        with pytest.raises(sqlite3.IntegrityError):
            _insert_terminal(
                store, event_id="e-terminal-dup", run_id="run-paused",
                sequence=8, event_type="error",
            )
        assert [row["event_id"] for row in _active_terminals(store, "run-paused")] == [
            "e-terminal"
        ]
        assert store._conn.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id = 'run-paused'"
            " AND event_type = 'run.paused'"
        ).fetchone()[0] == 2
    finally:
        store.close()

    assert CheckpointsStore(str(tmp_path / "ckpt.sqlite")).get_terminal_event(
        "run-paused"
    ) is not None


def test_bypassing_the_repository_a_second_terminal_is_rejected_by_sqlite(tmp_path):
    store = CheckpointsStore(str(tmp_path / "ckpt.sqlite"))
    try:
        _seed_conversation(store)
        _insert_terminal(store, event_id="t-1", run_id="run-x", sequence=3, event_type="done")

        # A writer that never went through RunRepository.finalize_run.
        with pytest.raises(sqlite3.IntegrityError):
            _insert_terminal(store, event_id="t-2", run_id="run-x", sequence=9, event_type="error")

        # The other run is untouched, so the constraint is per run.
        _insert_terminal(store, event_id="t-3", run_id="run-y", sequence=9, event_type="error")
        assert [row["event_id"] for row in _active_terminals(store, "run-x")] == ["t-1"]
    finally:
        store.close()


def test_duplicate_user_messages_and_duplicate_terminals_upgrade_together(tmp_path):
    path = str(tmp_path / "ckpt.sqlite")
    store = CheckpointsStore(path)
    try:
        _seed_conversation(store)
        store._conn.execute("BEGIN")
        # Pre-ticket-02 AND pre-S2: both constraints missing, both violated.
        store._conn.execute("DROP INDEX idx_messages_run_user")
        store._conn.execute(f"DROP INDEX IF EXISTS {TERMINAL_INDEX}")
        _insert(store, message_id="m-1", run_id="run-both", role="user", sequence=0)
        _insert(store, message_id="m-2", run_id="run-both", role="user", sequence=0)
        _insert(store, message_id="m-3", run_id="run-both", role="assistant", sequence=1)
        _insert_terminal(store, event_id="t-1", run_id="run-both", sequence=2, event_type="error")
        _insert_terminal(store, event_id="t-2", run_id="run-both", sequence=5, event_type="done")
        store._conn.execute("COMMIT")
    finally:
        store.close()

    upgraded = CheckpointsStore(path)
    try:
        rows = upgraded._conn.execute(
            "SELECT message_id, role FROM messages WHERE run_id = 'run-both'"
            " ORDER BY role, message_id"
        ).fetchall()
        # B2 contract unchanged: earliest user row survives, assistant kept.
        assert [(row["role"], row["message_id"]) for row in rows] == [
            ("assistant", "m-3"),
            ("user", "m-1"),
        ]
        assert [row["event_id"] for row in _active_terminals(upgraded, "run-both")] == ["t-1"]
        assert [row["event_id"] for row in _archived_terminals(upgraded, "run-both")] == ["t-2"]
        assert TERMINAL_INDEX in _index_names(upgraded)
        assert "idx_messages_run_user" in _index_names(upgraded)

        with pytest.raises(sqlite3.IntegrityError):
            _insert(upgraded, message_id="m-4", run_id="run-both", role="user", sequence=0)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_terminal(upgraded, event_id="t-3", run_id="run-both", sequence=7,
                             event_type="cancelled")
    finally:
        upgraded.close()


# ---- repeated upgrade: one historical pass, never two ----------------------


def _legacy_db_with_conflicts(tmp_path, runs: int) -> str:
    """A pre-S2 database: ``runs`` runs, each with three conflicting terminals."""
    path = str(tmp_path / "scale.sqlite")
    store = CheckpointsStore(path)
    try:
        store._conn.execute("BEGIN")
        store._conn.execute(f"DROP INDEX IF EXISTS {TERMINAL_INDEX}")
        for index in range(runs):
            run_id = f"run-{index}"
            for sequence, event_type, event_id in (
                (2, "error", f"{run_id}-t-error"),
                (5, "done", f"{run_id}-t-done"),
                (8, "cancelled", f"{run_id}-t-cancelled"),
            ):
                _insert_terminal(
                    store, event_id=event_id, run_id=run_id, sequence=sequence,
                    event_type=event_type,
                    payload={"code": event_type, "run": run_id},
                )
        store._conn.execute("COMMIT")
    finally:
        store.close()
    return path


def test_second_upgrade_keeps_archive_and_canonical_untouched(tmp_path):
    path = _make_legacy_terminal_db(
        tmp_path,
        [
            {"event_id": "t-1", "sequence": 2, "event_type": "error"},
            {"event_id": "t-2", "sequence": 4, "event_type": "done"},
            {"event_id": "t-3", "sequence": 6, "event_type": "cancelled"},
        ],
    )

    first = CheckpointsStore(path)
    try:
        archive_after_first = first._conn.execute(
            "SELECT event_id, canonical_event_id, archived_at, archive_reason"
            " FROM run_event_terminal_archive ORDER BY event_id"
        ).fetchall()
        canonical_after_first = first._conn.execute(
            "SELECT * FROM run_events WHERE run_id = 'run-1'"
        ).fetchall()
    finally:
        first.close()

    second = CheckpointsStore(path)
    try:
        archive_after_second = second._conn.execute(
            "SELECT event_id, canonical_event_id, archived_at, archive_reason"
            " FROM run_event_terminal_archive ORDER BY event_id"
        ).fetchall()
        canonical_after_second = second._conn.execute(
            "SELECT * FROM run_events WHERE run_id = 'run-1'"
        ).fetchall()

        # Durable facts: identical archive rows (same count, same ids, same
        # archived_at), identical canonical event and sequence.
        assert [tuple(row) for row in archive_after_second] == [
            tuple(row) for row in archive_after_first
        ]
        assert [tuple(row) for row in canonical_after_second] == [
            tuple(row) for row in canonical_after_first
        ]

        third = second._conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert third == "ok"
    finally:
        second.close()


def test_reopening_skips_the_historical_dedupe_entirely(tmp_path, monkeypatch):
    path = _make_legacy_terminal_db(
        tmp_path,
        [
            {"event_id": "t-1", "sequence": 2, "event_type": "error"},
            {"event_id": "t-2", "sequence": 4, "event_type": "done"},
        ],
    )

    first_boot: list[str] = []
    _trace_connect(monkeypatch, first_boot)
    store = CheckpointsStore(path)
    store.close()

    assert _statements_containing(first_boot, _ARCHIVE_INSERT_FRAGMENT), (
        "the first upgrade must archive the conflicts"
    )
    assert _statements_containing(first_boot, _INDEX_CREATE_FRAGMENT)

    monkeypatch.undo()
    second_boot: list[str] = []
    _trace_connect(monkeypatch, second_boot)
    store = CheckpointsStore(path)
    store.close()

    assert not _statements_containing(second_boot, _ARCHIVE_INSERT_FRAGMENT)
    assert not _statements_containing(second_boot, _INDEX_CREATE_FRAGMENT)
    assert not _statements_containing(second_boot, _DEDUPE_SCAN_FRAGMENT), (
        f"repeated start re-scanned the history: {second_boot}"
    )


def test_the_upgrade_never_runs_on_the_write_path(tmp_path, monkeypatch):
    """A constraint already installed means no migration work per request."""
    path = _make_legacy_terminal_db(
        tmp_path,
        [
            {"event_id": "t-old", "sequence": 2, "event_type": "error"},
            {"event_id": "t-old-2", "sequence": 4, "event_type": "done"},
        ],
    )
    boot: list[str] = []
    _trace_connect(monkeypatch, boot)
    store = CheckpointsStore(path)
    boot.clear()

    try:
        store.ensure_conversation(conversation_id="conv-live", profile_id="chat")
        store.ensure_user_message(
            conversation_id="conv-live", run_id="run-live", content="问", sequence=0
        )
        store.append_event(
            run_id="run-live", sequence=1, event_type="run_started",
            payload={"ok": True}, request_id="req-live", conversation_id="conv-live",
        )
        store.finalize_terminal(
            run_id="run-live", conversation_id="conv-live", request_id="req-live",
            terminal_type="done", payload={"code": "done"}, sequence=2,
            assistant_content="答",
        )
        store.finalize_terminal(
            run_id="run-live", conversation_id="conv-live", request_id="req-live",
            terminal_type="error", payload={"code": "late"}, sequence=3,
        )
        store.get_terminal_event("run-live")
        store.get_events_since(run_id="run-live", from_seq=0)
    finally:
        store.close()

    assert not _statements_containing(boot, _ARCHIVE_INSERT_FRAGMENT)
    assert not _statements_containing(boot, _INDEX_CREATE_FRAGMENT)
    # The dedupe scan reads EVERY run's terminals at once; a per-run event
    # query always filters on run_id, so this fragment proves a scan.
    assert not [
        sql for sql in boot
        if "PARTITION BY run_id" in sql
    ], f"an ordinary request ran the historical scan: {boot}"


def test_repeated_startup_archives_once_at_scale(tmp_path, monkeypatch):
    """Timing observation: the historical pass is paid once, ever."""
    path = _legacy_db_with_conflicts(tmp_path, runs=200)

    boot_one: list[str] = []
    _trace_connect(monkeypatch, boot_one)
    started = time.perf_counter()
    CheckpointsStore(path).close()
    first_upgrade_ms = (time.perf_counter() - started) * 1000.0

    monkeypatch.undo()
    boot_two: list[str] = []
    _trace_connect(monkeypatch, boot_two)
    started = time.perf_counter()
    store = CheckpointsStore(path)
    reopen_ms = (time.perf_counter() - started) * 1000.0

    try:
        archive_rows = store._conn.execute(
            "SELECT COUNT(*) FROM run_event_terminal_archive"
        ).fetchone()[0]
        active_terminals = store._conn.execute(
            "SELECT COUNT(*) FROM run_events WHERE event_type IN"
            " ('cancelled', 'done', 'error')"
        ).fetchone()[0]
        runs = store._conn.execute(
            "SELECT COUNT(DISTINCT run_id) FROM run_event_terminal_archive"
        ).fetchone()[0]
    finally:
        store.close()

    assert (archive_rows, active_terminals, runs) == (400, 200, 200)
    assert _statements_containing(boot_one, _ARCHIVE_INSERT_FRAGMENT)
    assert not _statements_containing(boot_two, _ARCHIVE_INSERT_FRAGMENT)
    assert not _statements_containing(boot_two, _DEDUPE_SCAN_FRAGMENT)
    print(
        f"[S2 upgrade timing] first_upgrade={first_upgrade_ms:.1f}ms "
        f"reopened={reopen_ms:.1f}ms archived_rows={archive_rows} "
        f"surplus_per_run=2 runs=200",
        flush=True,
    )


# ---- a failing upgrade leaves the database exactly as it was ----------------


def _assert_legacy_conflicts_still_there(path: str) -> None:
    """Probe the file with a constraint-free connection: no rollback trickery."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        terminals = conn.execute(
            "SELECT event_id FROM run_events WHERE event_type IN"
            " ('cancelled', 'done', 'error') ORDER BY sequence"
        ).fetchall()
        assert [row["event_id"] for row in terminals] == ["t-1", "t-2"]
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "run_event_terminal_archive" in tables:
            assert conn.execute(
                "SELECT COUNT(*) FROM run_event_terminal_archive"
            ).fetchone()[0] == 0
        indexes = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert TERMINAL_INDEX not in indexes
    finally:
        conn.close()


@pytest.mark.parametrize(
    "fragment",
    [
        pytest.param(_ARCHIVE_INSERT_FRAGMENT, id="archive-write-fails"),
        pytest.param(_INDEX_CREATE_FRAGMENT, id="index-creation-fails"),
    ],
)
def test_failed_upgrade_rolls_back_completely_and_is_retriable(
    tmp_path, monkeypatch, fragment
):
    path = _make_legacy_terminal_db(
        tmp_path,
        [
            {"event_id": "t-1", "sequence": 2, "event_type": "error"},
            {"event_id": "t-2", "sequence": 4, "event_type": "done"},
        ],
    )

    _inject_statement_fault(monkeypatch, fragment)
    with pytest.raises(RuntimeError, match="injected write failure"):
        CheckpointsStore(path)
    _assert_legacy_conflicts_still_there(path)

    # Same file, same data, no fault: the retry upgrades cleanly.
    monkeypatch.undo()
    store = CheckpointsStore(path)
    try:
        assert [row["event_id"] for row in _active_terminals(store, "run-1")] == ["t-1"]
        assert [row["event_id"] for row in _archived_terminals(store, "run-1")] == ["t-2"]
        assert TERMINAL_INDEX in _index_names(store)
    finally:
        store.close()
