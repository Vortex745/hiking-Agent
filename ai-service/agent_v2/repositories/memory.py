"""SQLite repository for anonymous subjects and long-term memory."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_settings (
    subject_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversation_owners (
    conversation_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    consent_type TEXT NOT NULL DEFAULT 'global_opt_in',
    consented_at TEXT,
    source_candidate_id TEXT,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_subject_status
ON memories(subject_id, status);

CREATE TABLE IF NOT EXISTS memory_candidates (
    candidate_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    preview TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decision_request_id TEXT,
    decided_at TEXT,
    memory_id TEXT,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_candidate_subject_status
ON memory_candidates(subject_id, status);
"""


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._ensure_memory_columns()

    def _ensure_memory_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        additions = {
            "consent_type": "TEXT NOT NULL DEFAULT 'global_opt_in'",
            "consented_at": "TEXT",
            "source_candidate_id": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE memories ADD COLUMN {name} {declaration}"
                )

    def close(self) -> None:
        self._conn.close()

    def get_or_create_subject(self, token_hash: str) -> str:
        row = self._conn.execute(
            "SELECT subject_id FROM subjects WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        if row:
            return row["subject_id"]
        subject_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "INSERT INTO subjects(subject_id, token_hash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (subject_id, token_hash, now, now),
            )
            self._conn.execute(
                "INSERT INTO memory_settings(subject_id, enabled, updated_at) "
                "VALUES (?, 1, ?)",
                (subject_id, now),
            )
            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            self._conn.execute("ROLLBACK")
            row = self._conn.execute(
                "SELECT subject_id FROM subjects WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if row:
                return row["subject_id"]
            raise
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return subject_id

    def rotate_token(self, subject_id: str, token_hash: str) -> None:
        self._conn.execute(
            "UPDATE subjects SET token_hash = ?, updated_at = ? WHERE subject_id = ?",
            (token_hash, _now_iso(), subject_id),
        )

    def memory_enabled(self, subject_id: str) -> bool:
        row = self._conn.execute(
            "SELECT enabled FROM memory_settings WHERE subject_id = ?", (subject_id,)
        ).fetchone()
        return bool(row and row["enabled"])

    def set_memory_enabled(self, subject_id: str, enabled: bool) -> None:
        self._conn.execute(
            "UPDATE memory_settings SET enabled = ?, updated_at = ? "
            "WHERE subject_id = ?",
            (int(enabled), _now_iso(), subject_id),
        )

    def claim_conversation(self, subject_id: str, conversation_id: str) -> bool:
        row = self._conn.execute(
            "SELECT subject_id FROM conversation_owners WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row:
            return row["subject_id"] == subject_id
        try:
            self._conn.execute(
                "INSERT INTO conversation_owners(conversation_id, subject_id, created_at) "
                "VALUES (?, ?, ?)",
                (conversation_id, subject_id, _now_iso()),
            )
            return True
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT subject_id FROM conversation_owners WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return bool(row and row["subject_id"] == subject_id)

    def owns_conversation(self, subject_id: str, conversation_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM conversation_owners "
            "WHERE conversation_id = ? AND subject_id = ?",
            (conversation_id, subject_id),
        ).fetchone()
        return row is not None

    def save_preference(self, subject_id: str, content: str) -> str | None:
        if not self.memory_enabled(subject_id):
            return None
        row = self._conn.execute(
            "SELECT memory_id FROM memories WHERE subject_id = ? "
            "AND category = 'preference' AND content = ? AND status = 'active'",
            (subject_id, content),
        ).fetchone()
        if row:
            return row["memory_id"]
        memory_id = str(uuid.uuid4())
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO memories(memory_id, subject_id, category, content, status, "
            "created_at, updated_at, expires_at, consent_type, consented_at) "
            "VALUES (?, ?, 'preference', ?, 'active', ?, ?, NULL, "
            "'global_opt_in', ?)",
            (memory_id, subject_id, content, now, now, now),
        )
        return memory_id

    def _expire_due(self) -> None:
        now = _now_iso()
        self._conn.execute(
            "UPDATE memories SET status = 'expired', content = '', updated_at = ? "
            "WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at <= ?",
            (now, now),
        )
        self._conn.execute(
            "UPDATE memory_candidates SET status = 'expired', content = '', "
            "preview = '', decided_at = ? WHERE status = 'pending' AND expires_at <= ?",
            (now, now),
        )

    @staticmethod
    def _memory_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        consent_type = item.pop("consent_type", "global_opt_in")
        consented_at = item.pop("consented_at", None) or item["created_at"]
        source_candidate_id = item.pop("source_candidate_id", None)
        item["consent"] = {
            "type": consent_type,
            "consented_at": consented_at,
            "candidate_id": source_candidate_id,
        }
        return item

    def list_active_memories(
        self, subject_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        self._expire_due()
        rows = self._conn.execute(
            "SELECT memory_id, category, content, created_at, updated_at, expires_at, "
            "consent_type, consented_at, source_candidate_id "
            "FROM memories WHERE subject_id = ? AND status = 'active' "
            "ORDER BY created_at, memory_id LIMIT ? OFFSET ?",
            (subject_id, limit, offset),
        ).fetchall()
        return [self._memory_dict(row) for row in rows]

    def count_active_memories(self, subject_id: str) -> int:
        self._expire_due()
        row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories "
            "WHERE subject_id = ? AND status = 'active'",
            (subject_id,),
        ).fetchone()
        return int(row["count"])

    def create_candidate(
        self,
        subject_id: str,
        *,
        category: str,
        content: str,
        preview: str,
        sensitivity: str,
    ) -> dict | None:
        if not self.memory_enabled(subject_id):
            return None
        candidate_id = str(uuid.uuid4())
        created_at = _now_iso()
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        self._conn.execute(
            "INSERT INTO memory_candidates(candidate_id, subject_id, category, "
            "content, preview, sensitivity, status, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                candidate_id, subject_id, category, content, preview,
                sensitivity, created_at, expires_at,
            ),
        )
        return {
            "candidate_id": candidate_id,
            "category": category,
            "preview": preview,
            "sensitivity": sensitivity,
            "status": "pending",
            "created_at": created_at,
            "expires_at": expires_at,
        }

    def get_candidate(self, subject_id: str, candidate_id: str) -> dict | None:
        self._expire_due()
        row = self._conn.execute(
            "SELECT * FROM memory_candidates WHERE candidate_id = ? AND subject_id = ?",
            (candidate_id, subject_id),
        ).fetchone()
        return dict(row) if row else None

    def decide_candidate(
        self,
        subject_id: str,
        candidate_id: str,
        decision: str,
        decision_request_id: str,
    ) -> tuple[dict, bool] | None:
        self._expire_due()
        row = self._conn.execute(
            "SELECT * FROM memory_candidates WHERE candidate_id = ? AND subject_id = ?",
            (candidate_id, subject_id),
        ).fetchone()
        if row is None:
            return None
        if row["status"] != "pending":
            if row["decision_request_id"] == decision_request_id:
                expected_status = "approved" if decision == "approve" else "rejected"
                if row["status"] != expected_status:
                    raise ValueError("decision_request_conflict")
                return self._candidate_result(row), True
            raise ValueError(f"candidate_{row['status']}")

        now = _now_iso()
        memory_id = None
        self._conn.execute("BEGIN")
        try:
            if decision == "approve":
                memory_id = str(uuid.uuid4())
                expires_at = None
                if row["category"] == "trip":
                    expires_at = (
                        datetime.now(timezone.utc) + timedelta(days=90)
                    ).isoformat()
                self._conn.execute(
                    "INSERT INTO memories(memory_id, subject_id, category, content, "
                    "status, created_at, updated_at, expires_at, consent_type, "
                    "consented_at, source_candidate_id) VALUES (?, ?, ?, ?, 'active', "
                    "?, ?, ?, 'candidate_approval', ?, ?)",
                    (
                        memory_id, subject_id, row["category"], row["content"],
                        now, now, expires_at, now, candidate_id,
                    ),
                )
            status = "approved" if decision == "approve" else "rejected"
            self._conn.execute(
                "UPDATE memory_candidates SET status = ?, content = '', preview = '', "
                "decision_request_id = ?, decided_at = ?, memory_id = ? "
                "WHERE candidate_id = ?",
                (status, decision_request_id, now, memory_id, candidate_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        updated = self._conn.execute(
            "SELECT * FROM memory_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return self._candidate_result(updated), False

    def _candidate_result(self, row: sqlite3.Row) -> dict:
        result = {
            "candidate_id": row["candidate_id"],
            "status": row["status"],
            "decided_at": row["decided_at"],
        }
        if row["memory_id"]:
            memory = self._conn.execute(
                "SELECT memory_id, category, content, created_at, updated_at, "
                "expires_at, consent_type, consented_at, source_candidate_id "
                "FROM memories WHERE memory_id = ?", (row["memory_id"],)
            ).fetchone()
            result["memory"] = self._memory_dict(memory) if memory else None
        return result

    def update_memory(
        self,
        subject_id: str,
        memory_id: str,
        *,
        content: str,
        category: str,
        ttl_days: int | None,
    ) -> dict | None:
        self._expire_due()
        expires_at = (
            (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
            if ttl_days is not None
            else None
        )
        self._conn.execute(
            "UPDATE memories SET content = ?, category = ?, expires_at = ?, "
            "updated_at = ? WHERE memory_id = ? AND subject_id = ? AND status = 'active'",
            (content, category, expires_at, _now_iso(), memory_id, subject_id),
        )
        row = self._conn.execute(
            "SELECT memory_id, category, content, created_at, updated_at, expires_at, "
            "consent_type, consented_at, source_candidate_id FROM memories "
            "WHERE memory_id = ? AND subject_id = ? AND status = 'active'",
            (memory_id, subject_id),
        ).fetchone()
        return self._memory_dict(row) if row else None

    def delete_memory(self, subject_id: str, memory_id: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE memories SET status = 'deleted', content = '', updated_at = ? "
            "WHERE memory_id = ? AND subject_id = ? AND status = 'active'",
            (_now_iso(), memory_id, subject_id),
        )
        return cursor.rowcount == 1

    def delete_all_memories(self, subject_id: str) -> int:
        cursor = self._conn.execute(
            "UPDATE memories SET status = 'deleted', content = '', updated_at = ? "
            "WHERE subject_id = ? AND status = 'active'",
            (_now_iso(), subject_id),
        )
        return cursor.rowcount
