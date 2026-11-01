"""账号表：登录 ID 大小写敏感唯一，密码只落 scrypt 哈希。"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    login_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    password_hash TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS auth_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DuplicateLoginIdError(Exception):
    """登录 ID 已存在（大小写敏感）。"""


class AuthStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def create_user(
        self,
        *,
        user_id: str,
        login_id: str,
        password_hash: str,
        subject_id: str,
    ) -> dict:
        now = _now_iso()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users(user_id, login_id, password_hash, subject_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, login_id, password_hash, subject_id, now),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateLoginIdError(login_id) from exc
        row = self.get_by_id(user_id)
        assert row is not None
        return row

    def get_by_login_id(self, login_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE login_id = ?", (login_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_id(self, user_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def count_users(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"]) if row else 0

    def touch_login(self, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (_now_iso(), user_id),
            )
