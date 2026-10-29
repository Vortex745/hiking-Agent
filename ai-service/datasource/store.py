"""SQLite repository for datasource config, sync cursors and sync logs.

This database is deliberately separate from the RAG index. ``synced_doc`` *is*
the cursor: every successful or deterministically skipped document gets a row,
so an interrupted run resumes from per-document state instead of a JSON blob.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from datasource.timeutil import now_iso, parse_iso

logger = logging.getLogger("ai-service.datasource.store")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS datasource (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    credentials_enc TEXT NOT NULL,
    settings_json TEXT NOT NULL DEFAULT '{}',
    selected_resources TEXT NOT NULL DEFAULT '[]',
    schedule TEXT NOT NULL DEFAULT '',
    sync_deletions INTEGER NOT NULL DEFAULT 1,
    sync_status TEXT NOT NULL DEFAULT 'idle',
    running_since TEXT,
    next_run_at TEXT,
    last_sync_at TEXT,
    last_sync_status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synced_doc (
    datasource_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    rev TEXT NOT NULL DEFAULT '',
    rag_source TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL DEFAULT '',
    last_seen_sync_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (datasource_id, external_id),
    FOREIGN KEY (datasource_id) REFERENCES datasource(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_synced_doc_resource
    ON synced_doc(datasource_id, resource_id);

CREATE TABLE IF NOT EXISTS sync_log (
    id TEXT PRIMARY KEY,
    datasource_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    discovered INTEGER NOT NULL DEFAULT 0,
    fetched INTEGER NOT NULL DEFAULT 0,
    created INTEGER NOT NULL DEFAULT 0,
    updated INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    skipped_by_type INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    message TEXT,
    failures_json TEXT NOT NULL DEFAULT '[]',
    enumeration_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (datasource_id) REFERENCES datasource(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sync_log_datasource
    ON sync_log(datasource_id, started_at DESC);
"""

_COUNT_FIELDS = (
    "discovered",
    "fetched",
    "created",
    "updated",
    "skipped",
    "skipped_by_type",
    "truncated",
    "failed",
    "deleted",
)

_UPDATE_FIELDS = {
    "name",
    "credentials_enc",
    "settings",
    "selected_resources",
    "schedule",
    "sync_deletions",
}


class DatasourceStore:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute("SELECT version FROM schema_meta").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.warning("datasource store close failed: %s", exc)

    # ------------------------------------------------------------------ datasource

    def create_datasource(
        self,
        *,
        type: str,
        name: str,
        credentials_enc: str,
        settings: dict[str, Any] | None = None,
        selected_resources: list[str] | None = None,
        schedule: str = "",
        sync_deletions: bool = True,
        next_run_at: str | None = None,
    ) -> dict[str, Any]:
        ds_id = uuid.uuid4().hex
        now = now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO datasource (
                    id, type, name, credentials_enc, settings_json,
                    selected_resources, schedule, sync_deletions, next_run_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ds_id,
                    type,
                    name,
                    credentials_enc,
                    json.dumps(settings or {}, ensure_ascii=False),
                    json.dumps(selected_resources or [], ensure_ascii=False),
                    schedule,
                    1 if sync_deletions else 0,
                    next_run_at,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        created = self.get_datasource(ds_id)
        assert created is not None
        return created

    def get_datasource(self, ds_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM datasource WHERE id = ?", (ds_id,)
            ).fetchone()
        return _datasource_row(row) if row is not None else None

    def list_datasources(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM datasource ORDER BY created_at ASC"
            ).fetchall()
        return [_datasource_row(row) for row in rows]

    def update_datasource(self, ds_id: str, **fields: Any) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in _UPDATE_FIELDS:
                raise ValueError(f"unsupported datasource field: {key}")
            if key in ("settings", "selected_resources"):
                updates[key + "_json" if key == "settings" else key] = json.dumps(value or ({} if key == "settings" else []), ensure_ascii=False)
            elif key == "sync_deletions":
                updates[key] = 1 if value else 0
            else:
                updates[key] = value
        if not updates:
            return self.get_datasource(ds_id)
        updates["updated_at"] = now_iso()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE datasource SET {assignments} WHERE id = ?",
                (*updates.values(), ds_id),
            )
            self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get_datasource(ds_id)

    def delete_datasource(self, ds_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM datasource WHERE id = ?", (ds_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def set_sync_status(self, ds_id: str, status: str, running_since: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE datasource SET sync_status = ?, running_since = ? WHERE id = ?",
                (status, running_since, ds_id),
            )
            self._conn.commit()

    def set_next_run(self, ds_id: str, next_run_at: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE datasource SET next_run_at = ? WHERE id = ?",
                (next_run_at, ds_id),
            )
            self._conn.commit()

    def set_last_sync(self, ds_id: str, at: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE datasource SET last_sync_at = ?, last_sync_status = ? WHERE id = ?",
                (at, status, ds_id),
            )
            self._conn.commit()

    def list_due(self, moment_iso: str | None = None) -> list[dict[str, Any]]:
        moment = parse_iso(moment_iso) if moment_iso else parse_iso(now_iso())
        due: list[dict[str, Any]] = []
        for row in self.list_datasources():
            if not row["schedule"] or not row["next_run_at"]:
                continue
            if row["sync_status"] == "running":
                continue
            try:
                if parse_iso(row["next_run_at"]) <= moment:
                    due.append(row)
            except ValueError:
                logger.warning("invalid next_run_at for datasource %s", row["id"])
        return due

    def recover_interrupted(self) -> int:
        now = now_iso()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE sync_log
                SET status = 'failed', finished_at = ?, error_code = 'interrupted',
                    message = '同步进程中断，未完成的文档将在下一轮重试'
                WHERE status = 'running'
                """,
                (now,),
            )
            self._conn.execute(
                "UPDATE datasource SET sync_status = 'idle', running_since = NULL "
                "WHERE sync_status = 'running'"
            )
            self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ synced docs

    def upsert_synced_doc(
        self,
        ds_id: str,
        *,
        resource_id: str,
        external_id: str,
        rev: str,
        rag_source: str,
        file_name: str,
        last_seen_sync_id: str | None = None,
    ) -> None:
        now = now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO synced_doc (
                    datasource_id, resource_id, external_id, rev, rag_source,
                    file_name, last_seen_sync_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(datasource_id, external_id) DO UPDATE SET
                    resource_id = excluded.resource_id,
                    rev = excluded.rev,
                    rag_source = excluded.rag_source,
                    file_name = excluded.file_name,
                    last_seen_sync_id = excluded.last_seen_sync_id,
                    updated_at = excluded.updated_at
                """,
                (
                    ds_id,
                    resource_id,
                    external_id,
                    rev,
                    rag_source,
                    file_name,
                    last_seen_sync_id,
                    now,
                ),
            )
            self._conn.commit()

    def touch_synced_doc(self, ds_id: str, external_id: str, sync_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE synced_doc SET last_seen_sync_id = ?, updated_at = ? "
                "WHERE datasource_id = ? AND external_id = ?",
                (sync_id, now_iso(), ds_id, external_id),
            )
            self._conn.commit()

    def list_synced_docs(self, ds_id: str, resource_id: str | None = None) -> list[dict[str, Any]]:
        if resource_id is None:
            sql = "SELECT * FROM synced_doc WHERE datasource_id = ? ORDER BY external_id"
            params: tuple[Any, ...] = (ds_id,)
        else:
            sql = (
                "SELECT * FROM synced_doc WHERE datasource_id = ? AND resource_id = ? "
                "ORDER BY external_id"
            )
            params = (ds_id, resource_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def delete_synced_doc(self, ds_id: str, external_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM synced_doc WHERE datasource_id = ? AND external_id = ?",
                (ds_id, external_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def count_synced_docs(self, ds_id: str, *, only_with_source: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM synced_doc WHERE datasource_id = ?"
        if only_with_source:
            sql += " AND rag_source != ''"
        with self._lock:
            row = self._conn.execute(sql, (ds_id,)).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ sync logs

    def start_sync_log(self, ds_id: str, trigger: str) -> str:
        log_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sync_log (id, datasource_id, trigger, status, started_at)
                VALUES (?, ?, ?, 'running', ?)
                """,
                (log_id, ds_id, trigger, now_iso()),
            )
            self._conn.commit()
        return log_id

    def finish_sync_log(
        self,
        log_id: str,
        *,
        status: str,
        counts: dict[str, int],
        failures: list[dict[str, Any]],
        enumeration: dict[str, Any],
        error_code: str | None = None,
        message: str = "",
    ) -> None:
        counters = {field: int(counts.get(field, 0)) for field in _COUNT_FIELDS}
        with self._lock:
            self._conn.execute(
                """
                UPDATE sync_log SET
                    status = ?, finished_at = ?,
                    discovered = ?, fetched = ?, created = ?, updated = ?,
                    skipped = ?, skipped_by_type = ?, truncated = ?, failed = ?,
                    deleted = ?, error_code = ?, message = ?,
                    failures_json = ?, enumeration_json = ?
                WHERE id = ?
                """,
                (
                    status,
                    now_iso(),
                    counters["discovered"],
                    counters["fetched"],
                    counters["created"],
                    counters["updated"],
                    counters["skipped"],
                    counters["skipped_by_type"],
                    counters["truncated"],
                    counters["failed"],
                    counters["deleted"],
                    error_code,
                    message,
                    json.dumps(failures[:100], ensure_ascii=False),
                    json.dumps(enumeration, ensure_ascii=False),
                    log_id,
                ),
            )
            self._conn.commit()

    def get_sync_log(self, log_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sync_log WHERE id = ?", (log_id,)).fetchone()
        return _sync_log_row(row) if row is not None else None

    def list_sync_logs(self, ds_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sync_log WHERE datasource_id = ? ORDER BY started_at DESC LIMIT ?",
                (ds_id, max(1, min(limit, 200))),
            ).fetchall()
        return [_sync_log_row(row) for row in rows]


def _datasource_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["settings"] = json.loads(data.pop("settings_json") or "{}")
    data["selected_resources"] = json.loads(data.pop("selected_resources") or "[]")
    data["sync_deletions"] = bool(data.get("sync_deletions", 1))
    return data


def _sync_log_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["failures"] = json.loads(data.pop("failures_json") or "[]")
    data["enumeration"] = json.loads(data.pop("enumeration_json") or "{}")
    return data
