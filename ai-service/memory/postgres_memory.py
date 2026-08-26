import json
import logging
from typing import Optional

from config import settings
from memory.base import ChatMemory

logger = logging.getLogger("ai-service.memory")
WINDOW_SIZE = 60


class PostgresChatMemory(ChatMemory):
    """Postgres/Neon-backed chat memory with a sliding message window."""

    def __init__(
        self,
        chat_id: str = "default",
        db_url: str | None = None,
        connect_timeout: int | None = None,
    ):
        self.chat_id = chat_id
        self.db_url = settings.database_url if db_url is None else db_url
        self.connect_timeout = (
            settings.database_connect_timeout_seconds
            if connect_timeout is None
            else connect_timeout
        )
        self._conn: Optional[object] = None
        self._ensure_table()

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg

            self._conn = psycopg.connect(self.db_url, connect_timeout=self.connect_timeout)
            self._conn.autocommit = True
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id_id
            ON chat_messages (chat_id, id)
            """
        )

    def add_message(self, role: str, content: str):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO chat_messages (chat_id, role, content, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (self.chat_id, role, content, json.dumps({})),
        )
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE chat_id = %s
              AND id NOT IN (
                SELECT id FROM chat_messages
                WHERE chat_id = %s
                ORDER BY id DESC
                LIMIT %s
              )
            """,
            (self.chat_id, self.chat_id, WINDOW_SIZE),
        )

    def get_messages(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE chat_id = %s
            ORDER BY id ASC
            """,
            (self.chat_id,),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def clear(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM chat_messages WHERE chat_id = %s", (self.chat_id,))
