import json
import logging
from typing import Optional

import httpx

from config import settings
from memory.base import ChatMemory

logger = logging.getLogger("ai-service.memory")
WINDOW_SIZE = 60

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis-py not installed, RedisChatMemory will raise ImportError on use")


class RedisCommandStore:
    """Small Redis command wrapper supporting protocol URLs and Upstash REST."""

    def __init__(
        self,
        redis_url: str | None = None,
        rest_url: str | None = None,
        rest_token: str | None = None,
    ):
        self.redis_url = settings.redis_url if redis_url is None else redis_url
        self.rest_url = settings.redis_rest_url if rest_url is None else rest_url
        self.rest_token = settings.redis_rest_token if rest_token is None else rest_token
        self._client: Optional["redis.Redis"] = None

    @property
    def mode(self) -> str:
        if self.redis_url:
            return "redis_url"
        if self.rest_url and self.rest_token:
            return "upstash_rest"
        return "unconfigured"

    def _ensure_client(self):
        if self._client is None:
            if not REDIS_AVAILABLE:
                raise ImportError("redis is required for Redis persistence. Install with: pip install redis")
            if not self.redis_url:
                raise RuntimeError("A Redis protocol URL is required")
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def command(self, *command):
        if self.mode == "redis_url":
            client = self._ensure_client()
            method_name = str(command[0]).lower()
            method = getattr(client, method_name)
            return method(*command[1:])
        if self.mode == "upstash_rest":
            headers = {"Authorization": f"Bearer {self.rest_token}"}
            with httpx.Client(timeout=5.0) as client:
                response = client.post(self.rest_url, json=list(command), headers=headers)
                response.raise_for_status()
                data = response.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"Upstash Redis command failed: {data['error']}")
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            return data
        raise RuntimeError("Redis persistence requires REDIS_URL/KV_URL or Upstash REST environment variables")

    def get_json(self, key: str):
        raw = self.command("GET", key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def set_json(self, key: str, value) -> None:
        self.command("SET", key, json.dumps(value, ensure_ascii=False))


class RedisChatMemory(ChatMemory):
    """Redis-based chat memory with protocol URL or Upstash REST support."""

    def __init__(
        self,
        chat_id: str = "default",
        redis_url: str | None = None,
        rest_url: str | None = None,
        rest_token: str | None = None,
    ):
        self.chat_id = chat_id
        self.redis_key = f"chat:{chat_id}:messages"
        self.redis_url = settings.redis_url if redis_url is None else redis_url
        self.rest_url = settings.redis_rest_url if rest_url is None else rest_url
        self.rest_token = settings.redis_rest_token if rest_token is None else rest_token
        self._store = RedisCommandStore(
            redis_url=self.redis_url,
            rest_url=self.rest_url,
            rest_token=self.rest_token,
        )

    @property
    def mode(self) -> str:
        return self._store.mode

    def add_message(self, role: str, content: str):
        """Add a message and maintain window size."""
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self._store.command("RPUSH", self.redis_key, msg)
        self._store.command("LTRIM", self.redis_key, -WINDOW_SIZE, -1)

    def get_messages(self) -> list[dict]:
        """Get all stored messages."""
        raw_messages = self._store.command("LRANGE", self.redis_key, 0, -1)

        messages = []
        for raw in raw_messages or []:
            try:
                messages.append(json.loads(raw))
            except (TypeError, json.JSONDecodeError):
                continue
        return messages

    def clear(self):
        """Clear all messages synchronously."""
        self._store.command("DEL", self.redis_key)
