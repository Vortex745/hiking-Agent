import logging

from memory.base import ChatMemory
from memory.file_memory import FileChatMemory
from memory.postgres_memory import PostgresChatMemory
from memory.redis_memory import RedisChatMemory

logger = logging.getLogger("ai-service.memory")


class FallbackChatMemory(ChatMemory):
    """Use Redis first, then fall back to file memory if Redis is unavailable."""

    def __init__(self, primary: ChatMemory, fallback: ChatMemory):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False

    def _call(self, method: str, *args):
        if not self._use_fallback:
            try:
                return getattr(self.primary, method)(*args)
            except Exception as exc:
                self._use_fallback = True
                logger.warning("Redis chat memory unavailable; falling back to file memory: %s", exc)
        return getattr(self.fallback, method)(*args)

    def add_message(self, role: str, content: str):
        return self._call("add_message", role, content)

    def get_messages(self) -> list[dict]:
        return self._call("get_messages")

    def clear(self):
        return self._call("clear")


def get_chat_memory(chat_id: str) -> ChatMemory:
    from config import settings

    fallback = FileChatMemory(chat_id=chat_id)
    durable_fallback: ChatMemory = fallback
    if settings.redis_configured:
        primary = RedisChatMemory(
            chat_id=chat_id,
            redis_url=settings.redis_url,
            rest_url=settings.redis_rest_url,
            rest_token=settings.redis_rest_token,
        )
        durable_fallback = FallbackChatMemory(primary, fallback)
    if settings.postgres_configured:
        try:
            primary = PostgresChatMemory(chat_id=chat_id)
            return FallbackChatMemory(primary, durable_fallback)
        except Exception as exc:
            logger.warning("Postgres chat memory unavailable; falling back: %s", exc)
    return durable_fallback


def chat_memory_status() -> dict:
    from config import settings

    if settings.postgres_configured:
        return {
            "backend": "postgres",
            "mode": "database_url",
            "source": settings.database_url_source,
        }

    mode = settings.redis_connection_mode
    if mode == "redis_url":
        return {
            "backend": "redis",
            "mode": "redis_url",
            "source": settings.redis_url_source,
        }
    if mode == "upstash_rest":
        return {
            "backend": "redis",
            "mode": "upstash_rest",
            "url_source": settings.redis_rest_url_source,
            "token_source": settings.redis_rest_token_source,
        }
    if mode == "incomplete_upstash_rest":
        return {
            "backend": "file",
            "mode": "file_fallback",
            "detail": "Upstash REST configuration is incomplete",
        }
    return {"backend": "file", "mode": "file_fallback"}
