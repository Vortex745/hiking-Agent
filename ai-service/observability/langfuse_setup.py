"""Langfuse LLM observability wiring for LangChain ChatOpenAI call sites.

Gate: callbacks stay empty unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
are both configured. Observability must never break the LLM call path.
"""

import logging
from contextvars import ContextVar

from config import settings

logger = logging.getLogger("ai-service.observability.langfuse")

_callbacks: list | None = None

# 当前聊天会话 id：用于在 Langfuse 中把同一会话的多次 LLM 调用聚合成一个 Session。
_session_id: ContextVar[str | None] = ContextVar("langfuse_session_id", default=None)


def set_langfuse_session(session_id: str | None):
    """在请求边界设置会话 id；返回 token 供 reset_langfuse_session 恢复。"""
    return _session_id.set(session_id)


def reset_langfuse_session(token) -> None:
    _session_id.reset(token)


def langchain_run_config() -> dict:
    """带 trace 属性 metadata 的 Runnable config；无会话时返回空 config。"""
    session_id = _session_id.get()
    if not session_id:
        return {}
    return {"metadata": {"langfuse_session_id": session_id}}


def get_langchain_callbacks() -> list:
    """Return LangChain callbacks for ChatOpenAI; empty list when disabled."""
    global _callbacks
    if _callbacks is not None:
        return _callbacks
    if not (
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    ):
        _callbacks = []
        return _callbacks
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        _callbacks = [CallbackHandler()]
        logger.info(
            "Langfuse tracing enabled (base_url=%s)", settings.langfuse_base_url
        )
    except Exception as e:
        logger.warning("Langfuse init failed, tracing disabled: %s", e)
        _callbacks = []
    return _callbacks
