"""运行期账号作用域：检索与工具按当前登录账号取数。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

current_owner_id: ContextVar[str | None] = ContextVar(
    "current_owner_id", default=None
)
current_user_settings: ContextVar[Any | None] = ContextVar(
    "current_user_settings", default=None
)


@contextmanager
def owner_scope(owner_id: str, effective_settings: Any | None = None) -> Iterator[None]:
    owner_token = current_owner_id.set(owner_id)
    settings_token = current_user_settings.set(effective_settings)
    try:
        yield
    finally:
        current_user_settings.reset(settings_token)
        current_owner_id.reset(owner_token)
