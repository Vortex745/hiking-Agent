"""账号私有数据目录。"""

from __future__ import annotations

import re
from pathlib import Path

_OWNER_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_owner_id(value: str) -> bool:
    return bool(_OWNER_ID_RE.fullmatch(value or ""))


def _owner(owner_id: str) -> str:
    if not _OWNER_ID_RE.fullmatch(owner_id or ""):
        raise ValueError("invalid owner id")
    return owner_id


def user_state_dir(settings, owner_id: str) -> Path:
    return Path(settings.agent_v2_state_dir) / "users" / _owner(owner_id)


def user_data_dir(settings, owner_id: str) -> Path:
    return Path(settings.rag_sqlite_db_path).parent / "users" / _owner(owner_id)


def user_rag_path(settings, owner_id: str) -> Path:
    return user_data_dir(settings, owner_id) / "rag.sqlite"


def user_config_path(settings, owner_id: str) -> Path:
    return user_data_dir(settings, owner_id) / "llm_config.json"


def user_mcp_path(settings, owner_id: str) -> Path:
    return user_state_dir(settings, owner_id) / "mcp_servers.json"
