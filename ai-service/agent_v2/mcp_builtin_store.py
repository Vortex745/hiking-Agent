"""账号级内置 MCP 凭据（如高德 API Key）。

普通 MCP 接口只返回“是否已配置”；浏览器地图专用接口仅返回当前账号
的 JSAPI 配置，不返回 Web 服务 Key。未登录的旧单机模式不参与本模块。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Mapping

from auth.paths import is_owner_id, user_builtin_mcp_path
from config import settings

logger = logging.getLogger("ai-service.agent_v2.mcp_builtin")
_VERSION = 1


def load_env(owner_id: str) -> dict[str, str]:
    if not is_owner_id(owner_id):
        return {}
    path = user_builtin_mcp_path(settings, owner_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    env = data.get("env") if isinstance(data, dict) else None
    if not isinstance(env, dict):
        return {}
    return {
        str(name): value
        for name, value in env.items()
        if isinstance(name, str) and isinstance(value, str) and value.strip()
    }


def save_env(owner_id: str, values: Mapping[str, str]) -> dict[str, str]:
    if not is_owner_id(owner_id):
        raise ValueError("invalid owner id")
    env = load_env(owner_id)
    for name, value in values.items():
        text = str(value).strip()
        if text:
            env[name] = text
        else:
            env.pop(name, None)
    _write(owner_id, env)
    return env


def clear_env(owner_id: str) -> None:
    if not is_owner_id(owner_id):
        raise ValueError("invalid owner id")
    _write(owner_id, {})


def _write(owner_id: str, env: Mapping[str, str]) -> None:
    path = user_builtin_mcp_path(settings, owner_id)
    if not env:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove empty builtin mcp config %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"version": _VERSION, "env": dict(env)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
