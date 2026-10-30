"""Local configuration and tested tool grants for user-added MCP servers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from config import settings
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from uniharness.mcp._client import McpClient


class LocalMcpClient(McpClient):
    def __init__(self, name: str, config: dict[str, Any], *, auth: httpx.Auth | None = None) -> None:
        super().__init__(name, config)
        self._auth = auth

    async def _open_transport(self):
        if self._config["type"] == "http":
            assert self._exit_stack is not None
            config = self._config
            http_client = await self._exit_stack.enter_async_context(
                httpx.AsyncClient(headers=dict(config.get("headers", {})), auth=self._auth)
            )
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(config["url"], http_client=http_client)
            )
            return read_stream, write_stream
        if self._config["type"] != "stdio":
            return await super()._open_transport()
        assert self._exit_stack is not None
        config = self._config
        stderr = self._exit_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
        return await self._exit_stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=config["command"], args=list(config.get("args", [])), env=config.get("env")
                ),
                errlog=stderr,
            )
        )


_OWNER_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _owner_dir(owner_id: str) -> Path:
    if not _OWNER_ID_RE.fullmatch(owner_id):
        raise ValueError("invalid owner id")
    return Path(settings.agent_v2_state_dir) / "users" / owner_id


def _path(owner_id: str | None = None) -> Path:
    if owner_id:
        return _owner_dir(owner_id) / "mcp_servers.json"
    return Path(settings.agent_v2_state_dir) / "mcp_servers.json"


def load_servers(owner_id: str | None = None) -> list[dict[str, Any]]:
    path = _path(owner_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["servers"]


def save_servers(servers: list[dict[str, Any]], owner_id: str | None = None) -> None:
    path = _path(owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"version": 1, "servers": servers}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def connection_fingerprint(server: dict[str, Any]) -> str:
    connection = {
        key: server.get(key)
        for key in ("transport", "url", "command", "args", "headers", "env")
    }
    return hashlib.sha256(
        json.dumps(connection, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def scoped_fingerprint(server: dict[str, Any], owner_id: str | None = None) -> str:
    base = connection_fingerprint(server)
    if not owner_id:
        return base
    return hashlib.sha256(f"{owner_id}\0{base}".encode("utf-8")).hexdigest()


def client_config(server: dict[str, Any]) -> dict[str, Any]:
    if server["transport"] == "http":
        return {"type": "http", "url": server["url"], "headers": server["headers"]}
    return {
        "type": "stdio",
        "command": server["command"],
        "args": server["args"],
        "env": server["env"],
    }


def public_server(server: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": server["id"],
        "name": server["name"],
        "builtin": False,
        "transport": server["transport"],
        "url": server["url"],
        "command": server["command"],
        "args": server["args"],
        "header_names": sorted(server["headers"]),
        "env_names": sorted(server["env"]),
        "enabled": server["enabled"],
        "status": server["status"],
        "tools": server["tools"],
    }


def all_scoped_fingerprints() -> set[str]:
    """所有账号 + 旧版配置的 OAuth 指纹，用于安全地清理已删除的服务。"""
    fingerprints = {
        scoped_fingerprint(server, None) for server in load_servers()
    }
    users_root = Path(settings.agent_v2_state_dir) / "users"
    if users_root.is_dir():
        for child in users_root.iterdir():
            if not child.is_dir() or not _OWNER_ID_RE.fullmatch(child.name):
                continue
            try:
                servers = load_servers(child.name)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            fingerprints.update(
                scoped_fingerprint(server, child.name) for server in servers
            )
    return fingerprints


def enabled_servers(
    runtime_settings: Any = settings, owner_id: str | None = None
) -> list[dict[str, Any]]:
    if getattr(runtime_settings, "agent_v2_sandbox_type", "") != "local_native":
        return []
    return [server for server in load_servers(owner_id) if server["enabled"]]
