"""Local configuration and tested tool grants for user-added MCP servers."""

from __future__ import annotations

import hashlib
import json
import os
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


def _path() -> Path:
    return Path(settings.agent_v2_state_dir) / "mcp_servers.json"


def load_servers() -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["servers"]


def save_servers(servers: list[dict[str, Any]]) -> None:
    path = _path()
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


def enabled_servers(runtime_settings: Any = settings) -> list[dict[str, Any]]:
    if getattr(runtime_settings, "agent_v2_sandbox_type", "") != "local_native":
        return []
    return [server for server in load_servers() if server["enabled"]]
