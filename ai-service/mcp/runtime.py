"""Runtime MCP tool provider."""

from __future__ import annotations

from typing import Any

from config import settings
from mcp.client import MCPClient


def _client_connected(client: MCPClient) -> bool:
    connected = getattr(client, "connected", None)
    if connected is not None:
        return bool(connected)
    return getattr(client, "process", None) is not None or bool(getattr(client, "http_url", None))


class MCPRuntime:
    """Lazy MCP server runtime with persistent clients."""

    def __init__(self, server_configs: dict[str, dict[str, Any]] | None = None):
        self.server_configs = server_configs if server_configs is not None else getattr(settings, "mcp_servers", {})
        self.clients: dict[str, MCPClient] = {}
        self.tools: dict[str, dict[str, dict[str, Any]]] = {}
        self.errors: dict[str, str] = {}
        self.started = False

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        for server_name, config in (self.server_configs or {}).items():
            url = config.get("url") if isinstance(config, dict) else None
            command = config.get("command") if isinstance(config, dict) else None
            if not (url or command):
                self.errors[server_name] = "missing command or url"
                continue
            args = config.get("args", []) or []
            env = config.get("env") or None
            headers = config.get("headers") or None
            client = MCPClient()
            if url:
                await client.connect_http(url, headers=headers)
            else:
                await client.connect_stdio(command, args, env=env)
            if not _client_connected(client):
                self.errors[server_name] = "connection failed"
                continue
            try:
                await client.initialize()
                tools = await client.list_tools()
            except Exception as exc:
                self.errors[server_name] = str(exc)
                await client.close()
                continue
            self.clients[server_name] = client
            self.tools[server_name] = {tool["name"]: tool for tool in tools}

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        await self.start()
        client = self.clients.get(server)
        if client is None:
            return {"isError": True, "message": f"MCP server '{server}' is not loaded"}
        if tool not in self.tools.get(server, {}):
            return {"isError": True, "message": f"MCP tool '{server}:{tool}' is not available"}
        return await client.call_tool(tool, arguments)

    async def close(self) -> None:
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
        self.tools.clear()
        self.started = False

    def health(self) -> dict[str, Any]:
        return {
            "configured": bool(self.server_configs),
            "loaded": bool(self.clients),
            "servers": sorted(self.clients),
            "tools": {
                server: sorted(tools)
                for server, tools in self.tools.items()
            },
            "errors": self.errors,
        }


_default_runtime: MCPRuntime | None = None


def get_mcp_runtime() -> MCPRuntime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = MCPRuntime()
    return _default_runtime


async def reset_mcp_runtime() -> None:
    global _default_runtime
    if _default_runtime is not None:
        await _default_runtime.close()
    _default_runtime = None
