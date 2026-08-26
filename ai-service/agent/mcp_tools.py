"""OpenManus-style MCP tools owned by the Agent runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from agent.tool_collection import ToolCollection
from mcp.client import MCPClient


def _client_connected(client: MCPClient) -> bool:
    connected = getattr(client, "connected", None)
    if connected is not None:
        return bool(connected)
    return getattr(client, "process", None) is not None or bool(getattr(client, "http_url", None))


def sanitize_mcp_tool_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:64] or "mcp_tool"


def _args_model(name: str, schema: dict[str, Any]) -> type[BaseModel] | None:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None

    required = set(schema.get("required") or [])
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    fields: dict[str, tuple[Any, Any]] = {}
    for prop, info in properties.items():
        prop_info = info if isinstance(info, dict) else {}
        py_type = type_map.get(prop_info.get("type", "string"), str)
        description = prop_info.get("description", "")
        if prop in required:
            fields[prop] = (py_type, Field(description=description))
        else:
            fields[prop] = (py_type | None, Field(default=None, description=description))
    return create_model(f"{sanitize_mcp_tool_name(name)}_schema", **fields)


def _result_text(result: Any) -> str:
    if isinstance(result, dict) and result.get("isError"):
        raise RuntimeError(str(result.get("message") or result))
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except TypeError:
        return str(result)


@dataclass
class MCPClientTool:
    name: str
    description: str
    parameters: dict[str, Any]
    client: MCPClient
    server_id: str
    original_name: str

    async def execute(self, **kwargs) -> str:
        result = await self.client.call_tool(self.original_name, kwargs)
        return _result_text(result)

    def to_param(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def to_langchain_tool(self) -> StructuredTool:
        async def runner(**kwargs) -> str:
            return await self.execute(**kwargs)

        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            args_schema=_args_model(self.name, self.parameters),
            coroutine=runner,
        )


class MCPClients(ToolCollection):
    """Collection of live MCP client tools, following the OpenManus lifecycle."""

    def __init__(self):
        super().__init__()
        self.clients: dict[str, MCPClient] = {}
        self.errors: dict[str, str] = {}

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        server_id: str = "",
        env: dict[str, Any] | None = None,
    ) -> None:
        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command
        if server_id in self.clients:
            await self.disconnect(server_id)

        client = MCPClient()
        await client.connect_stdio(command, args or [], env=env)
        if not _client_connected(client):
            self.errors[server_id] = "connection failed"
            return

        try:
            await client.initialize()
            tools = await client.list_tools()
        except Exception as exc:
            self.errors[server_id] = str(exc)
            await client.close()
            return

        self.clients[server_id] = client
        for tool in tools:
            original_name = str(tool.get("name", "")).strip()
            if not original_name:
                continue
            tool_name = sanitize_mcp_tool_name(f"mcp_{server_id}_{original_name}")
            self.add_tool(MCPClientTool(
                name=tool_name,
                description=tool.get("description", "") or f"MCP tool {server_id}:{original_name}",
                parameters=tool.get("inputSchema", {}) or {"type": "object", "properties": {}},
                client=client,
                server_id=server_id,
                original_name=original_name,
            ))

    async def connect_http(
        self,
        url: str,
        *,
        server_id: str = "",
        headers: dict[str, Any] | None = None,
    ) -> None:
        if not url:
            raise ValueError("Server URL is required.")

        server_id = server_id or url
        if server_id in self.clients:
            await self.disconnect(server_id)

        client = MCPClient()
        await client.connect_http(url, headers=headers)
        if not _client_connected(client):
            self.errors[server_id] = "connection failed"
            return

        try:
            await client.initialize()
            tools = await client.list_tools()
        except Exception as exc:
            self.errors[server_id] = str(exc)
            await client.close()
            return

        self.clients[server_id] = client
        for tool in tools:
            original_name = str(tool.get("name", "")).strip()
            if not original_name:
                continue
            tool_name = sanitize_mcp_tool_name(f"mcp_{server_id}_{original_name}")
            self.add_tool(MCPClientTool(
                name=tool_name,
                description=tool.get("description", "") or f"MCP tool {server_id}:{original_name}",
                parameters=tool.get("inputSchema", {}) or {"type": "object", "properties": {}},
                client=client,
                server_id=server_id,
                original_name=original_name,
            ))

    async def disconnect(self, server_id: str = "") -> None:
        if server_id:
            client = self.clients.pop(server_id, None)
            if client is not None:
                await client.close()
            self.tool_map = {
                name: tool
                for name, tool in self.tool_map.items()
                if getattr(tool, "server_id", None) != server_id
            }
            self.tools = tuple(self.tool_map.values())
            return

        for sid in sorted(list(self.clients)):
            await self.disconnect(sid)
        self.tool_map = {}
        self.tools = tuple()
