"""AI Hiking specialization of the OpenManus-style ToolCallAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.mcp_tools import MCPClientTool, MCPClients
from agent.tool_call_agent import ToolCallAgent
from agent.tool_collection import ToolCollection


@dataclass
class HikingManus(ToolCallAgent):
    name: str = "HikingManus"
    description: str = "A hiking-focused Manus agent using AI Hiking tools."
    llm: Any | None = None
    available_tools: ToolCollection = field(default_factory=ToolCollection)
    mcp_clients: MCPClients = field(default_factory=MCPClients)
    connected_servers: dict[str, str] = field(default_factory=dict)
    _initialized: bool = False
    max_steps: int = 20
    max_observe: int | None = 10000

    @classmethod
    async def create(
        cls,
        *,
        llm: Any,
        tools: list[Any],
        system_prompt: str | None = None,
        max_steps: int = 20,
        mcp_server_configs: dict[str, dict[str, Any]] | None = None,
    ) -> "HikingManus":
        instance = cls(
            llm=llm,
            available_tools=ToolCollection(*tools),
            system_prompt=system_prompt,
            max_steps=max_steps,
        )
        await instance.initialize_mcp_servers(mcp_server_configs or {})
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self, server_configs: dict[str, dict[str, Any]]) -> None:
        for server_id, server_config in (server_configs or {}).items():
            if not isinstance(server_config, dict):
                continue
            url = server_config.get("url")
            if url:
                await self.connect_mcp_http_server(
                    url,
                    server_id=server_id,
                    headers=server_config.get("headers") or None,
                )
                continue
            command = server_config.get("command")
            if not command:
                self.mcp_clients.errors[server_id] = "missing command or url"
                continue
            await self.connect_mcp_server(
                command,
                server_id=server_id,
                stdio_args=server_config.get("args", []) or [],
                env=server_config.get("env") or None,
            )

    async def connect_mcp_server(
        self,
        command: str,
        *,
        server_id: str = "",
        stdio_args: list[str] | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        await self.mcp_clients.connect_stdio(command, stdio_args or [], server_id=server_id, env=env)
        sid = server_id or command
        if sid not in self.mcp_clients.clients:
            return
        self.connected_servers[sid] = command
        new_tools = [
            tool
            for tool in self.mcp_clients.tools
            if isinstance(tool, MCPClientTool) and tool.server_id == sid
        ]
        self.available_tools.add_tools(*new_tools)

    async def connect_mcp_http_server(
        self,
        url: str,
        *,
        server_id: str = "",
        headers: dict[str, Any] | None = None,
    ) -> None:
        await self.mcp_clients.connect_http(url, server_id=server_id, headers=headers)
        sid = server_id or url
        if sid not in self.mcp_clients.clients:
            return
        self.connected_servers[sid] = url
        new_tools = [
            tool
            for tool in self.mcp_clients.tools
            if isinstance(tool, MCPClientTool) and tool.server_id == sid
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self) -> None:
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False
