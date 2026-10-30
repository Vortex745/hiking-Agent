"""Local-only MCP server management and connection testing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from agent_v2.mcp_config_store import (
    all_scoped_fingerprints,
    client_config,
    connection_fingerprint,
    load_servers,
    LocalMcpClient,
    public_server,
    save_servers,
    scoped_fingerprint,
)
from agent_v2.mcp_oauth_flow import (
    abort_oauth_flow,
    describe_tools,
    get_oauth_flow,
    pending_oauth_flows,
    resolve_oauth_callback,
    start_oauth_flow,
)
from agent_v2.mcp_registry import MCP_REGISTRY
from agent_v2.mcp_oauth_store import prune_oauth_tokens
from api.llm_config_router import require_local_request
from auth.deps import current_user_id
from config import settings

logger = logging.getLogger(__name__)


def require_local_mcp(request: Request) -> None:
    require_local_request(request)
    if settings.agent_v2_sandbox_type != "local_native":
        raise HTTPException(status_code=403, detail="MCP 管理仅在本机模式下可用")


mcp_router = APIRouter(prefix="/mcp", dependencies=[Depends(require_local_mcp)])


class ServerInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    transport: Literal["http", "stdio"]
    url: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    enabled: bool = False
    test_id: str | None = None
    server_id: str | None = None
    stdio_confirmed: bool = False

    @model_validator(mode="after")
    def validate_connection(self) -> "ServerInput":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("名称不能为空")
        if self.transport == "http":
            parsed = urlsplit(self.url.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("请输入有效的 HTTP URL，不要在 URL 中包含账户或密码")
        elif not self.command.strip():
            raise ValueError("命令不能为空")
        return self


# A test result grants only the exact connection that was tested. It is
# intentionally process-local; a restart requires a fresh test.
_tests: dict[str, tuple[str, list[dict]]] = {}


def _existing(servers: list[dict], server_id: str | None) -> dict | None:
    return next((server for server in servers if server["id"] == server_id), None)


def _connection(body: ServerInput, previous: dict | None) -> dict:
    same_transport = bool(previous and previous["transport"] == body.transport)
    credential_key = "headers" if body.transport == "http" else "env"
    supplied_credentials = body.headers if body.transport == "http" else body.env
    connection = {
        "name": body.name,
        "transport": body.transport,
        "url": body.url.strip() if body.transport == "http" else "",
        "command": body.command.strip() if body.transport == "stdio" else "",
        "args": body.args if body.transport == "stdio" else [],
        "headers": {},
        "env": {},
    }
    connection[credential_key] = (
        supplied_credentials if supplied_credentials is not None
        else previous[credential_key] if same_transport else {}
    )
    if same_transport and supplied_credentials is None and previous[credential_key]:
        destination = ("transport", "url", "command", "args")
        if any(connection[key] != previous[key] for key in destination):
            raise HTTPException(status_code=400, detail="连接地址或命令已更改，请重新填写凭据")
    return connection


def _test_key(owner_id: str, test_id: str) -> str:
    return f"{owner_id}:{test_id}"


@mcp_router.get("/servers")
async def list_mcp_servers(request: Request) -> dict:
    owner_id = current_user_id(request)
    builtins = [
        {
            "id": cfg.server_id,
            "name": cfg.description.split(" MCP", 1)[0],
            "builtin": True,
            "transport": cfg.transport,
            "status": "configured",
            "tools": list(cfg.tool_allowlist),
        }
        for cfg in MCP_REGISTRY.values()
    ]
    return {
        "servers": builtins
        + [public_server(server) for server in load_servers(owner_id or None)]
    }


def _mark_failed(previous: dict | None, connection: dict, owner_id: str) -> None:
    if previous and connection_fingerprint(previous) == connection_fingerprint(connection):
        previous["status"] = "failed"
        save_servers(
            [
                previous if item["id"] == previous["id"] else item
                for item in load_servers(owner_id or None)
            ],
            owner_id or None,
        )


async def test_mcp_server(body: ServerInput, owner_id: str = "") -> dict:
    previous = _existing(load_servers(owner_id or None), body.server_id)
    if body.server_id and previous is None:
        raise HTTPException(status_code=404, detail="MCP 服务不存在")
    connection = _connection(body, previous)
    if connection["transport"] == "stdio" and not body.stdio_confirmed:
        raise HTTPException(status_code=400, detail="请确认在本机启动该命令")
    if connection["transport"] == "http":
        flow = start_oauth_flow(connection, body.server_id, owner_id=owner_id)
        try:
            await asyncio.wait_for(asyncio.shield(flow.ready), timeout=15)
        except asyncio.TimeoutError:
            abort_oauth_flow(flow, "连接超时")
            _mark_failed(previous, connection, owner_id)
            raise HTTPException(status_code=502, detail="连接失败，请检查配置和服务状态") from None
        if flow.status == "auth_required":
            return {"status": "auth_required", "auth_url": flow.auth_url, "flow_id": flow.id}
        if flow.status != "success":
            _mark_failed(previous, connection, owner_id)
            raise HTTPException(status_code=502, detail="连接失败，请检查配置和服务状态")
        tools = flow.tools
    else:
        try:
            async with asyncio.timeout(15):
                async with LocalMcpClient("test", client_config(connection)) as client:
                    tools = describe_tools(client)
        except Exception:
            _mark_failed(previous, connection, owner_id)
            raise HTTPException(status_code=502, detail="连接失败，请检查配置和服务状态") from None
    test_id = uuid.uuid4().hex
    _tests[_test_key(owner_id, test_id)] = (
        scoped_fingerprint(connection, owner_id or None),
        tools,
    )
    return {"test_id": test_id, "tools": [tool["name"] for tool in tools], "status": "success"}


@mcp_router.post("/servers/test")
async def test_mcp_server_endpoint(body: ServerInput, request: Request) -> dict:
    return await test_mcp_server(body, current_user_id(request))


_OAUTH_CALLBACK_OK = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MCP 授权完成</title>
<body style="font-family:system-ui;padding:48px;color:#1f2937">
<h2>授权完成</h2><p>请返回应用，授权结果会自动同步。</p>
<script>setTimeout(function(){window.close()},800)</script></body></html>"""

_OAUTH_CALLBACK_FAILED = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MCP 授权未完成</title>
<body style="font-family:system-ui;padding:48px;color:#1f2937">
<h2>授权未完成</h2><p>请回到应用重新发起授权。</p></body></html>"""


@mcp_router.get("/oauth/callback", include_in_schema=False)
async def oauth_callback(
    request: Request,
    state: str = "",
    code: str = "",
    error: str = "",
    error_description: str = "",
) -> HTMLResponse:
    if error:
        flow = resolve_oauth_callback(state, "", error=error_description or error)
        logger.warning(
            "MCP OAuth callback denied: keys=%s state_len=%d flow=%s",
            sorted(request.query_params.keys()), len(state), bool(flow),
        )
        return HTMLResponse(_OAUTH_CALLBACK_FAILED)
    if not code:
        logger.warning(
            "MCP OAuth callback missing code: keys=%s state_len=%d pending=%d",
            sorted(request.query_params.keys()), len(state), len(pending_oauth_flows()),
        )
        return HTMLResponse(_OAUTH_CALLBACK_FAILED, status_code=400)
    if resolve_oauth_callback(state, code) is None:
        logger.warning(
            "MCP OAuth callback state not recognized: state_prefix=%r pending=%s",
            state[:12],
            [(flow.id[:8], (flow.expected_state or "")[:12]) for flow in pending_oauth_flows()],
        )
        return HTMLResponse(_OAUTH_CALLBACK_FAILED, status_code=400)
    return HTMLResponse(_OAUTH_CALLBACK_OK)


@mcp_router.get("/oauth/status/{flow_id}")
async def oauth_status(flow_id: str, request: Request) -> dict:
    flow = get_oauth_flow(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail="授权流程不存在或已过期")
    owner_id = current_user_id(request) or flow.owner_id
    payload: dict = {"status": flow.status}
    if flow.status == "success":
        if flow.test_id is None:
            flow.test_id = uuid.uuid4().hex
            _tests[_test_key(owner_id, flow.test_id)] = (flow.fingerprint, flow.tools)
        payload["test_id"] = flow.test_id
        payload["tools"] = [tool["name"] for tool in flow.tools]
    elif flow.status == "error":
        payload["detail"] = flow.error or "授权失败，请重试"
    return payload


def _save(body: ServerInput, server_id: str | None, owner_id: str = "") -> dict:
    servers = load_servers(owner_id or None)
    previous = _existing(servers, server_id)
    if server_id and previous is None:
        raise HTTPException(status_code=404, detail="MCP 服务不存在")
    connection = _connection(body, previous)
    fingerprint = scoped_fingerprint(connection, owner_id or None)
    test = _tests.get(_test_key(owner_id, body.test_id or ""))
    tested = bool(test and test[0] == fingerprint)
    unchanged = bool(previous and connection_fingerprint(previous) == connection_fingerprint(connection))
    if body.enabled and not tested and not (unchanged and previous["status"] == "success"):
        raise HTTPException(status_code=400, detail="启用前请先成功测试当前配置")
    server = {
        **connection,
        "id": server_id or f"custom_{uuid.uuid4().hex}",
        "enabled": body.enabled,
        "status": "success" if tested else previous["status"] if unchanged else "untested",
        "tools": [tool["name"] for tool in test[1]] if tested else previous["tools"] if unchanged else [],
        "tool_descriptors": test[1] if tested else previous.get("tool_descriptors", []) if unchanged else [],
        "revision": uuid.uuid4().hex,
    }
    if previous:
        servers[servers.index(previous)] = server
    else:
        servers.append(server)
    save_servers(servers, owner_id or None)
    prune_oauth_tokens(all_scoped_fingerprints())
    if tested:
        _tests.pop(_test_key(owner_id, body.test_id or ""), None)
    return public_server(server)


@mcp_router.post("/servers")
async def create_mcp_server(body: ServerInput, request: Request) -> dict:
    return _save(body, None, current_user_id(request))


@mcp_router.put("/servers/{server_id}")
async def update_mcp_server(server_id: str, body: ServerInput, request: Request) -> dict:
    return _save(body, server_id, current_user_id(request))


@mcp_router.delete("/servers/{server_id}")
async def delete_mcp_server(server_id: str, request: Request) -> dict:
    owner_id = current_user_id(request)
    servers = load_servers(owner_id or None)
    server = _existing(servers, server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="MCP 服务不存在")
    servers.remove(server)
    save_servers(servers, owner_id or None)
    prune_oauth_tokens(all_scoped_fingerprints())
    return {"deleted": True}
