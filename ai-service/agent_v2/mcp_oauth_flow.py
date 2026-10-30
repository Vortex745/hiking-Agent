"""Background OAuth authorization flows for user-added HTTP MCP servers.

One flow == one authorization attempt. The MCP SDK's ``OAuthClientProvider``
runs inside a background task; the flow hands the authorization URL to the
HTTP caller, waits for the loopback callback, then lets the provider exchange
the code and reconnect. Tokens are persisted by ``mcp_oauth_store``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

from agent_v2.mcp_config_store import LocalMcpClient, client_config, connection_fingerprint
from agent_v2.mcp_oauth_store import build_oauth_provider, save_authorization_server

logger = logging.getLogger(__name__)

_FLOW_TTL_SECONDS = 600.0
_MAX_OAUTH_SECONDS = 330.0
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class OAuthCallbackError(RuntimeError):
    """Raised inside the SDK provider when the authorization endpoint denies."""


@dataclass
class OAuthFlow:
    id: str
    fingerprint: str
    server_id: str | None
    status: Literal["starting", "auth_required", "success", "error"] = "starting"
    auth_url: str | None = None
    expected_state: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    test_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    ready: asyncio.Future = field(init=False, repr=False)
    callback: asyncio.Future = field(init=False, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.ready = loop.create_future()
        self.callback = loop.create_future()


_flows: dict[str, OAuthFlow] = {}
_by_fingerprint: dict[str, str] = {}
_by_state: dict[str, str] = {}
_consumed_states: dict[str, str] = {}


def start_oauth_flow(connection: dict[str, Any], server_id: str | None) -> OAuthFlow:
    _expire()
    fingerprint = connection_fingerprint(connection)
    existing_id = _by_fingerprint.get(fingerprint)
    if existing_id:
        existing = _flows.get(existing_id)
        if existing is not None and existing.status in {"starting", "auth_required"}:
            return existing
    flow = OAuthFlow(id=uuid.uuid4().hex, fingerprint=fingerprint, server_id=server_id)
    _flows[flow.id] = flow
    _by_fingerprint[fingerprint] = flow.id
    flow.task = asyncio.create_task(_run_flow(flow, connection))
    return flow


def get_oauth_flow(flow_id: str) -> OAuthFlow | None:
    _expire()
    return _flows.get(flow_id)


def abort_oauth_flow(flow: OAuthFlow, message: str) -> None:
    if flow.task is not None and not flow.task.done():
        flow.task.cancel()
    if flow.status != "success":
        flow.status = "error"
        flow.error = message
    _release(flow)
    _resolve(flow.ready)


def resolve_oauth_callback(state: str, code: str, *, error: str | None = None) -> OAuthFlow | None:
    _expire()
    flow_id = _by_state.pop(state, None) if state else None
    if flow_id is None and not state:
        pending = [flow for flow in _flows.values() if flow.status == "auth_required" and flow.expected_state]
        if len(pending) == 1:
            flow = pending[0]
            _consumed_states[flow.expected_state or ""] = flow.id
            if not flow.callback.done() and not (flow.task is not None and flow.task.done()):
                if error:
                    flow.callback.set_exception(OAuthCallbackError(error))
                else:
                    flow.callback.set_result((code, flow.expected_state))
            return flow
    if flow_id is None:
        consumed_id = _consumed_states.get(state)
        return _flows.get(consumed_id) if consumed_id else None
    _consumed_states[state] = flow_id
    flow = _flows.get(flow_id)
    if flow is None or flow.callback.done():
        return flow
    if flow.task is not None and flow.task.done():
        return flow
    if error:
        flow.callback.set_exception(OAuthCallbackError(error))
    else:
        flow.callback.set_result((code, state))
    return flow


def pending_oauth_flows() -> list[OAuthFlow]:
    _expire()
    return [flow for flow in _flows.values() if flow.status in {"starting", "auth_required"}]


def describe_tools(client: Any) -> list[dict[str, Any]]:
    def expand(value: Any, definitions: dict[str, Any]) -> Any:
        if isinstance(value, list):
            return [expand(item, definitions) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return expand(definitions[value["$ref"].rsplit("/", 1)[-1]], definitions)
        return {key: expand(item, definitions) for key, item in value.items() if key != "$defs"}

    descriptors: list[dict[str, Any]] = []
    for tool in client.tools:
        schema = tool.args_schema.model_json_schema()
        descriptors.append(
            {
                "name": tool.name.split("__", 2)[-1],
                "description": tool.description,
                "schema": expand(schema, schema.get("$defs", {})),
            }
        )
    return descriptors


async def _run_flow(flow: OAuthFlow, connection: dict[str, Any]) -> None:
    try:
        async with asyncio.timeout(_MAX_OAUTH_SECONDS):
            provider = await build_oauth_provider(
                connection,
                redirect_handler=lambda url: _capture_auth_url(flow, url),
                callback_handler=lambda: _await_callback(flow),
            )
            async with LocalMcpClient("oauth-test", client_config(connection), auth=provider) as client:
                flow.tools = describe_tools(client)
            save_authorization_server(flow.fingerprint, provider.context.oauth_metadata)
        flow.status = "success"
    except asyncio.CancelledError:
        if flow.status != "success":
            flow.status = "error"
            flow.error = "授权流程已取消"
        raise
    except TimeoutError:
        flow.status = "error"
        flow.error = "授权超时，请重试"
    except Exception:
        logger.warning("MCP OAuth flow failed", exc_info=True)
        flow.status = "error"
        flow.error = "授权失败，请检查服务授权配置后重试"
    finally:
        _release(flow)
        _resolve(flow.ready)


async def _capture_auth_url(flow: OAuthFlow, url: str) -> None:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme != "https" and not (scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS):
        raise OAuthCallbackError("授权地址必须使用 HTTPS")
    state = parse_qs(parsed.query).get("state", [None])[0]
    if not state:
        raise OAuthCallbackError("授权地址缺少 state")
    flow.auth_url = url
    flow.expected_state = state
    flow.status = "auth_required"
    _by_state[state] = flow.id
    _resolve(flow.ready)


async def _await_callback(flow: OAuthFlow) -> tuple[str, str | None]:
    result = await flow.callback
    return result


def _resolve(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


def _release(flow: OAuthFlow) -> None:
    if _by_fingerprint.get(flow.fingerprint) == flow.id:
        _by_fingerprint.pop(flow.fingerprint, None)
    if flow.callback.done() or flow.status != "auth_required":
        for state, flow_id in list(_by_state.items()):
            if flow_id == flow.id:
                _by_state.pop(state, None)


def _expire() -> None:
    now = time.monotonic()
    for flow_id, flow in list(_flows.items()):
        if now - flow.created_at <= _FLOW_TTL_SECONDS:
            continue
        if flow.status in {"starting", "auth_required"}:
            if flow.task is not None and not flow.task.done():
                flow.task.cancel()
        _flows.pop(flow_id, None)
        if _by_fingerprint.get(flow.fingerprint) == flow_id:
            _by_fingerprint.pop(flow.fingerprint, None)
        for state, mapped in list(_by_state.items()):
            if mapped == flow_id:
                _by_state.pop(state, None)
        for state, mapped in list(_consumed_states.items()):
            if mapped == flow_id:
                _consumed_states.pop(state, None)
