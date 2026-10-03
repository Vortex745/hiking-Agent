"""Server-side MCP registry for Chat v2 (Ticket 05).

The registry is the single source of truth for which MCP servers exist,
how to connect to them, and which tools they expose. Three guarantees
are enforced here so the product contract holds regardless of caller:

1. **Server-defined only** — clients cannot add servers, override
   connection parameters, or inject raw tool names. The ``MCP_REGISTRY``
   dict is a module-level constant; ``resolve_mcp_server`` rejects
   unknown ids.
2. **Credential isolation** — :class:`McpServerConfig` stores only the
   *names* of environment variables / header keys (``env_key_names``,
   ``headers_key_names``). The actual secret values are read from
   ``settings`` only inside :func:`expand_credentials`, which is called
   by the host-side connector. The config dataclass itself never holds
   a secret, so it can be logged, serialized into events, or attached
   to a run record without leakage.
3. **Bounded behavior** — every server config carries ``timeout_seconds``
   and ``max_output_bytes``. The host connector enforces both; tool
   results that exceed the cap are truncated and flagged. This prevents
   a misbehaving MCP server from blowing the SSE stream or memory.

Ticket 05 ships two servers: ``amap_weather`` and ``amap_map``. Both are
referenced by the ``route_plan`` profile's ``allowed_mcp_servers``
tuple. The ``chat`` profile keeps ``allowed_mcp_servers=None`` and so
never reaches this registry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class UnknownMcpServerError(KeyError):
    """Raised when a caller asks for an MCP server the registry does not define.

    Subclassing ``KeyError`` keeps the semantics obvious (the id is not in
    the registry dict) while giving callers a typed exception to catch.
    """


class MissingMcpCredentialError(RuntimeError):
    """Raised when ``expand_credentials`` cannot find a required credential.

    The message contains only key *names*; values are never echoed. The
    type is separate from :class:`UnknownMcpServerError` because the
    caller's response differs: unknown server → reject the request;
    missing credential → mark the tool unavailable and continue with
    explicit evidence-missing messaging.
    """


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """Server-side definition of one MCP server.

    Fields are grouped by purpose:

    * Identity — ``server_id``, ``description`` (user-safe, no secrets).
    * Transport — ``transport`` (``"stdio"`` or ``"http"``),
      ``command_or_url``, ``args``. These mirror the shape of the v1
      ``MCP_SERVERS`` env var so the host connector can reuse wiring.
    * Credentials — ``env_key_names`` and ``headers_key_names`` hold the
      *names* of env vars / HTTP headers that must be expanded at call
      time. Storing names (not values) is what lets this dataclass be
      logged and attached to run records without leaking secrets.
    * Tool surface — ``tool_allowlist`` is the tuple of tool names this
      server exposes. Tools outside the list are invisible even if the
      server advertises them.
    * Bounds — ``timeout_seconds`` caps a single tool call; ``max_output_bytes``
      caps the tool result size before it enters the SSE stream.

    The dataclass is frozen so a profile's ``allowed_mcp_servers`` tuple
    cannot be mutated to widen access at runtime.
    """

    server_id: str
    description: str
    transport: str  # "stdio" | "http"
    command_or_url: str
    args: tuple[str, ...] = ()
    env_key_names: tuple[str, ...] = ()
    headers_key_names: tuple[str, ...] = ()
    tool_allowlist: tuple[str, ...] = ()
    timeout_seconds: float = 10.0
    max_output_bytes: int = 64 * 1024  # 64 KiB default cap


# ---- Server registry ------------------------------------------------------
#
# Ticket 05 ships the two AMap MCP servers the route_plan profile needs.
# Both bounded surfaces use the official AMap server, pinned to the
# reviewed npm version. Credentials are expanded only on the host.
_AMAP_WEATHER = McpServerConfig(
    server_id="amap_weather",
    description="高德天气 MCP — 提供实时与预报天气数据",
    transport="stdio",
    command_or_url="npx",
    args=("-y", "@amap/amap-maps-mcp-server@0.0.8"),
    env_key_names=("AMAP_API_KEY",),
    headers_key_names=(),
    tool_allowlist=("maps_weather",),
    timeout_seconds=8.0,
    max_output_bytes=32 * 1024,  # 32 KiB — weather payloads are small
)

_AMAP_MAP = McpServerConfig(
    server_id="amap_map",
    description="高德地图 MCP — 提供路线规划、POI 检索与地理编码",
    transport="stdio",
    command_or_url="npx",
    args=("-y", "@amap/amap-maps-mcp-server@0.0.8"),
    env_key_names=("AMAP_API_KEY",),
    headers_key_names=(),
    tool_allowlist=(
        "maps_text_search",
        "maps_direction_walking",
        "maps_geo",
    ),
    timeout_seconds=12.0,
    max_output_bytes=64 * 1024,  # 64 KiB — route plans can be larger
)

MCP_REGISTRY: dict[str, McpServerConfig] = {
    _AMAP_WEATHER.server_id: _AMAP_WEATHER,
    _AMAP_MAP.server_id: _AMAP_MAP,
}


# ---- Resolution and credential expansion ----------------------------------


def resolve_mcp_server(server_id: str) -> McpServerConfig:
    """Return the frozen :class:`McpServerConfig` for ``server_id``.

    Raises :class:`UnknownMcpServerError` for ids not in ``MCP_REGISTRY``.
    The registry is a module-level constant; callers cannot add entries
    at runtime. This is the single entry point so the rejection rule is
    enforced in one place.
    """
    try:
        return MCP_REGISTRY[server_id]
    except KeyError as exc:
        raise UnknownMcpServerError(
            f"Unknown MCP server id: {server_id!r}. "
            f"Registered: {sorted(MCP_REGISTRY)}"
        ) from exc


def list_profile_mcp_tools(
    profile: Any,
) -> dict[str, tuple[str, ...]]:
    """Return ``{server_id: (tool_name, ...)}`` for the profile's MCP servers.

    A profile with ``allowed_mcp_servers=None`` (e.g. ``chat``) returns
    an empty dict — the profile has no MCP concept and never reaches the
    registry. A profile with an empty tuple returns an empty dict too,
    but for a different reason: MCP is enabled but no servers are
    allowlisted. The caller distinguishes via ``profile.allowed_mcp_servers
    is None``.

    Tools outside the server's ``tool_allowlist`` are not listed here,
    even if the server advertises them at runtime. This is the second
    layer of defense on top of the profile's server allowlist.
    """
    allowed_servers = getattr(profile, "allowed_mcp_servers", None)
    if not allowed_servers:
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for server_id in allowed_servers:
        config = resolve_mcp_server(server_id)
        result[server_id] = config.tool_allowlist
    return result


def check_mcp_available(config: McpServerConfig, settings: Any) -> bool:
    """True iff every required credential key is present in the environment.

    "Present" means the env var is set to a non-empty value. We do not
    validate the credential itself — that is the server's job. The check
    is cheap so the intake can call it to decide whether to mark a tool
    unavailable without spawning the MCP subprocess.

    Only ``env_key_names`` is checked here. HTTP header credentials are
    expanded by the connector from ``settings``; their availability is
    checked inside :func:`expand_credentials` because the values may
    come from a secret manager rather than env vars.
    """
    for key_name in config.env_key_names:
        value = os.getenv(key_name, "").strip()
        if not value:
            return False
    return True


def expand_credentials(
    config: McpServerConfig, settings: Any
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(env_dict, headers_dict)`` with actual secret values.

    Called only by the host-side MCP connector immediately before
    spawning the subprocess or opening the HTTP connection. The
    returned dicts are NOT logged, NOT attached to run records, NOT
    sent into the sandbox, and NOT emitted in events. The connector
    owns the secret lifecycle; this function only resolves names to
    values.

    Raises :class:`MissingMcpCredentialError` if any required key is
    absent. The error message lists the missing *names* (never values),
    so it is safe to surface in operator logs.
    """

    missing: list[str] = []
    env_dict: dict[str, str] = {}
    for key_name in config.env_key_names:
        value = os.getenv(key_name, "").strip()
        if not value:
            missing.append(key_name)
        else:
            env_dict[key_name] = value
            if key_name == "AMAP_API_KEY":
                # Preserve the deployment input name; the official server
                # reads this variable instead.
                env_dict["AMAP_MAPS_API_KEY"] = value

    headers_dict: dict[str, str] = {}
    for header_name in config.headers_key_names:
        # HTTP header credentials for v2 MCP servers are read from
        # settings attributes named after the header (uppercased). The
        # stub servers in Ticket 05 use stdio, so this branch is here
        # for forward compatibility with future HTTP MCP servers.
        attr = header_name.lower()
        value = str(getattr(settings, attr, "")).strip()
        if not value:
            missing.append(header_name)
        else:
            headers_dict[header_name] = value

    if missing:
        raise MissingMcpCredentialError(
            f"MCP server {config.server_id!r} is missing credentials: {missing}. "
            "Set the listed environment variables (names only — values are not "
            "echoed here) and retry."
        )
    return env_dict, headers_dict
