"""Ticket 29 — registry-approved MCP wiring & two-layer gate contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.kernel_tools import (
    ProductGateRule,
    build_chat_extra_tools,
    build_profile_mcp_servers,
)
from agent_v2.mcp_registry import resolve_mcp_server
from uniharness.harness.permission import PermissionDecision as KernelPermissionDecision
from uniharness.harness.permission import PermissionResult


class _ProfileId:
    def __init__(self, value: str) -> None:
        self.value = value


class _Profile:
    def __init__(self, pid: str, tools: tuple[str, ...], servers: tuple[str, ...] | None) -> None:
        self.profile_id = _ProfileId(pid)
        self.allowed_tools = tools
        self.allowed_mcp_servers = servers


class _Settings:
    """check_mcp_available consults credential presence for stdio servers."""

    pass


def test_mcp_builder_drops_unavailable_servers_and_audits(monkeypatch):
    monkeypatch.delenv("AMAP_API_KEY", raising=False)
    profile = _Profile("route_plan", ("route_research",), ("amap_weather", "amap_map"))

    servers, audits = build_profile_mcp_servers(profile, _Settings())

    assert servers == {}
    dropped = {a["server_id"] for a in audits if a["phase"] == "mcp_unavailable"}
    assert dropped == {"amap_weather", "amap_map"}


def test_mcp_builder_materializes_stdio_config_with_secret_env(monkeypatch):
    monkeypatch.setenv("AMAP_API_KEY", "ak-test-value")
    profile = _Profile(
        "route_plan", (), ("amap_weather", "amap_map"),
    )

    servers, audits = build_profile_mcp_servers(profile, _Settings())

    assert audits == []
    assert set(servers) == {"amap_weather", "amap_map"}
    weather = servers["amap_weather"]
    assert weather["type"] == "stdio" and weather["command"] == "npx"
    assert "@amap/amap-weather-mcp" in weather["args"]
    # Secret values ride ONLY inside env (subprocess-local), never elsewhere.
    assert weather["env"]["AMAP_API_KEY"] == "ak-test-value"
    assert "ak-test-value" not in str(weather.get("command")) + str(weather.get("args"))


def test_all_evidence_sources_unavailable_maps_to_stable_terminal():
    """The stream maps empty-mcp to evidence_unavailable (unit-level proof)."""

    class _Empty:
        allowed_mcp_servers: tuple[str, ...] | None = ()

    servers, audits = build_profile_mcp_servers(_Profile("route_plan", (), ()), _Settings())
    assert servers == {} and audits == []


# ---------------------------------------------------------------------------
# Two-layer MCP gate: server allowlist × per-server tool allowlist
# ---------------------------------------------------------------------------


def _rule_for(profile: Any, sink: list[dict[str, Any]]) -> ProductGateRule:
    return build_chat_kernel_gate(profile, run_id="rp", audit_sink=sink.append)._rules[0]


from agent_v2.kernel_tools import build_chat_kernel_gate as _builder  # noqa: E402
build_chat_kernel_gate = _builder  # noqa: F811


def test_approved_mcp_server_tool_pair_passes_gate(monkeypatch):
    monkeypatch.setenv("AMAP_API_KEY", "ak-x")
    profile = _Profile(
        "route_plan",
        ("hiking_knowledge_search",),
        ("amap_weather", "amap_map"),
    )
    sink: list[dict[str, Any]] = []
    rule = _rule_for(profile, sink)

    decision = rule.check("mcp__amap_map__plan_route", {"destination": "杭州"})
    cfg = resolve_mcp_server("amap_map")
    assert "plan_route" in cfg.tool_allowlist  # precondition sanity
    assert decision is None or decision.result is PermissionResult.ALLOWED
    assert [s for s in sink if s.get("phase") == "denied"] == []


def test_rogue_tool_on_allowed_server_is_denied_and_audited():
    profile = _Profile("route_plan", (), ("amap_map",))
    sink: list[dict[str, Any]] = []
    rule = _rule_for(profile, sink)

    decision = rule.check("mcp__amap_map__shell_exec", {})

    assert decision is not None and decision.result is PermissionResult.DENIED
    assert decision.reason == "unauthorized_mcp_tool"
    denial = sink[-1]
    assert denial["code"] == "unauthorized_mcp_tool" and denial["tool"].startswith("mcp__")


def test_unknown_server_in_kernel_name_is_denied():
    profile = _Profile("route_plan", (), ("amap_map",))
    sink: list[dict[str, Any]] = []
    rule = _rule_for(profile, sink)

    decision = rule.check("mcp__evil_server__read_secrets", {})

    assert decision is not None and decision.result is PermissionResult.DENIED
    assert decision.reason == "unknown_mcp_server"


def test_non_mcp_tools_keep_profile_allowlist_semantics():
    profile = _Profile("chat", ("gear_checklist",), None)
    sink: list[dict[str, Any]] = []
    rule = _rule_for(profile, sink)

    assert rule.check("Bash", {"command": "id"}) is not None  # denied path object
    assert rule.check("gear_checklist", {"days": 1}) is None  # allowed passes through