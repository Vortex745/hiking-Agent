"""Ticket 33 — fixed bounded subagent definitions wired to the root factory."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent_v2.kernel_tools import build_subagent_definitions
from agent_v2.profiles import resolve_profile


class _ProfileId:
    def __init__(self, value: str) -> None:
        self.value = value


def _profile(pid: str = "route_plan", mcp: tuple[str, ...] | None = ("amap_weather", "amap_map")) -> Any:
    return SimpleNamespace(profile_id=_ProfileId(pid), allowed_mcp_servers=mcp)


class _Settings:
    agent_v2_subagent_enabled = True


class _SettingsDisabled:
    agent_v2_subagent_enabled = False


_ROUTE_SURFACE = {
    "hiking_knowledge_search", "route_research", "web_scraping",
    "gear_checklist", "memory_search",
}


def test_disabled_flag_yields_root_only():
    defs, audits = build_subagent_definitions(
        _profile(), _SettingsDisabled(),
        extra_tool_names=_ROUTE_SURFACE, available_mcp_servers={"amap_weather", "amap_map"},
    )
    assert defs == {} and audits == []


def test_satisfiable_roles_injected_and_unsatisfiable_dropped_with_audit():
    defs, audits = build_subagent_definitions(
        _profile(), _Settings(),
        extra_tool_names=_ROUTE_SURFACE, available_mcp_servers={"amap_weather", "amap_map"},
    )

    assert set(defs) == {"researcher", "route_planner"}
    # researcher needs no MCP; route_planner needs both AMap servers.
    skipped = {a["role"] for a in audits if a["phase"] == "subagent_skipped"}
    assert skipped == {"artifact_writer"}  # sandbox surface not mounted on route_plan
    for d in defs.values():
        assert set(d.tools) <= _ROUTE_SURFACE


def test_partial_mcp_failure_drops_only_dependent_role():
    defs, audits = build_subagent_definitions(
        _profile(mcp=()), _Settings(),
        extra_tool_names=_ROUTE_SURFACE, available_mcp_servers=set(),
    )
    assert "researcher" in defs and "route_planner" not in defs
    skipped = {a["role"] for a in audits if a["phase"] == "subagent_skipped"}
    assert skipped == {"route_planner", "artifact_writer"}


def test_unknown_role_cannot_exist_by_construction():
    defs, _ = build_subagent_definitions(
        _profile(), _Settings(),
        extra_tool_names=_ROUTE_SURFACE, available_mcp_servers={"amap_weather", "amap_map"},
    )
    assert set(defs) <= {"researcher", "route_planner", "artifact_writer"}


def test_route_plan_hides_redundant_direct_tools():
    route_plan = resolve_profile("route_plan")
    chat = resolve_profile("chat")

    assert "web_search" not in route_plan.allowed_tools
    assert "risk_assessment" not in route_plan.allowed_tools
    assert "web_scraping" in route_plan.allowed_tools
    assert "risk_assessment" in chat.allowed_tools


# ---------------------------------------------------------------------------
# kernel turn wires defs → vendor factory (groups upgraded to task)
# ---------------------------------------------------------------------------


class _Settings:
    openai_api_key = "sk-fake"
    openai_base_url = "https://fake.invalid/v1"
    openai_model = "fake"
    agent_v2_uniharness_pin = os.getenv(
        "AGENT_V2_UNIHARNESS_PIN", "b026a26250487678aedc2beef0d0b6ae802de5c4"
    )
    agent_v2_subagent_enabled = True


def test_run_chat_turn_upgrades_task_group_only_with_defs(monkeypatch):
    captured: list[dict[str, Any]] = []

    async def _fake_create(model, computer, **kwargs):
        captured.append(kwargs)
        agent = SimpleNamespace()
        agent.aclose = _async_noop
        agent.ainvoke = _async_empty
        agent.tools = []
        return agent

    async def _async_noop():
        return None

    async def _async_empty(*a, **k):
        return {"messages": []}

    import uniharness.langchain.agent as vendor

    monkeypatch.setattr(vendor, "create_agent", _fake_create)

    from agent_v2.kernel import KernelModelFailure, run_chat_turn

    defs = {"researcher": object()}

    async def with_defs():
        with pytest.raises(KernelModelFailure):
            await run_chat_turn(
                system_prompt="p", message="m", settings=_Settings(),
                run_id="r", subagent_definitions=defs,
            )

    async def without_defs():
        with pytest.raises(KernelModelFailure):
            await run_chat_turn(
                system_prompt="p", message="m", settings=_Settings(), run_id="r2",
            )

    asyncio.run(with_defs())
    assert captured[-1]["builtin_tool_groups"] == ("task",)
    assert captured[-1]["agents"] == defs

    asyncio.run(without_defs())
    assert captured[-1]["builtin_tool_groups"] == ()
    assert captured[-1]["agents"] == {}
