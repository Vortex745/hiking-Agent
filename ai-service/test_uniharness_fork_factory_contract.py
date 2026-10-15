"""Ticket 25 — factory contract tests against the real vendored UniHarness fork.

Covers the pinned fork provenance chain (kernel identity, fail-closed
mismatch), the real ``create_agent`` factory surface (built-in tool groups,
PermissionGate semantics, bounded subagent definitions, domain tool adapter,
checkpointer/compaction/MCP wiring, Computer seam) and one REAL model-driven
tool selection + invocation through the actual kernel loop.

Network tests are skipped when credentials are absent so CI without secrets
stays green; the skip is recorded, never silently converted to a pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from uniharness.computer.base import Computer  # noqa: E402
from uniharness.harness.definition import AgentDefinition  # noqa: E402
from uniharness.harness.model import ModelProfile  # noqa: E402
from uniharness.harness.permission import (  # noqa: E402
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
)
from uniharness.langchain.agent import Agent, create_agent  # noqa: E402
from uniharness.tools.base import BaseAgentTool  # noqa: E402
from uniharness.types import CLIResult, ToolResult  # noqa: E402

from agent_v2.uniharness.kernel_identity import (  # noqa: E402
    VENDORED_UPSTREAM_REVISION,
    resolve_kernel_identity,
)
from agent_v2.kernel import KernelModelFailure, is_provider_throttled  # noqa: E402

_BUILTIN_TOOL_NAMES = {
    "Agent",
    "TaskOutput",
    "TaskStop",
    "TodoWrite",
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
}

_ENV_DELIMITER = "___ENV___"


def _build_chat_model() -> ChatOpenAI:
    """Build the OpenAI-compatible chat model from .env credentials."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=0,
    )


class _EnvProbeComputer:
    """Deterministic Computer double satisfying the kernel's Protocol seam.

    ``create_agent`` resolves the environment through a single batched shell
    command; the probe replays a fixed Unix-like answer so factory tests run
    fast and platform-independent. This exercises the required-Computer
    contract without spawning VMs or containers.
    """

    @property
    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self, command: str, *, timeout: float | None = None) -> CLIResult:
        parts = [
            "/workspace",
            "false",
            "linux",
            "bash",
            "probe-kernel 1.0",
            "2026-08-27T12:00:00+0000",
        ]
        return CLIResult(
            stdout=(f"\n{_ENV_DELIMITER}\n".join(parts)).replace("\n", " "),
            stderr="",
            exit_code=0,
        )

    async def upload(self, src: str, dst: str) -> None:
        raise AssertionError("factory contract tests never upload files")

    async def download(self, src: str, dst: str) -> None:
        raise AssertionError("factory contract tests never download files")


# The upstream Computer protocol must accept the probe structurally.
_: type[Computer] = _EnvProbeComputer


class _EchoParams(BaseModel):
    greeting: str = Field(description="Text to echo back.")


class EchoTool(BaseAgentTool[_EchoParams]):
    """Domain-tool adapter prototype: validates input, bounds output."""

    name = "echo_greeting"
    description = "Echo a greeting back deterministically."
    args_schema = _EchoParams

    def __init__(self) -> None:
        self.executed_with: list[str] = []

    async def execute(self, params: _EchoParams) -> ToolResult:
        self.executed_with.append(params.greeting)
        return ToolResult(output=f"echo:{params.greeting}")


class _DenyBashRule(SafetyRule):
    """Product-side rule proving kernel gate accepts external policies."""

    def check(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> PermissionDecision | None:
        if tool_name == "bash":
            return PermissionDecision(
                result=PermissionResult.DENIED,
                reason="product policy: bash denied in this profile",
            )
        return None


def _run(coro: Any) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


# ---------------------------------------------------------------------------
# Kernel identity (pinned fork provenance)
# ---------------------------------------------------------------------------


def _vendor_head() -> str:
    """Resolve the vendored fork HEAD via git rev-parse."""
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT / "third_party" / "uniharness",
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return out.stdout.strip()


def test_kernel_identity_matches_pinned_vendor_commit():
    identity = resolve_kernel_identity(_vendor_head())

    assert identity.compatible, identity.detail
    assert identity.fork_commit == _vendor_head()
    # The delta-inclusive fork HEAD differs from the upstream base revision.
    assert identity.fork_commit[:12] != VENDORED_UPSTREAM_REVISION[:12]
    assert identity.distribution_version  # installed distribution present
    assert identity.package_path  # resolvable source tree
    assert "stub" not in identity.detail.lower()


def test_kernel_identity_rejects_upstream_revision_as_operational_pin():
    # The operational pin is the delta-inclusive fork HEAD, NOT the raw
    # upstream revision; passing the upstream base must fail closed.
    identity = resolve_kernel_identity(VENDORED_UPSTREAM_REVISION)

    assert not identity.compatible
    assert identity.fork_commit == _vendor_head()
    assert "does not match configured pin" in identity.detail


def test_kernel_identity_fail_closed_on_foreign_pin():
    foreign = "deadbeef" + VENDORED_UPSTREAM_REVISION[8:]
    identity = resolve_kernel_identity(foreign)

    assert not identity.compatible
    assert identity.fork_commit == _vendor_head()
    assert "does not match configured pin" in identity.detail


# ---------------------------------------------------------------------------
# Real create_agent factory contract
# ---------------------------------------------------------------------------


def test_factory_builds_real_kernel_with_builtin_tool_surface():
    async def scenario():
        agent = await create_agent(
            _build_chat_model(),
            _EnvProbeComputer(),
            skill_paths=(),
        )
        try:
            assert isinstance(agent, Agent)
            names = {tool.name for tool in agent.tools}
            missing = _BUILTIN_TOOL_NAMES - names
            assert not missing, f"built-in tools missing from real kernel: {missing}"
            assert agent.graph is not None
            # No product stub type leaked into the kernel assembly.
            assert not hasattr(agent._context, "permission_gate")
        finally:
            await agent.aclose()

    _run(scenario())


def test_factory_wires_checkpointer_and_compaction_profile():
    async def scenario():
        checkpointer = InMemorySaver()
        profile = ModelProfile(model=_build_chat_model(), compaction_threshold=12345)
        agent = await create_agent(
            profile,
            _EnvProbeComputer(),
            skill_paths=(),
            checkpointer=checkpointer,
        )
        try:
            assert agent.model is profile
            assert agent.model.compaction_threshold == 12345
            assert agent.graph.checkpointer is checkpointer
        finally:
            await agent.aclose()

    _run(scenario())


def test_factory_mcp_connect_failure_is_skipped_not_fatal(
    caplog: pytest.LogCaptureFixture,
):
    """Pinned fork semantics: an unreachable MCP server is retried with
    backoff, then SKIPPED — the agent still builds and the product surfaces
    the source as ``mcp_unavailable`` evidence at chat time. A bare
    "fails fast" expectation here only ever passed on machines where the
    ``mcp`` SDK's own import chain is broken (see the recorded skip below).
    """
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(
            f"mcp client stack is not importable in this interpreter "
            f"(recorded, not a pass): {exc}"
        )

    caplog.set_level(logging.WARNING)

    async def scenario():
        agent = await asyncio.wait_for(
            create_agent(
                _build_chat_model(),
                _EnvProbeComputer(),
                skill_paths=(),
                mcp_servers={
                    "ghost": {"type": "http", "url": "http://127.0.0.1:9/mcp"},
                },
            ),
            timeout=30,
        )
        try:
            assert agent.graph is not None
        finally:
            await agent.aclose()

    _run(scenario())
    assert any(
        "Failed to connect to MCP server 'ghost'" in record.getMessage()
        for record in caplog.records
    ), "the dead server must be recorded as skipped, not silently dropped"


def test_bounded_subagent_definitions_validated_by_factory():
    researcher = AgentDefinition(
        description="bounded research helper",
        tools=("Glob", "Grep"),
    )

    async def happy():
        agent = await create_agent(
            _build_chat_model(), _EnvProbeComputer(), skill_paths=(), agents={"researcher": researcher}
        )
        try:
            assert agent.agents["researcher"] is researcher
        finally:
            await agent.aclose()

    _run(happy())

    async def unknown_tool():
        await create_agent(
            _build_chat_model(),
            _EnvProbeComputer(),
            skill_paths=(),
            agents={"bad": AgentDefinition(description="x", tools=("NoSuchTool",))},
        )

    with pytest.raises(ValueError, match="unknown tool"):
        _run(unknown_tool())


def test_subagents_cannot_spawn_lower_level_agents():
    async def scenario():
        await create_agent(
            _build_chat_model(),
            _EnvProbeComputer(),
            skill_paths=(),
            agents={"nested": AgentDefinition(description="x", tools=("Agent",))},
        )

    with pytest.raises(ValueError, match="sub-sub-agents are disabled"):
        _run(scenario())


def test_product_safety_rule_composes_with_kernel_permission_gate():
    async def scenario():
        gate = PermissionGate()
        gate.register_rule(_DenyBashRule())

        denied = await gate.check("bash", {"command": "echo hi"})
        allowed = await gate.check("Read", {"path": "/tmp/a.txt"})
        approval = await gate.check("needs_ladder", {})

        assert denied.result is PermissionResult.DENIED
        assert denied.reason == "product policy: bash denied in this profile"
        assert allowed.result is PermissionResult.ALLOWED
        assert approval.result is PermissionResult.ALLOWED  # default allow w/o rules

    _run(scenario())


def test_domain_tool_adapter_validates_input_and_bounds_output():
    tool = EchoTool()

    ok = asyncio.run(tool(greeting="ping"))
    assert ok.output == "echo:ping"
    assert tool.executed_with == ["ping"]

    bad = asyncio.run(tool(wrong="arg"))
    assert bad.error and "invalid parameter" in bad.error
    assert tool.executed_with == ["ping"]


# ---------------------------------------------------------------------------
# Real model-driven tool selection & invocation (AC4)
# ---------------------------------------------------------------------------

_HAS_LLM_CREDS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("OPENAI_BASE_URL"))
# Opt-in live check: strict provider qpm makes un-gated bursts flaky (429).
_LIVE_LLM = _HAS_LLM_CREDS and os.getenv("KERNEL_LIVE_E2E", "") == "1"


@pytest.mark.skipif(
    not _LIVE_LLM,
    reason="live model probe requires KERNEL_LIVE_E2E=1 plus OPENAI_* creds (recorded skip)",
)
def test_real_model_driven_tool_selection_and_invocation_through_kernel():
    echo = EchoTool()

    async def scenario():
        agent = await create_agent(
            _build_chat_model(),
            _EnvProbeComputer(),
            skill_paths=(),
            extra_tools=[echo],
            system_prompt=(
                "You are running a kernel integration probe.\n"
                "You MUST call the echo_greeting tool exactly once with "
                "greeting=\"kernel-probe\", then reply with the single word done."
            ),
        )
        try:
            assert any(t.name == "echo_greeting" for t in agent.tools)
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [("user", "Run the probe.")] }),
                timeout=240,
            )
        finally:
            await agent.aclose()
        return result

    try:
        result = _run(scenario())
    except KernelModelFailure as exc:
        if is_provider_throttled(exc):
            pytest.skip("provider qpm throttled (recorded skip)")
        raise
    messages = result["messages"]

    # The real kernel loop executed the selected tool adapter.
    assert echo.executed_with == ["kernel-probe"]
    tool_messages = [m for m in messages if getattr(m, "type", "") == "tool"]
    assert any("echo:kernel-probe" in str(m.content) for m in tool_messages)

    final_ai = [m for m in messages if getattr(m, "type", "") == "ai"]
    assert final_ai, "kernel produced no final AI message"
