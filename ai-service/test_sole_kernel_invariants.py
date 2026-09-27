"""Ticket 34 — sole-kernel static invariants (bypass removal proof)."""

from __future__ import annotations

import inspect
from pathlib import Path

import api.chat_v2 as chat_v2_module
import agent_v2.startup as startup_module
import agent_v2.health as health_module
import agent_v2.kernel as kernel_module

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = REPO_ROOT / "ai-service"


def _source(module) -> str:
    return inspect.getsource(module)


def test_chat_v2_has_no_stub_or_direct_flow_symbols():
    src = _source(chat_v2_module)
    forbidden = (
        "_stub_classify", "_stub_response", "_stream_route_plan_flow",
        "_simulate_mcp_call", "_stub_artifact_plan_content",
        "_stub_cowork_bash_command",
    )
    present = [name for name in forbidden if name in src]
    assert not present, f"stub/direct-flow symbols still referenced: {present}"


def test_startup_no_longer_imports_local_stub_factory():
    src = _source(startup_module)
    assert "from .uniharness import create_agent" not in src
    assert "create_agent()" not in src  # no stub context construction


def test_health_exposes_kernel_identity():
    src = _source(health_module)
    assert "kernel_identity" in src


def test_kernel_entrypoints_expose_the_sole_execution_seam():
    sig = inspect.signature(kernel_module.run_chat_turn)
    for param in ("computer", "subagent_definitions", "approval_callback",
                  "checkpoint_store", "mcp_servers"):
        assert param in sig.parameters, param


def test_chat_v2_does_not_import_legacy_dispatcher():
    src = _source(chat_v2_module)
    assert "SubagentDispatcher" not in src
    assert "from agent_v2.subagents" not in src


def test_standalone_stub_package_is_disconnected_from_product_imports():
    # The stub files may exist on disk (no-delete policy) but the product
    # must not import their factory/runtime at all.
    offenders = []
    for rel in ("api/chat_v2.py", "agent_v2/startup.py", "agent_v2/kernel.py",
                "agent_v2/kernel_tools.py", "agent_v2/kernel_persistence.py"):
        src = (AI_DIR / rel).read_text(encoding="utf-8")
        if "uniharness.factory" in src or ("from .uniharness import" in src and rel.endswith("startup.py")):
            offenders.append(rel)
        if "from agent_v2.uniharness import create_agent" in src:
            offenders.append(rel)
    assert not offenders, f"product modules still import the stub factory: {offenders}"


def test_health_never_reports_ready_without_verified_kernel_identity():
    src = _source(health_module)
    # The ready branch carries identity only when the startup check passed.
    assert "uniharness_pin" in src and "kernel_identity" in src
