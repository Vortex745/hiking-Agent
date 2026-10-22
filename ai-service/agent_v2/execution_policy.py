"""BCDEF ticket 08 — admission-time execution policy snapshot.

The policy frozen here is the run's whole execution contract: profile,
model identity, budget, capability inputs (tools / MCP / subagents),
lifecycle settings, effect/approval policy, schema version and source
version. It is built ONCE at admission from the current server config and
persisted in ``agent_control.sqlite.policy_snapshots`` (one row per run).

Execution reconstructs its profile and settings view from the STORED
snapshot, never from current config. Secrets (API keys, base URLs with
credentials) are deliberately absent from the payload; only the model
name is recorded.

Ticket 09 (versioned resume adapters) and ticket 10 (capability
projection) consume :meth:`RunRepository.resolve_policy_snapshot`; this
module only defines the frozen shape and the reconstruction helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from agent_v2.execution_contract import (
    EXECUTION_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    BoundaryError,
    ExecutionErrorCode,
    policy_hash,
)
from agent_v2.permissions import approval_risk_threshold, tool_risk_policy
from agent_v2.profiles import CapabilityProfile, ModelProfile, ProfileId

# Version of the SOURCE configuration that produced the policy, distinct
# from the execution schema version. Bump when the policy builder starts
# reading a different config surface.
POLICY_SOURCE_VERSION = "agent_v2.execution_policy/v1"
POLICY_KIND = "execution"


def build_execution_policy(
    profile: CapabilityProfile,
    settings: Any,
) -> dict[str, Any]:
    """Freeze the complete execution policy for one admission.

    Pure: reads the supplied profile/settings once and returns a
    JSON-representable mapping. It never reads credentials into the
    result and never calls the model.
    """
    model = profile.model
    policy: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "source_version": POLICY_SOURCE_VERSION,
        "policy_kind": POLICY_KIND,
        "profile_id": profile.profile_id.value,
        "model": {
            # Identity of the model actually used by the kernel at
            # admission time. No api_key/base_url/secret is recorded.
            "provider_model": str(getattr(settings, "openai_model", "") or ""),
            "main_model": model.main_model,
            "fast_model": model.fast_model,
            "context_window": model.context_window,
        },
        "budget": {
            "compaction_threshold": model.compaction_threshold,
            "output_limit": model.output_limit,
        },
        "capability": {
            "allowed_tools": list(profile.allowed_tools),
            "allowed_mcp_servers": (
                None
                if profile.allowed_mcp_servers is None
                else list(profile.allowed_mcp_servers)
            ),
            "tool_groups": ["task"] if getattr(
                settings, "agent_v2_subagent_enabled", False
            ) else [],
            "skills": [],
            "subagent_enabled": bool(
                getattr(settings, "agent_v2_subagent_enabled", False)
            ),
        },
        "lifecycle": {
            "agent_v2_enabled": bool(
                getattr(settings, "agent_v2_enabled", False)
            ),
            "thoughts_enabled": bool(
                getattr(settings, "agent_v2_thoughts_enabled", True)
            ),
            "cowork_policy": str(
                getattr(settings, "agent_v2_cowork_policy", "disabled")
            ),
        },
        "effect_policy": {
            "approval_risk_threshold": approval_risk_threshold(),
            "tool_risk": tool_risk_policy(),
        },
        "approval_policy": {
            "requires_approval_at_or_above": approval_risk_threshold(),
        },
        "compatibility": {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "supported_schema_versions": sorted(SUPPORTED_SCHEMA_VERSIONS),
            "policy_kind": POLICY_KIND,
            "source_version": POLICY_SOURCE_VERSION,
        },
    }
    # Fail fast if a future field is not canonicalizable: the snapshot is
    # part of the admission commit gate, so a broken policy must never be
    # silently stored.
    policy_hash(policy)
    return policy


def serialize_policy(policy: Mapping[str, Any]) -> str:
    """Canonical JSON of the frozen policy (the exact hashed content)."""
    return json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def profile_from_policy(policy: Mapping[str, Any]) -> CapabilityProfile:
    """Rebuild the execution capability profile from a frozen policy.

    Raises :class:`BoundaryError` ``policy_incompatible`` when a required
    field is missing or malformed. No default profile is ever guessed.
    """
    try:
        profile_id = ProfileId(str(policy["profile_id"]))
        model = policy["model"]
        capability = policy["capability"]
        allowed_tools = capability["allowed_tools"]
        allowed_mcp = capability["allowed_mcp_servers"]
        if not isinstance(allowed_tools, list) or not all(
            isinstance(tool, str) for tool in allowed_tools
        ):
            raise TypeError("allowed_tools must be a list of strings")
        if allowed_mcp is not None and (
            not isinstance(allowed_mcp, list)
            or not all(isinstance(server, str) for server in allowed_mcp)
        ):
            raise TypeError("allowed_mcp_servers must be a list or None")
        return CapabilityProfile(
            profile_id=profile_id,
            model=ModelProfile(
                profile_id=profile_id,
                main_model=str(model["main_model"]),
                fast_model=str(model["fast_model"]),
                context_window=int(model["context_window"]),
                compaction_threshold=int(policy["budget"]["compaction_threshold"]),
                output_limit=int(policy["budget"]["output_limit"]),
            ),
            allowed_tools=tuple(allowed_tools),
            allowed_mcp_servers=(
                None if allowed_mcp is None else tuple(allowed_mcp)
            ),
        )
    except BoundaryError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise BoundaryError(
            ExecutionErrorCode.POLICY_INCOMPATIBLE,
            f"stored policy cannot be reconstructed: {exc}",
        ) from exc


class FrozenSettings:
    """Read-through settings proxy with the admission-time overrides.

    The kernel receives this object so a config hot-update after admission
    cannot change the run's model name, thoughts/subagent switches or
    cowork policy. Credentials and unrelated settings are forwarded from
    the live object; they are never copied into the policy snapshot.
    """

    __slots__ = ("_base", "_overrides")

    def __init__(self, base: Any, overrides: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_base"), name)

    @property
    def frozen_overrides(self) -> dict[str, Any]:
        return dict(object.__getattribute__(self, "_overrides"))


def freeze_settings(settings: Any, policy: Mapping[str, Any]) -> FrozenSettings:
    """Project the live settings object through the frozen policy."""
    try:
        model = policy["model"]
        lifecycle = policy["lifecycle"]
        capability = policy["capability"]
    except (KeyError, TypeError) as exc:
        raise BoundaryError(
            ExecutionErrorCode.POLICY_INCOMPATIBLE,
            f"stored policy is missing execution settings: {exc}",
        ) from exc
    return FrozenSettings(
        settings,
        {
            "openai_model": str(model.get("provider_model", "") or ""),
            "agent_v2_thoughts_enabled": bool(
                lifecycle.get("thoughts_enabled", True)
            ),
            "agent_v2_subagent_enabled": bool(
                capability.get("subagent_enabled", False)
            ),
            "agent_v2_cowork_policy": str(
                lifecycle.get("cowork_policy", "disabled")
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class ResolvedPolicySnapshot:
    """Validated view of one persisted policy snapshot."""

    run_id: str
    policy: Mapping[str, Any]
    policy_hash: str
    schema_version: int
    source_version: str
    created_at: str
    compatibility: Mapping[str, Any]

    def profile(self) -> CapabilityProfile:
        return profile_from_policy(self.policy)

    def assert_compatible(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise BoundaryError(
                ExecutionErrorCode.POLICY_INCOMPATIBLE,
                f"policy schema {self.schema_version} is not supported",
            )


__all__ = [
    "FrozenSettings",
    "POLICY_KIND",
    "POLICY_SOURCE_VERSION",
    "ResolvedPolicySnapshot",
    "build_execution_policy",
    "freeze_settings",
    "profile_from_policy",
    "serialize_policy",
]
