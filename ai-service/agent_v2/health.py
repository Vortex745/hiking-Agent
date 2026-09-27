"""Chat v2 health status reporter.

The health endpoint is the only public surface where v2 state is visible
during the disabled-default period. The shape is intentionally simple so
the Gateway and monitoring can consume it without parsing logs.

Ticket 06 adds:

* ``persistence`` — whether the SQLite stores + writer lock are owned by
  this process. Operators use this to confirm single-writer ownership
  before routing v2 traffic.
* ``writer_id`` — UUID of the current writer. Survives process restart
  only if the OS lock is also re-acquired; mismatches between health
  reports from two processes signal a split-brain that the Gateway must
  not route to.
"""

from __future__ import annotations

from typing import Any

from agent_v2.startup import V2Runtime


def v2_health_status(app_state: Any, settings: Any) -> dict[str, Any]:
    """Return the v2 portion of the health payload.

    States:

    * ``disabled`` — v2 is off (default). No checkpoint, sandbox, or MCP.
    * ``ready`` — v2 is enabled, runtime assembled, writer lock held.
    * ``failed`` — v2 is enabled but assembly failed; callers must not
      route v2 traffic.

    The ``uniharness_pin`` field makes the dependency pin auditable from
    the health endpoint without inspecting the filesystem.
    """
    enabled = bool(getattr(settings, "agent_v2_enabled", False))
    runtime = getattr(app_state, "agent_v2_runtime", None)

    if not enabled:
        status = "disabled"
    elif runtime is not None:
        status = "ready"
    else:
        status = "failed"

    payload: dict[str, Any] = {
        "enabled": enabled,
        "status": status,
        "uniharness_pin": getattr(settings, "agent_v2_uniharness_pin", "unknown"),
    }

    # Ticket 25: surface the proven kernel identity so health can never be
    # mistaken for a stub build. A ready runtime always carries a verified
    # fork identity (startup fails closed otherwise).
    if isinstance(runtime, V2Runtime) and runtime.kernel_identity is not None:
        identity = runtime.kernel_identity
        payload["kernel_identity"] = {
            "distribution_version": identity.distribution_version,
            "fork_commit": identity.fork_commit,
            "compatible": identity.compatible,
        }

    # Ticket 06: persistence + writer lock audit fields. Only populate
    # when the runtime is the real V2Runtime (Ticket 06+); the Ticket 03
    # placeholder exposes neither.
    if isinstance(runtime, V2Runtime):
        payload["writer_id"] = runtime.writer_id
        payload["state_dir"] = runtime.state_dir
        payload["persistence"] = {
            "runs_repository": runtime.runs is not None,
            "writer_lock_held": runtime.writer_lock is not None,
        }
        if runtime.runs is not None:
            lock_row = runtime.runs.control.get_writer_lock()
            payload["persistence"]["writer_lock_row"] = lock_row
        sandbox_type = getattr(settings, "agent_v2_sandbox_type", "remote_e2b")
        payload["dependencies"] = {
            "model": {
                "status": "ready"
                if bool(getattr(settings, "openai_api_key", ""))
                else "unconfigured"
            },
            "mcp": {
                "status": "ready"
                if bool(getattr(settings, "mcp_servers", {}))
                else "disabled"
            },
            "e2b": {
                "status": "ready" if sandbox_type != "remote_e2b" else "selected"
            },
            "checkpoint": {
                "status": "ready"
                if runtime.runs is not None and runtime.runs.checkpoints is not None
                else "unavailable"
            },
            "control": {
                "status": "ready"
                if runtime.runs is not None and runtime.writer_lock is not None
                else "unavailable"
            },
            "memory": {
                "status": "ready" if runtime.memory is not None else "unavailable"
            },
            "artifact_store": {
                "status": "ready" if runtime.runs is not None else "unavailable"
            },
        }

    return payload
