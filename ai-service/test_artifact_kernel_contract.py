"""Ticket 30 — run-scoped sandbox bridge + artifact kernel-turn contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent_v2.kernel_tools import (
    ProductSandboxBridge,
    SandboxReadTool,
    SandboxWriteTool,
)
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import api.chat_v2 as chat_v2_module


class _FakeProductComputer:
    """Product-side Computer double recording every executed command."""

    sandbox_type = "local_native"

    def __init__(self) -> None:
        self.commands: list[tuple[str, str]] = []
        self.files: dict[str, str] = {}

    def create_workspace(self, run_id: str) -> str:  # pragma: no cover - unused here
        return f"/ws/{run_id}"

    def cleanup(self, workspace_path: str):  # pragma: no cover
        return {"files_deleted": 0}

    def destroy_sandbox(self, workspace_path: str):  # alias name
        return self.cleanup(workspace_path)

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        self.files[filename] = content
        return filename

    def build_manifest(self, workspace_path: str, relative_path: str, declared_mime: str):
        raise NotImplementedError("manifest handled by patched exporter in E2E test")

    def read_artifact_bytes(self, workspace_path: str, relative_path: str) -> bytes:
        return self.files[relative_path].encode()

    def run_bash(self, workspace_path: str, command: str, *, timeout_seconds: float = 20.0, max_output_bytes: int = 1_048_576):
        self.commands.append((workspace_path, command))
        # Simulate `printf %s <b64> | base64 -d > path`
        if "> " in command and "| base64 -d" in command:
            payload = command.split("printf %s ", 1)[1].split("|", 1)[0].strip().strip("'")
            import base64 as _b

            body = _b.b64decode(payload).decode()
            target = command.split("> ", 1)[1].strip().strip("'")
            self.files[target] = body
            from types import SimpleNamespace

            return SimpleNamespace(exit_code=0, stdout="", stderr="", duration_ms=1, truncated=False)
        from types import SimpleNamespace

        if command.startswith("cat -- "):
            target = command[len("cat -- "):].strip().strip("'")
            from types import SimpleNamespace

            data = self.files.get(target, "")
            ok = target in self.files
            return SimpleNamespace(
                exit_code=0 if ok else 1,
                stdout=data if ok else "",
                stderr="" if ok else "No such file",
                duration_ms=1,
                truncated=False,
            )
        raise AssertionError(f"unexpected sandbox command executed: {command!r}")


def _write_tool(computer) -> tuple[SandboxWriteTool, list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    bridge = ProductSandboxBridge(computer, "/ws/run-1")
    return SandboxWriteTool(bridge=bridge, run_id="run-1", audit_sink=audit.append), audit


def test_sandbox_write_happy_path_executes_escaped_command():
    computer = _FakeProductComputer()
    tool, audit = _write_tool(computer)

    result = asyncio.run(tool(path="plans/day1.md", content="计划\n; rm -rf /"))

    assert result.output == str(len("计划\n; rm -rf /".encode()))
    ws, cmd = computer.commands[-1]
    assert ws == "/ws/run-1"
    # The raw dangerous text never appears as an unescaped shell fragment.
    assert "rm -rf /" not in cmd.split("| base64 -d")[0]
    assert "rm -rf /" not in "".join(cmd.split("'")[1::2][:2])
    assert "base64 -d" in cmd
    assert computer.files["plans/day1.md"].endswith("; rm -rf /")
    entry = audit[-1]
    assert entry["phase"] == "result" and entry["ok"] is True and entry["run_id"] == "run-1"


@pytest.mark.parametrize("bad", ["/etc/passwd", "../../etc/passwd", "..\\windows\\x"])
def test_sandbox_write_rejects_escape_paths_before_execution(bad: str):
    computer = _FakeProductComputer()
    tool, audit = _write_tool(computer)

    result = asyncio.run(tool(path=bad, content="pwned"))

    assert result.error and "越出当前工作区" in result.error
    assert computer.commands == []  # nothing executed
    denial = audit[-1]
    assert denial["code"] == "unsafe_path" and denial["ok"] is False


def test_bridge_rejects_cross_boundary_transfers(tmp_path: Path):
    class _Noop:
        def run_bash(self, *a: Any, **k: Any):  # pragma: no cover
            raise AssertionError("must not execute")

    bridge = ProductSandboxBridge(_Noop(), "/ws/x")

    with pytest.raises(Exception):
        asyncio.run(bridge.upload("/host/secret", "/ws/x"))
    with pytest.raises(Exception):
        asyncio.run(bridge.download("/ws/x/a.txt", "/host/out"))


def test_sandbox_read_reports_missing_file_with_public_error():
    computer = _FakeProductComputer()
    audit: list[dict[str, Any]] = []
    bridge = ProductSandboxBridge(computer, "/ws/run-2")
    tool = SandboxReadTool(bridge=bridge, run_id="run-2", audit_sink=audit.append)

    result = asyncio.run(tool(path="missing.md"))

    assert result.error and "读取失败" in result.error
    assert "No such file" not in (result.error or "")  # raw stderr not leaked


# ---------------------------------------------------------------------------
# HTTP/SSE integration — real flow with the kernel turn stubbed at its seam
# ---------------------------------------------------------------------------


def _build_app(tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.runs import RunRepository

    repo = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )

    class _FakeRuntime:
        runs = repo
        memory = None
        telemetry = None
        capacity = None
        kernel_checkpoints = None

    app.state.agent_v2_runtime = _FakeRuntime()
    return app


async def _post(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        async with client.stream("POST", "/api/v2/chat/sse", json=payload) as resp:
            assert resp.status_code == 200
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
    return events


def test_artifact_sse_sequence_preserved_and_export_gated(tmp_path: Path, monkeypatch):
    """kernel turn feeds plan content; candidate/export/available stay gated."""
    monkeypatch.setattr(chat_v2_module.settings, "agent_v2_enabled", True)

    async def fake_turn(**kwargs: Any):
        audits = [
            {"tool": "sandbox_read", "phase": "denied", "ok": False, "code": "unsafe_path"},
        ]
        return "# 徒步计划\n内容由内核生成", audits

    from agent_v2 import kernel as km

    monkeypatch.setattr(km, "run_chat_turn", fake_turn)

    # Deterministic export: skip filesystem work, prove ordering + payload
    # shape. Ticket 41: the production export lives in agent_v2.artifacts
    # and the orchestrator resolves it lazily at call time, so patching
    # the module attribute is the stable seam.
    import agent_v2.artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "export_artifact_to_host", lambda **kw: {
        "artifact_id": "art-1",
        "public_filename": "hiking-plan.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 10,
        "sha256": "deadbeef",
        "expires_at": "2026-09-03T00:00:00+00:00",
    })

    class _PdfComputer(_FakeProductComputer):
        def create_workspace(self, run_id: str) -> str:
            return f"/ws/{run_id}"

        def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
            self.files[filename] = content
            return filename

    pdf = _PdfComputer()
    monkeypatch.setattr(chat_v2_module, "_build_artifact_computer", lambda *a, **k: pdf)
    # ALR ticket 03: the production cleanup adapter runs unpatched — it
    # translates the fake Computer's dict usage into the typed success
    # result and no longer performs any lease mutation of its own.

    # generate_pdf is HIGH-risk in the product table; the old flow pauses for
    # approval. This ticket keeps that semantics — the SSE test stubs the
    # gate to an ALLOW verdict so the downstream sequence is exercised.
    from agent_v2 import permissions as perms

    class _FakeGate:
        def evaluate(self, profile, tool_name, args=None):
            return perms.PermissionVerdict(
                decision=perms.PermissionDecision.ALLOW,
                risk=perms.RiskLevel.LOW,
                reason="allow",
            )

    monkeypatch.setattr(chat_v2_module, "PermissionGate", _FakeGate)

    app = _build_app(tmp_path)
    events = asyncio.run(_post(app, {
        "message": "规划杭州两日徒步并出 PDF 计划",
        "request_id": "req-art-kernel-1",
        "profile_id": "artifact",
    }))

    types = [e["type"] for e in events]
    assert "artifact_candidate" in types and "artifact.available" in types
    available = next(e for e in events if e["type"] == "artifact.available")
    assert available["payload"]["artifact_id"] == "art-1"
    assert "size_bytes" in available["payload"]
    terminals = [t for t in types if t in ("done", "error", "cancelled")]
    assert terminals[-1] == "done"

    seqs = [e["sequence"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    denied = next(e for e in events if e.get("type") == "tool_result" and e["payload"].get("status") == "denied")
    assert denied["payload"]["reason"] == "unsafe_path"
