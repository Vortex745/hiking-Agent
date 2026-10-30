"""Local MCP management through real stdio and streamable HTTP servers."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.mcp_router import mcp_router
from config import settings


SERVER = Path(__file__).parent / "test_fixtures" / "mcp_server.py"
SECRET = "mcp-contract-secret"


@pytest.fixture
def http_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MCP_TEST_EFFECT_FILE=str(tmp_path / "http-effects"), MCP_TEST_SECRET=SECRET)
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "--http-port", str(port)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("HTTP MCP fixture did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_discovery_persistence_and_local_boundary(tmp_path, monkeypatch, http_server, transport, capsys):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v2")
    body = {
        "name": "测试服务", "transport": transport,
        "url": http_server if transport == "http" else "",
        "command": sys.executable if transport == "stdio" else "",
        "args": [str(SERVER)] if transport == "stdio" else [],
        "headers": {"Authorization": SECRET} if transport == "http" else {},
        "env": {"MCP_TEST_SECRET": SECRET, "MCP_TEST_EFFECT_FILE": str(tmp_path / "stdio-effects")}
        if transport == "stdio" else {},
        "stdio_confirmed": True, "enabled": True,
    }

    async def scenario():
        async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 1234)), base_url="http://localhost") as client:
            if transport == "stdio":
                response = await client.post("/api/v2/mcp/servers/test", json={**body, "stdio_confirmed": False})
                assert response.status_code == 400
            tested = await client.post("/api/v2/mcp/servers/test", json=body)
            assert tested.status_code == 200, tested.text
            assert set(tested.json()["tools"]) == {"echo", "search_images"}
            created = await client.post("/api/v2/mcp/servers", json={**body, "test_id": tested.json()["test_id"]})
            assert created.status_code == 200, created.text
            server_id = created.json()["id"]
            listing = await client.get("/api/v2/mcp/servers")
            assert listing.status_code == 200
            assert listing.json()["servers"][-1]["status"] == "success"
            assert listing.json()["servers"][-1]["enabled"] is True
            assert any(server["builtin"] for server in listing.json()["servers"][:-1])
            assert SECRET not in listing.text
            assert SECRET not in created.text
            blocked = await client.get("/api/v2/mcp/servers", headers={"Origin": "https://example.com"})
            assert blocked.status_code == 403
            assert (await client.delete(f"/api/v2/mcp/servers/{server_id}")).status_code == 200
        async with AsyncClient(transport=ASGITransport(app=app, client=("192.0.2.1", 1234)), base_url="http://localhost") as remote:
            assert (await remote.get("/api/v2/mcp/servers")).status_code == 403

    asyncio.run(scenario())
    assert SECRET not in capsys.readouterr().err


def test_edit_without_credentials_and_failed_test_status(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v2")

    async def scenario():
        async with AsyncClient(transport=ASGITransport(app=app, client=("127.0.0.1", 1234)), base_url="http://localhost") as client:
            body = {"name": "offline", "transport": "http", "url": "http://127.0.0.1:1/mcp"}
            created = (await client.post("/api/v2/mcp/servers", json=body)).json()
            server_id = created["id"]
            changed = {**body, "url": "http://127.0.0.1:2/mcp", "headers": None, "env": None}
            assert (await client.put(f"/api/v2/mcp/servers/{server_id}", json=changed)).status_code == 200
            failed = await client.post("/api/v2/mcp/servers/test", json={**changed, "server_id": server_id})
            assert failed.status_code == 502
            listing = (await client.get("/api/v2/mcp/servers")).json()
            assert listing["servers"][-1]["status"] == "failed"

    asyncio.run(scenario())
