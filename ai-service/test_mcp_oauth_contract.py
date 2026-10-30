"""End-to-end OAuth authorization contract for HTTP MCP servers.

The fixture runs two real uvicorn servers on loopback:

* a FastMCP server gated by a bearer token, with local OAuth discovery,
  dynamic registration, authorize and token endpoints;
* the product's MCP router mounted in a minimal API app.

The test drives the same steps a browser would: test -> 401 challenge ->
authorize redirect -> loopback callback -> token exchange -> reconnect.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from api.mcp_router import mcp_router
from config import settings

ACCESS_TOKEN = "oauth-contract-access-token"


async def _protected_resource_metadata(request: Request) -> JSONResponse:
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "scopes_supported": ["mcp"],
        }
    )


async def _authorization_server_metadata(request: Request) -> JSONResponse:
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
        }
    )


async def _register(request: Request) -> JSONResponse:
    body = await request.json()
    body.setdefault("redirect_uris", [])
    body["client_id"] = "oauth-contract-client"
    body["token_endpoint_auth_method"] = "none"
    return JSONResponse(body)


async def _token(request: Request) -> JSONResponse:
    await request.form()
    return JSONResponse(
        {
            "access_token": ACCESS_TOKEN,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "mcp",
        }
    )


class _BearerGate:
    """Minimal ASGI gate: 401 with resource metadata until the bearer is valid."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
            if headers.get("authorization") != f"Bearer {ACCESS_TOKEN}":
                host, port = scope["server"]
                base = f"{scope.get('scheme', 'http')}://{host}:{port}"
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (
                                b"www-authenticate",
                                f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'.encode(),
                            ),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"error":"invalid_token"}'})
                return
        await self.app(scope, receive, send)


def _fake_oauth_mcp_app(*, drop_state: bool = False) -> Starlette:
    server = FastMCP("oauth-contract", host="127.0.0.1")

    @server.tool()
    def echo(text: str) -> str:
        """Echo the provided text."""
        return f"echo:{text}"

    mcp_app = server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        async with server.session_manager.run():
            yield

    async def authorize(request: Request) -> RedirectResponse:
        redirect_uri = request.query_params["redirect_uri"]
        separator = "&" if "?" in redirect_uri else "?"
        location = f"{redirect_uri}{separator}code=contract-code"
        if not drop_state:
            location += f"&state={quote(request.query_params.get('state', ''), safe='')}"
        return RedirectResponse(location, status_code=302)

    return Starlette(
        routes=[
            Route("/.well-known/oauth-protected-resource/mcp", _protected_resource_metadata),
            Route("/.well-known/oauth-authorization-server", _authorization_server_metadata),
            Route("/register", _register, methods=["POST"]),
            Route("/authorize", authorize),
            Route("/token", _token, methods=["POST"]),
            Mount("/", app=_BearerGate(mcp_app)),
        ],
        lifespan=lifespan,
    )


class _UvicornThread:
    def __init__(self, app) -> None:
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="on")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.base = ""

    def start(self) -> "_UvicornThread":
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("uvicorn did not start")
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def test_http_mcp_oauth_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    fake = _UvicornThread(_fake_oauth_mcp_app()).start()
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v2")
    api = _UvicornThread(app).start()
    monkeypatch.setenv("MCP_OAUTH_REDIRECT_URI", f"{api.base}/api/v2/mcp/oauth/callback")
    try:
        with httpx.Client(timeout=30) as client:
            body = {"name": "oauth", "transport": "http", "url": f"{fake.base}/mcp"}

            tested = client.post(f"{api.base}/api/v2/mcp/servers/test", json=body)
            assert tested.status_code == 200, tested.text
            payload = tested.json()
            assert payload["status"] == "auth_required"
            assert payload["auth_url"].startswith(f"{fake.base}/authorize?")

            page = client.get(payload["auth_url"], follow_redirects=True)
            assert page.status_code == 200
            assert "授权完成" in page.text

            status: dict = {}
            for _ in range(100):
                status = client.get(
                    f"{api.base}/api/v2/mcp/oauth/status/{payload['flow_id']}"
                ).json()
                if status["status"] in {"success", "error"}:
                    break
                time.sleep(0.1)
            assert status.get("status") == "success", status
            assert status["tools"] == ["echo"]

            saved = client.post(
                f"{api.base}/api/v2/mcp/servers", json={**body, "test_id": status["test_id"]}
            )
            assert saved.status_code == 200, saved.text
            server_id = saved.json()["id"]

            again = client.post(
                f"{api.base}/api/v2/mcp/servers/test", json={**body, "server_id": server_id}
            )
            assert again.status_code == 200, again.text
            assert again.json()["status"] == "success"

            stored = json.loads((tmp_path / "mcp_oauth.json").read_text(encoding="utf-8"))
            assert stored["servers"]
            assert ACCESS_TOKEN in json.dumps(stored)

            assert client.delete(f"{api.base}/api/v2/mcp/servers/{server_id}").status_code == 200
            assert json.loads((tmp_path / "mcp_oauth.json").read_text(encoding="utf-8"))["servers"] == {}
    finally:
        api.stop()
        fake.stop()


def test_http_mcp_oauth_flow_without_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    fake = _UvicornThread(_fake_oauth_mcp_app(drop_state=True)).start()
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v2")
    api = _UvicornThread(app).start()
    monkeypatch.setenv("MCP_OAUTH_REDIRECT_URI", f"{api.base}/api/v2/mcp/oauth/callback")
    try:
        with httpx.Client(timeout=30) as client:
            body = {"name": "oauth-nostate", "transport": "http", "url": f"{fake.base}/mcp"}
            tested = client.post(f"{api.base}/api/v2/mcp/servers/test", json=body)
            assert tested.status_code == 200, tested.text
            payload = tested.json()
            assert payload["status"] == "auth_required"

            page = client.get(payload["auth_url"], follow_redirects=True)
            assert page.status_code == 200
            assert "授权完成" in page.text, page.text

            status: dict = {}
            for _ in range(100):
                status = client.get(
                    f"{api.base}/api/v2/mcp/oauth/status/{payload['flow_id']}"
                ).json()
                if status["status"] in {"success", "error"}:
                    break
                time.sleep(0.1)
            assert status.get("status") == "success", status
            assert status["tools"] == ["echo"]
    finally:
        api.stop()
        fake.stop()


def test_oauth_callback_rejects_unknown_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    monkeypatch.setattr(settings, "agent_v2_sandbox_type", "local_native")
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v2")
    api = _UvicornThread(app).start()
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(
                f"{api.base}/api/v2/mcp/oauth/callback", params={"state": "missing", "code": "x"}
            )
            assert response.status_code == 400
    finally:
        api.stop()


def test_expired_persisted_token_is_forced_stale(tmp_path, monkeypatch):
    import asyncio
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(settings, "agent_v2_state_dir", str(tmp_path))
    from agent_v2.mcp_oauth_store import FileTokenStorage

    def write(updated_at: str) -> None:
        (tmp_path / "mcp_oauth.json").write_text(
            json.dumps({
                "version": 1,
                "servers": {
                    "fp": {
                        "tokens": {
                            "access_token": "access",
                            "token_type": "Bearer",
                            "expires_in": 3600,
                            "refresh_token": "refresh",
                        },
                        "updated_at": updated_at,
                    }
                },
            }),
            encoding="utf-8",
        )

    now = datetime.now(timezone.utc)
    write((now - timedelta(hours=2)).isoformat())
    expired = asyncio.run(FileTokenStorage("fp").get_tokens())
    assert expired is not None and expired.expires_in == -1

    write(now.isoformat())
    fresh = asyncio.run(FileTokenStorage("fp").get_tokens())
    assert fresh is not None and fresh.expires_in == 3600


def test_oauth_errors_are_recognized_through_exception_groups():
    from mcp.client.auth import OAuthFlowError

    from agent_v2.custom_mcp_tools import _is_oauth_error

    assert _is_oauth_error(OAuthFlowError("no handler"))
    assert _is_oauth_error(ExceptionGroup("group", [OAuthFlowError("no handler")]))
    assert not _is_oauth_error(RuntimeError("other"))
