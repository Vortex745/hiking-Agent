"""OAuth token storage for user-added HTTP MCP servers.

Tokens and dynamic-registration client credentials are persisted per
connection fingerprint under ``agent_state/mcp_oauth.json``. The file is
local-only infrastructure: it is never exposed through :func:`public_server`
and never echoed in HTTP responses or logs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken
from pydantic import AnyUrl
import httpx

from agent_v2.mcp_config_store import connection_fingerprint
from config import settings

_LOCK = threading.Lock()
_DEFAULT_REDIRECT_URI = "http://127.0.0.1:8000/api/v2/mcp/oauth/callback"
logger = logging.getLogger(__name__)


def redirect_uri() -> str:
    return os.getenv("MCP_OAUTH_REDIRECT_URI", _DEFAULT_REDIRECT_URI)


def _path() -> Path:
    return Path(settings.agent_v2_state_dir) / "mcp_oauth.json"


def _read() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"version": 1, "servers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "servers": {}}
    if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
        return {"version": 1, "servers": {}}
    return data


def _write(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry(data: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    servers = data.setdefault("servers", {})
    entry = servers.get(fingerprint)
    if not isinstance(entry, dict):
        entry = {}
        servers[fingerprint] = entry
    return entry


def _touch(entry: dict[str, Any]) -> None:
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()


def _token_age_seconds(updated_at: Any) -> float | None:
    if not isinstance(updated_at, str) or not updated_at:
        return None
    try:
        issued = datetime.fromisoformat(updated_at)
    except ValueError:
        return None
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - issued).total_seconds())


class FileTokenStorage(TokenStorage):
    """TokenStorage implementation keyed by the server connection fingerprint."""

    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    async def get_tokens(self) -> OAuthToken | None:
        entry = _read().get("servers", {}).get(self._fingerprint, {})
        raw = entry.get("tokens")
        if not raw:
            return None
        token = OAuthToken.model_validate(raw)
        # The SDK treats ``expires_in`` as "seconds from now" on every load,
        # so a persisted token would look fresh forever. Compare against the
        # stored issue time and force the SDK down the refresh path once the
        # lifetime has elapsed.
        age = _token_age_seconds(entry.get("updated_at"))
        if age is not None and token.expires_in is not None and age >= max(0, token.expires_in - 60):
            return token.model_copy(update={"expires_in": -1})
        return token

    async def set_tokens(self, tokens: OAuthToken) -> None:
        with _LOCK:
            data = _read()
            entry = _entry(data, self._fingerprint)
            entry["tokens"] = tokens.model_dump(mode="json")
            _touch(entry)
            _write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = _read().get("servers", {}).get(self._fingerprint, {}).get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        with _LOCK:
            data = _read()
            entry = _entry(data, self._fingerprint)
            entry["client_info"] = client_info.model_dump(mode="json")
            _touch(entry)
            _write(data)


def save_authorization_server(fingerprint: str, metadata: OAuthMetadata | None) -> None:
    if metadata is None:
        return
    with _LOCK:
        data = _read()
        entry = _entry(data, fingerprint)
        entry["authorization_server"] = metadata.model_dump(mode="json")
        _write(data)


def load_authorization_server(fingerprint: str) -> OAuthMetadata | None:
    raw = _read().get("servers", {}).get(fingerprint, {}).get("authorization_server")
    return OAuthMetadata.model_validate(raw) if raw else None


async def _fetch_authorization_server(issuer: str) -> OAuthMetadata | None:
    base = issuer.rstrip("/")
    urls = (
        f"{base}/.well-known/oauth-authorization-server",
        f"{base}/.well-known/openid-configuration",
    )
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for url in urls:
            try:
                response = await client.get(url)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                metadata = OAuthMetadata.model_validate_json(response.content)
            except ValueError:
                continue
            if str(metadata.issuer).rstrip("/") != base:
                continue
            return metadata
    return None


async def _refresh_expired_token(fingerprint: str) -> None:
    """Refresh a persisted token before the SDK sees it.

    The SDK only tracks expiry for tokens it obtained in-process; a token
    loaded from storage is assumed valid until the server answers 401, and
    its 401 path goes straight to interactive authorization. Refresh here so
    an expired access token is renewed against the cached token endpoint.
    """
    entry = _read().get("servers", {}).get(fingerprint, {})
    raw = entry.get("tokens") or {}
    if not raw.get("refresh_token"):
        return
    expires_in = raw.get("expires_in")
    age = _token_age_seconds(entry.get("updated_at"))
    if age is None or not isinstance(expires_in, int) or age < max(0, expires_in - 60):
        return
    storage = FileTokenStorage(fingerprint)
    client_info = await storage.get_client_info()
    if client_info is None or not client_info.client_id:
        return
    metadata = load_authorization_server(fingerprint)
    if metadata is None and client_info.issuer:
        metadata = await _fetch_authorization_server(str(client_info.issuer))
        if metadata is not None:
            save_authorization_server(fingerprint, metadata)
    if metadata is None:
        return
    data = {
        "grant_type": "refresh_token",
        "refresh_token": raw["refresh_token"],
        "client_id": client_info.client_id,
    }
    auth: httpx.BasicAuth | None = None
    if client_info.client_secret:
        if client_info.token_endpoint_auth_method == "client_secret_basic":
            auth = httpx.BasicAuth(client_info.client_id, client_info.client_secret)
        elif client_info.token_endpoint_auth_method == "client_secret_post":
            data["client_secret"] = client_info.client_secret
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(str(metadata.token_endpoint), data=data, auth=auth)
    except httpx.HTTPError:
        logger.warning("MCP OAuth refresh request failed for %s", fingerprint[:8])
        return
    if response.status_code != 200:
        logger.warning(
            "MCP OAuth refresh rejected for %s: HTTP %s",
            fingerprint[:8], response.status_code,
        )
        return
    try:
        token = OAuthToken.model_validate(response.json())
    except ValueError:
        logger.warning("MCP OAuth refresh returned an invalid token body for %s", fingerprint[:8])
        return
    await storage.set_tokens(token)


async def build_oauth_provider(
    server: dict[str, Any],
    *,
    redirect_handler: Callable[[str], Awaitable[None]] | None = None,
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None,
) -> OAuthClientProvider:
    """Build a provider with the authorization-server endpoint cached.

    The SDK's refresh path runs before any 401 discovery, so without a cached
    ``token_endpoint`` it would POST the refresh token to the MCP resource
    origin (or fall back to full authorization without a redirect handler).
    Cache the AS metadata from a prior authorization (or fetch it from the
    recorded issuer) so an expired access token refreshes correctly.
    """
    fingerprint = connection_fingerprint(server)
    await _refresh_expired_token(fingerprint)
    provider = OAuthClientProvider(
        server_url=server["url"],
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl(redirect_uri())],
            client_name="Hiking AI",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=FileTokenStorage(fingerprint),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=300.0,
    )
    cached = load_authorization_server(fingerprint)
    if cached is None:
        client_info = await provider.context.storage.get_client_info()
        if client_info is not None and client_info.issuer:
            cached = await _fetch_authorization_server(str(client_info.issuer))
            if cached is not None:
                save_authorization_server(fingerprint, cached)
    if cached is not None:
        provider.context.oauth_metadata = cached
        provider.context.auth_server_url = str(cached.issuer)
    return provider


async def oauth_auth(server: dict[str, Any]) -> OAuthClientProvider | None:
    """Runtime auth for HTTP servers: reuse stored tokens, refresh when expired.

    Without stored tokens the provider stays passive for servers that do not
    require authorization; if such a server answers 401 the connection fails
    closed instead of prompting (authorization only runs through the explicit
    /mcp/oauth flow in the management UI).
    """
    if server.get("transport") != "http" or not server.get("url"):
        return None
    return await build_oauth_provider(server)


def prune_oauth_tokens(keep_fingerprints: set[str]) -> None:
    with _LOCK:
        data = _read()
        servers = data.get("servers", {})
        stale = [key for key in servers if key not in keep_fingerprints]
        if not stale:
            return
        for key in stale:
            servers.pop(key, None)
        _write(data)


def clear_oauth_tokens(fingerprint: str) -> None:
    with _LOCK:
        data = _read()
        servers = data.get("servers", {})
        if servers.pop(fingerprint, None) is None:
            return
        _write(data)
