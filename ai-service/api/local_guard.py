"""Local-only guard shared by credential-management routes."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_ORIGIN_SCHEMES = frozenset({"http", "https"})
_CLIENT_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


def _is_allowed_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in _ALLOWED_ORIGIN_SCHEMES
        and parsed.hostname in _LOOPBACK_HOSTS
    )


def require_local_request(request: Request) -> None:
    """Reject non-loopback clients, foreign Origins and Vercel."""
    if os.getenv("VERCEL"):
        raise HTTPException(status_code=403, detail="该接口在 Vercel 环境不可用")

    client = request.client
    if client is None or client.host not in _CLIENT_LOOPBACK_HOSTS:
        raise HTTPException(status_code=403, detail="该接口仅允许本机访问")

    origin = request.headers.get("origin")
    if origin is not None and not _is_allowed_origin(origin):
        raise HTTPException(status_code=403, detail="Origin 必须为本机 http(s) 来源")
