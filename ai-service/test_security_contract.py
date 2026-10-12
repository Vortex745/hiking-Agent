"""Security contracts for externally reachable configuration surfaces."""

from __future__ import annotations

import pytest
import socket

from api.models_router import _validate_public_base_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "http://10.0.0.8/v1",
        "http://169.254.169.254/latest",
    ],
)
def test_models_fetch_rejects_private_or_local_urls(url: str) -> None:
    with pytest.raises(ValueError):
        _validate_public_base_url(url)


def test_models_fetch_rejects_credentials_and_query() -> None:
    with pytest.raises(ValueError):
        _validate_public_base_url("https://user:pass@example.com/v1")
    with pytest.raises(ValueError):
        _validate_public_base_url("https://example.com/v1?token=secret")


def test_models_fetch_accepts_public_https_origin(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443))])
    assert _validate_public_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_models_fetch_rejects_mixed_public_and_private_dns(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in ("93.184.215.14", "127.0.0.1")])
    with pytest.raises(ValueError, match="public address"):
        _validate_public_base_url("https://example.com/v1")
