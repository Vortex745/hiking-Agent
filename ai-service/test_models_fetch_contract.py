"""Model discovery with a stored section key.

The config page never echoes stored keys, so the decision model's "获取模型"
must be able to ask the backend to use its stored key. That is only safe for
loopback callers addressing the SAME stored host; otherwise it would forward
the stored credential to an attacker-controlled base URL.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.models_router as models_router
from config import settings

BASE_URL = "/api/v1/models/fetch"


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = (
            payload
            if payload is not None
            else {"data": [{"id": "Kev-4B"}, {"id": "SemIf"}]}
        )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    response = _FakeResponse()
    last_request: dict = {}

    def __init__(self, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, headers=None):
        type(self).last_request = {
            "url": url,
            "headers": headers,
            "timeout": self.timeout,
        }
        return type(self).response


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(models_router.models_router, prefix="/api/v1")
    return app


@pytest.fixture(autouse=True)
def _restore_process_state(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(
        settings, "decision_model_base_url", "https://api.siliconflow.cn/v1"
    )
    monkeypatch.setattr(settings, "decision_model_api_key", "sk-stored-decision")


@pytest.fixture()
def fake_upstream(monkeypatch):
    monkeypatch.setattr(models_router.socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("8.8.8.8", 443))])
    _FakeAsyncClient.response = _FakeResponse()
    _FakeAsyncClient.last_request = {}
    monkeypatch.setattr(models_router.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.fixture()
def loopback_client():
    with TestClient(_app(), client=("127.0.0.1", 50000)) as client:
        yield client


@pytest.fixture()
def remote_client():
    with TestClient(_app(), client=("10.0.0.5", 50000)) as client:
        yield client


def test_stored_decision_key_used_for_matching_base_url(
    loopback_client, fake_upstream
):
    response = loopback_client.post(
        BASE_URL,
        json={
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "section": "decision",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"models": ["Kev-4B", "SemIf"]}
    assert "sk-stored-decision" not in response.text
    assert fake_upstream.last_request["url"] == (
        "https://api.siliconflow.cn/v1/models"
    )
    assert fake_upstream.last_request["headers"] == {
        "Authorization": "Bearer sk-stored-decision"
    }
    assert fake_upstream.last_request["timeout"] == 15.0


def test_stored_key_not_forwarded_to_a_different_base_url(
    loopback_client, fake_upstream
):
    response = loopback_client.post(
        BASE_URL,
        json={
            "base_url": "https://attacker.example/v1",
            "api_key": "",
            "section": "decision",
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "base_url" in detail
    assert "api_key" in detail
    assert "sk-stored-decision" not in response.text
    assert fake_upstream.last_request == {}


def test_stored_key_requires_a_loopback_caller(remote_client, fake_upstream):
    response = remote_client.post(
        BASE_URL,
        json={
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "section": "decision",
        },
    )

    assert response.status_code == 403
    assert fake_upstream.last_request == {}


def test_empty_key_without_section_is_rejected(loopback_client, fake_upstream):
    response = loopback_client.post(
        BASE_URL,
        json={"base_url": "https://api.siliconflow.cn/v1", "api_key": ""},
    )

    assert response.status_code == 400
    assert "api_key" in response.json()["detail"]
    assert fake_upstream.last_request == {}


def test_missing_stored_key_is_rejected(loopback_client, fake_upstream, monkeypatch):
    monkeypatch.setattr(settings, "decision_model_api_key", "")

    response = loopback_client.post(
        BASE_URL,
        json={
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "section": "decision",
        },
    )

    assert response.status_code == 400
    assert "decision.api_key" in response.json()["detail"]
    assert fake_upstream.last_request == {}


def test_explicit_key_is_forwarded_even_from_a_remote_client(
    remote_client, fake_upstream
):
    response = remote_client.post(
        BASE_URL,
        json={
            "base_url": "https://provider.example/v1",
            "api_key": "explicit-key",
            "section": "decision",
        },
    )

    assert response.status_code == 200
    assert fake_upstream.last_request["url"] == "https://provider.example/v1/models"
    assert fake_upstream.last_request["headers"] == {
        "Authorization": "Bearer explicit-key"
    }
    assert "sk-stored-decision" not in str(fake_upstream.last_request)


def test_model_ids_are_sorted_and_invalid_entries_dropped(
    loopback_client, fake_upstream
):
    fake_upstream.response = _FakeResponse(
        payload={
            "data": [
                {"id": "z-model"},
                {"id": "a-model"},
                {"id": 123},
                {"id": ""},
                {},
            ]
        }
    )

    response = loopback_client.post(
        BASE_URL,
        json={
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "section": "decision",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"models": ["a-model", "z-model"]}


def test_empty_base_url_is_rejected(loopback_client, fake_upstream):
    response = loopback_client.post(
        BASE_URL,
        json={"base_url": "", "api_key": "explicit-key", "section": "decision"},
    )

    assert response.status_code == 400
    assert "base_url" in response.json()["detail"]
    assert fake_upstream.last_request == {}
