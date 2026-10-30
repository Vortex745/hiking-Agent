"""Ticket 02: LLM config API (GET/PUT/test + local-only guard).

Hermetic: every test uses an in-process FastAPI app, a temporary ``.env``
and temporary vec0 databases; process-wide settings/os.environ mutations
are restored after each test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.llm_config_router as llm_config_router
import config
from agent_v2.decisions import DecisionCallError
from config import settings
from rag import sqlite_schema

BASE_URL = "/api/v1/llm-config"
TEST_URL = f"{BASE_URL}/test"

SETTINGS_ATTRS = (
    "openai_base_url",
    "openai_api_key",
    "openai_model",
    "embedding_base_url",
    "embedding_api_key",
    "embedding_model",
    "embedding_dimensions",
    "rerank_base_url",
    "rerank_api_key",
    "rerank_model",
    "decision_model_base_url",
    "decision_model_api_key",
    "decision_model_name",
    "decision_model_mode",
    "rag_sqlite_db_path",
)

ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "RERANK_BASE_URL",
    "RERANK_API_KEY",
    "RERANK_MODEL",
    "DECISION_MODEL_BASE_URL",
    "DECISION_MODEL_API_KEY",
    "DECISION_MODEL_NAME",
    "DECISION_MODEL_MODE",
)


@pytest.fixture(autouse=True)
def _restore_process_state(monkeypatch):
    """update_runtime_config mutates the global singleton and os.environ."""
    monkeypatch.delenv("VERCEL", raising=False)
    saved_settings = {name: getattr(settings, name) for name in SETTINGS_ATTRS}
    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    yield
    for name, value in saved_settings.items():
        setattr(settings, name, value)
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(llm_config_router.llm_config_router, prefix="/api/v1")
    return app


@pytest.fixture()
def app():
    return _build_app()


@pytest.fixture()
def client(app):
    """Happy-path client: the source guard is bypassed explicitly."""
    app.dependency_overrides[llm_config_router.require_local_request] = lambda: None
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


@pytest.fixture()
def guarded_client(app):
    """Client that keeps the real source guard active."""
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


def _write_env(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_index_db(tmp_path: Path, dimensions: int) -> Path:
    db_path = tmp_path / "rag.sqlite"
    conn = sqlite_schema.connect(db_path, embedding_dimensions=dimensions)
    conn.close()
    return db_path


def test_get_masks_keys_and_reports_env_metadata(tmp_path, client, monkeypatch):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=stored-model\n")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.example/v1")
    monkeypatch.setattr(settings, "openai_model", "llm-model")
    monkeypatch.setattr(settings, "openai_api_key", "sk-llm-1234567890")
    monkeypatch.setattr(settings, "embedding_base_url", "https://emb.example/v1")
    monkeypatch.setattr(settings, "embedding_model", "emb-model")
    monkeypatch.setattr(settings, "embedding_api_key", "short")
    monkeypatch.setattr(settings, "embedding_dimensions", 1024)
    monkeypatch.setattr(settings, "rerank_base_url", "https://rerank.example/v1")
    monkeypatch.setattr(settings, "rerank_model", "rerank-model")
    monkeypatch.setattr(settings, "rerank_api_key", "")
    monkeypatch.setattr(settings, "decision_model_base_url", "https://api.siliconflow.cn/v1")
    monkeypatch.setattr(settings, "decision_model_name", "Kev-4b")
    monkeypatch.setattr(settings, "decision_model_api_key", "sk-decision-1234567890")
    monkeypatch.setattr(settings, "decision_model_mode", "shadow")

    response = client.get(BASE_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["llm"] == {
        "base_url": "https://llm.example/v1",
        "model": "llm-model",
        "api_key_masked": "sk-l...7890",
        "api_key_set": True,
    }
    assert body["embedding"]["api_key_masked"] == "已设置"
    assert body["embedding"]["api_key_set"] is True
    assert body["embedding"]["dimensions"] == 1024
    assert body["rerank"]["api_key_masked"] == ""
    assert body["rerank"]["api_key_set"] is False
    assert body["decision"] == {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Kev-4b",
        "api_key_masked": "sk-d...7890",
        "api_key_set": True,
        "mode": "shadow",
    }
    assert body["index_dimensions"] == 1024
    assert body["updated_at"] == config.env_file_updated_at()
    assert body["updated_at"].endswith("Z")
    assert "sk-llm-1234567890" not in response.text


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("get", BASE_URL, {}),
        ("put", BASE_URL, {"json": {}}),
        ("post", TEST_URL, {"json": {}}),
    ],
)
def test_guard_rejects_non_loopback_client(method, path, kwargs, app):
    with TestClient(app, client=("10.0.0.5", 50000)) as test_client:
        response = getattr(test_client, method)(path, **kwargs)

    assert response.status_code == 403


def test_guard_rejects_disallowed_origin(guarded_client):
    response = guarded_client.get(BASE_URL, headers={"Origin": "https://evil.example"})

    assert response.status_code == 403


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://localhost:4173",
        "http://[::1]:8080",
        "http://localhost",
    ],
)
def test_guard_allows_loopback_origin_on_any_port(origin, guarded_client):
    response = guarded_client.get(BASE_URL, headers={"Origin": origin})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "not-a-url",
        "http://",
        "null",
        "ftp://localhost:5173",
        "http://localhost:bad-port",
        "http://localhost.evil.example:5174",
    ],
)
def test_guard_rejects_malformed_or_non_loopback_origin(origin, guarded_client):
    response = guarded_client.get(BASE_URL, headers={"Origin": origin})

    assert response.status_code == 403


def test_guard_rejects_vercel_environment(guarded_client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")

    response = guarded_client.get(BASE_URL)

    assert response.status_code == 403


def test_get_reports_null_metadata_without_env_or_index(
    tmp_path, client, monkeypatch
):
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(tmp_path / "missing.sqlite"))

    response = client.get(BASE_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["updated_at"] is None
    assert body["index_dimensions"] is None


def test_guard_allows_loopback_without_origin(guarded_client):
    response = guarded_client.get(BASE_URL)

    assert response.status_code == 200


def test_put_does_not_recommend_reindex_when_embedding_endpoint_is_unchanged(
    tmp_path, client, monkeypatch
):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=old-model\n")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    monkeypatch.setattr(settings, "embedding_base_url", "https://emb.new/v1")
    monkeypatch.setattr(settings, "embedding_model", "emb-new")
    payload = _valid_put_payload()
    payload["llm"]["model"] = "llm-renamed"

    response = client.put(BASE_URL, json=payload)

    assert response.status_code == 200
    assert response.json()["reindex_recommended"] is False


def _valid_put_payload(dimensions: int = 1024) -> dict:
    return {
        "llm": {
            "base_url": "https://llm.new/v1",
            "model": "llm-new",
            "api_key": "sk-llm-new-1234567890",
        },
        "embedding": {
            "base_url": "https://emb.new/v1",
            "model": "emb-new",
            "dimensions": dimensions,
            "api_key": "emb-new-123456",
        },
        "rerank": {
            "base_url": "https://rerank.new/v1",
            "model": "rerank-new",
        },
    }


def test_put_writes_env_preserves_comments_and_syncs_settings(
    tmp_path, client, monkeypatch
):
    env_path = _write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "# llm settings",
                "OPENAI_BASE_URL='https://old.example/v1'",
                "OPENAI_API_KEY='sk-old-llm'",
                "OPENAI_MODEL=old-model",
                "",
                "# unrelated block",
                "UNRELATED_KEY=keep-me",
                "EMBEDDING_DIMENSIONS=1024",
                "",
            ]
        ),
    )
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    monkeypatch.setattr(settings, "embedding_base_url", "https://emb.old/v1")
    monkeypatch.setattr(settings, "embedding_model", "emb-old")

    response = client.put(BASE_URL, json=_valid_put_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["llm"]["base_url"] == "https://llm.new/v1"
    assert body["llm"]["model"] == "llm-new"
    assert body["llm"]["api_key_masked"] == "sk-l...7890"
    assert body["embedding"]["dimensions"] == 1024
    assert body["index_dimensions"] == 1024
    assert body["updated_at"] is not None
    assert body["reindex_recommended"] is True
    assert "sk-llm-new-1234567890" not in response.text

    text = env_path.read_text(encoding="utf-8")
    assert "# llm settings" in text
    assert "# unrelated block" in text
    assert "UNRELATED_KEY=keep-me" in text
    values = dotenv_values(env_path)
    assert values["OPENAI_BASE_URL"] == "https://llm.new/v1"
    assert values["OPENAI_MODEL"] == "llm-new"
    assert values["OPENAI_API_KEY"] == "sk-llm-new-1234567890"
    assert values["EMBEDDING_MODEL"] == "emb-new"
    assert values["EMBEDDING_DIMENSIONS"] == "1024"
    assert values["RERANK_MODEL"] == "rerank-new"

    assert settings.openai_base_url == "https://llm.new/v1"
    assert settings.openai_model == "llm-new"
    assert settings.openai_api_key == "sk-llm-new-1234567890"
    assert settings.embedding_base_url == "https://emb.new/v1"
    assert settings.embedding_model == "emb-new"
    assert settings.embedding_dimensions == 1024
    assert settings.rerank_base_url == "https://rerank.new/v1"
    assert settings.rerank_model == "rerank-new"


def test_put_with_empty_api_key_keeps_stored_key(tmp_path, client, monkeypatch):
    env_path = _write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "OPENAI_BASE_URL='https://old.example/v1'",
                "OPENAI_API_KEY='sk-stored-llm-key'",
                "OPENAI_MODEL=old-model",
                "EMBEDDING_API_KEY='emb-stored-key'",
                "",
            ]
        ),
    )
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    settings.openai_api_key = "sk-stored-llm-key"
    settings.embedding_api_key = "emb-stored-key"

    response = client.put(
        BASE_URL,
        json={
            "llm": {
                "base_url": "https://old.example/v1",
                "model": "new-model",
                "api_key": "",
            },
            "embedding": {
                "base_url": "https://emb.example/v1",
                "model": "emb-model",
                "dimensions": 1024,
                "api_key": "   ",
            },
            "rerank": {
                "base_url": "https://rerank.example/v1",
                "model": "rerank-model",
            },
        },
    )

    assert response.status_code == 200
    values = dotenv_values(env_path)
    assert values["OPENAI_API_KEY"] == "sk-stored-llm-key"
    assert values["EMBEDDING_API_KEY"] == "emb-stored-key"
    assert values["OPENAI_MODEL"] == "new-model"
    assert settings.openai_api_key == "sk-stored-llm-key"
    assert settings.embedding_api_key == "emb-stored-key"
    assert response.json()["llm"]["api_key_masked"] == "sk-s...-key"


def test_put_rejects_dimension_mismatch_before_writing(tmp_path, client, monkeypatch):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=before-model\n")
    before = env_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    settings.openai_model = "settings-before"

    response = client.put(BASE_URL, json=_valid_put_payload(dimensions=1536))

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "1536" in detail
    assert "1024" in detail
    assert "需重建索引" in detail
    assert env_path.read_text(encoding="utf-8") == before
    assert settings.openai_model == "settings-before"


@pytest.mark.parametrize("dimensions", [63, 4097])
def test_put_rejects_dimensions_outside_range(
    tmp_path, client, monkeypatch, dimensions
):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=before-model\n")
    before = env_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))

    response = client.put(BASE_URL, json=_valid_put_payload(dimensions=dimensions))

    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == before


def test_put_rejects_blank_required_base_url(tmp_path, client, monkeypatch):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=before-model\n")
    before = env_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    payload = _valid_put_payload()
    payload["llm"]["base_url"] = "   "

    response = client.put(BASE_URL, json=payload)

    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == before


def test_put_rejects_missing_embedding_dimensions(tmp_path, client, monkeypatch):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=before-model\n")
    before = env_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    payload = _valid_put_payload()
    payload["embedding"].pop("dimensions")

    response = client.put(BASE_URL, json=payload)

    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == before


class _FakeUpstreamResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": []}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    response = _FakeUpstreamResponse()
    error: Exception | None = None
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
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


@pytest.fixture()
def fake_upstream(monkeypatch):
    _FakeAsyncClient.response = _FakeUpstreamResponse(
        payload={"data": [{"id": "m-a"}, {"id": "m-b"}, {"id": "m-c"}]}
    )
    _FakeAsyncClient.error = None
    _FakeAsyncClient.last_request = {}
    monkeypatch.setattr(llm_config_router.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.mark.parametrize(
    "section, base_attr, key_attr",
    [
        ("llm", "openai_base_url", "openai_api_key"),
        ("embedding", "embedding_base_url", "embedding_api_key"),
        ("rerank", "rerank_base_url", "rerank_api_key"),
    ],
)
def test_probe_falls_back_to_stored_base_url_and_key(
    section, base_attr, key_attr, tmp_path, client, monkeypatch, fake_upstream
):
    env_path = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, base_attr, f"https://{section}.example/v1")
    monkeypatch.setattr(settings, key_attr, f"{section}-stored-secret")

    response = client.post(TEST_URL, json={"section": section, "api_key": ""})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0
    assert body["model_count"] == 3
    assert body["models"] == ["m-a", "m-b", "m-c"]
    assert fake_upstream.last_request["url"] == f"https://{section}.example/v1/models"
    assert fake_upstream.last_request["headers"] == {
        "Authorization": f"Bearer {section}-stored-secret"
    }
    assert fake_upstream.last_request["timeout"] == 15.0
    # Probes never persist or mutate configuration.
    assert not env_path.exists()
    assert getattr(settings, key_attr) == f"{section}-stored-secret"


def test_probe_prefers_explicit_base_url_and_key(client, monkeypatch, fake_upstream):
    monkeypatch.setattr(settings, "openai_base_url", "https://stored.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "stored-key")

    response = client.post(
        TEST_URL,
        json={
            "section": "llm",
            "base_url": "https://explicit.example/v1/",
            "api_key": "explicit-key",
        },
    )

    assert response.status_code == 200
    assert fake_upstream.last_request["url"] == "https://explicit.example/v1/models"
    assert fake_upstream.last_request["headers"] == {
        "Authorization": "Bearer explicit-key"
    }


def test_probe_rejects_new_base_url_without_api_key(client, monkeypatch, fake_upstream):
    """Never forward the stored key to a host chosen by the request."""
    monkeypatch.setattr(settings, "openai_base_url", "https://stored.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "stored-secret-key")

    response = client.post(
        TEST_URL,
        json={"section": "llm", "base_url": "https://attacker.example/v1"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "base_url" in detail
    assert "api_key" in detail
    assert "stored-secret-key" not in response.text
    # The stored key was never sent anywhere.
    assert fake_upstream.last_request == {}


def test_probe_treats_trailing_slash_as_same_base_url(client, monkeypatch, fake_upstream):
    monkeypatch.setattr(settings, "openai_base_url", "https://stored.example/v1/")
    monkeypatch.setattr(settings, "openai_api_key", "stored-secret-key")

    response = client.post(
        TEST_URL,
        json={"section": "llm", "base_url": "https://stored.example/v1"},
    )

    assert response.status_code == 200
    assert fake_upstream.last_request["url"] == "https://stored.example/v1/models"
    assert fake_upstream.last_request["headers"] == {
        "Authorization": "Bearer stored-secret-key"
    }


def test_probe_omits_authorization_without_any_key(client, monkeypatch, fake_upstream):
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "")

    response = client.post(TEST_URL, json={"section": "llm"})

    assert response.status_code == 200
    assert fake_upstream.last_request["headers"] == {}
    assert "Authorization" not in fake_upstream.last_request["headers"]


def test_probe_returns_model_ids_from_upstream_data(client, monkeypatch, fake_upstream):
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "stored-key")
    fake_upstream.response = _FakeUpstreamResponse(
        payload={
            "data": [
                {"id": "model-a"},
                {"id": 123},
                "not-a-dict",
                {"id": ""},
                {"id": "model-b"},
                {},
            ]
        }
    )

    response = client.post(TEST_URL, json={"section": "llm"})

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == ["model-a", "model-b"]
    assert body["model_count"] == 6


@pytest.mark.parametrize(
    "upstream_status, expected_status",
    [(401, 401), (403, 403), (429, 502), (500, 502)],
)
def test_probe_maps_upstream_errors(
    upstream_status, expected_status, client, monkeypatch, fake_upstream
):
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "stored-key")
    fake_upstream.response = _FakeUpstreamResponse(status_code=upstream_status)

    response = client.post(TEST_URL, json={"section": "llm"})

    assert response.status_code == expected_status


def test_probe_maps_connection_errors_to_502(client, monkeypatch, fake_upstream):
    monkeypatch.setattr(settings, "openai_base_url", "https://llm.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "stored-key")
    fake_upstream.error = llm_config_router.httpx.ConnectError("boom")

    response = client.post(TEST_URL, json={"section": "llm"})

    assert response.status_code == 502
    assert "stored-key" not in response.text


def test_put_decision_section_writes_env_and_snapshot(tmp_path, client, monkeypatch):
    env_path = _write_env(
        tmp_path / ".env",
        "OPENAI_MODEL=old-model\nEMBEDDING_DIMENSIONS=1024\n",
    )
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    payload = _valid_put_payload()
    payload["decision"] = {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Kev-4b",
        "api_key": "sk-decision-1234567890",
        "mode": "shadow",
    }

    response = client.put(BASE_URL, json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["base_url"] == "https://api.siliconflow.cn/v1"
    assert body["decision"]["model"] == "Kev-4b"
    assert body["decision"]["mode"] == "shadow"
    assert body["decision"]["api_key_masked"] == "sk-d...7890"
    assert "sk-decision-1234567890" not in response.text
    values = dotenv_values(env_path)
    assert values["DECISION_MODEL_BASE_URL"] == "https://api.siliconflow.cn/v1"
    assert values["DECISION_MODEL_API_KEY"] == "sk-decision-1234567890"
    assert values["DECISION_MODEL_NAME"] == "Kev-4b"
    assert values["DECISION_MODEL_MODE"] == "shadow"
    assert settings.decision_model_mode == "shadow"
    assert settings.decision_model_api_key == "sk-decision-1234567890"


def test_put_decision_rejects_invalid_mode_without_writing(
    tmp_path, client, monkeypatch
):
    env_path = _write_env(tmp_path / ".env", "EMBEDDING_DIMENSIONS=1024\n")
    before = env_path.read_text(encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(_make_index_db(tmp_path, 1024)))
    monkeypatch.setattr(settings, "decision_model_mode", "off")
    payload = _valid_put_payload()
    payload["decision"] = {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Kev-4b",
        "mode": "sometimes",
    }

    response = client.put(BASE_URL, json=payload)

    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == before
    assert settings.decision_model_mode == "off"


def test_probe_decision_uses_systemone_contract(client, monkeypatch):
    calls: dict = {}

    async def fake_probe(**kwargs):
        calls.update(kwargs)
        return 123, "ok"

    monkeypatch.setattr(llm_config_router, "probe_systemone", fake_probe)
    monkeypatch.setattr(
        settings, "decision_model_base_url", "https://api.siliconflow.cn/v1"
    )
    monkeypatch.setattr(settings, "decision_model_api_key", "stored-decision-key")
    monkeypatch.setattr(settings, "decision_model_name", "Kev-4b")

    response = client.post(TEST_URL, json={"section": "decision"})

    assert response.status_code == 200
    body = response.json()
    assert body == {"latency_ms": 123, "model_count": 1, "models": ["ok"]}
    assert calls == {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "stored-decision-key",
        "model": "Kev-4b",
        "timeout_seconds": 15.0,
    }


def test_probe_decision_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(
        settings, "decision_model_base_url", "https://api.siliconflow.cn/v1"
    )
    monkeypatch.setattr(settings, "decision_model_api_key", "")

    response = client.post(TEST_URL, json={"section": "decision"})

    assert response.status_code == 400
    assert "api_key" in response.json()["detail"]


def test_probe_decision_maps_upstream_auth_to_401(client, monkeypatch):
    async def fake_probe(**kwargs):
        raise DecisionCallError("upstream_auth", 401)

    monkeypatch.setattr(llm_config_router, "probe_systemone", fake_probe)
    monkeypatch.setattr(
        settings, "decision_model_base_url", "https://api.siliconflow.cn/v1"
    )
    monkeypatch.setattr(settings, "decision_model_api_key", "stored-decision-key")

    response = client.post(TEST_URL, json={"section": "decision"})

    assert response.status_code == 401
    assert "stored-decision-key" not in response.text
