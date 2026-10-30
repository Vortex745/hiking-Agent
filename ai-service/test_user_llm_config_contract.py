"""账号级模型配置：覆盖、继承、加密与 HTTP 隔离。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from auth import user_config
from auth.paths import user_config_path, user_rag_path
from config import settings

OWNER_A = "a" * 32
OWNER_B = "b" * 32


@pytest.fixture()
def temp_rag_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(tmp_path / "data" / "rag.sqlite"))
    user_config._SECRET_BOXES.clear()
    yield tmp_path


def test_overrides_inherit_encrypt_and_reset(temp_rag_path) -> None:
    base_url = settings.openai_base_url
    model = settings.openai_model
    changed = user_config.save_overrides(
        settings,
        OWNER_A,
        {
            "llm": {
                "base_url": "https://account.example/v1",
                "model": "account-model",
                "api_key": "sk-account-secret",
            }
        },
    )
    assert "llm.api_key" in changed

    effective = user_config.effective_settings(settings, OWNER_A)
    assert effective.openai_base_url == "https://account.example/v1"
    assert effective.openai_model == "account-model"
    assert effective.openai_api_key == "sk-account-secret"
    # 未覆盖的段继承系统默认
    assert effective.embedding_model == settings.embedding_model
    assert user_config.section_sources(settings, OWNER_A)["llm"] == "account"
    assert user_config.section_sources(settings, OWNER_A)["embedding"] == "system"

    file_text = user_config_path(settings, OWNER_A).read_text(encoding="utf-8")
    assert "sk-account-secret" not in file_text
    assert json.loads(file_text)["sections"]["llm"]["api_key"]

    # 另一个账号完全继承系统默认
    other = user_config.effective_settings(settings, OWNER_B)
    assert other.openai_model == model
    assert other.openai_base_url == base_url
    assert user_config.section_sources(settings, OWNER_B)["llm"] == "system"

    user_config.reset_overrides(settings, OWNER_A)
    assert user_config.effective_settings(settings, OWNER_A).openai_model == model
    assert not user_config_path(settings, OWNER_A).exists()


def test_values_equal_to_system_default_are_not_stored(temp_rag_path) -> None:
    user_config.save_overrides(
        settings,
        OWNER_A,
        {
            "embedding": {
                "base_url": settings.embedding_base_url,
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dimensions,
                "api_key": "",
            }
        },
    )
    assert user_config.section_sources(settings, OWNER_A)["embedding"] == "system"
    assert not user_config_path(settings, OWNER_A).exists()


def test_account_base_url_change_never_inherits_the_system_key(
    temp_rag_path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "openai_base_url", "https://system.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "sk-system-secret")
    monkeypatch.setattr(settings, "openai_model", "system-model")

    # 只改 model：base 未变，可以继续用系统 key
    user_config.save_overrides(settings, OWNER_A, {"llm": {"model": "a-model"}})
    assert user_config.effective_settings(settings, OWNER_A).openai_api_key == "sk-system-secret"

    # 改 base 但不给账号自己的 key：不能把系统 key 发到账号指定的主机
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"llm": {"base_url": "https://account.example/v1", "api_key": ""}},
    )
    effective = user_config.effective_settings(settings, OWNER_A)
    assert effective.openai_base_url == "https://account.example/v1"
    assert effective.openai_api_key == ""

    # 账号自己提供 key 后才生效
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"llm": {"api_key": "sk-account-key"}},
    )
    assert user_config.effective_settings(settings, OWNER_A).openai_api_key == "sk-account-key"


def _client() -> TestClient:
    from api.llm_config_router import llm_config_router

    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request, call_next):
        request.state.user_id = request.headers.get("x-test-user", "")
        return await call_next(request)

    app.include_router(llm_config_router, prefix="/api/v1")
    return TestClient(app, client=("127.0.0.1", 50000))


def _payload(model: str) -> dict:
    return {
        "llm": {"base_url": "https://account.example/v1", "model": model, "api_key": "sk-user-secret"},
        "embedding": {
            "base_url": settings.embedding_base_url,
            "model": settings.embedding_model,
            "api_key": "",
            "dimensions": settings.embedding_dimensions,
        },
        "rerank": {
            "base_url": settings.rerank_base_url,
            "model": settings.rerank_model,
            "api_key": "",
        },
        "decision": {
            "base_url": settings.decision_model_base_url,
            "model": settings.decision_model_name,
            "api_key": "",
            "mode": settings.decision_model_mode,
        },
    }


def test_http_config_is_scoped_to_the_logged_in_account(temp_rag_path) -> None:
    client = _client()
    system_model = settings.openai_model

    put = client.put(
        "/api/v1/llm-config",
        json=_payload("account-model"),
        headers={"x-test-user": OWNER_A},
    )
    assert put.status_code == 200, put.text
    body_a = client.get("/api/v1/llm-config", headers={"x-test-user": OWNER_A}).json()
    body_b = client.get("/api/v1/llm-config", headers={"x-test-user": OWNER_B}).json()

    assert body_a["llm"]["model"] == "account-model"
    assert body_a["llm"]["api_key_set"] is True
    assert body_a["sources"]["llm"] == "account"
    assert body_b["llm"]["model"] == system_model
    assert body_b["sources"]["llm"] == "system"
    assert "sk-user-secret" not in put.text

    # 全局 .env 未被账号写入污染
    assert settings.openai_model == system_model

    reset = client.delete("/api/v1/llm-config", headers={"x-test-user": OWNER_A})
    assert reset.status_code == 200
    assert reset.json()["llm"]["model"] == system_model
    assert reset.json()["sources"]["llm"] == "system"


def test_reset_refuses_to_break_an_existing_account_index(temp_rag_path) -> None:
    from rag import sqlite_schema

    dimensions = 512 if settings.embedding_dimensions != 512 else 768
    path = user_rag_path(settings, OWNER_A)
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_schema.connect(path, embedding_dimensions=dimensions).close()

    client = _client()
    response = client.delete("/api/v1/llm-config", headers={"x-test-user": OWNER_A})
    assert response.status_code == 409, response.text
    assert "维度" in response.text
