"""账号级模型配置：完全自持、加密、空态与 HTTP 隔离。"""

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


@pytest.fixture()
def system_credentials(monkeypatch):
    monkeypatch.setattr(settings, "openai_base_url", "https://system.example/v1")
    monkeypatch.setattr(settings, "openai_api_key", "sk-system-secret")
    monkeypatch.setattr(settings, "openai_model", "system-model")
    monkeypatch.setattr(settings, "embedding_base_url", "https://system.example/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "sk-system-embedding")
    monkeypatch.setattr(settings, "embedding_model", "system-embedding")
    monkeypatch.setattr(settings, "rerank_base_url", "https://system.example/v1")
    monkeypatch.setattr(settings, "rerank_api_key", "sk-system-rerank")
    monkeypatch.setattr(settings, "rerank_model", "system-rerank")
    monkeypatch.setattr(settings, "decision_model_base_url", "https://system.example/v1")
    monkeypatch.setattr(settings, "decision_model_api_key", "sk-system-decision")


def test_account_without_overrides_is_empty_and_never_inherits_system(
    temp_rag_path, system_credentials
) -> None:
    effective = user_config.effective_settings(settings, OWNER_A)

    assert effective.openai_base_url == ""
    assert effective.openai_api_key == ""
    assert effective.openai_model == ""
    assert effective.embedding_base_url == ""
    assert effective.embedding_api_key == ""
    assert effective.embedding_model == ""
    assert effective.rerank_base_url == ""
    assert effective.rerank_api_key == ""
    assert effective.rerank_model == ""
    assert effective.decision_model_base_url == ""
    assert effective.decision_model_api_key == ""
    assert effective.decision_model_mode == "off"

    assert user_config.section_sources(settings, OWNER_A) == {
        "llm": "unset",
        "embedding": "unset",
        "rerank": "unset",
        "decision": "unset",
    }
    assert user_config.user_config_updated_at(settings, OWNER_A) is None


def test_overrides_are_account_owned_encrypted_and_reset_to_empty(
    temp_rag_path, system_credentials
) -> None:
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
    # 未配置的段保持为空，不继承系统默认
    assert effective.embedding_base_url == ""
    assert effective.embedding_api_key == ""
    assert user_config.section_sources(settings, OWNER_A)["llm"] == "account"
    assert user_config.section_sources(settings, OWNER_A)["embedding"] == "unset"
    assert user_config.user_config_updated_at(settings, OWNER_A) is not None

    file_text = user_config_path(settings, OWNER_A).read_text(encoding="utf-8")
    assert "sk-account-secret" not in file_text
    assert json.loads(file_text)["sections"]["llm"]["api_key"]

    other = user_config.effective_settings(settings, OWNER_B)
    assert other.openai_model == ""
    assert other.openai_base_url == ""
    assert other.openai_api_key == ""

    user_config.reset_overrides(settings, OWNER_A)
    after = user_config.effective_settings(settings, OWNER_A)
    assert after.openai_model == ""
    assert after.openai_api_key == ""
    assert not user_config_path(settings, OWNER_A).exists()


def test_empty_section_clears_and_empty_key_keeps_the_stored_one(
    temp_rag_path, system_credentials
) -> None:
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"llm": {"base_url": "https://account.example/v1", "model": "m1", "api_key": "sk-1"}},
    )
    # 只改模型且 api_key 留空：保留已存 key
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"llm": {"base_url": "https://account.example/v1", "model": "m2", "api_key": ""}},
    )
    effective = user_config.effective_settings(settings, OWNER_A)
    assert effective.openai_model == "m2"
    assert effective.openai_api_key == "sk-1"

    # 空的 base_url/model 表示清空该段（连带 key）
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"llm": {"base_url": "", "model": "", "api_key": ""}},
    )
    effective = user_config.effective_settings(settings, OWNER_A)
    assert effective.openai_model == ""
    assert effective.openai_api_key == ""
    assert user_config.section_sources(settings, OWNER_A)["llm"] == "unset"


def _client() -> TestClient:
    from api.llm_config_router import llm_config_router

    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request, call_next):
        request.state.user_id = request.headers.get("x-test-user", "")
        return await call_next(request)

    app.include_router(llm_config_router, prefix="/api/v1")
    return TestClient(app, client=("127.0.0.1", 50000))


def test_http_config_is_empty_for_new_account_and_scoped_to_login(
    temp_rag_path, system_credentials
) -> None:
    client = _client()

    body_a = client.get("/api/v1/llm-config", headers={"x-test-user": OWNER_A}).json()
    assert body_a["llm"] == {
        "base_url": "",
        "model": "",
        "api_key_masked": "",
        "api_key_set": False,
    }
    assert body_a["embedding"]["dimensions"] is None
    assert body_a["sources"] == {
        "llm": "unset",
        "embedding": "unset",
        "rerank": "unset",
        "decision": "unset",
    }
    assert "sk-system-secret" not in json.dumps(body_a)

    put = client.put(
        "/api/v1/llm-config",
        json={
            "llm": {
                "base_url": "https://account.example/v1",
                "model": "account-model",
                "api_key": "sk-user-secret",
            },
            "embedding": {"base_url": "", "model": "", "api_key": ""},
            "rerank": {"base_url": "", "model": "", "api_key": ""},
            "decision": {"base_url": "", "model": "", "api_key": "", "mode": "off"},
        },
        headers={"x-test-user": OWNER_A},
    )
    assert put.status_code == 200, put.text
    saved = put.json()
    assert saved["llm"]["model"] == "account-model"
    assert saved["llm"]["api_key_masked"] == "已设置"
    assert saved["llm"]["api_key_set"] is True
    assert saved["embedding"]["dimensions"] is None
    assert saved["sources"]["llm"] == "account"
    assert saved["sources"]["embedding"] == "unset"
    assert "sk-user-secret" not in put.text

    body_b = client.get("/api/v1/llm-config", headers={"x-test-user": OWNER_B}).json()
    assert body_b["llm"]["model"] == ""
    assert body_b["sources"]["llm"] == "unset"
    # 写入只落在账号文件，不污染 .env 对应的全局设置
    assert settings.openai_model == "system-model"
    assert settings.openai_api_key == "sk-system-secret"


def test_http_reset_clears_even_when_an_account_index_exists(
    temp_rag_path, system_credentials
) -> None:
    from rag import sqlite_schema

    path = user_rag_path(settings, OWNER_A)
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_schema.connect(path, embedding_dimensions=512).close()
    user_config.save_overrides(
        settings,
        OWNER_A,
        {"embedding": {"base_url": "https://emb.example/v1", "model": "e1", "dimensions": 512}},
    )

    client = _client()
    response = client.delete("/api/v1/llm-config", headers={"x-test-user": OWNER_A})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["embedding"]["base_url"] == ""
    assert body["embedding"]["dimensions"] is None
    assert body["sources"]["embedding"] == "unset"


def test_http_partial_section_rejects_missing_model(
    temp_rag_path, system_credentials
) -> None:
    client = _client()
    response = client.put(
        "/api/v1/llm-config",
        json={"llm": {"base_url": "https://account.example/v1", "model": ""}},
        headers={"x-test-user": OWNER_A},
    )
    assert response.status_code == 400
    assert "model" in response.text
    assert user_config.section_sources(settings, OWNER_A)["llm"] == "unset"
