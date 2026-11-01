"""JWT 账号契约：注册、登录、Cookie/Bearer、大小写与鉴权门。

Seam 是 HTTP 层（/api/auth/*）与账号身份（subject）本身，不触碰密码哈希实现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from agent_v2.repositories.memory import MemoryStore
from auth.router import auth_router
from auth.service import AuthService, DuplicateLoginId, InvalidCredentials
from auth.store import AuthStore


def _client(tmp_path: Path) -> tuple[TestClient, AuthService]:
    auth_db = tmp_path / "auth.sqlite"
    memory = MemoryStore(str(tmp_path / "agent_memory.sqlite"))
    service = AuthService(AuthStore(str(auth_db)), memory, secret="test-secret-0123456789abcdef0123456789", ttl_seconds=3600)
    app = FastAPI()
    app.state.auth_service = service
    app.include_router(auth_router, prefix="/api")
    return TestClient(app), service


def _register(client: TestClient, login_id: str = "Abc1", password: str = "password123"):
    return client.post(
        "/api/auth/register", json={"login_id": login_id, "password": password}
    )


def test_register_logs_in_with_cookie_and_bearer(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = _register(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["login_id"] == "Abc1"
    assert body["access_token"]
    assert body["user_id"]

    cookie = response.headers["set-cookie"]
    assert "hiking_auth=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie or "samesite=lax" in cookie.lower()

    me = client.get("/api/auth/me")
    assert me.status_code == 200, me.text
    assert me.json()["login_id"] == "Abc1"

    bearer = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert bearer.status_code == 200
    assert bearer.json()["user_id"] == body["user_id"]


def test_login_id_is_case_sensitive_and_unique(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    assert _register(client, "Abc1").status_code == 201
    assert _register(client, "Abc1").status_code == 409
    assert _register(client, "abc1").status_code == 201

    wrong_case = client.post(
        "/api/auth/login", json={"login_id": "ABC1", "password": "password123"}
    )
    assert wrong_case.status_code == 401
    exact = client.post(
        "/api/auth/login", json={"login_id": "abc1", "password": "password123"}
    )
    assert exact.status_code == 200


def test_login_rejects_bad_password_without_user_enumeration(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    _register(client)
    wrong = client.post(
        "/api/auth/login", json={"login_id": "Abc1", "password": "password124"}
    )
    unknown = client.post(
        "/api/auth/login", json={"login_id": "Nope1", "password": "password123"}
    )
    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


@pytest.mark.parametrize(
    "login_id,password",
    [
        ("a", "password123"),
        ("abcdefg", "password123"),
        ("bad id", "password123"),
        ("Abc1", "short7!"),
        ("Abc1", "way-too-long-password"),
        ("Abc1", "pass word"),
    ],
)
def test_registration_validation(tmp_path: Path, login_id: str, password: str) -> None:
    client, _ = _client(tmp_path)
    response = client.post(
        "/api/auth/register", json={"login_id": login_id, "password": password}
    )
    assert response.status_code == 422


def test_unauthenticated_requests_are_rejected_and_logout_clears_session(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path)
    assert client.get("/api/auth/me").status_code == 401

    _register(client)
    assert client.get("/api/auth/me").status_code == 200

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_tampered_token_is_rejected(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    response = _register(client)
    token = response.json()["access_token"]
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tampered}"})
    assert me.status_code == 401


def test_password_hash_never_contains_plaintext(tmp_path: Path) -> None:
    client, service = _client(tmp_path)
    _register(client, "Abc1", "TopSecret42")
    row = service.store.get_by_login_id("Abc1")
    assert row is not None
    assert "TopSecret42" not in row["password_hash"]


def test_duplicate_race_is_mapped_to_duplicate_error(tmp_path: Path) -> None:
    client, service = _client(tmp_path)
    _register(client)
    with pytest.raises(DuplicateLoginId):
        service.register("Abc1", "password123")


def test_invalid_credentials_error_is_stable(tmp_path: Path) -> None:
    client, service = _client(tmp_path)
    _register(client)
    with pytest.raises(InvalidCredentials):
        service.login("Abc1", "wrongpass1")


def test_secret_file_is_created_once_and_reused(tmp_path: Path) -> None:
    from auth.security import load_or_create_secret

    path = tmp_path / "auth.key"
    first = load_or_create_secret(path)
    second = load_or_create_secret(path)
    assert first == second
    assert path.read_text(encoding="utf-8").strip() == first


def test_main_registration_never_inherits_legacy_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    import main
    from agent_v2 import mcp_config_store
    from agent_v2.mcp_builtin_store import load_env, save_env
    from auth.paths import user_rag_path
    from auth.user_config import save_overrides
    from config import settings
    from datasource.store import DatasourceStore
    from rag.sqlite_store import SQLiteStore

    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir()
    state.mkdir()
    for name, value in {
        "auth_db_path": str(data / "auth.sqlite"),
        "auth_secret_path": str(data / "auth.key"),
        "auth_jwt_secret": "fixture-secret-0123456789abcdef0123456789",
        "rag_sqlite_db_path": str(data / "rag.sqlite"),
        "datasource_db_path": str(data / "datasource.sqlite"),
        "datasource_secret_key": "",
        "agent_v2_state_dir": str(state),
        "agent_v2_enabled": False,
        "agent_v2_sandbox_type": "local_native",
        "embedding_dimensions": 8,
        "langfuse_enabled": False,
        "openai_base_url": "https://global.invalid/v1",
        "openai_model": "global-model",
        "openai_api_key": "fixture-global-key",
    }.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("AMAP_API_KEY", "fixture-global-amap")

    legacy_server = {
        "id": "legacy", "name": "旧 MCP", "transport": "stdio", "url": "",
        "command": "python", "args": [], "env": {}, "headers": {},
        "enabled": False, "status": "untested", "tools": [],
        "tool_descriptors": [], "revision": "fixture-revision",
    }
    legacy_path = state / "mcp_servers.json"
    legacy_path.write_text(
        json.dumps({"version": 1, "servers": [legacy_server]}), encoding="utf-8"
    )
    legacy_bytes = legacy_path.read_bytes()
    SQLiteStore(settings.rag_sqlite_db_path, embedding_dimensions=8).close()
    store = DatasourceStore(settings.datasource_db_path)
    legacy_source = store.create_datasource(
        type="notion", name="旧知识来源", credentials_enc="fixture", owner_id=""
    )
    store.close()

    with TestClient(main.app, client=("127.0.0.1", 50000)) as client:
        first = _register(client, "Alice1")
        assert first.status_code == 201, first.text
        owner_a = first.json()["user_id"]
        servers = client.get("/api/v2/mcp/servers").json()["servers"]
        assert [s["id"] for s in servers if not s["builtin"]] == []
        assert {s["status"] for s in servers if s["builtin"]} == {"unconfigured"}
        assert client.get("/api/v1/datasources").json() == []
        assert not user_rag_path(settings, owner_a).exists()
        config = client.get("/api/v1/llm-config").json()
        assert set(config["sources"].values()) == {"unset"}
        assert config["llm"]["model"] == ""
        assert config["llm"]["api_key_set"] is False

        save_overrides(settings, owner_a, {"llm": {
            "base_url": "https://alice.invalid/v1", "model": "alice-model",
            "api_key": "fixture-alice-key",
        }})
        mcp_config_store.save_servers([{**legacy_server, "id": "alice", "name": "A 的 MCP"}], owner_a)
        save_env(owner_a, {"AMAP_API_KEY": "fixture-alice-amap"})
        store = main.app.state.datasource_service.store
        owned_source = store.create_datasource(
            type="notion", name="A 的知识来源", credentials_enc="fixture", owner_id=owner_a
        )

        assert client.post("/api/auth/logout").status_code == 200
        second = _register(client, "Bob123")
        assert second.status_code == 201, second.text
        owner_b = second.json()["user_id"]
        assert owner_a != owner_b
        assert client.get("/api/v1/llm-config").json()["llm"]["model"] == ""
        assert client.get("/api/v1/datasources").json() == []
        assert client.get(f"/api/v1/datasources/{owned_source['id']}").status_code == 404
        servers = client.get("/api/v2/mcp/servers").json()["servers"]
        assert all(s["builtin"] for s in servers)
        assert {s["status"] for s in servers} == {"unconfigured"}
        assert load_env(owner_b) == {}
        assert not user_rag_path(settings, owner_b).exists()
        assert store.get_datasource(legacy_source["id"])["owner_id"] == ""

        saved = client.put("/api/v1/llm-config", json={"llm": {
            "base_url": "https://bob.invalid/v1", "model": "bob-model",
            "api_key": "fixture-bob-key",
        }})
        assert saved.status_code == 200, saved.text
        saved = client.post("/api/v2/mcp/servers", json={
            "name": "B 的 MCP", "transport": "stdio", "command": "python",
        })
        assert saved.status_code == 200, saved.text
        saved = client.put("/api/v2/mcp/builtin/credentials", json={
            "env": {"AMAP_API_KEY": "fixture-bob-amap"},
        })
        assert saved.status_code == 200, saved.text
        saved = client.post("/api/v1/datasources", json={
            "name": "B 的知识来源", "type": "notion",
            "credentials": {"token": "fixture-bob-token"},
            "selected_resources": ["fixture-page"],
        })
        assert saved.status_code == 201, saved.text
        source_b = saved.json()["id"]

    with TestClient(main.app, client=("127.0.0.1", 50000)) as client:
        logged_in = client.post("/api/auth/login", json={
            "login_id": "Alice1", "password": "password123",
        })
        assert logged_in.status_code == 200, logged_in.text
        assert client.get("/api/v1/llm-config").json()["llm"]["model"] == "alice-model"
        servers = client.get("/api/v2/mcp/servers").json()["servers"]
        assert [s["id"] for s in servers if not s["builtin"]] == ["alice"]
        assert {s["status"] for s in servers if s["builtin"]} == {"configured"}
        assert [s["id"] for s in client.get("/api/v1/datasources").json()] == [owned_source["id"]]
        assert load_env(owner_a) == {"AMAP_API_KEY": "fixture-alice-amap"}
        assert legacy_path.read_bytes() == legacy_bytes
        assert Path(settings.rag_sqlite_db_path).is_file()

        assert client.post("/api/auth/logout").status_code == 200
        assert client.post("/api/auth/login", json={
            "login_id": "Bob123", "password": "password123",
        }).status_code == 200
        assert client.get("/api/v1/llm-config").json()["llm"]["model"] == "bob-model"
        assert [s["name"] for s in client.get("/api/v2/mcp/servers").json()["servers"]
                if not s["builtin"]] == ["B 的 MCP"]
        assert [s["id"] for s in client.get("/api/v1/datasources").json()] == [source_b]
        assert load_env(owner_b) == {"AMAP_API_KEY": "fixture-bob-amap"}
        assert client.delete("/api/v1/llm-config").status_code == 200
        assert client.get("/api/v1/llm-config").json()["llm"]["model"] == ""
        assert client.post("/api/auth/logout").status_code == 200
        assert client.post("/api/auth/login", json={
            "login_id": "Alice1", "password": "password123",
        }).status_code == 200
        assert client.get("/api/v1/llm-config").json()["llm"]["model"] == "alice-model"


def test_main_app_data_routes_require_login(tmp_path: Path) -> None:
    import main

    service = AuthService(
        AuthStore(str(tmp_path / "auth.sqlite")),
        MemoryStore(str(tmp_path / "agent_memory.sqlite")),
        secret="test-secret",
        ttl_seconds=3600,
    )
    previous = getattr(main.app.state, "auth_service", None)
    main.app.state.auth_service = service
    try:
        client = TestClient(main.app, client=("127.0.0.1", 50000))
        for path in (
            "/api/v1/llm-config",
            "/api/v1/datasources",
            "/api/v2/mcp/servers",
            "/api/v2/chat/snapshot/run-test",
            "/api/v2/memory/settings",
        ):
            response = client.get(path)
            assert response.status_code == 401, (path, response.status_code, response.text)
    finally:
        main.app.state.auth_service = previous
        service.close()
