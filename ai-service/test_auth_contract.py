"""JWT 账号契约：注册、登录、Cookie/Bearer、大小写与鉴权门。

Seam 是 HTTP 层（/api/auth/*）与账号身份（subject）本身，不触碰密码哈希实现。
"""

from __future__ import annotations

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
