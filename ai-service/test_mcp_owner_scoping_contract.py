"""自定义 MCP 按账号隔离：配置、工具构建、执行策略与旧数据认领。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from agent_v2 import mcp_config_store
from agent_v2.custom_mcp_tools import build_custom_mcp_tools
from agent_v2.execution_policy import build_execution_policy
from agent_v2.profiles import ProfileId, resolve_profile
from auth.legacy import claim_legacy_data
from auth.store import AuthStore

OWNER_A = "a" * 32
OWNER_B = "b" * 32


def _server(server_id: str = "custom_1", name: str = "A 的服务") -> dict:
    return {
        "id": server_id,
        "name": name,
        "transport": "stdio",
        "url": "",
        "command": "python",
        "args": ["-m", "test_fixtures.mcp_server"],
        "headers": {},
        "env": {},
        "enabled": True,
        "status": "success",
        "tools": ["ping"],
        "tool_descriptors": [
            {
                "name": "ping",
                "description": "ping",
                "schema": {"type": "object", "properties": {}},
            }
        ],
        "revision": "rev1",
    }


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        agent_v2_state_dir=str(tmp_path / "state"),
        rag_sqlite_db_path=str(tmp_path / "data" / "rag.sqlite"),
        agent_v2_sandbox_type="local_native",
        agent_v2_subagent_enabled=False,
        agent_v2_enabled=True,
        agent_v2_cowork_policy="disabled",
        openai_model="fixture-model",
    )


def test_config_files_are_scoped_by_owner(tmp_path, monkeypatch) -> None:
    runtime_settings = _settings(tmp_path)
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_state_dir", runtime_settings.agent_v2_state_dir)
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_sandbox_type", "local_native")

    mcp_config_store.save_servers([_server()], OWNER_A)
    assert [item["id"] for item in mcp_config_store.load_servers(OWNER_A)] == ["custom_1"]
    assert mcp_config_store.load_servers(OWNER_B) == []
    assert mcp_config_store.load_servers() == []

    assert len(mcp_config_store.enabled_servers(runtime_settings, OWNER_A)) == 1
    assert mcp_config_store.enabled_servers(runtime_settings, OWNER_B) == []

    assert mcp_config_store.scoped_fingerprint(_server(), OWNER_A) != mcp_config_store.scoped_fingerprint(
        _server(), OWNER_B
    )


def test_execution_policy_and_tools_only_see_owner_servers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_sandbox_type", "local_native")
    mcp_config_store.save_servers([_server()], OWNER_A)
    settings = _settings(tmp_path)

    policy_a = build_execution_policy(resolve_profile(ProfileId.CHAT), settings, owner_id=OWNER_A)
    policy_b = build_execution_policy(resolve_profile(ProfileId.CHAT), settings, owner_id=OWNER_B)
    assert "custom_1" in policy_a["capability"]["allowed_mcp_servers"]
    assert "custom_1" not in (policy_b["capability"]["allowed_mcp_servers"] or [])

    projection = SimpleNamespace(
        mcp_services=[SimpleNamespace(revision="rev1", server_id="custom_1", tools=["ping"])],
        run_id="run-1",
    )
    tools = build_custom_mcp_tools(projection, runs_repo=None, audit_sink=lambda _: None, owner_id=OWNER_A)
    assert [tool.name for tool in tools] == ["mcp__custom_1__ping"]

    try:
        build_custom_mcp_tools(projection, runs_repo=None, audit_sink=lambda _: None, owner_id=OWNER_B)
    except ValueError:
        pass
    else:
        raise AssertionError("另一个账号不应看到 A 的 MCP 配置")


def test_legacy_mcp_is_claimed_only_once(tmp_path) -> None:
    settings = _settings(tmp_path)
    state = Path(settings.agent_v2_state_dir)
    state.mkdir(parents=True)
    (state / "mcp_servers.json").write_text(
        json.dumps({"version": 1, "servers": [_server()]}), encoding="utf-8"
    )
    store = AuthStore(tmp_path / "auth.sqlite")
    try:
        claim_legacy_data(settings, store, OWNER_A)
        claim_legacy_data(settings, store, OWNER_B)
        assert store.get_meta("legacy_owner_id") == OWNER_A
    finally:
        store.close()

    assert (state / "users" / OWNER_A / "mcp_servers.json").is_file()
    assert not (state / "users" / OWNER_B / "mcp_servers.json").exists()
    assert (state / "mcp_servers.json").is_file(), "旧文件只能复制，不能移动"


def test_http_server_list_is_scoped_by_logged_in_account(tmp_path, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.mcp_router import mcp_router

    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_sandbox_type", "local_native")
    mcp_config_store.save_servers([_server()], OWNER_A)

    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request, call_next):
        request.state.user_id = request.headers.get("x-test-user", "")
        return await call_next(request)

    app.include_router(mcp_router, prefix="/api/v2")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response_a = client.get("/api/v2/mcp/servers", headers={"x-test-user": OWNER_A})
    response_b = client.get("/api/v2/mcp/servers", headers={"x-test-user": OWNER_B})
    assert response_a.status_code == 200, response_a.text
    assert response_b.status_code == 200, response_b.text

    names_a = {server["name"] for server in response_a.json()["servers"]}
    names_b = {server["name"] for server in response_b.json()["servers"]}
    assert "A 的服务" in names_a
    assert "A 的服务" not in names_b
    # 高德内置服务两边共享
    assert {"高德天气", "高德地图"} <= names_a
    assert {"高德天气", "高德地图"} <= names_b


def test_builtin_credentials_are_account_scoped_and_never_echoed(
    tmp_path, monkeypatch
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agent_v2.kernel_tools import build_profile_mcp_servers
    from agent_v2.mcp_builtin_store import load_env
    from api.mcp_router import mcp_router

    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_sandbox_type", "local_native")
    monkeypatch.delenv("AMAP_API_KEY", raising=False)
    mcp_config_store.save_servers([_server("custom_a", "A 自定义")], OWNER_A)

    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request, call_next):
        request.state.user_id = request.headers.get("x-test-user", "")
        return await call_next(request)

    app.include_router(mcp_router, prefix="/api/v2")
    client = TestClient(app, client=("127.0.0.1", 50000))

    def builtins(owner: str) -> dict[str, dict]:
        response = client.get("/api/v2/mcp/servers", headers={"x-test-user": owner})
        assert response.status_code == 200, response.text
        return {
            server["id"]: server
            for server in response.json()["servers"]
            if server["builtin"]
        }

    before = builtins(OWNER_A)
    assert set(before) == {"amap_weather", "amap_map"}
    assert {server["status"] for server in before.values()} == {"unconfigured"}
    assert before["amap_weather"]["command"] == "npx"
    assert "@amap/amap-weather-mcp" in before["amap_weather"]["args"]
    assert before["amap_weather"]["env_names"] == ["AMAP_API_KEY"]

    saved = client.put(
        "/api/v2/mcp/builtin/credentials",
        json={"env": {"AMAP_API_KEY": "ak-account-a"}},
        headers={"x-test-user": OWNER_A},
    )
    assert saved.status_code == 200, saved.text
    assert "ak-account-a" not in saved.text
    saved_servers = saved.json()["servers"]
    assert {s["status"] for s in saved_servers if s["builtin"]} == {"configured"}
    assert "custom_a" in {s["id"] for s in saved_servers}

    assert {s["status"] for s in builtins(OWNER_B).values()} == {"unconfigured"}
    assert load_env(OWNER_A) == {"AMAP_API_KEY": "ak-account-a"}

    profile = SimpleNamespace(
        profile_id="route_plan",
        allowed_mcp_servers=("amap_weather", "amap_map"),
    )
    servers_a, audits_a = build_profile_mcp_servers(
        profile, SimpleNamespace(), owner_id=OWNER_A
    )
    assert audits_a == []
    assert servers_a["amap_map"]["env"] == {"AMAP_API_KEY": "ak-account-a"}

    servers_b, audits_b = build_profile_mcp_servers(
        profile, SimpleNamespace(), owner_id=OWNER_B
    )
    assert servers_b == {}
    assert {audit["server_id"] for audit in audits_b} == {
        "amap_weather",
        "amap_map",
    }

    assert client.put(
        "/api/v2/mcp/builtin/credentials",
        json={"env": {"UNKNOWN_KEY": "x"}},
        headers={"x-test-user": OWNER_A},
    ).status_code == 400
    assert client.put(
        "/api/v2/mcp/builtin/credentials",
        json={"env": {}},
        headers={"x-test-user": OWNER_A},
    ).status_code == 400

    cleared = client.delete(
        "/api/v2/mcp/builtin/credentials", headers={"x-test-user": OWNER_A}
    )
    assert cleared.status_code == 200, cleared.text
    cleared_servers = cleared.json()["servers"]
    assert {s["status"] for s in cleared_servers if s["builtin"]} == {"unconfigured"}
    assert "custom_a" in {s["id"] for s in cleared_servers}
    assert load_env(OWNER_A) == {}


def test_oauth_status_cannot_be_polled_by_another_account(tmp_path, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agent_v2 import mcp_oauth_flow
    from api.mcp_router import mcp_router

    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_state_dir", str(tmp_path / "state"))
    monkeypatch.setattr(mcp_config_store.settings, "agent_v2_sandbox_type", "local_native")
    flow = SimpleNamespace(
        id="flow-a",
        owner_id=OWNER_A,
        status="success",
        test_id="test-a",
        fingerprint="fp-a",
        tools=[{"name": "ping"}],
        created_at=time.monotonic(),
        expected_state=None,
        callback=None,
        task=None,
    )
    mcp_oauth_flow._flows[flow.id] = flow
    app = FastAPI()

    @app.middleware("http")
    async def inject_user(request, call_next):
        request.state.user_id = request.headers.get("x-test-user", "")
        return await call_next(request)

    app.include_router(mcp_router, prefix="/api/v2")
    client = TestClient(app, client=("127.0.0.1", 50000))
    try:
        assert client.get(
            "/api/v2/mcp/oauth/status/flow-a", headers={"x-test-user": OWNER_B}
        ).status_code == 404
        response = client.get(
            "/api/v2/mcp/oauth/status/flow-a", headers={"x-test-user": OWNER_A}
        )
        assert response.status_code == 200
        assert response.json()["tools"] == ["ping"]
    finally:
        mcp_oauth_flow._flows.pop(flow.id, None)
