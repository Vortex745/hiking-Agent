"""HTTP contract tests for the local-only datasource management API."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.datasource_router import datasource_router
from datasource.connectors.fake import FakeDoc, FakeState, register_fake
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8
TOKEN = "tok-secret-12345"


@pytest.fixture
def ctx(tmp_path: Path):
    state = FakeState(
        validate_token=TOKEN,
        docs=[FakeDoc(external_id="doc-1", title="Route A", content="alpine lake")],
    )
    ds_store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    registry = ConnectorRegistry()
    register_fake(registry, state)
    service = DatasourceService(
        ds_store, registry, ingestor, SecretBox.from_key_bytes(b"k" * 32)
    )
    app = FastAPI()
    app.state.datasource_service = service
    app.include_router(datasource_router, prefix="/api/v1")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client, service, state, tmp_path
    service.close()


def _payload(**overrides):
    payload = {
        "type": "fake",
        "name": "Fake",
        "credentials": {"token": TOKEN},
        "settings": {},
        "selected_resources": ["root"],
    }
    payload.update(overrides)
    return payload


def _create(client, **overrides):
    response = client.post("/api/v1/datasources", json=_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def test_types_and_non_loopback_guard(ctx):
    client, _, _, _ = ctx
    response = client.get("/api/v1/datasources/types")
    assert response.status_code == 200
    assert response.json()[0]["type"] == "fake"

    app = client.app
    with TestClient(app, client=("10.0.0.5", 50000)) as foreign:
        denied = foreign.get("/api/v1/datasources/types")
    assert denied.status_code == 403


def test_create_list_edit_and_secret_redaction(ctx):
    client, _, _, tmp_path = ctx
    created = _create(client)
    assert created["credentials_configured"] == {"token": True}
    assert TOKEN not in str(created)

    listed = client.get("/api/v1/datasources").json()
    assert len(listed) == 1 and listed[0]["id"] == created["id"]

    fetched = client.get(f"/api/v1/datasources/{created['id']}")
    assert fetched.status_code == 200
    assert TOKEN not in fetched.text

    patched = client.patch(
        f"/api/v1/datasources/{created['id']}", json={"name": "Renamed"}
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Renamed"
    assert patched.json()["credentials_configured"] == {"token": True}

    bad_cron = client.patch(
        f"/api/v1/datasources/{created['id']}", json={"schedule": "not a cron"}
    )
    assert bad_cron.status_code == 422

    for suffix in ("", "-wal", "-shm"):
        path = tmp_path / f"datasource.sqlite{suffix}"
        if path.exists():
            assert TOKEN.encode() not in path.read_bytes()


def test_preview_validate_resources_and_ancestors(ctx):
    client, _, _, _ = ctx
    good = client.post(
        "/api/v1/datasources/preview/validate",
        json={"type": "fake", "credentials": {"token": TOKEN}, "settings": {}},
    )
    assert good.status_code == 200 and good.json()["ok"] is True

    bad = client.post(
        "/api/v1/datasources/preview/validate",
        json={"type": "fake", "credentials": {"token": "bad"}, "settings": {}},
    )
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "auth_failed"

    roots = client.post(
        "/api/v1/datasources/preview/resources",
        json={"type": "fake", "credentials": {"token": TOKEN}, "settings": {}},
    )
    assert roots.status_code == 200
    assert roots.json()[0]["id"] == "root"

    children = client.post(
        "/api/v1/datasources/preview/resources",
        json={
            "type": "fake",
            "credentials": {"token": TOKEN},
            "settings": {},
            "parent_id": "root",
        },
    )
    assert children.status_code == 200
    assert children.json()[0]["id"] == "doc-1"

    ancestors = client.post(
        "/api/v1/datasources/preview/ancestors",
        json={
            "type": "fake",
            "credentials": {"token": TOKEN},
            "settings": {},
            "ids": ["doc-1"],
        },
    )
    assert ancestors.status_code == 200
    assert {node["id"] for node in ancestors.json()} == {"doc-1", "root"}


def test_manual_sync_logs_and_documents(ctx):
    client, _, _, _ = ctx
    created = _create(client)
    started = client.post(f"/api/v1/datasources/{created['id']}/sync")
    assert started.status_code == 202
    sync_id = started.json()["sync_id"]

    log = {}
    for _ in range(100):
        log = client.get(f"/api/v1/datasources/{created['id']}/syncs/{sync_id}").json()
        if log["status"] != "running":
            break
        time.sleep(0.02)
    assert log["status"] == "success", log
    assert log["counts"]["created"] == 1

    logs = client.get(f"/api/v1/datasources/{created['id']}/syncs").json()
    assert logs[0]["id"] == sync_id

    documents = client.get(f"/api/v1/datasources/{created['id']}/documents").json()
    assert documents[0]["external_id"] == "doc-1"
    assert documents[0]["file_name"] == "Route A.md"
    assert documents[0]["rag_source"] == f"ds:{created['id']}:doc-1"


def test_saved_resources_and_validate(ctx):
    client, _, _, _ = ctx
    created = _create(client)
    assert client.post(f"/api/v1/datasources/{created['id']}/validate").status_code == 200

    roots = client.get(f"/api/v1/datasources/{created['id']}/resources").json()
    assert roots[0]["id"] == "root"
    ancestors = client.get(
        f"/api/v1/datasources/{created['id']}/ancestors", params={"ids": "doc-1"}
    ).json()
    assert {node["id"] for node in ancestors} == {"doc-1", "root"}


def test_edit_credentials_merge_and_saved_validation(ctx):
    client, _, _, _ = ctx
    created = _create(client)

    blank = client.patch(
        f"/api/v1/datasources/{created['id']}", json={"credentials": {"token": ""}}
    )
    assert blank.status_code == 200, blank.text
    assert blank.json()["credentials_configured"] == {"token": True}
    assert client.post(f"/api/v1/datasources/{created['id']}/validate").status_code == 200

    bad = client.patch(
        f"/api/v1/datasources/{created['id']}", json={"credentials": {"token": "bad"}}
    )
    assert bad.status_code == 200, bad.text
    denied = client.post(f"/api/v1/datasources/{created['id']}/validate")
    assert denied.status_code == 400
    assert denied.json()["detail"]["code"] == "auth_failed"

    restored = client.patch(
        f"/api/v1/datasources/{created['id']}", json={"credentials": {"token": TOKEN}}
    )
    assert restored.status_code == 200
    assert client.post(f"/api/v1/datasources/{created['id']}/validate").status_code == 200


def test_concurrent_sync_returns_409(ctx):
    client, service, _, _ = ctx
    created = _create(client)
    service.store.set_sync_status(created["id"], "running", "2026-01-01T00:00:00+08:00")
    conflict = client.post(f"/api/v1/datasources/{created['id']}/sync")
    assert conflict.status_code == 409
    service.store.set_sync_status(created["id"], "idle")


def test_delete_with_and_without_purge(ctx):
    client, service, _, _ = ctx
    created = _create(client)
    service.run_sync_now(created["id"])
    source = f"ds:{created['id']}:doc-1"
    assert service.ingestor.store.get_parent(source) is not None

    deleted = client.delete(f"/api/v1/datasources/{created['id']}")
    assert deleted.status_code == 200
    assert service.ingestor.store.get_parent(source) is not None
    assert client.get(f"/api/v1/datasources/{created['id']}").status_code == 404

    second = _create(client)
    service.run_sync_now(second["id"])
    source2 = f"ds:{second['id']}:doc-1"
    assert service.ingestor.store.get_parent(source2) is not None
    purged = client.delete(
        f"/api/v1/datasources/{second['id']}", params={"purge_docs": "true"}
    )
    assert purged.status_code == 200
    assert service.ingestor.store.get_parent(source2) is None
