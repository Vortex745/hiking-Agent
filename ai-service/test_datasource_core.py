"""Contract tests for datasource core: cron, credential encryption and store."""

from __future__ import annotations

from datetime import datetime

import pytest

from datasource.errors import CredentialsUnreadable
from datasource.scheduler import next_run_at, validate_cron
from datasource.secrets import SecretBox
from datasource.store import DatasourceStore


def test_cron_validates_and_finds_next_minute():
    validate_cron("*/5 * * * *")
    with pytest.raises(ValueError):
        validate_cron("not-a-cron")
    with pytest.raises(ValueError):
        validate_cron("")

    assert next_run_at("*/5 * * * *", datetime(2026, 10, 30, 10, 1)) == datetime(
        2026, 10, 30, 10, 5
    )


def test_cron_weekday_and_month_boundaries():
    # 2026-10-30 is a Friday; next weekday 09:00 is Monday 2026-11-02.
    assert next_run_at("0 9 * * 1-5", datetime(2026, 10, 30, 10, 0)) == datetime(
        2026, 11, 2, 9, 0
    )
    # Sunday is 0 in cron.
    assert next_run_at("30 6 * * 0", datetime(2026, 10, 30, 10, 0)) == datetime(
        2026, 11, 1, 6, 30
    )


def test_secret_box_roundtrip_and_wrong_key():
    box = SecretBox.from_key_bytes(b"k" * 32)
    token = box.encrypt({"app_id": "abc", "app_secret": "super-secret"})
    assert "super-secret" not in token
    assert box.decrypt(token) == {"app_id": "abc", "app_secret": "super-secret"}

    other = SecretBox.from_key_bytes(b"x" * 32)
    with pytest.raises(CredentialsUnreadable):
        other.decrypt(token)


def test_secret_box_key_file_is_reused(tmp_path):
    key_path = tmp_path / "datasource.key"
    first = SecretBox.from_settings("", key_path)
    token = first.encrypt({"token": "t"})
    second = SecretBox.from_settings("", key_path)
    assert second.decrypt(token) == {"token": "t"}
    assert key_path.read_bytes() == key_path.read_bytes()


def test_store_crud_synced_docs_and_logs(tmp_path):
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ds = store.create_datasource(
        type="fake",
        name="Fake",
        credentials_enc="enc",
        settings={"base_url": "https://example.test"},
        selected_resources=["root"],
        schedule="*/5 * * * *",
        sync_deletions=True,
    )
    assert store.get_datasource(ds["id"])["name"] == "Fake"
    assert store.list_datasources()[0]["selected_resources"] == ["root"]

    updated = store.update_datasource(ds["id"], name="Renamed", sync_deletions=False)
    assert updated["name"] == "Renamed"
    assert updated["sync_deletions"] is False

    store.upsert_synced_doc(
        ds["id"],
        resource_id="root",
        external_id="doc-1",
        rev="1",
        rag_source=f"ds:{ds['id']}:doc-1",
        file_name="Doc.md",
    )
    store.upsert_synced_doc(
        ds["id"],
        resource_id="root",
        external_id="doc-1",
        rev="2",
        rag_source=f"ds:{ds['id']}:doc-1",
        file_name="Doc.md",
    )
    rows = store.list_synced_docs(ds["id"])
    assert len(rows) == 1 and rows[0]["rev"] == "2"
    assert store.count_synced_docs(ds["id"], only_with_source=True) == 1

    log_id = store.start_sync_log(ds["id"], "manual")
    store.finish_sync_log(
        log_id,
        status="success",
        counts={"discovered": 1, "fetched": 1, "created": 1},
        failures=[],
        enumeration={"root": {"complete": True}},
    )
    log = store.get_sync_log(log_id)
    assert log["status"] == "success" and log["created"] == 1
    assert store.list_sync_logs(ds["id"])[0]["id"] == log_id
    store.close()


def test_store_due_running_and_recovery(tmp_path):
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ds = store.create_datasource(
        type="fake",
        name="Due",
        credentials_enc="enc",
        selected_resources=["root"],
        schedule="* * * * *",
        next_run_at="2020-01-01T00:00:00+08:00",
    )
    assert [row["id"] for row in store.list_due("2026-01-01T00:00:00+08:00")] == [ds["id"]]

    store.set_sync_status(ds["id"], "running", "2026-01-01T00:00:00+08:00")
    assert store.list_due("2026-01-01T00:00:00+08:00") == []

    log_id = store.start_sync_log(ds["id"], "schedule")
    assert store.recover_interrupted() == 1
    assert store.get_sync_log(log_id)["status"] == "failed"
    assert store.get_sync_log(log_id)["error_code"] == "interrupted"
    assert store.get_datasource(ds["id"])["sync_status"] == "idle"
    store.close()


def test_claim_sync_is_atomic(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ds = store.create_datasource(
        type="fake", name="A", credentials_enc="enc", selected_resources=["root"]
    )
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        return store.claim_sync(ds["id"], "manual")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sum(1 for result in results if result) == 1
    store.close()


def test_deleting_datasource_cascades_rows(tmp_path):
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ds = store.create_datasource(
        type="fake", name="A", credentials_enc="enc", selected_resources=["root"]
    )
    store.upsert_synced_doc(
        ds["id"],
        resource_id="root",
        external_id="d",
        rev="1",
        rag_source="s",
        file_name="d.md",
    )
    store.start_sync_log(ds["id"], "manual")
    assert store.delete_datasource(ds["id"]) is True
    assert store.get_datasource(ds["id"]) is None
    assert store.list_synced_docs(ds["id"]) == []
    assert store.list_sync_logs(ds["id"]) == []
    store.close()
