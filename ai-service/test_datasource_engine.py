"""End-to-end sync engine acceptance with the fake connector."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from datasource.connectors.fake import FakeDoc, FakeState, register_fake
from datasource.errors import ValidationError
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from datasource.types import ResourceNode
from rag.hybrid_retrieval import search_child_chunks
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8


def _build(tmp_path: Path, state: FakeState, **service_kwargs):
    ds_store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite",
        DIMENSIONS,
        embedding_provider=FakeEmbedding(DIMENSIONS),
    )
    registry = ConnectorRegistry()
    register_fake(registry, state)
    service = DatasourceService(
        ds_store,
        registry,
        ingestor,
        SecretBox.from_key_bytes(b"k" * 32),
        **service_kwargs,
    )
    return service, ds_store, ingestor.store


def _create(service: DatasourceService) -> dict:
    return service.create_datasource(
        type="fake",
        name="Fake",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root"],
    )


def _docs(*specs: tuple[str, str, str]) -> FakeState:
    return FakeState(
        docs=[
            FakeDoc(external_id=external_id, title=title, content=content)
            for external_id, title, content in specs
        ]
    )


def test_sync_creates_documents_and_retrieval_hits(tmp_path):
    service, _, rag_store = _build(
        tmp_path, _docs(("doc-1", "Route A", "alpine lake trail"), ("doc-2", "Route B", "ridge camp"))
    )
    ds = _create(service)
    log_id = service.run_sync_now(ds["id"])
    log = service.get_sync_log(ds["id"], log_id)

    assert log["status"] == "success"
    assert log["created"] == 2
    assert service.get_datasource(ds["id"])["doc_count"] == 2
    assert rag_store.get_parent(f"ds:{ds['id']}:doc-1")["file_name"] == "Route A.md"

    hits = search_child_chunks(
        rag_store.read_conn,
        FakeEmbedding(DIMENSIONS).embed_query("alpine lake trail"),
        query_text="alpine lake trail",
        k=4,
        embedding_dimensions=DIMENSIONS,
    )
    assert any(hit["source"] == f"ds:{ds['id']}:doc-1" for hit in hits)
    service.close()


def test_second_sync_skips_unchanged_without_fetching(tmp_path):
    state = _docs(("doc-1", "A", "a"), ("doc-2", "B", "b"))
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    service.run_sync_now(ds["id"])
    calls_after_first = len(state.fetch_calls)
    log_id = service.run_sync_now(ds["id"])
    log = service.get_sync_log(ds["id"], log_id)

    assert log["status"] == "success"
    assert log["skipped"] == 2
    assert log["created"] == 0
    assert len(state.fetch_calls) == calls_after_first
    service.close()


def test_changed_document_replaces_old_copy(tmp_path):
    state = _docs(("doc-1", "A", "old content"), ("doc-2", "B", "b"))
    service, store, rag_store = _build(tmp_path, state)
    ds = _create(service)
    service.run_sync_now(ds["id"])

    state.docs[0].content = "new content"
    state.docs[0].rev = "2"
    log_id = service.run_sync_now(ds["id"])
    log = service.get_sync_log(ds["id"], log_id)

    assert log["updated"] == 1
    assert service.get_datasource(ds["id"])["doc_count"] == 2
    parent = rag_store.get_parent(f"ds:{ds['id']}:doc-1")
    assert "new content" in rag_store.read_conn.execute(
        "SELECT content FROM parent WHERE source = ?", (parent["source"],)
    ).fetchone()[0]
    service.close()


def test_fetch_failure_does_not_advance_cursor(tmp_path):
    state = _docs(("doc-1", "A", "a"))
    state.docs[0].fail = True
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    first = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))

    assert first["failed"] == 1
    assert store.list_synced_docs(ds["id"]) == []

    state.docs[0].fail = False
    second = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert second["created"] == 1
    assert len(store.list_synced_docs(ds["id"])) == 1
    service.close()


def test_partial_enumeration_never_deletes(tmp_path):
    state = _docs(("doc-1", "A", "a"), ("doc-2", "B", "b"))
    state.roots.append(ResourceNode(id="root2", title="第二空间", kind="space"))
    state.docs[1].resource_id = "root2"
    service, store, _ = _build(tmp_path, state)
    ds = service.create_datasource(
        type="fake",
        name="Fake",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root", "root2"],
    )
    service.run_sync_now(ds["id"])
    state.fail_enumerate.add("root2")
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))

    assert log["status"] == "partial"
    assert log["deleted"] == 0
    assert len(store.list_synced_docs(ds["id"])) == 2
    service.close()


def test_empty_enumeration_trips_deletion_guard(tmp_path):
    state = _docs(("doc-1", "A", "a"), ("doc-2", "B", "b"))
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    service.run_sync_now(ds["id"])
    state.docs.clear()
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))

    assert log["status"] == "partial"
    assert log["error_code"] == "deletion_guard_tripped"
    assert log["deleted"] == 0
    assert len(store.list_synced_docs(ds["id"])) == 2
    service.close()


def test_bulk_deletion_guard_and_normal_deletion(tmp_path):
    state = _docs(*[(f"doc-{i}", f"D{i}", "x") for i in range(10)])
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    service.run_sync_now(ds["id"])

    # One deletion is below max(5, 20% of 10) = 5.
    state.docs = [doc for doc in state.docs if doc.external_id != "doc-0"]
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert log["deleted"] == 1
    assert len(store.list_synced_docs(ds["id"])) == 9

    # Removing six of nine exceeds the guard.
    state.docs = state.docs[:3]
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert log["error_code"] == "deletion_guard_tripped"
    assert log["deleted"] == 0
    assert len(store.list_synced_docs(ds["id"])) == 9
    service.close()


def test_scheduler_starts_due_datasource_and_advances_next_run(tmp_path):
    state = _docs(("doc-1", "A", "a"))
    service, store, _ = _build(tmp_path, state)
    ds = service.create_datasource(
        type="fake",
        name="Scheduled",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root"],
        schedule="*/5 * * * *",
    )
    store.set_next_run(ds["id"], "2020-01-01T00:00:00+08:00")

    started = service.run_due_once("2026-01-01T00:00:00+08:00")
    assert len(started) == 1
    for _ in range(100):
        log = service.store.get_sync_log(started[0])
        if log["status"] != "running":
            break
        import time

        time.sleep(0.02)
    assert service.store.get_sync_log(started[0])["status"] == "success"
    next_run = service.get_datasource(ds["id"])["next_run_at"]
    assert next_run is not None and next_run > "2026-01-01"
    service.close()


def test_scheduler_loop_starts_due_source_and_stops(tmp_path):
    state = _docs(("doc-1", "A", "a"))
    service, store, _ = _build(tmp_path, state, scheduler_interval_seconds=0.05)
    ds = service.create_datasource(
        type="fake",
        name="Loop",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root"],
        schedule="* * * * *",
    )
    store.set_next_run(ds["id"], "2020-01-01T00:00:00+08:00")

    async def run():
        service.start_scheduler()
        logs = []
        for _ in range(100):
            logs = store.list_sync_logs(ds["id"])
            if logs and logs[0]["status"] != "running":
                break
            await asyncio.sleep(0.02)
        await service.stop_scheduler()
        return logs

    logs = asyncio.run(run())
    assert len(logs) == 1
    assert logs[0]["status"] == "success"
    assert logs[0]["created"] == 1
    service.close()


def test_restart_persists_cursor_and_skips(tmp_path):
    state = _docs(("doc-1", "A", "a"), ("doc-2", "B", "b"))
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    service.run_sync_now(ds["id"])
    fetch_calls = len(state.fetch_calls)
    service.close()

    restarted, _, _ = _build(tmp_path, state)
    log = restarted.get_sync_log(ds["id"], restarted.run_sync_now(ds["id"]))
    assert log["skipped"] == 2
    assert log["created"] == 0
    assert len(state.fetch_calls) == fetch_calls
    assert restarted.get_datasource(ds["id"])["doc_count"] == 2
    restarted.close()


def test_two_sources_queue_on_single_worker(tmp_path):
    state = _docs(("doc-1", "A", "a"))
    state.roots.append(ResourceNode(id="root2", title="第二空间", kind="space"))
    state.docs.append(FakeDoc(external_id="doc-2", title="B", content="b", resource_id="root2"))
    service, store, _ = _build(tmp_path, state)
    first = service.create_datasource(
        type="fake",
        name="First",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root"],
        schedule="* * * * *",
    )
    second = service.create_datasource(
        type="fake",
        name="Second",
        credentials={"token": "ok"},
        settings={},
        selected_resources=["root2"],
        schedule="* * * * *",
    )
    store.set_next_run(first["id"], "2020-01-01T00:00:00+08:00")
    store.set_next_run(second["id"], "2020-01-01T00:00:00+08:00")
    started = service.run_due_once("2026-01-01T00:00:00+08:00")
    assert len(started) == 2
    for sync_id in started:
        for _ in range(100):
            log = store.get_sync_log(sync_id)
            if log["status"] != "running":
                break
            import time

            time.sleep(0.02)
        assert store.get_sync_log(sync_id)["status"] == "success"
    assert service.get_datasource(first["id"])["doc_count"] == 1
    assert service.get_datasource(second["id"])["doc_count"] == 1
    service.close()


def test_corrupt_schedule_does_not_claim_sync(tmp_path):
    service, store, _ = _build(tmp_path, _docs(("doc-1", "A", "a")))
    ds = _create(service)
    store.update_datasource(ds["id"], schedule="bad cron")
    with pytest.raises(ValidationError):
        service.start_sync(ds["id"], "manual")
    assert store.get_datasource(ds["id"])["sync_status"] == "idle"
    assert store.list_sync_logs(ds["id"]) == []
    service.close()


def test_new_service_recovers_interrupted_run(tmp_path):
    state = _docs(("doc-1", "A", "a"))
    service, store, _ = _build(tmp_path, state)
    ds = _create(service)
    log_id = store.start_sync_log(ds["id"], "manual")
    store.set_sync_status(ds["id"], "running", "2026-01-01T00:00:00+08:00")

    restarted, _, _ = _build(tmp_path, state)
    assert store.get_sync_log(log_id)["status"] == "failed"
    assert store.get_datasource(ds["id"])["sync_status"] == "idle"
    restarted.close()
    service.close()
