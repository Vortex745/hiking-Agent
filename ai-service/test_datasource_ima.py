"""Tencent IMA connector contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from datasource.connectors.ima import ImaConnector, logical_key
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8
BASE = "https://ima.test"
SIGN = "sign-1"


def _media():
    return {
        "d1": {
            "media_type": 1,
            "title": "Doc 1",
            "url_info": {
                "url": "https://ima.test/files/d1.pdf",
                "headers": {"X-IMA-Sign": SIGN},
            },
        },
        "d2": {
            "media_type": 1,
            "title": "Doc 2",
            "url_info": {
                "url": "https://ima.test/files/d2.md",
                "headers": {"X-IMA-Sign": SIGN},
            },
        },
        "n1": {
            "media_type": 11,
            "title": "Note 1",
            "notebook_ext_info": {"notebook_id": "nb1"},
        },
        "d3": {
            "media_type": 2,
            "title": "Web",
            "url_info": {"url": "https://ima.test/files/d3.html", "headers": {}},
        },
        "d4": {"media_type": 12, "title": "Chat"},
    }


def _handler(state):
    def handler(request):
        path = request.url.path
        if path == "/openapi/wiki/v1/get_addable_knowledge_base_list":
            return _json({"code": 0, "data": {"items": state.get("kbs", [])}})
        if path == "/openapi/wiki/v1/search_knowledge_base":
            return _json({"code": 0, "data": {"items": [{"kb_id": "kb1", "kb_name": "KB"}]}})
        if path == "/openapi/wiki/v1/get_knowledge_list":
            body = json.loads(request.content or b"{}")
            folder = body.get("folder_id", "")
            if state.get("loop") and folder == "":
                return _json(
                    {
                        "code": 0,
                        "data": {
                            "items": [{"media_id": "f1", "title": "Folder", "media_type": 99}],
                            "next_cursor": "same",
                            "is_end": False,
                        },
                    }
                )
            return _json(
                {
                    "code": 0,
                    "data": {"items": state.get("folders", {}).get(folder, []), "is_end": True},
                }
            )
        if path == "/openapi/wiki/v1/get_media_info":
            body = json.loads(request.content or b"{}")
            media_id = str(body.get("media_id") or "")
            state.setdefault("media_calls", []).append(media_id)
            if media_id in state.get("fail_media", set()):
                return _json({"code": 500, "msg": "boom"}, status=500)
            return _json({"code": 0, "data": state["media"].get(media_id, {})})
        if path == "/openapi/note/v1/get_doc_content":
            return _json(
                {"code": 0, "data": {"content": state.get("note_content", "# Note\nhello")}}
            )
        if path.startswith("/files/"):
            name = path.rsplit("/", 1)[-1]
            if not name.endswith(".html") and request.headers.get("x-ima-sign") != SIGN:
                return _json({"msg": "missing header"}, status=403)
            if name.endswith(".pdf"):
                return httpx.Response(
                    200, content=b"%PDF-1.4", headers={"Content-Type": "application/pdf"}
                )
            if name.endswith(".md"):
                return httpx.Response(
                    200,
                    content=state.get("markdown", "# Doc\nhello").encode(),
                    headers={"Content-Type": "text/markdown"},
                )
            return httpx.Response(
                200, content=b"<html/>", headers={"Content-Type": "text/html"}
            )
        return _json({"code": 404, "msg": "not found"})

    return handler


def _json(payload, status=200):
    return httpx.Response(status, json=payload)


def _connector(handler):
    return ImaConnector(
        {"client_id": "cid", "api_key": "key"},
        {"base_url": BASE},
        transport=httpx.MockTransport(handler),
    )


def _folders_state():
    return {
        "kbs": [{"kb_id": "kb1", "kb_name": "KB"}],
        "folders": {
            "": [
                {"media_id": "f1", "title": "Folder", "media_type": 99},
                {"media_id": "d1", "title": "Doc 1", "media_type": 1},
            ],
            "f1": [
                {"media_id": "d2", "title": "Doc 2", "media_type": 1},
                {"media_id": "n1", "title": "Note 1", "media_type": 11},
            ],
        },
        "media": _media(),
    }


def test_kb_fallback_and_folder_tree():
    state = {"kbs": [], "folders": {}, "media": _media()}
    connector = _connector(_handler(state))
    assert [node.id for node in connector.list_resources(None)] == ["kb1"]
    state["folders"] = _folders_state()["folders"]
    assert [node.id for node in connector.list_resources("kb1")] == ["kb1:f1"]
    assert connector.list_resources("kb1:f1") == []
    assert connector.resolve_ancestors(["kb1:f1"]) == []
    connector.close()


def test_enumerate_logical_keys_and_folder_recursion():
    connector = _connector(_handler(_folders_state()))
    result = connector.enumerate("kb1")
    assert result.complete is True
    keys = {node.id: node for node in result.nodes}
    assert keys[logical_key("kb1", "", "Doc 1")].rev == "d1"
    assert keys[logical_key("kb1", "f1", "Doc 2")].rev == "d2"
    assert keys[logical_key("kb1", "f1", "Note 1")].rev == "n1"
    child_only = connector.enumerate("kb1:f1")
    assert {node.id for node in child_only.nodes} == {
        logical_key("kb1", "f1", "Doc 2"),
        logical_key("kb1", "f1", "Note 1"),
    }
    connector.close()


def test_logical_key_ignores_media_type_and_media_id():
    first = logical_key("kb1", "f1", "Doc")
    assert first == logical_key("kb1", "f1", "Doc")
    assert first != logical_key("kb1", "f2", "Doc")
    assert first != logical_key("kb1", "f1", "Other")
    assert first.startswith("ima_") and len(first) == 36


def test_fetch_pdf_markdown_note_and_skip():
    connector = _connector(_handler(_folders_state()))
    nodes = {node.id: node for node in connector.enumerate("kb1").nodes}
    pdf = connector.fetch(nodes[logical_key("kb1", "", "Doc 1")])
    assert pdf.item is not None
    assert pdf.item.content_type == "pdf"
    assert pdf.item.data == b"%PDF-1.4"
    assert pdf.item.file_name == "Doc 1.pdf"
    markdown = connector.fetch(nodes[logical_key("kb1", "f1", "Doc 2")])
    assert markdown.item is not None
    assert markdown.item.content.startswith("# Doc")
    assert markdown.item.file_name == "Doc 2.md"
    note = connector.fetch(nodes[logical_key("kb1", "f1", "Note 1")])
    assert note.item is not None and note.item.content == "# Note\nhello"

    state = _folders_state()
    state["folders"][""].append({"media_id": "d3", "title": "Web", "media_type": 2})
    state["folders"][""].append({"media_id": "d4", "title": "Chat", "media_type": 12})
    connector2 = _connector(_handler(state))
    nodes2 = {node.id: node for node in connector2.enumerate("kb1").nodes}
    assert connector2.fetch(nodes2[logical_key("kb1", "", "Web")]).skip_reason == "unsupported_type"
    assert connector2.fetch(nodes2[logical_key("kb1", "", "Chat")]).skip_reason == "unsupported_type"
    connector.close()
    connector2.close()


def test_pagination_loop_guard():
    state = _folders_state()
    state["loop"] = True
    connector = _connector(_handler(state))
    result = connector.enumerate("kb1")
    assert result.complete is False
    assert result.error == "vendor_error"
    connector.close()


def _make_service(tmp_path: Path, state):
    registry = ConnectorRegistry()
    from datasource.connectors import ima

    ima.register(registry, transport=httpx.MockTransport(_handler(state)))
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    service = DatasourceService(store, registry, ingestor, SecretBox.from_key_bytes(b"k" * 32))
    ds = service.create_datasource(
        type="ima",
        name="IMA",
        credentials={"client_id": "cid", "api_key": "key"},
        settings={"base_url": BASE},
        selected_resources=["kb1"],
    )
    return service, ingestor, ds


def test_engine_media_id_churn_is_update(tmp_path: Path):
    state = {
        "kbs": [{"kb_id": "kb1", "kb_name": "KB"}],
        "folders": {"": [{"media_id": "m1", "title": "Doc", "media_type": 1}]},
        "media": {
            "m1": {
                "media_type": 1,
                "title": "Doc",
                "url_info": {"url": "https://ima.test/files/m1.md", "headers": {"X-IMA-Sign": SIGN}},
            }
        },
        "markdown": "# Doc\nfirst",
    }
    service, ingestor, ds = _make_service(tmp_path, state)
    first = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert first["created"] == 1
    assert service.get_datasource(ds["id"])["doc_count"] == 1

    state["folders"][""] = [{"media_id": "m2", "title": "Doc", "media_type": 1}]
    state["media"]["m2"] = {
        "media_type": 1,
        "title": "Doc",
        "url_info": {"url": "https://ima.test/files/m2.md", "headers": {"X-IMA-Sign": SIGN}},
    }
    state["markdown"] = "# Doc\nsecond"
    second = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert second["updated"] == 1
    assert service.get_datasource(ds["id"])["doc_count"] == 1
    parent = ingestor.store.get_parent(f"ds:{ds['id']}:{logical_key('kb1', '', 'Doc')}")
    assert parent is not None
    service.close()


def test_note_update_time_triggers_update(tmp_path: Path):
    state = {
        "kbs": [{"kb_id": "kb1", "kb_name": "KB"}],
        "folders": {
            "": [
                {
                    "media_id": "n1",
                    "title": "Note",
                    "media_type": 11,
                    "update_time": "t1",
                }
            ]
        },
        "media": {
            "n1": {
                "media_type": 11,
                "title": "Note",
                "notebook_ext_info": {"notebook_id": "nb1"},
            }
        },
        "note_content": "# Note\none",
    }
    service, ingestor, ds = _make_service(tmp_path, state)
    first = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert first["created"] == 1

    state["folders"][""][0]["update_time"] = "t2"
    state["note_content"] = "# Note\ntwo"
    second = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert second["updated"] == 1
    assert service.get_datasource(ds["id"])["doc_count"] == 1
    parent = ingestor.store.get_parent(
        f"ds:{ds['id']}:{logical_key('kb1', '', 'Note')}"
    )
    assert parent is not None
    service.close()


def test_deterministic_skip_does_not_refetch(tmp_path: Path):
    state = {
        "kbs": [{"kb_id": "kb1", "kb_name": "KB"}],
        "folders": {"": [{"media_id": "s1", "title": "Chat", "media_type": 12}]},
        "media": {"s1": {"media_type": 12, "title": "Chat"}},
    }
    service, _, ds = _make_service(tmp_path, state)
    first = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert first["skipped_by_type"] == 1
    calls_after_first = len(state.get("media_calls", []))
    second = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert second["skipped"] == 1
    assert len(state.get("media_calls", [])) == calls_after_first
    service.close()


def test_transient_failure_retries_next_run(tmp_path: Path):
    state = {
        "kbs": [{"kb_id": "kb1", "kb_name": "KB"}],
        "folders": {"": [{"media_id": "f1", "title": "Flaky", "media_type": 1}]},
        "media": {
            "f1": {
                "media_type": 1,
                "title": "Flaky",
                "url_info": {"url": "https://ima.test/files/f1.md", "headers": {"X-IMA-Sign": SIGN}},
            }
        },
        "fail_media": {"f1"},
    }
    service, _, ds = _make_service(tmp_path, state)
    first = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert first["failed"] == 1
    assert service.store.list_synced_docs(ds["id"]) == []

    state["fail_media"] = set()
    second = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert second["created"] == 1
    service.close()
