"""Feishu Wiki connector contract tests at the HTTP boundary."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from datasource.connectors.feishu import FeishuConnector
from datasource.errors import VendorError
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from datasource.types import ResourceNode
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8
BASE = "https://open.feishu.test"


def _response(payload, status=200, headers=None):
    return httpx.Response(status, json=payload, headers=headers or {})


def _connector(handler, *, sleeps=None, wall_clock=None):
    recorder = (lambda seconds: sleeps.append(seconds)) if sleeps is not None else (lambda _: None)
    return FeishuConnector(
        {"app_id": "app", "app_secret": "secret"},
        {"base_url": BASE, "requests_per_minute": "600000"},
        transport=httpx.MockTransport(handler),
        sleep_fn=recorder,
        clock_fn=lambda: 100.0,
        wall_clock_fn=wall_clock or (lambda: 1000.0),
    )


def _base_handler(token_counter=None):
    def handler(request):
        path = request.url.path
        if path == "/open-apis/auth/v3/tenant_access_token/internal":
            if token_counter is not None:
                token_counter.append(1)
            return _response({"code": 0, "tenant_access_token": "tenant", "expire": 7200})
        if path == "/open-apis/wiki/v2/spaces":
            return _response(
                {"code": 0, "data": {"items": [{"space_id": "s1", "name": "Space 1"}], "has_more": False}}
            )
        if path == "/open-apis/wiki/v2/spaces/s1/nodes":
            parent = request.url.params.get("parent_node_token", "")
            if parent == "":
                items = [
                    {
                        "node_token": "n1",
                        "obj_token": "obj1",
                        "obj_type": "docx",
                        "title": "路线",
                        "has_child": True,
                        "obj_edit_time": "111",
                    },
                    {
                        "node_token": "n2",
                        "obj_token": "obj2",
                        "obj_type": "sheet",
                        "title": "表格",
                        "has_child": False,
                        "obj_edit_time": "222",
                    },
                ]
            elif parent == "n1":
                items = [
                    {
                        "node_token": "n3",
                        "obj_token": "obj3",
                        "obj_type": "docx",
                        "title": "子文档",
                        "has_child": False,
                        "obj_edit_time": "333",
                    }
                ]
            else:
                items = []
            return _response({"code": 0, "data": {"items": items, "has_more": False}})
        if path == "/open-apis/wiki/v2/spaces/get_node":
            token = request.url.params.get("token")
            if token == "n2":
                node = {"node_token": "n2", "obj_token": "obj2", "obj_type": "sheet", "title": "表格",
                        "parent_node_token": "n1", "obj_edit_time": "222"}
            elif token == "n1":
                node = {"node_token": "n1", "obj_token": "obj1", "obj_type": "docx", "title": "路线",
                        "parent_node_token": "", "obj_edit_time": "111"}
            else:
                return _response({"code": 400, "msg": "not found"}, 404)
            return _response({"code": 0, "data": {"node": node}})
        if path == "/open-apis/docs/v1/content":
            token = request.url.params.get("doc_token")
            if token == "obj1":
                return _response({"code": 0, "data": {"title": "路线", "content": "# 路线\nhello"}})
            if token == "obj3":
                return _response({"code": 0, "data": {"title": "子文档", "content": "child"}})
            return _response({"code": 400, "msg": "not found"}, 404)
        return _response({"code": 400, "msg": "unknown"}, 404)

    return handler


def test_token_is_cached_and_tree_is_lazy():
    token_calls: list[int] = []
    connector = _connector(_base_handler(token_calls))
    spaces = connector.list_resources(None)
    assert [space.id for space in spaces] == ["s1"]
    assert [node.id for node in connector.list_resources("s1")] == ["s1:n1", "s1:n2"]
    assert [node.id for node in connector.list_resources("s1:n1")] == ["s1:n3"]
    assert len(token_calls) == 1
    connector.close()


def test_ancestors_walk_parent_tokens():
    connector = _connector(_base_handler())
    ancestors = connector.resolve_ancestors(["s1:n2"])
    assert [node.id for node in ancestors] == ["s1", "s1:n1", "s1:n2"]
    connector.close()


def test_enumerate_and_fetch_rules():
    connector = _connector(_base_handler())
    enumerated = connector.enumerate("s1")
    assert enumerated.complete is True
    ids = [node.id for node in enumerated.nodes]
    assert ids == ["s1:n1", "s1:n2", "s1:n3"]
    assert enumerated.nodes[0].rev == "111"

    item = connector.fetch(enumerated.nodes[0]).item
    assert item is not None
    assert item.content == "# 路线\nhello"
    assert item.file_name == "路线.md"
    assert connector.fetch(enumerated.nodes[1]).skip_reason == "unsupported_type"
    connector.close()


def test_partial_listing_marks_incomplete():
    def handler(request):
        if request.url.path.endswith("/nodes") and request.url.params.get("parent_node_token") == "n1":
            return _response({"code": 500, "msg": "boom"}, 500)
        return _base_handler()(request)

    connector = _connector(handler)
    enumerated = connector.enumerate("s1")
    assert [node.id for node in enumerated.nodes] == ["s1:n1", "s1:n2"]
    assert enumerated.complete is False
    assert enumerated.error in {"vendor_error", "enumeration_incomplete"}
    connector.close()


def test_rate_limit_retry_after_and_reset_header():
    sleeps: list[float] = []
    state = {"calls": 0}

    def handler(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["calls"] += 1
            if state["calls"] == 1:
                return _response({"code": 99991400}, 429, {"Retry-After": "3"})
        return _base_handler()(request)

    connector = _connector(handler, sleeps=sleeps)
    assert connector.list_resources(None)[0].id == "s1"
    assert any(abs(value - 3.0) < 0.01 for value in sleeps)

    sleeps.clear()
    state["calls"] = 0

    def handler_reset(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["calls"] += 1
            if state["calls"] == 1:
                return _response({"code": 99991400}, 429, {"x-ogw-ratelimit-reset": "1005"})
        return _base_handler()(request)

    connector2 = _connector(handler_reset, sleeps=sleeps, wall_clock=lambda: 1000.0)
    assert connector2.list_resources(None)[0].id == "s1"
    assert any(abs(value - 5.0) < 0.01 for value in sleeps)
    connector.close()
    connector2.close()


def test_transport_error_retries_and_succeeds():
    sleeps: list[float] = []
    state = {"calls": 0}

    def handler(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["calls"] += 1
            if state["calls"] == 1:
                raise httpx.ConnectError("boom", request=request)
        return _base_handler()(request)

    connector = _connector(handler, sleeps=sleeps)
    assert connector.list_resources(None)[0].id == "s1"
    assert any(abs(value - 2.0) < 0.01 for value in sleeps)
    connector.close()


def test_transport_retries_three_times_with_2_4_8():
    sleeps: list[float] = []
    state = {"spaces": 0}

    def handler(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["spaces"] += 1
            if state["spaces"] <= 3:
                raise httpx.ConnectError("boom", request=request)
        return _base_handler()(request)

    connector = _connector(handler, sleeps=sleeps)
    assert connector.list_resources(None)[0].id == "s1"
    for expected in (2.0, 4.0, 8.0):
        assert any(abs(value - expected) < 0.01 for value in sleeps)
    connector.close()


def test_5xx_retries_only_once():
    sleeps: list[float] = []
    state = {"spaces": 0}

    def handler(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["spaces"] += 1
            if state["spaces"] <= 2:
                return _response({"code": 500}, 500)
        return _base_handler()(request)

    connector = _connector(handler, sleeps=sleeps)
    with pytest.raises(VendorError):
        connector.list_resources(None)
    assert sum(1 for value in sleeps if abs(value - 2.0) < 0.01) == 1
    connector.close()


def test_5xx_retries_once():
    sleeps: list[float] = []
    state = {"calls": 0}

    def handler(request):
        if request.url.path == "/open-apis/wiki/v2/spaces":
            state["calls"] += 1
            if state["calls"] == 1:
                return _response({"code": 500}, 500)
        return _base_handler()(request)

    connector = _connector(handler, sleeps=sleeps)
    assert connector.list_resources(None)[0].id == "s1"
    assert any(abs(value - 2.0) < 0.01 for value in sleeps)
    connector.close()


def test_engine_sync_ingests_docx(tmp_path: Path):
    registry = ConnectorRegistry()
    from datasource.connectors import feishu

    feishu.register(registry, transport=httpx.MockTransport(_base_handler()))
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    service = DatasourceService(store, registry, ingestor, SecretBox.from_key_bytes(b"k" * 32))
    ds = service.create_datasource(
        type="feishu",
        name="飞书",
        credentials={"app_id": "app", "app_secret": "secret"},
        settings={"base_url": BASE},
        selected_resources=["s1"],
    )
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert log["status"] == "success"
    assert log["created"] == 2
    parent = ingestor.store.get_parent(f"ds:{ds['id']}:s1:n1")
    assert parent is not None
    assert parent["file_name"] == "路线.md"
    service.close()
