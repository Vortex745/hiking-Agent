"""Yuque connector contract tests at the vendor HTTP boundary."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from datasource.connectors import yuque
from datasource.errors import VendorError
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from datasource.types import ResourceNode
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8


def _response(payload, status=200):
    return httpx.Response(status, json=payload)


def _connector(handler, **kwargs):
    return yuque.YuqueConnector(
        {"token": "yuque-token"},
        {"base_url": "https://www.yuque.test"},
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_team_token_lists_only_books():
    def handler(request):
        if request.url.path == "/api/v2/user":
            return _response({"data": {"id": 1, "login": "team", "type": "Group"}})
        if request.url.path == "/api/v2/groups/team/repos":
            return _response(
                {
                    "data": [
                        {"id": 10, "name": "Book A", "type": "Book"},
                        {"id": 11, "name": "Sheet A", "type": "Sheet"},
                    ]
                }
            )
        return _response({}, 404)

    connector = _connector(handler)
    assert connector.validate() is None
    nodes = connector.list_resources(None)
    assert [node.id for node in nodes] == ["10"]
    assert nodes[0].title == "Book A"
    assert connector.list_resources("10") == []
    assert connector.resolve_ancestors(["10"]) == []
    connector.close()


def test_personal_token_and_missing_groups_are_empty():
    def handler(request):
        path = request.url.path
        if path == "/api/v2/user":
            return _response({"data": {"id": 2, "login": "alice", "type": "User"}})
        if path == "/api/v2/users/alice/repos":
            return _response({"data": [{"id": 20, "name": "My Book", "type": "Book"}]})
        if path == "/api/v2/users/2/groups":
            return _response({}, 404)
        return _response({}, 404)

    connector = _connector(handler)
    assert [node.id for node in connector.list_resources(None)] == ["20"]
    connector.close()


def test_group_permission_denied_is_skipped():
    def handler(request):
        path = request.url.path
        if path == "/api/v2/user":
            return _response({"data": {"id": 3, "login": "bob", "type": "User"}})
        if path == "/api/v2/users/bob/repos":
            return _response({"data": []})
        if path == "/api/v2/users/3/groups":
            return _response({"data": [{"login": "g1"}, {"login": "g2"}]})
        if path == "/api/v2/groups/g1/repos":
            return _response({"message": "forbidden"}, 403)
        if path == "/api/v2/groups/g2/repos":
            return _response({"data": [{"id": 30, "name": "G2 Book", "type": "Book"}]})
        return _response({}, 404)

    connector = _connector(handler)
    assert [node.id for node in connector.list_resources(None)] == ["30"]
    connector.close()


def test_enumerate_and_fetch_skip_rules():
    def handler(request):
        path = request.url.path
        if path == "/api/v2/repos/book-1/docs":
            return _response(
                {
                    "data": [
                        {
                            "id": 1,
                            "title": "路线",
                            "type": "Doc",
                            "status": 1,
                            "format": "markdown",
                            "content_updated_at": "2026-01-01T00:00:00+08:00",
                        },
                        {"id": 2, "title": "表格", "type": "Sheet", "status": 1, "format": "markdown"},
                        {"id": 3, "title": "草稿", "type": "Doc", "status": 0, "format": "markdown"},
                        {"id": 4, "title": "HTML", "type": "Doc", "status": 1, "format": "html"},
                    ]
                }
            )
        if path == "/api/v2/repos/docs/1":
            return _response(
                {
                    "data": {
                        "title": "路线",
                        "body": "# 路线\nhello route",
                        "format": "markdown",
                        "content_updated_at": "2026-02-02T00:00:00+08:00",
                    }
                }
            )
        return _response({}, 404)

    connector = _connector(handler)
    enumerated = connector.enumerate("book-1")
    assert len(enumerated.nodes) == 4
    assert enumerated.complete is True
    assert enumerated.nodes[0].rev == "2026-01-01T00:00:00+08:00"

    item = connector.fetch(enumerated.nodes[0]).item
    assert item is not None
    assert item.content == "# 路线\nhello route"
    assert item.file_name == "路线.md"
    assert item.rev == "2026-02-02T00:00:00+08:00"
    assert item.resource_id == "book-1"

    assert connector.fetch(enumerated.nodes[1]).skip_reason == "unsupported_type"
    assert connector.fetch(enumerated.nodes[2]).skip_reason == "draft"
    assert connector.fetch(enumerated.nodes[3]).skip_reason == "unsupported_type"
    connector.close()


def test_detail_calls_are_throttled():
    sleeps: list[float] = []

    def handler(request):
        if request.url.path == "/api/v2/repos/docs/1":
            return _response({"data": {"title": "A", "body": "a", "format": "markdown"}})
        return _response({}, 404)

    connector = _connector(handler, sleep_fn=lambda seconds: sleeps.append(seconds))
    node = ResourceNode(
        id="1",
        title="A",
        parent_id="book-1",
        meta={"type": "Doc", "status": 1, "format": "markdown", "book_id": "book-1"},
    )
    connector.fetch(node)
    connector.fetch(node)
    assert sleeps and 0.29 <= sleeps[0] <= 0.31
    connector.close()


def test_detail_failure_raises_stable_error():
    sleeps: list[float] = []

    def handler(request):
        if request.url.path == "/api/v2/repos/docs/1":
            return _response({"message": "boom"}, 500)
        return _response({}, 404)

    connector = _connector(handler, sleep_fn=lambda seconds: sleeps.append(seconds))
    node = ResourceNode(
        id="1",
        title="A",
        parent_id="book-1",
        meta={"type": "Doc", "status": 1, "format": "markdown", "book_id": "book-1"},
    )
    with pytest.raises(VendorError):
        connector.fetch(node)
    assert len(sleeps) == 2  # two retries after the first failure
    connector.close()


def test_pagination_detects_truncation():
    def handler(request):
        offset = int(request.url.params.get("offset", "0"))
        if request.url.path == "/api/v2/repos/book-1/docs":
            docs = [
                {
                    "id": offset + index,
                    "title": f"Doc {offset + index}",
                    "type": "Doc",
                    "status": 1,
                    "format": "markdown",
                    "content_updated_at": "2026-01-01",
                }
                for index in range(100)
            ]
            return _response({"data": docs})
        return _response({}, 404)

    connector = _connector(handler)
    enumerated = connector.enumerate("book-1")
    assert len(enumerated.nodes) == 1000
    assert enumerated.complete is False
    assert enumerated.truncated is True
    connector.close()


def test_yuque_sync_ingests_markdown(tmp_path: Path):
    def handler(request):
        path = request.url.path
        if path == "/api/v2/repos/book-1/docs":
            return _response(
                {
                    "data": [
                        {
                            "id": 1,
                            "title": "路线",
                            "type": "Doc",
                            "status": 1,
                            "format": "markdown",
                            "content_updated_at": "2026-01-01",
                        }
                    ]
                }
            )
        if path == "/api/v2/repos/docs/1":
            return _response(
                {"data": {"title": "路线", "body": "# 路线\nhello route", "format": "markdown"}}
            )
        return _response({}, 404)

    registry = ConnectorRegistry()
    yuque.register(registry, transport=httpx.MockTransport(handler))
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    service = DatasourceService(
        store, registry, ingestor, SecretBox.from_key_bytes(b"k" * 32)
    )
    ds = service.create_datasource(
        type="yuque",
        name="语雀",
        credentials={"token": "yuque-token"},
        settings={},
        selected_resources=["book-1"],
    )
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert log["status"] == "success"
    assert log["created"] == 1
    parent = ingestor.store.get_parent(f"ds:{ds['id']}:1")
    assert parent is not None
    assert parent["file_name"] == "路线.md"
    service.close()
