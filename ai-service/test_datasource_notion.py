"""Notion renderer, search tree, page fetch and engine integration tests."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from datasource.connectors.notion import NotionConnector
from datasource.connectors.notion_markdown import render_blocks
from datasource.ingest import RagIngestor
from datasource.registry import ConnectorRegistry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore
from rag.ingestion import FakeEmbedding

DIMENSIONS = 8
BASE = "https://notion.test"


def _response(payload, status=200):
    return httpx.Response(status, json=payload)


def _renderer_blocks():
    return [
        {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "路线"}]}},
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"plain_text": "hello "},
                    {"plain_text": "world", "annotations": {"bold": True}},
                ]
            },
        },
        {
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"plain_text": "一"}]},
            "children": [
                {
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"plain_text": "二"}]},
                }
            ],
        },
        {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "带水"}], "checked": True}},
        {"type": "quote", "quote": {"rich_text": [{"plain_text": "注意安全"}]}},
        {"type": "code", "code": {"rich_text": [{"plain_text": "print(1)"}], "language": "python"}},
        {"type": "callout", "callout": {"rich_text": [{"plain_text": "提示"}]}},
        {
            "type": "toggle",
            "toggle": {"rich_text": [{"plain_text": "更多"}]},
            "children": [
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "内容"}]}}
            ],
        },
        {"type": "divider", "divider": {}},
        {
            "type": "table",
            "table": {},
            "children": [
                {"type": "table_row", "table_row": {"cells": [[{"plain_text": "名称"}], [{"plain_text": "数量"}]]}},
                {"type": "table_row", "table_row": {"cells": [[{"plain_text": "水"}], [{"plain_text": "2"}]]}},
            ],
        },
        {"type": "image", "image": {"caption": [{"plain_text": "图"}]}},
        {"type": "child_page", "child_page": {"title": "子页"}},
        {"type": "file", "file": {"name": "gpx"}},
        {"type": "bookmark", "bookmark": {"url": "https://example.test"}},
    ]


def test_renderer_snapshot():
    expected = "\n\n".join(
        [
            "## 路线",
            "hello **world**",
            "- 一\n  - 二",
            "- [x] 带水",
            "> 注意安全",
            "```python\nprint(1)\n```",
            "> 提示",
            "- 更多\n内容",
            "---",
            "| 名称 | 数量 |\n|---|---|\n| 水 | 2 |",
            "[图片: 图]",
            "[子页面: 子页]",
            "[附件: gpx]",
            "[链接](https://example.test)",
        ]
    )
    assert render_blocks(_renderer_blocks()) == expected


def _search_payload():
    return {
        "results": [
            {
                "object": "page",
                "id": "p1",
                "last_edited_time": "2026-01-01",
                "properties": {"title": {"type": "title", "title": [{"plain_text": "路线"}]}},
                "parent": {"type": "workspace"},
            },
            {
                "object": "page",
                "id": "p2",
                "last_edited_time": "2026-01-02",
                "properties": {"title": {"type": "title", "title": [{"plain_text": "子页"}]}},
                "parent": {"type": "page_id", "page_id": "p1"},
            },
            {
                "object": "database",
                "id": "db1",
                "last_edited_time": "2026-01-03",
                "title": [{"plain_text": "装备库"}],
                "parent": {"type": "workspace"},
            },
        ],
        "has_more": False,
    }


def _handler(token_calls=None, sleeps=None):
    def handler(request):
        path = request.url.path
        if path == "/v1/search":
            if token_calls is not None:
                token_calls.append(1)
            return _response(_search_payload())
        if path == "/v1/blocks/p1/children":
            return _response(
                {
                    "results": [
                        {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "路线"}]}},
                        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "hello"}]}},
                    ],
                    "has_more": False,
                }
            )
        if path == "/v1/blocks/p2/children":
            return _response({"results": [], "has_more": False})
        if path == "/v1/blocks/db1/children":
            return _response({"results": [], "has_more": False})
        return _response({}, 404)

    return handler


def _connector(handler, *, sleeps=None):
    recorder = (lambda seconds: sleeps.append(seconds)) if sleeps is not None else (lambda _: None)
    return NotionConnector(
        {"token": "notion-token"},
        {"base_url": BASE, "requests_per_minute": "600000"},
        transport=httpx.MockTransport(handler),
        sleep_fn=recorder,
    )


def test_search_tree_and_enumerate():
    connector = _connector(_handler())
    nodes = connector.list_resources(None)
    assert [node.id for node in nodes] == ["p1", "db1"]
    assert next(node for node in nodes if node.id == "db1").kind == "database"
    assert [node.id for node in connector.enumerate("p1").nodes] == ["p1", "p2"]
    assert [node.id for node in connector.enumerate("db1").nodes] == ["db1"]
    assert [node.id for node in connector.list_resources("p1")] == ["p2"]
    assert connector.list_resources("p2") == []
    ancestors = connector.resolve_ancestors(["p2"])
    assert [node.id for node in ancestors] == ["p1", "p2"]
    assert ancestors[0].parent_id is None
    assert ancestors[1].parent_id == "p1"
    connector.close()


def test_fetch_page_and_database_skip():
    connector = _connector(_handler())
    roots = connector.list_resources(None)
    nodes = {node.id: node for node in roots}
    nodes.update({node.id: node for node in connector.list_resources("p1")})
    outcome = connector.fetch(nodes["p1"])
    assert outcome.item is not None
    assert "## 路线" in outcome.item.content
    assert "hello" in outcome.item.content
    assert outcome.item.file_name == "路线.md"
    assert connector.fetch(nodes["db1"]).skip_reason == "db_unsupported"
    empty = connector.fetch(nodes["p2"]).item
    assert empty is not None and empty.content == ""
    connector.close()


def test_block_truncation_is_reported():
    def handler(request):
        if request.url.path == "/v1/search":
            return _response(_search_payload())
        if request.url.path == "/v1/blocks/p1/children":
            return _response(
                {
                    "results": [
                        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "x"}]}}
                        for _ in range(100)
                    ],
                    "has_more": True,
                    "next_cursor": "next",
                }
            )
        return _response({"results": [], "has_more": False})

    connector = _connector(handler)
    nodes = {node.id: node for node in connector.list_resources(None)}
    outcome = connector.fetch(nodes["p1"])
    assert outcome.truncated is True
    connector.close()


def test_rate_limit_retries_with_retry_after():
    sleeps: list[float] = []
    state = {"calls": 0}

    def handler(request):
        if request.url.path == "/v1/search":
            state["calls"] += 1
            if state["calls"] == 1:
                return httpx.Response(429, json={}, headers={"Retry-After": "2"})
            return _response(_search_payload())
        return _response({}, 404)

    connector = _connector(handler, sleeps=sleeps)
    assert connector.list_resources(None)[0].id == "p1"
    assert any(abs(value - 2.0) < 0.01 for value in sleeps)
    connector.close()


def test_engine_sync_pages_and_skips_database(tmp_path: Path):
    registry = ConnectorRegistry()
    from datasource.connectors import notion

    notion.register(registry, transport=httpx.MockTransport(_handler()))
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    ingestor = RagIngestor(
        tmp_path / "rag.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    service = DatasourceService(store, registry, ingestor, SecretBox.from_key_bytes(b"k" * 32))
    ds = service.create_datasource(
        type="notion",
        name="Notion",
        credentials={"token": "notion-token"},
        settings={"base_url": BASE},
        selected_resources=["p1"],
    )
    log = service.get_sync_log(ds["id"], service.run_sync_now(ds["id"]))
    assert log["status"] == "success"
    assert log["created"] == 2
    parent = ingestor.store.get_parent(f"ds:{ds['id']}:p2")
    assert parent is not None
    assert parent["file_name"] == "子页.md"
    service.close()

    registry2 = ConnectorRegistry()
    notion.register(registry2, transport=httpx.MockTransport(_handler()))
    store2 = DatasourceStore(tmp_path / "datasource2.sqlite")
    ingestor2 = RagIngestor(
        tmp_path / "rag2.sqlite", DIMENSIONS, embedding_provider=FakeEmbedding(DIMENSIONS)
    )
    service2 = DatasourceService(
        store2, registry2, ingestor2, SecretBox.from_key_bytes(b"k" * 32)
    )
    ds2 = service2.create_datasource(
        type="notion",
        name="Notion DB",
        credentials={"token": "notion-token"},
        settings={"base_url": BASE},
        selected_resources=["db1"],
    )
    log2 = service2.get_sync_log(ds2["id"], service2.run_sync_now(ds2["id"]))
    assert log2["created"] == 0
    assert log2["skipped_by_type"] == 1
    service2.close()
