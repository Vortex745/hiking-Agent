"""Notion page connector with a pinned API version and page-only v1 scope."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import httpx

from datasource.registry import ConnectorRegistry
from datasource.connectors.notion_markdown import render_blocks, render_rich_text
from datasource.errors import AuthError, NotFoundError
from datasource.http import HttpClient, json_response
from datasource.ingest import sanitize_file_name
from datasource.ratelimit import TokenBucket
from datasource.types import (
    AuthField,
    ConnectorMetadata,
    EnumerationResult,
    FetchedItem,
    FetchOutcome,
    ResourceNode,
    SettingField,
)

logger = logging.getLogger("ai-service.datasource.notion")

NOTION_VERSION = "2026-03-11"
DEFAULT_BASE_URL = "https://api.notion.com"
MAX_SEARCH_PAGES = 10
MAX_BLOCK_PAGES = 10
MAX_BLOCKS = 1000
MAX_DEPTH = 5
RECURSIVE_TYPES = {"table", "toggle", "column_list", "column", "synced_block"}

NOTION_METADATA = ConnectorMetadata(
    type="notion",
    label="Notion",
    auth_fields=(
        AuthField(
            key="token",
            label="Integration Token",
            secret=True,
            help="在 Notion 创建 Integration，并把目标页面 Add connections",
        ),
    ),
    setting_fields=(
        SettingField(key="base_url", label="API 地址", default=DEFAULT_BASE_URL),
        SettingField(
            key="requests_per_minute",
            label="每分钟请求上限",
            default="180",
            help="Notion 建议约 3 req/s",
        ),
    ),
)


class NotionClient:
    def __init__(
        self,
        token: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        rate_per_minute: int = 180,
    ) -> None:
        self._http = HttpClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            transport=transport,
            sleep_fn=sleep_fn,
        )
        self._limiter = TokenBucket(rate_per_minute, sleep_fn=sleep_fn)

    def search_all(self) -> tuple[list[dict[str, Any]], bool]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_SEARCH_PAGES):
            body: dict[str, Any] = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            self._limiter.acquire()
            payload = json_response(self._http.request("POST", "/v1/search", json=body))
            results.extend(payload.get("results") or [])
            if not payload.get("has_more"):
                return results, False
            cursor = payload.get("next_cursor")
            if not cursor:
                return results, False
        return results, True

    def block_children(self, block_id: str) -> tuple[list[dict[str, Any]], bool]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(MAX_BLOCK_PAGES):
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            self._limiter.acquire()
            payload = json_response(
                self._http.request(
                    "GET", f"/v1/blocks/{block_id}/children", params=params
                )
            )
            results.extend(payload.get("results") or [])
            if not payload.get("has_more"):
                return results, False
            cursor = payload.get("next_cursor")
            if not cursor:
                return results, False
        return results, True

    def close(self) -> None:
        self._http.close()


class NotionConnector:
    metadata = NOTION_METADATA

    def __init__(
        self,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        token = str(credentials.get("token") or "")
        if not token:
            raise AuthError("缺少 Notion Integration Token")
        base_url = str(settings.get("base_url") or DEFAULT_BASE_URL)
        try:
            rate = int(str(settings.get("requests_per_minute") or "180"))
        except ValueError:
            rate = 180
        self._client = NotionClient(
            token,
            base_url,
            transport=transport,
            sleep_fn=sleep_fn,
            rate_per_minute=rate,
        )
        self._search_cache: list[dict[str, Any]] | None = None
        self._search_truncated = False
        self._fetch_truncated = False

    def validate(self) -> None:
        results, _ = self._client.search_all()
        if results is None:
            raise NotFoundError("Notion 搜索失败")

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        nodes = self._nodes()
        ids = {node.id for node in nodes}
        if parent_id is None:
            return [
                replace(node, parent_id=None)
                for node in nodes
                if not node.parent_id or node.parent_id not in ids
            ]
        return [
            replace(node, parent_id=parent_id)
            for node in nodes
            if node.parent_id == parent_id
        ]

    def resolve_ancestors(self, ids: Sequence[str]) -> list[ResourceNode]:
        if not ids:
            return []
        nodes = self._nodes()
        by_id = {node.id: node for node in nodes}
        result: list[ResourceNode] = []
        seen: set[str] = set()
        for object_id in ids:
            chain: list[ResourceNode] = []
            visited: set[str] = set()
            current = by_id.get(object_id)
            while current is not None and current.id not in visited and len(chain) < 50:
                visited.add(current.id)
                chain.append(current)
                current = by_id.get(current.parent_id) if current.parent_id else None
            for index, node in enumerate(reversed(chain)):
                if node.id in seen:
                    continue
                seen.add(node.id)
                result.append(replace(node, parent_id=None) if index == 0 else node)
        return result

    def enumerate(self, root: str) -> EnumerationResult:
        nodes = self._nodes()
        by_id = {node.id: node for node in nodes}
        if root not in by_id:
            return EnumerationResult(
                nodes=(),
                complete=not self._search_truncated,
                truncated=self._search_truncated,
            )
        root_node = by_id[root]
        if root_node.kind != "page":
            return EnumerationResult(
                nodes=(root_node,),
                complete=not self._search_truncated,
                truncated=self._search_truncated,
            )
        children: dict[str, list[ResourceNode]] = {}
        for node in nodes:
            if node.parent_id and node.parent_id in by_id:
                children.setdefault(node.parent_id, []).append(node)
        collected: list[ResourceNode] = []
        queue = [root_node]
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current.id in seen:
                continue
            seen.add(current.id)
            collected.append(current)
            queue.extend(children.get(current.id, []))
            if len(collected) >= MAX_BLOCKS:
                self._search_truncated = True
                break
        return EnumerationResult(
            nodes=tuple(collected),
            complete=not self._search_truncated,
            truncated=self._search_truncated,
        )

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        if str(node.meta.get("object") or "") != "page":
            return FetchOutcome(skipped=True, skip_reason="db_unsupported")
        self._fetch_truncated = False
        budget = [MAX_BLOCKS]
        blocks = self._fetch_children(node.id, 0, budget)
        markdown = render_blocks(blocks)
        title = node.title or node.id
        return FetchOutcome(
            item=FetchedItem(
                external_id=node.id,
                resource_id=str(node.meta.get("parent_id") or ""),
                title=title,
                content=markdown,
                content_type="markdown",
                file_name=sanitize_file_name(title, ".md"),
                rev=node.rev,
                metadata={"channel": "notion"},
            ),
            truncated=self._fetch_truncated,
        )

    def close(self) -> None:
        self._client.close()

    def _fetch_children(
        self, block_id: str, depth: int, budget: list[int]
    ) -> list[dict[str, Any]]:
        if depth > MAX_DEPTH:
            self._fetch_truncated = True
            return []
        blocks, page_truncated = self._client.block_children(block_id)
        if page_truncated:
            self._fetch_truncated = True
        for block in blocks:
            if budget[0] <= 0:
                self._fetch_truncated = True
                break
            budget[0] -= 1
            block_type = str(block.get("type") or "")
            if block.get("has_children") and block_type in RECURSIVE_TYPES:
                block["children"] = self._fetch_children(
                    str(block.get("id") or ""), depth + 1, budget
                )
            else:
                block["children"] = []
        return blocks

    def _nodes(self) -> list[ResourceNode]:
        if self._search_cache is None:
            self._search_cache, self._search_truncated = self._client.search_all()
        nodes: list[ResourceNode] = []
        for obj in self._search_cache:
            if obj.get("in_trash"):
                continue
            object_type = str(obj.get("object") or "")
            if object_type not in {"page", "database", "data_source"}:
                continue
            object_id = str(obj.get("id") or "")
            if not object_id:
                continue
            parent = obj.get("parent") or {}
            if object_type == "data_source":
                parent = obj.get("database_parent") or parent
            parent_id = (
                parent.get("page_id")
                or parent.get("database_id")
                or parent.get("data_source_id")
                or None
            )
            nodes.append(
                ResourceNode(
                    id=object_id,
                    title=_title_of(obj, object_id),
                    kind="page" if object_type == "page" else "database",
                    parent_id=str(parent_id) if parent_id else None,
                    has_children=object_type == "page",
                    rev=str(obj.get("last_edited_time") or ""),
                    meta={"object": object_type, "parent_id": parent_id or ""},
                )
            )
        return nodes


def _title_of(obj: Mapping[str, Any], fallback: str) -> str:
    title = obj.get("title")
    if isinstance(title, list):
        rendered = render_rich_text(title)
        if rendered:
            return rendered
    name = obj.get("name")
    if name:
        return str(name)
    properties = obj.get("properties") or {}
    if isinstance(properties, dict):
        for value in properties.values():
            if not isinstance(value, dict) or value.get("type") != "title":
                continue
            rendered = render_rich_text(value.get("title") or [])
            if rendered:
                return rendered
    return fallback


def register(
    registry: ConnectorRegistry,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    registry.register(
        NOTION_METADATA,
        lambda credentials, settings: NotionConnector(
            credentials, settings, transport=transport, sleep_fn=sleep_fn
        ),
    )
