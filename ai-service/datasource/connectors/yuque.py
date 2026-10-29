"""Yuque connector: team/personal token, Book resources, Markdown docs."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Sequence

import httpx

from datasource.registry import ConnectorRegistry
from datasource.errors import AuthError, PermissionDenied, UnsupportedType
from datasource.http import HttpClient, json_response
from datasource.ingest import sanitize_file_name
from datasource.types import (
    AuthField,
    ConnectorMetadata,
    EnumerationResult,
    FetchedItem,
    FetchOutcome,
    ResourceNode,
    SettingField,
)

logger = logging.getLogger("ai-service.datasource.yuque")

DEFAULT_BASE_URL = "https://www.yuque.com"
PAGE_SIZE = 100
MAX_ITEMS = 1000
DETAIL_INTERVAL_SECONDS = 0.3

YUQUE_METADATA = ConnectorMetadata(
    type="yuque",
    label="语雀",
    auth_fields=(
        AuthField(
            key="token",
            label="访问令牌",
            secret=True,
            help="个人或团队 token，只读权限即可",
        ),
    ),
    setting_fields=(
        SettingField(
            key="base_url",
            label="站点地址",
            default=DEFAULT_BASE_URL,
            help="企业或私有化域名可覆盖",
        ),
    ),
)


class YuqueConnector:
    metadata = YUQUE_METADATA

    def __init__(
        self,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        base_url = str(settings.get("base_url") or DEFAULT_BASE_URL)
        token = str(credentials.get("token") or "")
        if not token:
            raise AuthError("缺少语雀访问令牌")
        self._http = HttpClient(
            base_url=base_url,
            headers={
                "X-Auth-Token": token,
                "Accept": "application/json",
                "User-Agent": "ai-hiking-datasource/1.0",
            },
            transport=transport,
            sleep_fn=sleep_fn,
        )
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last_detail_at: float | None = None
        self._identity: dict[str, Any] | None = None

    def validate(self) -> None:
        self._load_identity()

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        if parent_id is not None:
            return []
        identity = self._load_identity()
        login = str(identity.get("login") or "")
        repos: list[dict[str, Any]] = []
        if str(identity.get("type") or "").lower() == "group":
            repos.extend(self._safe_repos(f"/api/v2/groups/{login}/repos"))
        else:
            repos.extend(self._safe_repos(f"/api/v2/users/{login}/repos"))
            for group_login in self._group_logins(identity):
                repos.extend(self._safe_repos(f"/api/v2/groups/{group_login}/repos"))
        nodes: list[ResourceNode] = []
        seen: set[str] = set()
        for repo in repos:
            if str(repo.get("type") or "").lower() != "book":
                continue
            repo_id = str(repo.get("id") or "")
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            nodes.append(
                ResourceNode(
                    id=repo_id,
                    title=str(repo.get("name") or repo.get("slug") or repo_id),
                    kind="book",
                    has_children=True,
                    meta={"namespace": repo.get("namespace") or ""},
                )
            )
        return nodes

    def resolve_ancestors(self, ids: Sequence[str]) -> list[ResourceNode]:
        return []

    def enumerate(self, root: str) -> EnumerationResult:
        docs, truncated = self._paged(f"/api/v2/repos/{root}/docs")
        nodes = tuple(
            ResourceNode(
                id=str(doc.get("id") or ""),
                title=str(doc.get("title") or doc.get("slug") or doc.get("id") or "Untitled"),
                kind="document",
                parent_id=root,
                has_children=False,
                rev=str(doc.get("content_updated_at") or doc.get("updated_at") or ""),
                meta={
                    "type": doc.get("type") or "",
                    "status": doc.get("status"),
                    "format": doc.get("format") or "",
                    "book_id": root,
                    "slug": doc.get("slug") or "",
                },
            )
            for doc in docs
            if doc.get("id") is not None
        )
        return EnumerationResult(nodes=nodes, complete=not truncated, truncated=truncated)

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        doc_type = str(node.meta.get("type") or "").strip().lower()
        if doc_type != "doc":
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        status = node.meta.get("status")
        if status is not None and str(status) != "1":
            return FetchOutcome(skipped=True, skip_reason="draft")
        node_format = str(node.meta.get("format") or "").lower()
        if node_format and node_format not in {"markdown", "lake"}:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        self._throttle()
        payload = self._http.get_json(f"/api/v2/repos/docs/{node.id}")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise UnsupportedType("语雀详情响应缺少正文")
        file_format = str(data.get("format") or node.meta.get("format") or "").lower()
        if file_format not in {"markdown", "lake"}:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        title = str(data.get("title") or node.title or node.id)
        return FetchOutcome(
            item=FetchedItem(
                external_id=node.id,
                resource_id=str(node.meta.get("book_id") or node.parent_id or ""),
                title=title,
                content=str(data.get("body") or ""),
                content_type="markdown",
                file_name=sanitize_file_name(title, ".md"),
                rev=str(data.get("content_updated_at") or node.rev or ""),
                updated_at=data.get("content_updated_at") or data.get("updated_at"),
                metadata={"channel": "yuque"},
            )
        )

    def close(self) -> None:
        self._http.close()

    def _load_identity(self) -> dict[str, Any]:
        if self._identity is not None:
            return self._identity
        payload = self._http.get_json("/api/v2/user")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not data.get("login"):
            raise AuthError("语雀凭据无效")
        self._identity = data
        return data

    def _group_logins(self, identity: Mapping[str, Any]) -> list[str]:
        user_id = identity.get("id")
        if user_id is None:
            return []
        try:
            payload = self._http.get_json(f"/api/v2/users/{user_id}/groups")
        except PermissionDenied:
            return []
        except Exception as exc:
            if getattr(exc, "code", "") == "not_found":
                return []
            raise
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        return [str(group.get("login")) for group in data if group.get("login")]

    def _safe_repos(self, path: str) -> list[dict[str, Any]]:
        try:
            items, _ = self._paged(path)
            return items
        except PermissionDenied:
            return []
        except Exception as exc:
            if getattr(exc, "code", "") == "not_found":
                return []
            raise

    def _paged(self, path: str) -> tuple[list[dict[str, Any]], bool]:
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._http.get_json(
                path, params={"limit": PAGE_SIZE, "offset": offset}
            )
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict):
                data = data.get("data")
            if not isinstance(data, list):
                return items, False
            items.extend(item for item in data if isinstance(item, dict))
            if len(data) < PAGE_SIZE:
                return items, False
            offset += PAGE_SIZE
            if offset >= MAX_ITEMS:
                return items, True

    def _throttle(self) -> None:
        current = self._clock()
        if self._last_detail_at is not None:
            remaining = DETAIL_INTERVAL_SECONDS - (current - self._last_detail_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_detail_at = self._clock()


def register(
    registry: ConnectorRegistry,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> None:
    registry.register(
        YUQUE_METADATA,
        lambda credentials, settings: YuqueConnector(
            credentials,
            settings,
            transport=transport,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
        ),
    )
