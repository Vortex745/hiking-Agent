"""Feishu / Lark Wiki connector with lazy nodes, Markdown docs and quota retry."""

from __future__ import annotations

import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping, Sequence

import httpx

from datasource.registry import ConnectorRegistry
from datasource.errors import AuthError, NetworkError, NotFoundError, RateLimited, VendorError
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

logger = logging.getLogger("ai-service.datasource.feishu")

DEFAULT_BASE_URL = "https://open.feishu.cn"
MAX_NODES = 5000
PAGE_SIZE = 50
SUPPORTED_DOC_TYPES = {"docx", "doc"}
SKIPPED_OBJ_TYPES = {"sheet", "bitable", "file", "slides", "mindnote"}

FEISHU_METADATA = ConnectorMetadata(
    type="feishu",
    label="飞书 / Lark",
    auth_fields=(
        AuthField(key="app_id", label="App ID", required=True),
        AuthField(key="app_secret", label="App Secret", secret=True, required=True),
    ),
    setting_fields=(
        SettingField(
            key="base_url",
            label="开放平台地址",
            default=DEFAULT_BASE_URL,
            help="Lark 使用 https://open.larksuite.com",
        ),
        SettingField(
            key="requests_per_minute",
            label="每分钟请求上限",
            default="80",
            help="飞书按接口分桶限流；客户端会按响应头额外等待",
        ),
    ),
)


class FeishuClient:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_clock_fn: Callable[[], float] = time.time,
        rate_per_minute: int = 80,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._http = HttpClient(
            base_url=base_url,
            transport=transport,
            max_attempts=1,
            sleep_fn=sleep_fn,
        )
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._wall_clock = wall_clock_fn
        self._limiter = TokenBucket(
            rate_per_minute, sleep_fn=sleep_fn, clock_fn=clock_fn
        )
        self._tenant_token = ""
        self._token_expires_at = 0.0

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> dict[str, Any]:
        attempts = 4
        server_retries = 0
        for attempt in range(attempts):
            try:
                token = self._token()
            except NetworkError:
                if attempt + 1 < attempts:
                    self._sleep(float(2 ** (attempt + 1)))
                    continue
                raise
            self._limiter.acquire()
            try:
                response = self._http.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except NetworkError:
                if attempt + 1 < attempts:
                    self._sleep(float(2 ** (attempt + 1)))
                    continue
                raise
            if response.status_code == 401 and attempt + 1 < attempts:
                self._tenant_token = ""
                continue
            if response.status_code == 429:
                if attempt + 1 < attempts:
                    wait = max(
                        feishu_rate_limit_wait(response, self._wall_clock()),
                        float(2 ** (attempt + 1)),
                    )
                    self._sleep(wait)
                    continue
                raise RateLimited("飞书限流")
            if response.status_code >= 500:
                if server_retries < 1 and attempt + 1 < attempts:
                    server_retries += 1
                    self._sleep(2.0)
                    continue
            payload = json_response(response)
            code = payload.get("code", 0) if isinstance(payload, dict) else 0
            if code == 99991400:
                if attempt + 1 < attempts:
                    self._sleep(float(2**attempt))
                    continue
                raise RateLimited("飞书限流")
            if code in {99991661, 99991663, 99991668, 99991671}:
                raise AuthError("飞书应用凭据无效或已过期")
            if code != 0:
                raise VendorError(f"飞书接口错误（code {code}）")
            return payload
        raise RateLimited("飞书限流")

    def _token(self) -> str:
        now = self._clock()
        if self._tenant_token and now < self._token_expires_at - 300:
            return self._tenant_token
        self._limiter.acquire()
        response = self._http.request(
            "POST",
            "/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        payload = json_response(response)
        if payload.get("code", 0) != 0:
            raise AuthError("飞书应用凭据无效")
        token = payload.get("tenant_access_token")
        if not token:
            raise AuthError("飞书未返回 tenant_access_token")
        self._tenant_token = str(token)
        expire = int(payload.get("expire") or 7200)
        self._token_expires_at = now + max(expire, 60)
        return self._tenant_token

    def close(self) -> None:
        self._http.close()


class FeishuConnector:
    metadata = FEISHU_METADATA

    def __init__(
        self,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        wall_clock_fn: Callable[[], float] = time.time,
    ) -> None:
        app_id = str(credentials.get("app_id") or "")
        app_secret = str(credentials.get("app_secret") or "")
        if not app_id or not app_secret:
            raise AuthError("缺少飞书 App ID 或 App Secret")
        base_url = str(settings.get("base_url") or DEFAULT_BASE_URL)
        try:
            rate = int(str(settings.get("requests_per_minute") or "80"))
        except ValueError:
            rate = 80
        self._client = FeishuClient(
            app_id,
            app_secret,
            base_url,
            transport=transport,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            wall_clock_fn=wall_clock_fn,
            rate_per_minute=rate,
        )

    def validate(self) -> None:
        self._spaces(limit=1)

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        if parent_id is None:
            return [
                ResourceNode(
                    id=str(space.get("space_id") or ""),
                    title=str(space.get("name") or space.get("space_id") or "Space"),
                    kind="space",
                    has_children=True,
                )
                for space in self._spaces()
                if space.get("space_id")
            ]
        if ":" in parent_id:
            space_id, node_token = parent_id.split(":", 1)
        else:
            space_id, node_token = parent_id, ""
        items, _, _ = self._nodes(space_id, node_token)
        return [self._resource(space_id, item, parent_id) for item in items if item.get("node_token")]

    def resolve_ancestors(self, ids: Sequence[str]) -> list[ResourceNode]:
        result: list[ResourceNode] = []
        seen: set[str] = set()

        def add(node: ResourceNode) -> None:
            if node.id not in seen:
                seen.add(node.id)
                result.append(node)

        for resource_id in ids:
            if ":" not in resource_id:
                add(ResourceNode(id=resource_id, title=resource_id, kind="space"))
                continue
            space_id, node_token = resource_id.split(":", 1)
            chain: list[ResourceNode] = []
            current = node_token
            visited: set[str] = set()
            try:
                while current and current not in visited and len(visited) < 50:
                    visited.add(current)
                    node = self._get_node(current)
                    chain.append(self._resource(space_id, node, None))
                    current = str(node.get("parent_node_token") or "")
            except NotFoundError:
                continue
            add(ResourceNode(id=space_id, title=space_id, kind="space"))
            for node in reversed(chain):
                add(node)
        return result

    def enumerate(self, root: str) -> EnumerationResult:
        nodes: list[ResourceNode] = []
        seen: set[str] = set()
        complete = True
        truncated = False
        error = ""
        queue: list[tuple[str, str]] = []

        if ":" in root:
            space_id, node_token = root.split(":", 1)
            try:
                node = self._get_node(node_token)
            except Exception as exc:
                return EnumerationResult(nodes=(), complete=False, error=_code(exc))
            resource = self._resource(space_id, node, None)
            nodes.append(resource)
            seen.add(resource.id)
            queue.append((node_token, resource.id))
        else:
            space_id = root
            queue.append(("", space_id))

        try:
            while queue:
                parent_token, parent_id = queue.pop(0)
                items, page_complete, page_truncated = self._nodes(space_id, parent_token)
                if not page_complete:
                    complete = False
                    error = "enumeration_incomplete"
                if page_truncated:
                    truncated = True
                for item in items:
                    token = str(item.get("node_token") or "")
                    if not token or token in seen:
                        continue
                    seen.add(token)
                    resource = self._resource(space_id, item, parent_id)
                    nodes.append(resource)
                    if item.get("has_child"):
                        queue.append((token, resource.id))
                    if len(nodes) >= MAX_NODES:
                        truncated = True
                        complete = False
                        error = "enumeration_truncated"
                        queue.clear()
                        break
        except Exception as exc:
            complete = False
            error = _code(exc)
        return EnumerationResult(
            nodes=tuple(nodes),
            complete=complete,
            truncated=truncated,
            error=error,
        )

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        obj_type = str(node.meta.get("obj_type") or "").strip().lower()
        if obj_type not in SUPPORTED_DOC_TYPES:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        obj_token = str(node.meta.get("obj_token") or "")
        if not obj_token:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        payload = self._client.request(
            "GET",
            "/open-apis/docs/v1/content",
            params={
                "doc_token": obj_token,
                "doc_type": obj_type,
                "content_type": "markdown",
                "lang": "zh",
            },
        )
        data = payload.get("data") or {}
        title = str(data.get("title") or node.title or obj_token)
        return FetchOutcome(
            item=FetchedItem(
                external_id=node.id,
                resource_id=str(node.meta.get("space_id") or ""),
                title=title,
                content=str(data.get("content") or ""),
                content_type="markdown",
                file_name=sanitize_file_name(title, ".md"),
                rev=node.rev,
                metadata={"channel": "feishu"},
            )
        )

    def close(self) -> None:
        self._client.close()

    def _spaces(self, limit: int = 0) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"page_size": PAGE_SIZE}
            if page_token:
                params["page_token"] = page_token
            payload = self._client.request("GET", "/open-apis/wiki/v2/spaces", params=params)
            data = payload.get("data") or {}
            items.extend(data.get("items") or [])
            if limit and len(items) >= limit:
                return items[:limit]
            if not data.get("has_more"):
                return items
            next_token = str(data.get("page_token") or "")
            if not next_token or next_token in seen_tokens:
                return items
            seen_tokens.add(next_token)
            page_token = next_token

    def _nodes(
        self, space_id: str, parent_token: str
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        items: list[dict[str, Any]] = []
        page_token = ""
        truncated = False
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"page_size": PAGE_SIZE}
            if parent_token:
                params["parent_node_token"] = parent_token
            if page_token:
                params["page_token"] = page_token
            payload = self._client.request(
                "GET", f"/open-apis/wiki/v2/spaces/{space_id}/nodes", params=params
            )
            data = payload.get("data") or {}
            items.extend(data.get("items") or [])
            if len(items) >= MAX_NODES:
                truncated = True
                break
            if not data.get("has_more"):
                break
            next_token = str(data.get("page_token") or "")
            if not next_token:
                break
            if next_token in seen_tokens:
                truncated = True
                break
            seen_tokens.add(next_token)
            page_token = next_token
        return items, not truncated, truncated

    def _get_node(self, node_token: str) -> dict[str, Any]:
        payload = self._client.request(
            "GET",
            "/open-apis/wiki/v2/spaces/get_node",
            params={"token": node_token},
        )
        node = (payload.get("data") or {}).get("node") or {}
        if not node:
            raise NotFoundError("飞书 Wiki 节点不存在")
        return node

    def _resource(
        self, space_id: str, node: Mapping[str, Any], parent_id: str | None
    ) -> ResourceNode:
        token = str(node.get("node_token") or "")
        obj_type = str(node.get("obj_type") or "docx").lower()
        parent_token = str(node.get("parent_node_token") or "")
        node_parent = parent_id
        if node_parent is None and parent_token:
            node_parent = f"{space_id}:{parent_token}"
        return ResourceNode(
            id=f"{space_id}:{token}",
            title=str(node.get("title") or token or "Untitled"),
            kind=obj_type,
            parent_id=node_parent,
            has_children=bool(node.get("has_child")),
            rev=str(node.get("obj_edit_time") or ""),
            meta={
                "obj_type": obj_type,
                "obj_token": str(node.get("obj_token") or ""),
                "space_id": space_id,
                "parent_node_token": parent_token,
            },
        )


def feishu_rate_limit_wait(response: httpx.Response, now_epoch: float) -> float:
    waits: list[float] = []
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            waits.append(float(retry_after))
        except ValueError:
            try:
                waits.append(
                    max(0.0, parsedate_to_datetime(retry_after).timestamp() - now_epoch)
                )
            except (TypeError, ValueError):
                pass
    reset = response.headers.get("x-ogw-ratelimit-reset")
    if reset:
        try:
            value = float(reset)
            if value > 1_000_000_000_000:
                value /= 1000.0
            if value > 1_000_000_000 or value > now_epoch:
                waits.append(max(0.0, value - now_epoch))
            else:
                waits.append(value)
        except ValueError:
            pass
    return min(max(max(waits, default=1.0), 1.0), 60.0)


def _code(exc: BaseException) -> str:
    return getattr(exc, "code", "vendor_error")


def register(
    registry: ConnectorRegistry,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    wall_clock_fn: Callable[[], float] = time.time,
) -> None:
    registry.register(
        FEISHU_METADATA,
        lambda credentials, settings: FeishuConnector(
            credentials,
            settings,
            transport=transport,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            wall_clock_fn=wall_clock_fn,
        ),
    )
