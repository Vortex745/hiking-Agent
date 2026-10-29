"""Tencent IMA connector: logical keys, three-state failures and paging guards."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Mapping

import httpx

from datasource.registry import ConnectorRegistry
from datasource.errors import AuthError, UnsupportedType, VendorError
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

logger = logging.getLogger("ai-service.datasource.ima")

DEFAULT_BASE_URL = "https://ima.qq.com"
PAGE_SIZE = 50
MAX_ITEMS = 10000
MAX_PAGE_JUMPS = 10000
FOLDER_MEDIA_TYPE = 99
NOTE_MEDIA_TYPE = 11
SKIPPED_MEDIA_TYPES = {9, 12, 16}

IMA_METADATA = ConnectorMetadata(
    type="ima",
    label="腾讯 IMA",
    auth_fields=(
        AuthField(key="client_id", label="Client ID", required=True),
        AuthField(key="api_key", label="API Key", secret=True, required=True),
    ),
    setting_fields=(SettingField(key="base_url", label="API 地址", default=DEFAULT_BASE_URL),),
)


class ImaClient:
    def __init__(
        self,
        client_id: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = HttpClient(
            base_url=base_url,
            headers={
                "ima-openapi-clientid": client_id,
                "ima-openapi-apikey": api_key,
                "Content-Type": "application/json",
            },
            transport=transport,
            sleep_fn=sleep_fn,
        )
        self._transport = transport

    def post(self, action: str, payload: Mapping[str, Any], namespace: str = "wiki") -> Any:
        response = self._http.request(
            "POST", f"/openapi/{namespace}/v1/{action}", json=dict(payload)
        )
        body = json_response(response)
        code = body.get("code", 0) if isinstance(body, dict) else 0
        if code != 0:
            if code in {401, 403, 10001}:
                raise AuthError("IMA 凭据无效或没有权限")
            raise VendorError(f"IMA 接口错误（code {code}）")
        return body.get("data") if isinstance(body, dict) else body

    def download(self, url: str, headers: Mapping[str, str]) -> tuple[bytes, str]:
        client = httpx.Client(
            transport=self._transport, timeout=60.0, follow_redirects=True
        )
        try:
            response = client.get(url, headers=dict(headers))
        except httpx.TransportError as exc:
            raise VendorError("IMA 文件下载失败") from exc
        finally:
            client.close()
        if response.status_code >= 400:
            raise VendorError(f"IMA 文件下载失败（HTTP {response.status_code}）")
        return response.content, response.headers.get("content-type", "")

    def close(self) -> None:
        self._http.close()


class ImaConnector:
    metadata = IMA_METADATA

    def __init__(
        self,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        client_id = str(credentials.get("client_id") or "")
        api_key = str(credentials.get("api_key") or "")
        if not client_id or not api_key:
            raise AuthError("缺少 IMA Client ID 或 API Key")
        base_url = str(settings.get("base_url") or DEFAULT_BASE_URL)
        self._client = ImaClient(
            client_id, api_key, base_url, transport=transport, sleep_fn=sleep_fn
        )

    def validate(self) -> None:
        self._list_kbs(limit=1, fallback=False)

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        if parent_id is None:
            return [
                ResourceNode(
                    id=str(kb.get("kb_id") or kb.get("knowledge_base_id") or ""),
                    title=_kb_title(kb),
                    kind="knowledge_base",
                    has_children=True,
                )
                for kb in self._list_kbs()
                if kb.get("kb_id") or kb.get("knowledge_base_id")
            ]
        kb_id, folder_id = _split_root(parent_id)
        items, _ = self._list_page(kb_id, folder_id)
        return [
            ResourceNode(
                id=f"{kb_id}:{item.get('media_id')}",
                title=_item_title(item),
                kind="folder",
                parent_id=parent_id,
                has_children=True,
                rev=str(item.get("media_id") or ""),
            )
            for item in items
            if _is_folder(item) and item.get("media_id")
        ]

    def resolve_ancestors(self, ids: list[str]) -> list[ResourceNode]:
        return []

    def enumerate(self, root: str) -> EnumerationResult:
        kb_id, folder_id = _split_root(root)
        collected: list[tuple[str, Mapping[str, Any]]] = []
        stack = [folder_id]
        complete = True
        truncated = False
        error = ""
        visited_folders: set[str] = set()
        try:
            while stack:
                current = stack.pop()
                if current in visited_folders:
                    continue
                visited_folders.add(current)
                items, page_truncated = self._list_page(kb_id, current)
                if page_truncated:
                    truncated = True
                for item in items:
                    if _is_folder(item):
                        if item.get("media_id"):
                            stack.append(str(item["media_id"]))
                    else:
                        collected.append((current, item))
                if len(collected) >= MAX_ITEMS:
                    truncated = True
                    complete = False
                    error = "enumeration_truncated"
                    break
        except Exception as exc:
            complete = False
            error = _code(exc)
        by_key: dict[str, ResourceNode] = {}
        for parent_folder_id, item in collected:
            title = _item_title(item)
            media_id = str(item.get("media_id") or "")
            rev = str(
                item.get("update_time")
                or item.get("updated_at")
                or item.get("modify_time")
                or media_id
            )
            external_id = logical_key(kb_id, parent_folder_id, title)
            by_key[external_id] = ResourceNode(
                id=external_id,
                title=title,
                kind="item",
                parent_id=root,
                rev=rev,
                meta={
                    "kb_id": kb_id,
                    "media_id": media_id,
                    "parent_folder_id": parent_folder_id,
                    "media_type": item.get("media_type"),
                },
            )
        return EnumerationResult(
            nodes=tuple(by_key.values()),
            complete=complete and not truncated,
            truncated=truncated,
            error=error,
        )

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        media_id = str(node.meta.get("media_id") or "")
        if not media_id:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        data = self._client.post("get_media_info", {"media_id": media_id})
        if not isinstance(data, dict):
            raise VendorError("IMA 媒体详情格式异常")
        media_type = int(data.get("media_type") or node.meta.get("media_type") or 0)
        title = str(data.get("title") or node.title or media_id)
        if media_type == NOTE_MEDIA_TYPE or data.get("notebook_ext_info"):
            notebook = data.get("notebook_ext_info") or {}
            notebook_id = str(notebook.get("notebook_id") or "")
            if not notebook_id:
                return FetchOutcome(skipped=True, skip_reason="unsupported_type")
            note = self._client.post(
                "get_doc_content", {"note_id": notebook_id}, namespace="note"
            )
            content = note.get("content") if isinstance(note, dict) else note
            return FetchOutcome(
                item=FetchedItem(
                    external_id=node.id,
                    resource_id=node.meta.get("kb_id") or "",
                    title=title,
                    content=str(content or ""),
                    content_type="markdown",
                    file_name=sanitize_file_name(title, ".md"),
                    rev=node.rev or media_id,
                    metadata={"channel": "ima", "kind": "note"},
                )
            )
        if media_type in SKIPPED_MEDIA_TYPES:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        url_info = data.get("url_info") or {}
        url = str(url_info.get("url") or "")
        if not url:
            return FetchOutcome(skipped=True, skip_reason="unsupported_type")
        headers = url_info.get("headers") or {}
        body, content_type = self._client.download(url, headers)
        mime = content_type.split(";")[0].strip().lower()
        if mime == "application/pdf":
            if not body:
                return FetchOutcome(skipped=True, skip_reason="empty")
            return FetchOutcome(
                item=FetchedItem(
                    external_id=node.id,
                    resource_id=node.meta.get("kb_id") or "",
                    title=title,
                    content="",
                    content_type="pdf",
                    data=body,
                    file_name=sanitize_file_name(title, ".pdf"),
                    rev=node.rev or media_id,
                    metadata={"channel": "ima"},
                )
            )
        if mime in {"text/markdown", "text/plain"}:
            text = body.decode("utf-8", errors="replace")
            return FetchOutcome(
                item=FetchedItem(
                    external_id=node.id,
                    resource_id=node.meta.get("kb_id") or "",
                    title=title,
                    content=text,
                    content_type="markdown",
                    file_name=sanitize_file_name(title, ".md"),
                    rev=node.rev or media_id,
                    metadata={"channel": "ima"},
                )
            )
        return FetchOutcome(skipped=True, skip_reason="unsupported_type")

    def close(self) -> None:
        self._client.close()

    def _list_kbs(self, limit: int = 50, fallback: bool = True) -> list[dict[str, Any]]:
        data = self._client.post(
            "get_addable_knowledge_base_list", {"cursor": "", "limit": limit}
        )
        items = _items_of(data)
        if not items and fallback:
            data = self._client.post(
                "search_knowledge_base", {"query": "", "cursor": "", "limit": 10}
            )
            items = _items_of(data)
        return items

    def _list_page(self, kb_id: str, folder_id: str) -> tuple[list[dict[str, Any]], bool]:
        items: list[dict[str, Any]] = []
        cursor = ""
        seen: set[str] = set()
        jumps = 0
        while True:
            data = self._client.post(
                "get_knowledge_list",
                {
                    "knowledge_base_id": kb_id,
                    "folder_id": folder_id,
                    "cursor": cursor,
                    "limit": PAGE_SIZE,
                },
            )
            items.extend(_items_of(data))
            if len(items) >= MAX_ITEMS:
                return items, True
            if data.get("is_end"):
                return items, False
            next_cursor = str(data.get("next_cursor") or "")
            if not next_cursor:
                return items, False
            if next_cursor in seen or jumps >= MAX_PAGE_JUMPS:
                raise VendorError("IMA 分页游标异常")
            seen.add(next_cursor)
            cursor = next_cursor
            jumps += 1


def logical_key(kb_id: str, parent_folder_id: str, title: str) -> str:
    raw = f"{kb_id}\x1f{parent_folder_id}\x1f{title}".encode("utf-8")
    return "ima_" + hashlib.sha256(raw).hexdigest()[:32]


def _split_root(root: str) -> tuple[str, str]:
    if ":" in root:
        kb_id, folder_id = root.split(":", 1)
        return kb_id, folder_id
    return root, ""


def _items_of(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    for key in ("items", "knowledge_base_list", "list"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _is_folder(item: Mapping[str, Any]) -> bool:
    try:
        return int(item.get("media_type") or 0) == FOLDER_MEDIA_TYPE or bool(
            item.get("is_folder")
        )
    except (TypeError, ValueError):
        return bool(item.get("is_folder"))


def _item_title(item: Mapping[str, Any]) -> str:
    return str(
        item.get("title")
        or item.get("name")
        or item.get("file_name")
        or item.get("media_id")
        or "Untitled"
    )


def _kb_title(kb: Mapping[str, Any]) -> str:
    return str(
        kb.get("kb_name")
        or kb.get("name")
        or kb.get("title")
        or kb.get("kb_id")
        or kb.get("knowledge_base_id")
        or "Knowledge Base"
    )


def _code(exc: BaseException) -> str:
    return getattr(exc, "code", "vendor_error")


def register(
    registry: ConnectorRegistry,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    registry.register(
        IMA_METADATA,
        lambda credentials, settings: ImaConnector(
            credentials, settings, transport=transport, sleep_fn=sleep_fn
        ),
    )
