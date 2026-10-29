"""Shared HTTP client for external connectors.

Retries transient transport errors, 429 and 5xx responses. Vendor response
bodies are never placed into user-visible errors; connectors map status codes
to stable DatasourceError codes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx

from datasource.errors import (
    AuthError,
    NetworkError,
    NotFoundError,
    PermissionDenied,
    RateLimited,
    VendorError,
)

logger = logging.getLogger("ai-service.datasource.http")

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_AFTER = 60.0


class HttpClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers or {},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._max_attempts = max(1, max_attempts)
        self._sleep = sleep_fn

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.request(
                    method.upper(), path, params=params, json=json, headers=headers
                )
            except httpx.TransportError as exc:
                last_error = exc
                if attempt + 1 >= self._max_attempts:
                    raise NetworkError("网络请求失败") from exc
                self._sleep(float(2**attempt))
                continue
            if response.status_code in _RETRY_STATUSES and attempt + 1 < self._max_attempts:
                delay = _retry_after(response)
                self._sleep(delay if delay is not None else float(2**attempt))
                continue
            return response
        raise NetworkError("网络请求失败") from last_error

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return json_response(self.request("GET", path, params=params))

    def post_json(self, path: str, payload: Any) -> Any:
        return json_response(self.request("POST", path, json=payload))

    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:
            logger.warning("http client close failed: %s", exc)


def json_response(response: httpx.Response) -> Any:
    if response.status_code == 401:
        raise AuthError("凭据无效或已过期")
    if response.status_code == 403:
        raise PermissionDenied("没有访问该资源的权限")
    if response.status_code == 404:
        raise NotFoundError("请求的资源不存在")
    if response.status_code == 429:
        raise RateLimited("平台限流")
    if response.status_code >= 500:
        raise VendorError(f"平台服务异常（HTTP {response.status_code}）")
    if response.status_code < 200 or response.status_code >= 300:
        raise VendorError(f"平台请求失败（HTTP {response.status_code}）")
    try:
        return response.json()
    except ValueError as exc:
        raise VendorError("平台返回了无法解析的响应") from exc


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), _MAX_RETRY_AFTER)
    except ValueError:
        return None
