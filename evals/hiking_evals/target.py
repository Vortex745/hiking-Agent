"""统一目标适配器：本地 ASGI 应用 与 外部 HTTP/SSE 服务。

两种适配方式共享同一套请求/响应/异常模型：
- ASGITarget: 持有 fixtures 构建的 FastAPI app，通过 httpx.ASGITransport 调用。
- HTTPTarget: 通过 base_url 调用外部服务（请求/解析逻辑与 ASGI 完全一致）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from httpx import ASGITransport, AsyncClient, Response

from .models import ActionSpec, ResponseRecord


class ActionError(Exception):
    """动作执行失败（连接失败、非法响应等，非业务断言失败）。"""


def parse_sse_frames(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """解析 SSE 文本缓冲：返回 (已解析 data 事件, 剩余缓冲)。"""
    events: list[dict[str, Any]] = []
    while "\n\n" in buffer:
        frame, buffer = buffer.split("\n\n", 1)
        for line in frame.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError as exc:
                    raise ActionError(f"invalid SSE data frame: {payload[:200]} ({exc})") from exc
    return events, buffer


def build_request_dict(action: ActionSpec) -> dict[str, Any]:
    return {
        "method": action.method,
        "path": action.path,
        "json": action.json,
        "cookies": action.cookies,
        "headers": action.headers,
        "query": action.query,
        "sse": action.sse,
    }


async def _execute_http(
    action: ActionSpec,
    client: AsyncClient,
) -> ResponseRecord:
    record = ResponseRecord(action=action.name, request=build_request_dict(action))
    params = dict(action.query) if action.query else None
    try:
        headers = dict(action.headers) if action.headers else None
        if action.sse:
            async with client.stream(
                action.method,
                action.path,
                json=action.json,
                params=params,
                headers=headers,
            ) as resp:
                record.status = resp.status_code
                record.headers = {k: v for k, v in resp.headers.items()}
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    events, buffer = parse_sse_frames(buffer)
                    record.events.extend(events)
                # 流结束但缓冲残留 → 帧不完整
                if buffer.strip():
                    record.error = f"trailing incomplete SSE frame: {buffer[:200]!r}"
        else:
            resp: Response = await client.request(
                action.method,
                action.path,
                json=action.json,
                params=params,
                headers=headers,
            )
            record.status = resp.status_code
            record.headers = {k: v for k, v in resp.headers.items()}
            record.body_text = resp.text
            if "application/json" in resp.headers.get("content-type", ""):
                try:
                    record.body_json = resp.json()
                except ValueError:
                    record.error = "response claimed JSON but failed to parse"
    except ActionError as exc:
        record.error = str(exc)
    except Exception as exc:  # 连接失败/超时等
        record.error = f"{type(exc).__name__}: {exc}"
    return record


class ASGITarget:
    """针对本地 ASGI 应用执行动作。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://asgi-eval"
        )

    async def execute(self, action: ActionSpec) -> ResponseRecord:
        if action.type != "http":
            raise ActionError(f"ASGITarget cannot execute action type {action.type!r}")
        # cookies 显式传入（不做会话保持，保证请求可复盘）
        self._client.cookies.update(action.cookies or {})
        try:
            record = await _execute_http(action, self._client)
        finally:
            self._client.cookies.clear()
        return record

    async def close(self) -> None:
        await self._client.aclose()


class HTTPTarget:
    """针对外部 HTTP/SSE 服务执行动作。"""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = AsyncClient(base_url=base_url, timeout=timeout)

    async def execute(self, action: ActionSpec) -> ResponseRecord:
        if action.type != "http":
            raise ActionError(
                f"HTTPTarget cannot execute local action type {action.type!r}; "
                "component fixtures require --adapter asgi"
            )
        self._client.cookies.update(action.cookies or {})
        try:
            record = await _execute_http(action, self._client)
        finally:
            self._client.cookies.clear()
        return record

    async def close(self) -> None:
        await self._client.aclose()


# ---- Chat 终态提取（T09/T10 共用边界） -------------------------------------


TERMINAL_DONE = "done"
TERMINAL_ERROR = "error"
TERMINAL_CANCELLED = "cancelled"


@dataclass(frozen=True)
class ChatTurnOutcome:
    """一次 Chat v2 SSE 回合的公开终态。

    语义质量判定只允许在此终态基础上进行：``gradable`` 要求正常终态
    （``done``）、非空回答、无传输错误。错误终态、空回答或传输失败
    都不能被包装成语义通过（它们对应契约失败 + not_evaluated）。
    """

    answer: str = ""
    terminal_type: str = ""
    terminal_code: str = ""
    terminal_message: str = ""
    transport_error: str = ""
    http_status: int | None = None
    run_id: str = ""
    conversation_id: str = ""
    event_count: int = 0

    @property
    def gradable(self) -> bool:
        return (
            self.terminal_type == TERMINAL_DONE
            and bool(self.answer.strip())
            and not self.transport_error
        )

    def describe(self) -> str:
        """稳定、无凭证的一行原因（供报告与 not_evaluated 依据）。"""
        if self.transport_error:
            return f"transport_error: {self.transport_error}"
        if self.terminal_type == TERMINAL_ERROR:
            code = self.terminal_code or "error"
            return f"target terminal error: {code}"
        if self.terminal_type != TERMINAL_DONE:
            return f"target terminal not completed: {self.terminal_type or 'missing'}"
        if not self.answer.strip():
            return "target returned an empty answer"
        return "target completed normally with a non-empty answer"


def extract_chat_outcome(record: ResponseRecord) -> ChatTurnOutcome:
    """从一次 SSE 记录提取公开终态。

    - 回答文本 = 全部 ``text`` 事件拼接（trim 后）；
    - 终态 = 最后一个事件（done / error / cancelled）；
    - ``error`` 事件的 code/message 与传输错误被显式保留，绝不吞掉，
      更不得回落成"空回答但有效"。
    """
    parts: list[str] = []
    terminal_type = ""
    terminal_code = ""
    terminal_message = ""
    run_id = ""
    conversation_id = ""
    for event in record.events:
        etype = str(event.get("type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if etype == "text":
            parts.append(str(payload.get("content") or ""))
        if etype in (TERMINAL_DONE, TERMINAL_ERROR, TERMINAL_CANCELLED):
            terminal_type = etype
            if etype == TERMINAL_ERROR:
                terminal_code = str(payload.get("code") or "")
                terminal_message = str(payload.get("message") or "")
        if not run_id and event.get("run_id"):
            run_id = str(event["run_id"])
        if not conversation_id and event.get("conversation_id"):
            conversation_id = str(event["conversation_id"])
    return ChatTurnOutcome(
        answer="\n".join(part for part in parts if part).strip(),
        terminal_type=terminal_type,
        terminal_code=terminal_code,
        terminal_message=terminal_message,
        transport_error=str(record.error or ""),
        http_status=record.status,
        run_id=run_id,
        conversation_id=conversation_id,
        event_count=len(record.events),
    )
