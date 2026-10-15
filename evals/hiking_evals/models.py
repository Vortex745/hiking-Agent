"""Versioned eval case / run config / run record models.

格式契约（format_version=1）：

EvalCase（cases/*.json）:
    {
      "format_version": 1,
      "suite": "chat_v2_protocol",
      "id": "chat-v2-valid-request",
      "scenario": "场景名",
      "severity": "critical|major|minor",
      "description": "...",
      "tags": ["protocol"],
      "fixture": { ... },        # 由 hiking_evals.fixtures 解释
      "actions": [ ... ],       # 顺序执行的请求/组件动作
      "expectations": [ ... ]    # scorer 配置
    }

RunConfig（CLI 参数的序列化形态）:
    {"format_version": 1, "adapter": "asgi"|"http", "base_url": ...,
     "suites": [...], "case_filter": [...], "seed": 0, "live": false}

RunRecord（runs/{run_id}/run.json）:
    {"format_version": 1, "run_id": ..., "started_at": ..., "config": ...,
     "environment": {...}, "cases": [ {case 摘要, records, checks} ]}

所有 dict 字段都只含 JSON 可序列化类型，保证可复盘、可 diff。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

CASE_FORMAT_VERSION = 1
RUN_CONFIG_FORMAT_VERSION = 1
RUN_RECORD_FORMAT_VERSION = 1

SEVERITIES = ("critical", "major", "minor")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


# ---- Case / action / expectation ------------------------------------------


@dataclass
class ActionSpec:
    """一个顺序动作：HTTP 请求（含 SSE）或组件级查询。

    HTTP 请求: method/path/json/cookies/headers/query, sse=True 时解析 SSE 帧。
    组件查询: type="rag_query", question/chat_id/k 由 fixtures 提供的 runtime 执行。
    capture: {变量名: 相对本次响应的取值路径}，供后续动作引用 "{变量名}"。
    """

    name: str
    type: str = "http"  # http | rag_query
    method: str = "POST"
    path: str = ""
    json: dict[str, Any] | None = None
    cookies: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    query: dict[str, Any] | None = None
    sse: bool = False
    question: str = ""
    chat_id: str = ""
    k: int = 4
    capture: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpec":
        return cls(
            name=data["name"],
            type=data.get("type", "http"),
            method=data.get("method", "POST"),
            path=data.get("path", ""),
            json=data.get("json"),
            cookies=data.get("cookies"),
            headers=data.get("headers"),
            query=data.get("query"),
            sse=bool(data.get("sse", False)),
            question=data.get("question", ""),
            chat_id=data.get("chat_id", ""),
            k=int(data.get("k", 4)),
            capture=dict(data.get("capture") or {}),
            note=data.get("note", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "method": self.method,
            "path": self.path,
            "json": self.json,
            "cookies": self.cookies,
            "headers": self.headers,
            "query": self.query,
            "sse": self.sse,
            "question": self.question,
            "chat_id": self.chat_id,
            "k": self.k,
            "capture": self.capture,
            "note": self.note,
        }


@dataclass
class Expectation:
    """一条 scorer 配置：对指定动作（或全量记录）执行断言。"""

    scorer: str
    action: str = ""  # 空 = 面向整个 case 的记录集
    severity: str = "major"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Expectation":
        return cls(
            scorer=data["scorer"],
            action=data.get("action", ""),
            severity=data.get("severity", "major"),
            params=dict(data.get("params") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "action": self.action,
            "severity": self.severity,
            "params": self.params,
        }


@dataclass
class EvalCase:
    format_version: int
    suite: str
    id: str
    scenario: str
    severity: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    fixture: dict[str, Any] = field(default_factory=dict)
    actions: list[ActionSpec] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        case = cls(
            format_version=int(data["format_version"]),
            suite=data["suite"],
            id=data["id"],
            scenario=data.get("scenario", ""),
            severity=data.get("severity", "major"),
            description=data.get("description", ""),
            tags=list(data.get("tags") or []),
            fixture=dict(data.get("fixture") or {}),
            actions=[ActionSpec.from_dict(a) for a in data.get("actions") or []],
            expectations=[Expectation.from_dict(e) for e in data.get("expectations") or []],
        )
        case.validate()
        return case

    def validate(self) -> None:
        if self.format_version != CASE_FORMAT_VERSION:
            raise ValueError(
                f"case {self.id!r}: unsupported format_version "
                f"{self.format_version} (expected {CASE_FORMAT_VERSION})"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(f"case {self.id!r}: severity must be one of {SEVERITIES}")
        names = [a.name for a in self.actions]
        if len(set(names)) != len(names):
            raise ValueError(f"case {self.id!r}: duplicate action names")
        for exp in self.expectations:
            if exp.action and exp.action not in names:
                raise ValueError(
                    f"case {self.id!r}: expectation references unknown action {exp.action!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "suite": self.suite,
            "id": self.id,
            "scenario": self.scenario,
            "severity": self.severity,
            "description": self.description,
            "tags": self.tags,
            "fixture": self.fixture,
            "actions": [a.to_dict() for a in self.actions],
            "expectations": [e.to_dict() for e in self.expectations],
        }


# ---- Run config ------------------------------------------------------------


@dataclass
class RunConfig:
    format_version: int = RUN_CONFIG_FORMAT_VERSION
    adapter: str = "asgi"  # asgi | http
    base_url: str = ""
    suites: list[str] = field(default_factory=list)
    case_filter: list[str] = field(default_factory=list)
    seed: int = 0
    live: bool = False
    max_failures: int = 0
    min_pass_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "adapter": self.adapter,
            "base_url": self.base_url,
            "suites": self.suites,
            "case_filter": self.case_filter,
            "seed": self.seed,
            "live": self.live,
            "max_failures": self.max_failures,
            "min_pass_rate": self.min_pass_rate,
        }


# ---- Execution records ------------------------------------------------------


@dataclass
class ResponseRecord:
    """一个动作的统一执行记录（HTTP/SSE 与组件动作共用）。"""

    action: str
    request: dict[str, Any]
    status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body_json: Any = None
    body_text: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)  # SSE 事件（已解析）
    rag_answer: str = ""
    rag_docs: list[dict[str, Any]] = field(default_factory=list)
    rag_process: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 内核层工具调用日志
    error: str = ""
    duration_ms: float = 0.0
    # 协议未暴露 token/成本时保持 null；payload 携带时记录（可复盘）
    token_usage: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "request": self.request,
            "status": self.status,
            "headers": self.headers,
            "body_json": self.body_json,
            "body_text": self.body_text[:4000],
            "events": self.events,
            "rag_answer": self.rag_answer,
            "rag_docs": self.rag_docs,
            "rag_process": self.rag_process,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "cost": self.cost,
            "artifact_refs": self.artifact_refs,
        }


@dataclass
class Check:
    """一条 scorer 断言结果。"""

    scorer: str
    action: str
    severity: str
    passed: bool
    expected: Any
    actual: Any
    evidence: str = ""  # 首个违反点 + 原始上下文

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "action": self.action,
            "severity": self.severity,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
        }


@dataclass
class CaseResult:
    case: EvalCase
    status: str  # passed | failed | skipped
    checks: list[Check] = field(default_factory=list)
    records: list[ResponseRecord] = field(default_factory=list)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0
    env_audit: dict[str, Any] = field(default_factory=dict)  # fixture 环境元数据（Artifact 清理等）

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.case.suite,
            "id": self.case.id,
            "scenario": self.case.scenario,
            "severity": self.case.severity,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "checks": [c.to_dict() for c in self.checks],
            "records": [r.to_dict() for r in self.records],
            "env_audit": self.env_audit,
        }


@dataclass
class RunRecord:
    format_version: int = RUN_RECORD_FORMAT_VERSION
    run_id: str = field(default_factory=new_run_id)
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    randomness: dict[str, Any] = field(default_factory=dict)
    case_results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "environment": self.environment,
            "randomness": self.randomness,
            "cases": [c.to_dict() for c in self.case_results],
        }


# ---- 通用取值路径 -----------------------------------------------------------


def resolve_path(data: Any, path: str) -> tuple[bool, Any]:
    """微型取值路径：a.b.c / items[*].content / events[0].run_id。

    返回 (found, value)。找不到时 found=False。
    """
    node = data
    rest = path.strip()
    if not rest:
        return True, node
    # 展开成 token 序列
    tokens: list[str] = []
    buf = ""
    for ch in rest:
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
        elif ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
        elif ch == "]":
            tokens.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    for tok in tokens:
        if node is None:
            return False, None
        if tok == "*":
            if not isinstance(node, list):
                return False, None
            node = node  # 供下一段作用于每个元素；本实现仅支持末尾通配
            continue
        if isinstance(tok, str) and tok.lstrip("-").isdigit() and isinstance(node, list):
            idx = int(tok)
            if -len(node) <= idx < len(node):
                node = node[idx]
            else:
                return False, None
            continue
        if isinstance(node, dict):
            if tok not in node:
                return False, None
            node = node[tok]
        else:
            return False, None
    return True, node


def collect_wildcard(data: Any, path: str) -> list[Any]:
    """收集 items[*].content 形式的所有值（[*] 只支持一处）。"""
    if "[*]" not in path:
        found, value = resolve_path(data, path)
        return [value] if found else []
    prefix, suffix = path.split("[*]", 1)
    found, base = resolve_path(data, prefix)
    if not found or not isinstance(base, list):
        return []
    values: list[Any] = []
    for item in base:
        if not suffix:
            values.append(item)
        else:
            sub = suffix.lstrip(".")
            f, v = resolve_path(item, sub)
            if f:
                values.append(v)
    return values


class Timer:
    """简单耗时器（毫秒）。"""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
