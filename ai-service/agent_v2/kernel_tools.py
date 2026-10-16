"""Bounded domain-tool adapters and the product PermissionGate bridge (T28).

Every hiking capability exposed to the UniHarness kernel goes through a
:class:`BoundedDomainTool` adapter:

* input validation via the kernel's pydantic ``args_schema`` contract;
* an execution timeout and an output-size cap;
* normalized public errors — raw exceptions never reach the model or the
  wire;
* one audit record per invocation bound to the public run identity
  (``run_id``, tool name, duration, outcome code) appended to a shared
  sink supplied per turn.

Adapters are EXECUTION LEAVES: they delegate to the existing product
implementation functions without copying algorithms, and they cannot call
chat back, start another agent runtime, choose profiles, or orchestrate
other tools.

The :class:`ProductGateRule` bridges the product's single
:class:`agent_v2.permissions.PermissionGate` into the kernel gate so every
kernel-initiated tool call receives ONE product verdict — unknown tools,
tools outside the profile allowlist, unclassified tools and high-risk
escalations all fail closed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
import shlex
import time
from typing import Any, Callable

from pydantic import BaseModel, Field
from uniharness.harness.permission import PermissionDecision, PermissionResult
from uniharness.harness.permission import SafetyRule as KernelSafetyRule
from uniharness.tools.base import BaseAgentTool
from uniharness.types import ToolResult

from agent_v2.kernel import KernelAssemblyError  # noqa: F401  (module-level re-export for builders)

logger = logging.getLogger("ai-service.agent_v2.kernel_tools")

# Output size cap for any wrapped domain result rendered to the model.
MAX_TOOL_OUTPUT_CHARS = 8_000
# Wall-clock cap for a single adapter execution.
TOOL_EXECUTION_TIMEOUT_SECONDS = 30.0

AuditSink = Callable[[dict[str, Any]], None]

# Tool names handled OUTSIDE the per-profile adapter registries (Ticket 32).
MEMORY_TOOL_NAMES: frozenset[str] = frozenset({"memory_save", "memory_search"})


class _GearChecklistParams(BaseModel):
    days: int | None = Field(default=None, ge=1, le=30)
    season: str | None = None
    experience: str | None = None
    camping: bool | None = None
    gear_level: str | None = None


class _RiskAssessmentParams(BaseModel):
    destination: str | None = None
    weather: str | None = None
    route: str | None = None
    experience: str | None = None
    days: int | None = Field(default=None, ge=1, le=60)


class _KnowledgeSearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    k: int | None = Field(default=None, ge=1, le=8)


class BoundedDomainTool(BaseAgentTool[Any]):
    """Kernel-execution-leaf base enforcing bounds + audit on subclasses."""

    # Subclasses set:
    delegate_name: str = ""
    timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS

    def __init__(
        self,
        *,
        run_id: str | None,
        audit_sink: AuditSink | None,
        delegate: Callable[..., Any],
    ) -> None:
        self._run_id = run_id or ""
        self._audit_sink = audit_sink
        self._delegate = delegate

    def _record(self, entry: dict[str, Any]) -> None:
        if self._audit_sink is not None:
            record = {"run_id": self._run_id, "tool": self.name}
            record.update(entry)
            try:
                self._audit_sink(record)
            except Exception:
                logger.exception("tool audit sink failure (non-fatal)")

    def _success_audit_fields(self, raw: Any) -> dict[str, Any]:
        """Return minimal, non-public metadata needed by the stream guard."""
        return {}

    async def execute(self, params: Any) -> ToolResult:
        started = time.perf_counter()
        args = params.model_dump(exclude_none=True) if isinstance(params, BaseModel) else {}
        try:
            raw = await asyncio.wait_for(
                self._delegate(**args),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record({
                "phase": "result",
                "ok": False,
                "code": "tool_timeout",
                "duration_ms": duration_ms,
            })
            return ToolResult(error=f"工具 {self.name} 执行超时，请稍后重试。")
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.exception("domain tool %s failed", self.name)
            self._record({
                "phase": "result",
                "ok": False,
                "code": "tool_error",
                "duration_ms": duration_ms,
            })
            return ToolResult(error=f"工具 {self.name} 执行失败，请稍后重试。")

        duration_ms = int((time.perf_counter() - started) * 1000)
        output = _normalize_result(raw)
        audit_entry = {
            "phase": "result",
            "ok": True,
            "code": None,
            "duration_ms": duration_ms,
        }
        audit_entry.update(self._success_audit_fields(raw))
        self._record(audit_entry)
        return ToolResult(output=output)


def _normalize_result(raw: Any) -> str:
    """Flatten the product dict/str result into a bounded model-facing string."""
    import json

    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = json.dumps(raw, ensure_ascii=False)
        except TypeError:
            text = str(raw)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + "\n…(输出已截断)"
    return text


class GearChecklistTool(BoundedDomainTool):
    name = "gear_checklist"
    description = "按天数、季节、经验和露营偏好生成保守的徒步装备清单。"
    args_schema = _GearChecklistParams
    delegate_name = "gear_checklist"

    def _success_audit_fields(self, raw: Any) -> dict[str, Any]:
        # Ticket 02: the turn evidence gate renders gear answers from
        # CONTROLLED fields, so the audit trail must carry a validated,
        # bounded copy of the structured result. An unusable result is
        # recorded as invalid — it must never launder a fake success.
        controlled = _controlled_gear_result(self.name, raw)
        return {"controlled": controlled} if controlled is not None else {}


class RiskAssessmentTool(BoundedDomainTool):
    name = "risk_assessment"
    description = "对目的地/天气/路线/经验进行保守的风险评估并给出建议。"
    args_schema = _RiskAssessmentParams
    delegate_name = "risk_assessment"

    def _success_audit_fields(self, raw: Any) -> dict[str, Any]:
        controlled = _controlled_risk_result(self.name, raw)
        return {"controlled": controlled} if controlled is not None else {}


class HikingKnowledgeSearchTool(BoundedDomainTool):
    name = "hiking_knowledge_search"
    description = "在徒步知识库中检索与查询相关的权威知识片段（RAG）。"
    args_schema = _KnowledgeSearchParams
    delegate_name = "hiking_knowledge_search"

    def _success_audit_fields(self, raw: Any) -> dict[str, Any]:
        chunks = raw.get("chunks") if isinstance(raw, Mapping) else None
        if not isinstance(chunks, list):
            chunks = []
        sources = [
            str(chunk.get("source"))
            for chunk in chunks
            if isinstance(chunk, Mapping) and isinstance(chunk.get("source"), str)
        ]
        # Spec #13 phase 2 (T05): bounded chunk summaries ride in the audit
        # trail so the RAG-page SSE adapter can render the documents panel.
        # has_chunks/sources stay authoritative for the grounding gate.
        chunk_summaries = [
            {
                "title": str(chunk.get("title") or ""),
                "source": str(chunk.get("source") or ""),
                "preview": str(chunk.get("preview") or "")[:220],
            }
            for chunk in chunks
            if isinstance(chunk, Mapping)
        ]
        return {
            "evidence": {
                "has_chunks": bool(chunks),
                "sources": sources,
                "chunks": chunk_summaries,
            }
        }


# ---- Ticket 02: controlled domain results ------------------------------------
#
# The turn evidence gate only lets deterministic tool results shape the
# public answer. These builders validate + bound the raw delegate output
# into a ``controlled`` payload; anything that fails validation is reported
# as ``{"ok": False, "code": "invalid_controlled_result"}`` so the gate can
# never render a pseudo success. Sizes are capped so the audit trail (which
# also rides on public tool_result events) stays bounded.

_CONTROLLED_MAX_CATEGORIES = 50
_CONTROLLED_MAX_ITEMS = 40
_CONTROLLED_MAX_TEXT = 300


def _bounded_text(value: Any, *, limit: int = _CONTROLLED_MAX_TEXT) -> str:
    text = value if isinstance(value, str) else ""
    return text[:limit]


def _bounded_items(value: Any) -> list[str]:
    """Non-empty strings of one list, capped at ``_CONTROLLED_MAX_ITEMS``
    (each entry capped at 120 chars). Non-list input yields ``[]``."""
    if not isinstance(value, list):
        return []
    return [_bounded_text(item, limit=120) for item in value[:_CONTROLLED_MAX_ITEMS] if item]


def _controlled_gear_result(tool_name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("ok") is not True:
        return _invalid_controlled(tool_name)
    checklist_raw = raw.get("checklist")
    if not isinstance(checklist_raw, Mapping) or not checklist_raw:
        return _invalid_controlled(tool_name)
    checklist: dict[str, list[str]] = {}
    for category, items in list(checklist_raw.items())[:_CONTROLLED_MAX_CATEGORIES]:
        bounded = _bounded_items(items)
        if bounded:
            checklist[_bounded_text(category, limit=40)] = bounded
    if not checklist:
        return _invalid_controlled(tool_name)
    return {
        "tool": tool_name,
        "ok": True,
        "days": raw.get("days") if isinstance(raw.get("days"), int) else None,
        "season": _bounded_text(raw.get("season"), limit=40) or "未指定",
        "experience": _bounded_text(raw.get("experience"), limit=40) or "未指定",
        "camping": bool(raw.get("camping")),
        "checklist": checklist,
        "limitations": _bounded_text(raw.get("message")),
    }


_RISK_LEVELS = frozenset({"low", "medium", "high"})


def _controlled_risk_result(tool_name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("ok") is not True:
        return _invalid_controlled(tool_name)
    level = raw.get("risk_level")
    if not isinstance(level, str) or level not in _RISK_LEVELS:
        return _invalid_controlled(tool_name)
    reasons = _bounded_items(raw.get("reasons"))
    if not reasons:
        return _invalid_controlled(tool_name)
    return {
        "tool": tool_name,
        "ok": True,
        "destination": _bounded_text(raw.get("destination"), limit=80) or "未指定",
        "risk_level": level,
        "reasons": reasons,
        "cancel_conditions": _bounded_items(raw.get("cancel_conditions")),
        "alternatives": _bounded_items(raw.get("alternatives")),
        "recommendation": _bounded_text(raw.get("recommendation")),
    }


def _invalid_controlled(tool_name: str) -> dict[str, Any]:
    return {"tool": tool_name, "ok": False, "code": "invalid_controlled_result"}


# Profile-aware registry: implemented adapters only. Anything allowed but
# neither here nor in the deferred set makes surface validation fail closed.
CHAT_TOOL_ADAPTERS: dict[str, type[BoundedDomainTool]] = {
    GearChecklistTool.delegate_name: GearChecklistTool,
    RiskAssessmentTool.delegate_name: RiskAssessmentTool,
    HikingKnowledgeSearchTool.delegate_name: HikingKnowledgeSearchTool,
}


# ---------------------------------------------------------------------------
# Ticket 30: run-scoped sandbox bridge + sandbox tool adapters
# ---------------------------------------------------------------------------


class ProductSandboxBridge:
    """Expose the product run-scoped Computer through the upstream protocol.

    The kernel decides WHAT sandbox command to run; the product Computer
    owns WHERE and HOW it executes (workspace chdir, timeout, env isolation).
    No host paths cross this boundary — exports go through the product
    artifact service only.
    """

    def __init__(self, computer: Any, workspace_path: str) -> None:
        self._computer = computer
        self.workspace_hint = str(workspace_path)

    @property
    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self, command: str, *, timeout: float | None = None):
        from uniharness.exceptions import CLIError

        from agent_v2.computer import SandboxUnavailableError

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._computer.run_bash,
                    self.workspace_hint,
                    command,
                    timeout_seconds=(timeout / 1000) if timeout else 20.0,
                ),
                timeout=(timeout / 1000 + 5.0) if timeout else 30.0,
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            if isinstance(exc, SandboxUnavailableError):
                # Ticket 01 AC1: the indirect (kernel tool) call keeps the
                # SAME refusal identity as a direct adapter call. The kernel
                # protocol still sees the CLIError it knows how to render,
                # but the stable ``sandbox_unavailable`` code rides on its
                # message and the original refusal stays as ``__cause__``.
                raise CLIError(f"{SandboxUnavailableError.code}: {exc}") from exc
            msg = "sandbox command timed out" if isinstance(exc, TimeoutError) else "sandbox command failed"
            raise CLIError(msg) from exc
        return result

    async def upload(self, src: str, dst: str) -> None:
        raise _CLI("cross-boundary upload is not permitted; write inside the workspace")

    async def download(self, src: str, dst: str) -> None:
        raise _CLI("cross-boundary download is not permitted; export goes through the product artifact service")


def _CLI(message: str):
    from uniharness.exceptions import CLIError

    return CLIError(message)


def _safe_rel(path: str) -> bool:
    """Reject absolute paths and traversal outside the run workspace."""
    if not path or path.startswith(("/", "\\")) or (len(path) > 1 and path[1] == ":"):
        return False
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    return ".." not in parts


class _SandboxPathParams(BaseModel):
    path: str = Field(..., min_length=1, max_length=512)


class _SandboxWriteParams(BaseModel):
    path: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., max_length=200_000)


class _SandboxEditParams(_SandboxWriteParams):
    old: str = Field(..., min_length=1)


class _MemorySearchParams(BaseModel):
    query: str | None = Field(default=None, max_length=200)


class _BashParams(BaseModel):
    command: str = Field(..., min_length=1, max_length=4000)


class _MemorySaveParams(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)


class MemorySaveTool(BoundedDomainTool):
    """Persist an ordinary preference for the SERVER-resolved subject (T32).

    Fail-closed ladder: no subject/service → refuse; not opted-in → refuse;
    sensitive content → refuse (consent candidates go through the product
    API only, never through this tool).
    """

    name = "memory_save"
    description = "把用户的一句普通偏好保存到其长期记忆（未开启或敏感内容会被拒绝）。"
    args_schema = _MemorySaveParams
    delegate_name = "memory_save"

    def __init__(self, *, memory: Any, subject_id: str | None, run_id: str | None, audit_sink: AuditSink | None) -> None:
        super().__init__(delegate=lambda *a, **k: None, run_id=run_id, audit_sink=audit_sink)
        self._memory = memory
        self._subject = subject_id

    async def execute(self, params: _MemorySaveParams) -> ToolResult:
        if not self._memory or not self._subject:
            self._record({"phase": "denied", "ok": False, "code": "no_subject", "duration_ms": 0})
            return ToolResult(error="缺少已认证主体，无法保存记忆。")
        if not self._memory.memory_enabled(self._subject):
            self._record({"phase": "denied", "ok": False, "code": "memory_not_opted_in", "duration_ms": 0})
            return ToolResult(error="该主体未开启长期记忆，已拒绝保存。")
        from agent_v2.memory_tools import extract_sensitive_memory

        if extract_sensitive_memory(params.content) is not None:
            self._record({"phase": "denied", "ok": False, "code": "sensitive_requires_consent", "duration_ms": 0})
            return ToolResult(error="内容含敏感信息，请通过记忆确认流程逐条批准后保存。")
        memory_id = self._memory.save_preference(self._subject, params.content)
        self._record({"phase": "result", "ok": memory_id is not None, "code": None, "duration_ms": 0})
        return ToolResult(output=f"saved:{memory_id}" if memory_id else "保存失败。")


class MemorySearchTool(BoundedDomainTool):
    """List the subject's active long-term memories (bounded output)."""

    name = "memory_search"
    description = "检索当前主体的活跃长期偏好记忆。"
    args_schema = _MemorySearchParams
    delegate_name = "memory_search"

    def __init__(self, *, memory: Any, subject_id: str | None, run_id: str | None, audit_sink: AuditSink | None) -> None:
        super().__init__(delegate=lambda *a, **k: None, run_id=run_id, audit_sink=audit_sink)
        self._memory = memory
        self._subject = subject_id

    async def execute(self, params: _MemorySearchParams) -> ToolResult:
        if not self._memory or not self._subject:
            self._record({"phase": "denied", "ok": False, "code": "no_subject", "duration_ms": 0})
            return ToolResult(error="缺少已认证主体，无法检索记忆。")
        items = self._memory.list_active_memories(self._subject)
        self._record({"phase": "result", "ok": True, "code": None, "duration_ms": 0})
        lines = [str(item.get("content", "")) for item in items]
        text = "；".join(lines) if lines else "（无活跃记忆）"
        return ToolResult(output=text[:MAX_TOOL_OUTPUT_CHARS])





class CoworkBashTool(BoundedDomainTool):
    """cowork-only Bash executed inside the run-scoped sandbox, wrapped by the
    product effect ledger (Ticket 31).

    Guard order per call:

    1. effect ledger must contain a row for (run, tool, canonical args) —
       the durable approval flow creates it in ``authorized``; a model call
       WITHOUT prior approval is rejected here (fail closed);
    2. CAS ``authorized → executing`` (only one executor);
    3. execute via the bridge, then ``committed`` (exit 0) or ``failed``;
    4. crash AFTER ``executing`` → ``outcome_unknown`` — recorded, never
       auto-replayed (the tool returns a stable public error).
    """

    name = "sandbox_bash"
    description = "在当前 run 的隔离沙箱工作区内执行一条 bash 命令（需事先批准）。"
    args_schema = _BashParams
    delegate_name = "sandbox_bash"

    def __init__(
        self,
        *,
        bridge: Any,
        run_id: str | None,
        effects: Any | None,  # RunRepository
        audit_sink: AuditSink | None,
    ) -> None:
        async def _exec(command: str) -> Any:
            return await bridge.run(command)

        super().__init__(delegate=_exec, run_id=run_id or "", audit_sink=audit_sink)
        self._effects = effects
        self._command = ""

    def _record(self, entry: dict[str, Any]) -> None:
        entry.setdefault("tool", self.name)
        super()._record(entry)

    async def execute(self, params: _BashParams) -> ToolResult:
        from agent_v2.effects import EffectStatus, compute_effect_key, serialize_result

        started = time.perf_counter()
        self._command = params.command
        if self._effects is None or not self._run_id:
            return ToolResult(error="effect ledger 不可用，拒绝执行 Bash。")

        effect_key = compute_effect_key(
            run_id=self._run_id,
            tool_name=self.name,
            args={"command": params.command},  # canonical args envelope
        )
        rec = self._effects.get_effect(effect_key)
        if rec is None:
            self._record({"phase": "denied", "ok": False, "code": "no_authorized_effect", "duration_ms": 0})
            return ToolResult(error="sandbox_bash 需要先经 durable approval（缺少授权记录），已拒绝执行。")

        if rec.status == EffectStatus.COMMITTED.value:
            recovered = self._effects.get_effect(effect_key)
            import json as _json

            stored = _json.loads(recovered.result_json) if recovered and recovered.result_json else {}
            return ToolResult(output=stored.get("stdout", "") or "")
        if rec.status == EffectStatus.EXECUTING.value:
            # Prior crash mid-execution → outcome_unknown semantics forbid
            # silent replay; requires human reconciliation.
            self._record({"phase": "denied", "ok": False, "code": "outcome_unknown", "duration_ms": 0})
            return ToolResult(error="上次执行结果未知（outcome_unknown），需人工处理，禁止自动重放。")
        if rec.status != EffectStatus.AUTHORIZED.value:
            self._record({"phase": "denied", "ok": False, "code": f"effect_{rec.status}", "duration_ms": 0})
            return ToolResult(error=f"effect 状态 {rec.status} 不允许执行。")

        _, won = self._effects.transition_effect_to_executing(effect_key)
        if not won:
            self._record({"phase": "denied", "ok": False, "code": "effect_race_lost", "duration_ms": 0})
            return ToolResult(error="该操作已由另一执行者接管。")

        try:
            res = await self._delegate(params.command)
        except Exception as exc:
            self._effects.mark_effect_outcome_unknown(effect_key, type(exc).__name__)
            self._record({"phase": "result", "ok": False, "code": "outcome_unknown", "duration_ms": 0})
            return ToolResult(error="执行结果未知（outcome_unknown），需人工处理，禁止自动重放。")

        exit_code = getattr(res, "exit_code", 1)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result_dict = {
            "exit_code": exit_code,
            "stdout": getattr(res, "stdout", "") or "",
            "stderr": getattr(res, "stderr", "") or "",
            "duration_ms": duration_ms,
            "truncated": bool(getattr(res, "truncated", False)),
        }
        if exit_code == 0:
            self._effects.commit_effect(effect_key, serialize_result(result_dict))
            self._record({"phase": "result", "ok": True, "code": None, "duration_ms": duration_ms})
            return ToolResult(output=result_dict["stdout"] or "(no output)")
        self._effects.mark_effect_failed(effect_key)
        self._record({"phase": "result", "ok": False, "code": f"exit_{exit_code}", "duration_ms": duration_ms})
        return ToolResult(error=f"命令失败（exit={exit_code}）。")


class SandboxReadTool(BoundedDomainTool):
    """Read a UTF-8 text file INSIDE the run workspace (path-guarded)."""

    name = "sandbox_read"
    description = "读取当前 run 工作区内相对路径的文本文件内容。禁止绝对路径与 '..'。"
    args_schema = _SandboxPathParams
    delegate_name = "sandbox_read"

    def __init__(self, *, bridge: Any, run_id: str | None, audit_sink: AuditSink | None) -> None:
        async def _exec(command: str) -> Any:
            return await bridge.run(command)

        super().__init__(delegate=_exec, run_id=run_id, audit_sink=audit_sink)

    async def execute(self, params: _SandboxPathParams) -> ToolResult:
        _started = time.perf_counter()
        if not _safe_rel(params.path):
            self._record({"phase": "denied", "ok": False, "code": "unsafe_path", "duration_ms": 0})
            return ToolResult(error=f"路径 {params.path!r} 越出当前工作区，已拒绝。")
        res = await self._delegate(f"cat -- {shlex.quote(params.path)}")
        stdout = getattr(res, "stdout", "")
        if getattr(res, "exit_code", 0) != 0:
            self._record({"phase": "result", "ok": False, "code": "tool_error", "duration_ms": 0})
            return ToolResult(error="读取失败（文件不存在或不可读）。")
        self._record({"phase": "result", "ok": True, "code": None,
                      "duration_ms": int((time.perf_counter() - _started) * 1000)})
        return ToolResult(output=stdout)


class SandboxWriteTool(BoundedDomainTool):
    """Write a UTF-8 text file INSIDE the run workspace (path-guarded).

    Command construction is fully escaped: the content travels base64-encoded
    through a shlex-quoted pipeline, so neither model-supplied text nor paths
    can break out of the workspace chdir performed by the product
    ``Computer.run_bash``.
    """

    name = "sandbox_write"
    description = "在工作区相对路径写入/覆盖文本文件。禁止绝对路径与 '..'。"
    args_schema = _SandboxWriteParams
    delegate_name = "sandbox_write"

    def __init__(self, *, bridge: Any, run_id: str | None, audit_sink: AuditSink | None) -> None:
        # Delegate = a closure over the bridge producing an escaped command.
        async def _exec(command: str) -> Any:
            return await bridge.run(command)

        super().__init__(delegate=_exec, run_id=run_id, audit_sink=audit_sink)

    def _record_denied(self, message: str) -> ToolResult:
        self._record({"phase": "denied", "ok": False, "code": "unsafe_path", "duration_ms": 0})
        return ToolResult(error=message)

    async def execute(self, params: _SandboxWriteParams) -> ToolResult:
        _started = time.perf_counter()
        if not _safe_rel(params.path):
            return self._record_denied(f"路径 {params.path!r} 越出当前工作区，已拒绝。")
        import base64

        encoded = base64.b64encode(params.content.encode()).decode()
        command = f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(params.path)}"
        res = await self._delegate(command)
        stdout = getattr(res, "stdout", "") or ""
        self._record({"phase": "result", "ok": True, "code": None,
                      "duration_ms": int((time.perf_counter() - _started) * 1000)})
        return ToolResult(output=str(len(stdout.strip()) or len(params.content.encode())))



class _RouteResearchParams(BaseModel):
    destination: str = Field(..., min_length=1, max_length=80)
    date: str | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    focus: str | None = None


class _WebSearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


class _WebScrapingParams(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


def _make_simple_adapter(name: str, description: str, params: type[BaseModel], delegate: str) -> type[BoundedDomainTool]:
    body: dict[str, Any] = {
        "name": name,
        "description": description,
        "args_schema": params,
        "delegate_name": delegate,
    }
    return type(name.capitalize() + "Tool", (BoundedDomainTool,), body)


RouteResearchTool = _make_simple_adapter(
    "route_research",
    "检索目的地徒步路线候选并返回结构化事实（里程/爬升/下撤点等）。调用产品既有实现，不自行编造。",
    _RouteResearchParams,
    "route_research",
)
WebSearchTool = _make_simple_adapter("web_search", "通过产品 Tavily 通道执行网络搜索。", _WebSearchParams, "web_search")
WebScrapingTool = _make_simple_adapter("web_scraping", "抓取指定网页正文（经产品通道清洗为文本）。", _WebScrapingParams, "web_scraping")

PROFILE_TOOL_ADAPTERS: dict[str, dict[str, type[BoundedDomainTool]]] = {
    "chat": {
        GearChecklistTool.delegate_name: GearChecklistTool,
        RiskAssessmentTool.delegate_name: RiskAssessmentTool,
        HikingKnowledgeSearchTool.delegate_name: HikingKnowledgeSearchTool,
    },
    "route_plan": {
        GearChecklistTool.delegate_name: GearChecklistTool,
        RiskAssessmentTool.delegate_name: RiskAssessmentTool,
        HikingKnowledgeSearchTool.delegate_name: HikingKnowledgeSearchTool,
        RouteResearchTool.delegate_name: RouteResearchTool,
        WebSearchTool.delegate_name: WebSearchTool,
        WebScrapingTool.delegate_name: WebScrapingTool,
    },
    # Spec #13 phase 2: rag profile — knowledge search is the only tool;
    # retrieval planning lives inside the delegate, not in the tool surface.
    "rag": {
        HikingKnowledgeSearchTool.delegate_name: HikingKnowledgeSearchTool,
    },
}

_CHAT_TOOL_ADAPTERS = PROFILE_TOOL_ADAPTERS["chat"]


# ---------------------------------------------------------------------------
# Ticket 33: fixed bounded subagent definitions for the root factory
# ---------------------------------------------------------------------------


def build_subagent_definitions(
    profile: Any,
    settings: Any,
    *,
    extra_tool_names: set[str],
    available_mcp_servers: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map the product's fixed roles to vendored ``AgentDefinition``s.

    Returns ``(definitions, audit_entries)``. A role is included ONLY when
    every tool in its allowlist is present on the current kernel surface
    and every MCP server it needs is already wired on the root — an
    unsatisfiable role is dropped with an audit entry instead of breaking
    the whole run (partial failure must not destroy the root turn). The
    model or a client can never invent roles: only keys of this map exist.
    """
    from agent_v2.subagents import SUBAGENT_DEFINITIONS
    from uniharness.harness.definition import AgentDefinition

    if not getattr(settings, "agent_v2_subagent_enabled", False):
        return {}, []

    defs: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    for role, sub in SUBAGENT_DEFINITIONS.items():
        missing_tools = [t for t in sub.allowed_tools if t not in extra_tool_names]
        missing_mcp = [
            s for s in (sub.allowed_mcp_servers or ())
            if s not in available_mcp_servers
        ]
        if missing_tools or missing_mcp:
            audits.append({
                "phase": "subagent_skipped",
                "ok": False,
                "code": "subagent_surface_unavailable",
                "role": role.value,
                "missing_tools": missing_tools,
                "missing_mcp": missing_mcp,
            })
            continue
        defs[role.value] = AgentDefinition(
            description=sub.system_duty,
            system_prompt=sub.system_duty,
            tools=tuple(sub.allowed_tools),
            model=None,  # inherit the root ModelProfile (server-owned budget)
        )
    return defs, audits


def _unwrapped_delegate(tool_name: str) -> Callable[..., Any]:
    """Resolve the EXISTING product implementation function (no copying)."""
    if tool_name == "gear_checklist":
        from tools.hiking_domain import gear_checklist

        return _unwrap_langchain_tool(gear_checklist)
    if tool_name == "risk_assessment":
        from tools.hiking_domain import risk_assessment

        return _unwrap_langchain_tool(risk_assessment)
    if tool_name == "hiking_knowledge_search":
        from tools.hiking_knowledge import hiking_knowledge_search

        return _unwrap_langchain_tool(hiking_knowledge_search)
    if tool_name == "route_research":
        from tools.hiking_domain import route_research

        return _unwrap_langchain_tool(route_research)
    if tool_name == "web_search":
        from tools.web_search import web_search

        return _unwrap_langchain_tool(web_search)
    if tool_name == "web_scraping":
        from tools.web_scraping import web_scraping

        return _unwrap_langchain_tool(web_scraping)
    raise KernelAssemblyError(f"no product delegate registered for {tool_name!r}")


_build_profile_mcp_unavailable_note = "MCP-server-unavailable"


def build_profile_mcp_servers(
    profile: Any, settings: Any
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Materialize uni-kernel MCP configs for the profile's allowlist.

    Returns ``(kernel_servers_config, audit_entries)``. Unavailable servers
    (missing credentials / unreachable per ``check_mcp_available``) are
    DROPPED with an audit entry so the model gets explicit degradation
    context instead of a mid-run failure; fabricating evidence is never an
    option. When credential expansion fails for any included server that is
    also fail-closed via :class:`KernelAssemblyError`.
    """
    from agent_v2.mcp_registry import (
        check_mcp_available,
        expand_credentials,
        resolve_mcp_server,
    )

    allowed = getattr(profile, "allowed_mcp_servers", None)
    if not allowed:
        return {}, []
    servers: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for server_id in sorted(allowed):
        cfg = resolve_mcp_server(server_id)
        try:
            available = check_mcp_available(cfg, settings)
            if not available:
                raise KernelAssemblyError(f"MCP server {server_id} 不可用")
            env, headers = expand_credentials(cfg, settings)
            if missing := [k for k in cfg.env_key_names if not env.get(k)]:
                raise KernelAssemblyError(f"MCP server {server_id} 缺少凭证: {missing}")
        except KernelAssemblyError as exc:
            logger.warning("mcp preflight drop %s: %s", server_id, exc)
            audits.append({
                "phase": "mcp_unavailable",
                "ok": False,
                "code": "mcp_server_unavailable",
                "server_id": server_id,
            })
            continue
        if cfg.transport == "stdio":
            servers[cfg.server_id] = {
                "type": "stdio",
                "command": cfg.command_or_url,
                "args": list(cfg.args),
                "env": env,
                **({"headers": headers} if headers else {}),
            }
        elif cfg.transport in {"sse", "http"}:
            servers[cfg.server_id] = {
                "type": cfg.transport,
                "url": cfg.command_or_url,
                **({"env": env} if env else {}),
                **({"headers": headers} if headers else {}),
            }
        else:
            raise KernelAssemblyError(f"unsupported mcp transport for {server_id}")
    return servers, audits


def _split_mcp_tool_name(tool_name: str) -> tuple[str, str] | None:
    """Parse kernel-style ``mcp__<server>__<tool…>`` names."""
    if not tool_name.startswith("mcp__"):
        return None
    rest = tool_name[len("mcp__") :]
    if "__" not in rest:
        return None
    server, tool = rest.split("__", 1)
    return server or "", tool or ""


def build_chat_extra_tools(
    profile: Any,
    *,
    run_id: str | None,
    audit_sink: AuditSink | None,
    memory_service: Any | None = None,
    subject_id: str | None = None,
) -> list[BoundedDomainTool]:
    """Instantiate adapters for the profile's implementable tool surface.

    Fail-closed check: every ``profile.allowed_tools`` name must be either an
    implemented adapter, a deferred product-layer capability (memory_*), or
    the caller explicitly narrows allowed_tools upstream. Otherwise assembly
    refuses to start (unknown surface ⇒ no silent partial exposure).
    """
    from agent_v2.kernel import KernelAssemblyError

    profile_key = getattr(getattr(profile, "profile_id", None), "value", str(getattr(profile, "profile_id", "")))
    registry = PROFILE_TOOL_ADAPTERS.get(profile_key)
    if registry is None:
        raise KernelAssemblyError(f"no kernel tool adapters registered for profile {profile_key!r}")

    memory_names = {"memory_save", "memory_search"}
    missing = sorted(
        set(profile.allowed_tools)
        - set(registry)
        - memory_names
    )
    if missing:
        raise KernelAssemblyError(
            f"profile '{profile_key}' "
            f"允许了未实现内核 adapter 的工具：{missing}"
        )
    tools: list[BoundedDomainTool] = [
        registry[name](run_id=run_id, audit_sink=audit_sink, delegate=_unwrapped_delegate(name))
        for name in sorted(set(profile.allowed_tools) & set(registry))
    ]
    # Ticket 32: memory tools require BOTH the product service and a
    # server-resolved subject; without either they are not constructed and
    # the kernel never sees a way to touch long-term memory.
    if memory_names & set(profile.allowed_tools):
        if memory_service is None or not subject_id:
            raise KernelAssemblyError(
                "profile 允许 memory_* 但缺少已认证主体或 memory service —— fail closed"
            )
        if "memory_save" in profile.allowed_tools:
            tools.append(MemorySaveTool(
                memory=memory_service, subject_id=subject_id,
                run_id=run_id, audit_sink=audit_sink,
            ))
        if "memory_search" in profile.allowed_tools:
            tools.append(MemorySearchTool(
                memory=memory_service, subject_id=subject_id,
                run_id=run_id, audit_sink=audit_sink,
            ))
    return tools


def _unwrap_langchain_tool(obj: Any) -> Callable[..., Any]:
    """``@tool``-decorated product functions arrive as StructuredTool.

    The adapter must call the ORIGINAL coroutine directly (no LangChain
    machinery in the execution leaf)."""
    if hasattr(obj, "coroutine") and obj.coroutine is not None:
        return obj.coroutine
    if hasattr(obj, "func") and callable(obj.func):
        return obj.func
    if callable(obj):
        return obj
    raise KernelAssemblyError(f"cannot unwrap delegate {obj!r}")


def _delegate_for(tool_name: str) -> Callable[..., Any]:
    """Resolve the EXISTING product implementation function (no copying)."""
    if tool_name == "gear_checklist":
        from tools.hiking_domain import gear_checklist

        return _unwrap_langchain_tool(gear_checklist)
    if tool_name == "risk_assessment":
        from tools.hiking_domain import risk_assessment

        return _unwrap_langchain_tool(risk_assessment)
    if tool_name == "hiking_knowledge_search":
        from tools.hiking_knowledge import hiking_knowledge_search

        return _unwrap_langchain_tool(hiking_knowledge_search)
    raise KernelAssemblyError(f"no product delegate registered for {tool_name!r}")


class ProductGateRule(KernelSafetyRule):
    """Bridge product PermissionGate verdicts into the kernel gate.

    Registered as THE rule of the kernel PermissionGate handed to
    ``create_agent(permission_gate=...)`` so every kernel tool call is
    decided by exactly one product evaluation:

    * ALLOW → returns None (chain continues → upstream default allow);
    * DENY  → kernel DENIED carrying the stable public reason;
    * NEEDS_APPROVAL → DENIED for now (durable approval arrives with T31);
      recorded in the shared audit sink so denial evidence is observable.
    """

    def __init__(
        self,
        profile: Any,
        *,
        run_id: str | None,
        audit_sink: AuditSink | None,
    ) -> None:
        from agent_v2.permissions import PermissionGate as ProductGate

        self._product_gate = ProductGate()
        self._profile = profile
        self._run_id = run_id or ""
        self._audit_sink = audit_sink

    def check(self, tool_name: str, tool_args: dict[str, Any]) -> PermissionDecision | None:
        # Ticket 29: MCP tool names are policy-checked against the profile's
        # server allowlist AND the server's declared tool_allowlist — the
        # model or a client can never widen either layer.
        mcp = _split_mcp_tool_name(tool_name)
        if mcp is not None:
            server_id, tool = mcp
            allowed_servers = tuple(getattr(self._profile, "allowed_mcp_servers", None) or ())
            if server_id not in allowed_servers:
                self._record_denial(tool_name, "unknown_mcp_server")
                return PermissionDecision(result=PermissionResult.DENIED, reason="unknown_mcp_server")
            from agent_v2.mcp_registry import resolve_mcp_server

            cfg = resolve_mcp_server(server_id)
            if tool not in cfg.tool_allowlist:
                self._record_denial(tool_name, "unauthorized_mcp_tool")
                return PermissionDecision(
                    result=PermissionResult.DENIED, reason="unauthorized_mcp_tool"
                )
            # Approved pair → judged by the PRODUCT risk classifier so
            # high-risk MCP tools require the same approval ladder as the
            # rest of the surface; low-risk pairs pass without noise.
            from tools.risk_classifier import (
                RiskLevel as _RiskLevel,
                classify_tool as _classify,
                requires_confirmation as _needs_approval,
            )

            level = _classify(tool_name.split("__")[-1], default=_RiskLevel.HIGH)
            if _needs_approval(level):
                self._record_denial(tool_name, "high_risk_requires_approval")
                return PermissionDecision(
                    result=PermissionResult.DENIED,
                    reason="该操作需要审批，当前阶段暂不可用。",
                )
            return None

        verdict = self._product_gate.evaluate(self._profile, tool_name, tool_args)

        if verdict.decision.value == "allow":
            return None
        if verdict.decision.value == "deny":
            self._record_denial(tool_name, verdict.reason)
            return PermissionDecision(result=PermissionResult.DENIED, reason=verdict.reason)
        # needs_approval: T31 will map durable approvals; today fail closed.
        self._record_denial(tool_name, verdict.reason)
        return PermissionDecision(
            result=PermissionResult.DENIED,
            reason="该操作需要审批，当前阶段暂不可用。",
        )

    def _record_denial(self, tool_name: str, reason: str) -> None:
        if self._audit_sink is None:
            return
        try:
            self._audit_sink({
                "run_id": self._run_id,
                "tool": tool_name,
                "phase": "denied",
                "ok": False,
                "code": reason,
                "duration_ms": 0,
            })
        except Exception:
            logger.exception("gate denial audit failed (non-fatal)")


def build_chat_kernel_gate(profile: Any, *, run_id: str | None, audit_sink: AuditSink | None) -> Any:
    """Build the kernel PermissionGate carrying the product bridge rule."""
    from uniharness.harness.permission import PermissionGate as KernelGate

    gate = KernelGate()
    gate.register_rule(ProductGateRule(profile, run_id=run_id, audit_sink=audit_sink))
    return gate
