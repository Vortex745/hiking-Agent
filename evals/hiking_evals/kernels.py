"""可控内核替身：scripted 内核与 agent-policy 内核。

评测不依赖真实 LLM（除非 --live）。内核替身遵循
``agent_v2.run_orchestration.KernelTurn`` 契约：
``async def kernel_turn(**kwargs) -> (response_text, audit_entries)``。

两种模式（fixture.kernel 配置）：

1. scripted —— 完全脚本化：
   {"mode": "script", "text": "...", "audits": [{tool, ok, code, duration_ms}]}
   {"mode": "script", "error": {"code": "model_unavailable", "message": "..."}}

2. agent_policy —— 确定性"模型决策"替身：按规则选工具、经真实
   args_schema 校验参数、真实执行 BoundedDomainTool（含超时/错误
   归一化路径），再按模板合成最终回答。工具调用全部记录到
   call_log，供 scorer 检查工具选择/参数/次数/顺序。
"""

from __future__ import annotations

import time
from typing import Any

from .models import ResponseRecord


class ScriptedKernel:
    """脚本化内核回合。所有调用入参（含未来可能出现的历史输入）记录到
    ``call_kwargs``，供评测观测上下文接线是否存在——评测只记录事实，
    不裁决产品（产品验收属于后续票）。
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self._script = script
        self.calls = 0
        self.call_kwargs: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        self.calls += 1
        self.call_kwargs.append(dict(kwargs))
        if "error" in self._script:
            from agent_v2.kernel import KernelModelFailure

            err = self._script["error"]
            raise KernelModelFailure(err.get("code", "model_unavailable"), err.get("message", ""))
        audits = [dict(a) for a in self._script.get("audits") or []]
        return str(self._script.get("text", "")), audits


class AgentPolicyKernel:
    """确定性 agent 策略内核：真实执行 profile 暴露的工具面。

    规则: [{"when_message_contains": "装备", "tool": "gear_checklist",
            "args": {...}, "invalid_args": {...}(可选，跳过 schema 校验直传)}]
    模板: 最终回答文本，占位符 {tool_output:<name>} / {tool_error:<name>}。
    """

    def __init__(self, policy: dict[str, Any]) -> None:
        self._rules = list(policy.get("rules") or [])
        self._template = str(policy.get("template", ""))
        self._unknown_tool_message = str(
            policy.get("unknown_tool_message", "所需工具当前不可用，已停止回答。")
        )
        self.call_log: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        message = str(kwargs.get("message", ""))
        extra_tools = list(kwargs.get("extra_tools") or ())
        audits: list[dict[str, Any]] = []
        substitutions: dict[str, str] = {}

        for rule in self._rules:
            if not self._rule_matches(message, rule):
                continue
            tool_name = str(rule.get("tool", ""))
            tool = next((t for t in extra_tools if t.name == tool_name), None)
            if tool is None:
                audits.append({
                    "tool": tool_name,
                    "phase": "result",
                    "ok": False,
                    "code": "tool_unavailable",
                    "duration_ms": 0,
                })
                substitutions[f"tool_error:{tool_name}"] = self._unknown_tool_message
                self.call_log.append({
                    "tool": tool_name,
                    "args": None,
                    "ok": False,
                    "error": "tool_unavailable",
                    "output": None,
                })
                continue
            args_source = (
                rule["invalid_args"]
                if "invalid_args" in rule
                else (rule.get("args") or {})
            )
            args = dict(args_source)
            timeout_override = rule.get("timeout_seconds")
            if timeout_override is not None:
                # 仅在本次调用内收紧工具超时（评测注入用，调用后恢复）
                original_timeout = tool.timeout_seconds
                tool.timeout_seconds = float(timeout_override)
                try:
                    entry = await self._execute_tool(tool, args)
                finally:
                    tool.timeout_seconds = original_timeout
            else:
                entry = await self._execute_tool(tool, args)
            self.call_log.append(entry)
            audits.append(entry["audit"])
            if entry["ok"]:
                substitutions[f"tool_output:{tool_name}"] = entry["output"] or ""
            else:
                substitutions[f"tool_error:{tool_name}"] = entry["error"] or ""

        text = self._template
        for key, value in substitutions.items():
            text = text.replace("{" + key + "}", value)
        return text, audits

    @staticmethod
    def _rule_matches(message: str, rule: dict[str, Any]) -> bool:
        contains = rule.get("when_message_contains")
        if contains is None:
            return True
        return all(str(part) in message for part in contains)

    async def _execute_tool(self, tool: Any, args: dict[str, Any]) -> dict[str, Any]:
        """经真实 args_schema 构造参数并执行，返回 call log 条目。"""
        started = time.perf_counter()
        try:
            params = tool.args_schema(**args)
        except Exception as exc:
            return {
                "tool": tool.name,
                "args": args,
                "ok": False,
                "error": f"invalid_args: {type(exc).__name__}: {exc}",
                "output": None,
                "audit": {
                    "tool": tool.name,
                    "phase": "result",
                    "ok": False,
                    "code": "invalid_args",
                    "duration_ms": 0,
                },
            }
        try:
            result = await tool.execute(params)
        except Exception as exc:
            return {
                "tool": tool.name,
                "args": args,
                "ok": False,
                "error": f"execution_error: {type(exc).__name__}: {exc}",
                "output": None,
                "audit": {
                    "tool": tool.name,
                    "phase": "result",
                    "ok": False,
                    "code": "tool_error",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            }
        duration_ms = int((time.perf_counter() - started) * 1000)
        if getattr(result, "error", None):
            # BoundedDomainTool 归一化后的公共错误（超时/失败）
            code = "tool_timeout" if "超时" in result.error else "tool_error"
            return {
                "tool": tool.name,
                "args": args,
                "ok": False,
                "error": result.error,
                "output": None,
                "audit": {
                    "tool": tool.name,
                    "phase": "result",
                    "ok": False,
                    "code": code,
                    "duration_ms": duration_ms,
                },
            }
        output = str(getattr(result, "output", "") or "")
        return {
            "tool": tool.name,
            "args": args,
            "ok": True,
            "error": None,
            "output": output[:4000],
            "audit": {
                "tool": tool.name,
                "phase": "result",
                "ok": True,
                "code": None,
                "duration_ms": duration_ms,
            },
        }


def build_kernel(fixture: dict[str, Any]) -> ScriptedKernel | AgentPolicyKernel | None:
    """按 fixture.kernel 配置构建内核替身；live 模式由 FixtureEnv 负责跳过。"""
    kernel_cfg = fixture.get("kernel") or {}
    mode = kernel_cfg.get("mode")
    if mode == "script":
        return ScriptedKernel(kernel_cfg)
    if mode == "agent_policy":
        return AgentPolicyKernel(kernel_cfg)
    return None


def attach_kernel_record(record: ResponseRecord, kernel: Any) -> None:
    """把内核层工具调用日志写入动作记录（供 scorer 与复盘）。

    内核实例跨动作共享且 call_log 只增不清空，这里按游标只附加自
    上次附加以来的新条目，避免多动作 case 的日志互相污染。
    """
    log = getattr(kernel, "call_log", None)
    if not log:
        return
    cursor = getattr(kernel, "_attach_cursor", 0)
    fresh = log[cursor:]
    if fresh:
        record.tool_calls = [dict(e) for e in fresh]
    kernel._attach_cursor = len(log)
