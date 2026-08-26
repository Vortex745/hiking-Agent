from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentExitStatus(str, Enum):
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STUCK = "stuck"
    ERROR = "error"


@dataclass(slots=True)
class AgentExit:
    status: AgentExitStatus
    reason: str
    final_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_event(self) -> dict[str, Any] | None:
        if not self.final_text:
            return None
        return {
            "type": "text",
            "content": self.final_text,
            "metadata": {
                "phase": "task_exit",
                "status": self.status.value,
                "reason": self.reason,
                **self.metadata,
            },
        }

    def done_event(self) -> dict[str, Any]:
        return {
            "type": "done",
            "content": "",
            "metadata": {
                "status": self.status.value,
                "reason": self.reason,
                **self.metadata,
            },
        }


class AgentTaskExitController:
    """Translate runtime stop conditions into user-facing Agent exits."""

    _BUDGET_MARKERS = (
        "need more steps",
        "max steps",
        "step budget",
    )

    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def completed(self, reason: str = "agent_completed") -> AgentExit:
        return AgentExit(status=AgentExitStatus.COMPLETED, reason=reason)

    def done_event(
        self,
        status: str = AgentExitStatus.COMPLETED.value,
        reason: str = "agent_completed",
        **metadata: Any,
    ) -> dict[str, Any]:
        try:
            exit_status = AgentExitStatus(status)
        except ValueError:
            exit_status = AgentExitStatus.ERROR
        return AgentExit(status=exit_status, reason=reason, metadata=metadata).done_event()

    def from_exception(
        self,
        exc: Exception,
        *,
        context: Any | None = None,
        current_step: int = 0,
    ) -> AgentExit | None:
        if not self._is_budget_exhausted(exc):
            return None
        return AgentExit(
            status=AgentExitStatus.BUDGET_EXHAUSTED,
            reason="step_budget_exhausted",
            final_text=self._budget_exhausted_message(context),
            metadata={
                "current_step": current_step,
                "max_steps": self.max_steps,
            },
        )

    def from_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        current_step: int = 0,
    ) -> AgentExit | None:
        if tool_name != "terminate":
            return None

        reason = self._extract_terminate_reason(content)
        if self._looks_waiting_for_user(reason):
            status = AgentExitStatus.WAITING_FOR_USER
            final_text = f"我需要你补充一个关键信息后再继续：{reason}"
        elif reason:
            status = AgentExitStatus.COMPLETED
            final_text = f"任务已结束：{reason}"
        else:
            status = AgentExitStatus.COMPLETED
            final_text = "任务已完成。"

        return AgentExit(
            status=status,
            reason=reason or "terminate_called",
            final_text=final_text,
            metadata={
                "tool": "terminate",
                "current_step": current_step,
                "max_steps": self.max_steps,
            },
        )

    def _is_budget_exhausted(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__} {str(exc)}".lower()
        return any(marker in text for marker in self._BUDGET_MARKERS)

    def _budget_exhausted_message(self, context: Any | None) -> str:
        if self._has_prefetched_evidence(context):
            return (
                "本轮执行已达到执行步数上限，我已停止继续调用工具，避免继续空转。"
                "当前已拿到部分定位或天气信息，但还没完成最终风险判断；"
                "请重新发送问题，或补充目的地、日期和路线强度后我会继续给出保守建议。"
            )
        return (
            "本轮执行已达到执行步数上限，我已停止继续调用工具，避免继续空转。"
            "请缩小问题范围，或补充目的地、日期、路线强度等关键信息后继续。"
        )

    def _has_prefetched_evidence(self, context: Any | None) -> bool:
        results = getattr(context, "prefetched_tool_results", None) if context is not None else None
        return bool(results)

    def _extract_terminate_reason(self, content: str) -> str:
        text = " ".join(str(content or "").split())
        marker = "原因:"
        if marker in text:
            reason = text.split(marker, 1)[1].strip()
            return reason.removesuffix("）").removesuffix(")").strip()
        if text.startswith("任务已被 Agent 终止"):
            return ""
        return text

    def _looks_waiting_for_user(self, reason: str) -> bool:
        return any(marker in reason for marker in ("补充", "等待用户", "需要用户", "澄清", "无法继续"))
