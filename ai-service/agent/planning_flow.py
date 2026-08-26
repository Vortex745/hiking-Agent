"""OpenManus-style planning tool and serial planning flow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PlanStepStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def active_statuses(cls) -> set[str]:
        return {cls.NOT_STARTED.value, cls.IN_PROGRESS.value}


@dataclass
class PlanningTool:
    name: str = "planning"
    description: str = "Create and manage plans with step statuses."
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_plan_id: str | None = None

    async def execute(self, command: str, **kwargs: Any) -> dict[str, Any]:
        if command == "create":
            return self._create(kwargs["plan_id"], kwargs.get("title") or "Untitled", kwargs.get("steps") or [])
        if command == "mark_step":
            return self._mark_step(
                kwargs.get("plan_id") or self.current_plan_id,
                int(kwargs["step_index"]),
                kwargs["step_status"],
                kwargs.get("step_notes", ""),
            )
        if command == "get":
            return self._get(kwargs.get("plan_id") or self.current_plan_id)
        raise ValueError(f"Unsupported planning command: {command}")

    def _create(self, plan_id: str, title: str, steps: list[str]) -> dict[str, Any]:
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": list(steps),
            "step_statuses": [PlanStepStatus.NOT_STARTED.value for _ in steps],
            "step_notes": ["" for _ in steps],
        }
        self.plans[plan_id] = plan
        self.current_plan_id = plan_id
        return {"ok": True, "plan": plan}

    def _mark_step(self, plan_id: str | None, index: int, status: str, notes: str = "") -> dict[str, Any]:
        if not plan_id or plan_id not in self.plans:
            raise ValueError("Plan not found")
        plan = self.plans[plan_id]
        plan["step_statuses"][index] = status
        if notes:
            plan["step_notes"][index] = notes
        return {"ok": True, "plan": plan}

    def _get(self, plan_id: str | None) -> dict[str, Any]:
        if not plan_id or plan_id not in self.plans:
            raise ValueError("Plan not found")
        return {"ok": True, "plan": self.plans[plan_id]}


@dataclass
class PlanningFlow:
    """Serially execute the first active plan step with a supplied agent."""

    agent: Any
    planning_tool: PlanningTool = field(default_factory=PlanningTool)
    active_plan_id: str = field(default_factory=lambda: f"plan_{int(time.time())}")

    async def create_plan(self, title: str, steps: list[str]) -> dict[str, Any]:
        return await self.planning_tool.execute(
            "create",
            plan_id=self.active_plan_id,
            title=title,
            steps=steps,
        )

    async def execute_next(self) -> str:
        plan = self.planning_tool.plans[self.active_plan_id]
        for index, status in enumerate(plan["step_statuses"]):
            if status in PlanStepStatus.active_statuses():
                await self.planning_tool.execute(
                    "mark_step",
                    plan_id=self.active_plan_id,
                    step_index=index,
                    step_status=PlanStepStatus.IN_PROGRESS.value,
                )
                result = await self.agent.run(plan["steps"][index])
                await self.planning_tool.execute(
                    "mark_step",
                    plan_id=self.active_plan_id,
                    step_index=index,
                    step_status=PlanStepStatus.COMPLETED.value,
                )
                return result
        return "Plan completed."
