"""OpenManus-style ReAct split: think, then act."""

from __future__ import annotations

from abc import abstractmethod

from agent.base_agent import BaseAgent


class ReActAgent(BaseAgent):

    @abstractmethod
    async def think(self) -> bool:
        """Decide whether an action is needed."""

    @abstractmethod
    async def act(self) -> str:
        """Execute the selected action."""

    async def step(self) -> str:
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
