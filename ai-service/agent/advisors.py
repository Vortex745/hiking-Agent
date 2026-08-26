import logging

logger = logging.getLogger("ai-service.agent")


class LoggerAdvisor:
    """Logs each step of the agent's reasoning process."""

    def __init__(self):
        self.steps = []

    def on_step(self, step_type, content: str | None = None):
        """Record a step in the agent's execution.

        Accepts the legacy ``(step_type, content)`` shape and the newer
        structured dict shape used by tests and stream records.
        """
        if isinstance(step_type, dict):
            step = dict(step_type)
            action = step.get("action") or step.get("type") or "step"
            text = str(step.get("content", ""))
        else:
            action = str(step_type)
            text = content or ""
            step = {"type": action, "content": text}

        self.steps.append(step)
        logger.info("[Agent Step - %s] %s", action, text[:200])

    def get_steps(self) -> list:
        return [dict(step) for step in self.steps]

    def clear(self):
        self.steps = []


class ReReadAdvisor:
    """Provides the last N messages for re-reading context."""

    def __init__(self, window_size: int | None = None, recent_n: int | None = None):
        self.recent_n = recent_n if recent_n is not None else (window_size if window_size is not None else 5)
        self.window_size = self.recent_n

    def get_recent_context(self, history: list) -> list:
        """Return recent conversation messages, capped at recent_n user/assistant turns."""
        limit = max(self.recent_n * 2, 0)
        if limit == 0:
            return []
        recent = history[-limit:] if len(history) > limit else history
        return [dict(msg) if isinstance(msg, dict) else msg for msg in recent]
