"""React guard: stuck detection and repeat-call prevention.

StuckDetector — detects when the ReAct loop is making no progress
(repeated assistant content or repeated tool observations).

RepeatCallDetector — prevents the same tool+args from being called
twice within a single turn.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


# Same observation/content repeated this many times → stuck.
_STUCK_THRESHOLD = 3


class StuckDetector:
    """Detect when the ReAct loop is making no progress.

    Tracks tool observations and assistant content. If the same
    (tool_name, content) pair or the same assistant text appears
    _STUCK_THRESHOLD times, the loop is considered stuck.
    """

    def __init__(self, threshold: int = _STUCK_THRESHOLD) -> None:
        self._threshold = threshold
        self._observations: list[tuple[str, str]] = []
        self._assistant_contents: list[str] = []

    def record_observation(self, tool_name: str, content: str) -> None:
        self._observations.append((tool_name, content))

    def record_assistant_content(self, content: str) -> None:
        self._assistant_contents.append(content)

    def is_stuck(self) -> bool:
        """Return True if the loop appears to be stuck."""
        # Check repeated observations
        obs_counter = Counter(self._observations)
        for (_, _), count in obs_counter.items():
            if count >= self._threshold:
                return True

        # Check repeated assistant content
        text_counter = Counter(self._assistant_contents)
        for _, count in text_counter.items():
            if count >= self._threshold:
                return True

        return False

    def stuck_reason(self) -> str:
        """Return a human-readable explanation of why we're stuck."""
        obs_counter = Counter(self._observations)
        for (tool_name, content), count in obs_counter.items():
            if count >= self._threshold:
                return f"工具 {tool_name} 返回相同结果 {count} 次（重复），内容: {content[:50]}"

        text_counter = Counter(self._assistant_contents)
        for content, count in text_counter.items():
            if count >= self._threshold:
                return f"助手重复生成相同内容 {count} 次: {content[:50]}"

        return ""

    def reset(self) -> None:
        self._observations.clear()
        self._assistant_contents.clear()


class RepeatCallDetector:
    """Prevent the same tool+args from being called twice in one turn.

    Uses a frozen JSON representation of args for comparison, so
    key ordering does not matter.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @staticmethod
    def _key(tool_name: str, args: dict[str, Any]) -> str:
        # Sort keys so {"a": 1, "b": 2} == {"b": 2, "a": 1}
        return f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

    def record(self, tool_name: str, args: dict[str, Any]) -> None:
        self._seen.add(self._key(tool_name, args))

    def is_repeat(self, tool_name: str, args: dict[str, Any]) -> bool:
        return self._key(tool_name, args) in self._seen

    def reset(self) -> None:
        self._seen.clear()
