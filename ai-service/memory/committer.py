"""Post-run memory commit rules for hiking agent interactions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryCandidate:
    type: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8

    def to_item(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
        }


class MemoryCommitter:
    """Extract stable user/trip memories after the final response exists."""

    def extract_candidates(
        self,
        history: list[dict],
        query: str,
        final_response: str = "",
        task_state: dict | None = None,
    ) -> list[dict]:
        text = "\n".join(
            str(msg.get("content", ""))
            for msg in (history or [])[-6:]
            if msg.get("role") == "user"
        )
        if query:
            text = f"{text}\n{query}"

        candidates: list[MemoryCandidate] = []
        candidates.extend(self._profile_candidates(text))
        candidates.extend(self._trip_candidates(task_state or {}))

        deduped: dict[tuple[str, str, str], MemoryCandidate] = {}
        for item in candidates:
            key = (item.type, item.predicate, item.object)
            current = deduped.get(key)
            if current is None or item.confidence > current.confidence:
                deduped[key] = item
        return [candidate.to_item() for candidate in deduped.values()]

    def _profile_candidates(self, text: str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []

        origin = re.search(r"我(?:一般|通常|平时)?从([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})出发", text)
        if origin:
            candidates.append(MemoryCandidate("ProfileMemory", "用户", "常驻出发地", origin.group(1), 0.9))

        if "膝盖不好" in text or "膝盖不太好" in text or "膝盖疼" in text:
            candidates.append(MemoryCandidate("ProfileMemory", "用户", "身体限制", "膝盖不适", 0.9))

        if "没有露营经验" in text or "没露营经验" in text:
            candidates.append(MemoryCandidate("ProfileMemory", "用户", "露营经验", "无", 0.9))

        if any(word in text for word in ("新手", "第一次徒步", "没徒步经验", "没有徒步经验")):
            candidates.append(MemoryCandidate("ProfileMemory", "用户", "徒步经验", "新手", 0.8))

        if "轻装" in text:
            candidates.append(MemoryCandidate("ProfileMemory", "用户", "装备偏好", "轻装", 0.75))

        return candidates

    def _trip_candidates(self, task_state: dict) -> list[MemoryCandidate]:
        slots = task_state.get("slots") or {}
        if hasattr(slots, "to_dict"):
            slots = slots.to_dict()
        if not isinstance(slots, dict):
            return []

        stable = {
            key: value
            for key, value in slots.items()
            if key in {"destination", "date", "days", "origin", "experience", "camping", "group_size", "gear_level"}
            and value not in (None, "", [])
        }
        if not stable:
            return []

        return [
            MemoryCandidate(
                "TripMemory",
                "当前行程",
                "结构化状态",
                json.dumps(stable, ensure_ascii=False, sort_keys=True),
                0.8,
            )
        ]
