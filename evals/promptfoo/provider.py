"""Promptfoo provider for the real Chat v2 ASGI/SSE path."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = REPO_ROOT / ".scratch" / "hiking-ai-evals"
AI_SERVICE = REPO_ROOT / "ai-service"
for candidate in (str(EVALS_ROOT), str(AI_SERVICE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hiking_evals.fixtures import FixtureEnv
from hiking_evals.models import ActionSpec
from hiking_evals.target import ASGITarget


async def evaluate(prompt: str, variables: dict[str, Any]) -> dict[str, Any]:
    case_id = str(variables.get("case_id") or "promptfoo")
    with tempfile.TemporaryDirectory(prefix=f"promptfoo-{case_id}-") as temp_dir:
        environment = FixtureEnv({}, Path(temp_dir), live=True)
        target = ASGITarget(environment.build_app())
        try:
            record = await target.execute(
                ActionSpec(
                    name="chat",
                    method="POST",
                    path="/api/v2/chat/sse",
                    json={
                        "message": prompt,
                        "request_id": f"pf-{uuid.uuid4().hex}",
                        "conversation_id": f"pf-{case_id}-{uuid.uuid4().hex[:8]}",
                        "profile_id": "chat",
                    },
                    sse=True,
                )
            )
            event_types = [str(event.get("type")) for event in record.events]
            text = "\n".join(
                str((event.get("payload") or {}).get("content") or "")
                for event in record.events
                if event.get("type") == "text"
            ).strip()
            terminal = record.events[-1] if record.events else {}
            return {
                "output": text or record.body_text or record.error,
                "metadata": {
                    "http_status": record.status,
                    "event_types": event_types,
                    "terminal_type": terminal.get("type"),
                    "terminal_code": (terminal.get("payload") or {}).get("code"),
                    "transport_error": record.error or None,
                },
            }
        finally:
            await target.close()
            environment.close()


def main() -> None:
    payload = json.load(sys.stdin)
    result = asyncio.run(evaluate(str(payload["prompt"]), dict(payload.get("vars") or {})))
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
