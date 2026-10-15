"""Promptfoo provider for the real Chat v2 ASGI/SSE path.

Uses the TRACKED stable evaluation package (``evals/hiking_evals``); there
is no scratch fallback. The provider is a live entry: it drives the real
Chat v2 app, so a runnable model provider is required (offline contract
checks live in ``evals/test_evaluation_contract.py``).

Honest output contract: the SSE outcome is extracted with the shared
``extract_chat_outcome``. Error terminals and empty answers yield an EMPTY
output (Promptfoo assertions fail — never a false green); the failure
detail stays in ``metadata``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = REPO_ROOT / "evals"
AI_SERVICE = REPO_ROOT / "ai-service"
for candidate in (str(EVALS_ROOT), str(AI_SERVICE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hiking_evals.fixtures import FixtureEnv
from hiking_evals.models import ActionSpec
from hiking_evals.target import ASGITarget, extract_chat_outcome


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
            outcome = extract_chat_outcome(record)
            return {
                "output": outcome.answer,
                "metadata": {
                    "http_status": outcome.http_status,
                    "event_types": [
                        str(event.get("type")) for event in record.events
                    ],
                    "terminal_type": outcome.terminal_type,
                    "terminal_code": outcome.terminal_code or None,
                    "gradable": outcome.gradable,
                    "run_id": outcome.run_id,
                    "transport_error": outcome.transport_error or None,
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
