"""DeepEval trace-based evaluation of the real Chat v2 ASGI/SSE path."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe

ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = ROOT / ".scratch" / "hiking-ai-evals"
AI_SERVICE = ROOT / "ai-service"
for path in (EVALS_ROOT, AI_SERVICE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hiking_evals.fixtures import FixtureEnv
from hiking_evals.models import ActionSpec
from hiking_evals.target import ASGITarget

from metrics import TRACE_METRICS


DATASET = EvaluationDataset()
DATASET.add_goldens_from_json_file(file_path=str(Path(__file__).with_name(".dataset.json")))


@observe(type="agent", metrics=TRACE_METRICS)
def run_traced_chat(prompt: str) -> str:
    """Execute one real Chat v2 request inside a DeepEval trace."""

    async def _run() -> str:
        with tempfile.TemporaryDirectory(prefix="deepeval-chat-") as temp_dir:
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
                            "request_id": f"deepeval-{uuid.uuid4().hex}",
                            "conversation_id": f"deepeval-{uuid.uuid4().hex[:12]}",
                            "profile_id": "chat",
                        },
                        sse=True,
                    )
                )
                return "\n".join(
                    str((event.get("payload") or {}).get("content") or "")
                    for event in record.events
                    if event.get("type") == "text"
                ).strip() or record.body_text or record.error
            finally:
                await target.close()
                environment.close()

    return asyncio.run(_run())


@pytest.mark.parametrize("golden", DATASET.goldens, ids=lambda golden: golden.name or golden.input[:24])
def test_hiking_agent_trace(golden: Golden) -> None:
    run_traced_chat(golden.input)
    assert_test(golden=golden, metrics=TRACE_METRICS)
