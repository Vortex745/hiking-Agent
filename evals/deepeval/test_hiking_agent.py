"""DeepEval trace-based evaluation of the real Chat v2 ASGI/SSE path.

Defects pinned by the eval-contract suite (evals/test_evaluation_contract.py):

* the previous runner SWALLOWED SSE ``error`` terminals / empty answers and
  returned ``""`` — a failed target run looked like an empty-but-valid
  answer (false green: ``passed=4`` with ``metricsData=[]``);
* ``deepeval test run --ignore-errors`` turned judge failures into passes.

Now the shared ``extract_chat_outcome`` / ``assert_gradable`` contract is
enforced: only a normal ``done`` terminal with a non-empty answer proceeds
to scoring. Error terminals, empty answers, and transport failures RAISE
(no false green; judge/metric failures surface as real failures).

Live opt-in: this suite drives the REAL Chat v2 app and the Jev judge, so
it runs only when explicitly enabled. Default offline entry is
``evals/test_evaluation_contract.py`` (no external model/judge calls):

    $env:HIKING_EVAL_LIVE = "1"; $env:TYPESAFE_API_KEY = "..."
    E:/miniconda3/Scripts/deepeval.exe test run evals/deepeval/test_hiking_agent.py --identifier "hiking-agent-jev-round"

Do NOT pass ``--ignore-errors``: judge/service failures must stay visible.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe

ROOT = Path(__file__).resolve().parents[2]
EVALS_ROOT = ROOT / "evals"
AI_SERVICE = ROOT / "ai-service"
for path in (EVALS_ROOT, AI_SERVICE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hiking_evals.fixtures import FixtureEnv
from hiking_evals.models import ActionSpec
from hiking_evals.target import ASGITarget, extract_chat_outcome
from hiking_evals.status import assert_gradable

from metrics import build_trace_metrics

pytestmark = pytest.mark.skipif(
    os.environ.get("HIKING_EVAL_LIVE") != "1",
    reason=(
        "live DeepEval run disabled: set HIKING_EVAL_LIVE=1 together with a "
        "working model provider and judge credentials (offline contract "
        "entry: evals/test_evaluation_contract.py)"
    ),
)


DATASET = EvaluationDataset()
DATASET.add_goldens_from_json_file(file_path=str(Path(__file__).with_name(".dataset.json")))


@observe(type="agent")
def run_traced_chat(prompt: str) -> str:
    """Execute one real Chat v2 request inside a DeepEval trace.

    Raises :class:`hiking_evals.status.TargetNotGradable` when the run did
    not complete with a valid answer — the failure must never be swallowed
    into an empty string.
    """

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
                            "request_id": f"eval-deepeval-{uuid.uuid4().hex}",
                            "conversation_id": f"deepeval-{uuid.uuid4().hex[:12]}",
                            "profile_id": "chat",
                        },
                        sse=True,
                    )
                )
                return assert_gradable(extract_chat_outcome(record))
            finally:
                await target.close()
                environment.close()

    return asyncio.run(_run())


@pytest.mark.parametrize(
    "golden", DATASET.goldens, ids=lambda golden: golden.name or golden.input[:24]
)
def test_hiking_agent_trace(golden: Golden, trace_metrics) -> None:
    run_traced_chat(golden.input)
    assert_test(golden=golden, metrics=trace_metrics)


@pytest.fixture(scope="module")
def trace_metrics():
    return build_trace_metrics()
