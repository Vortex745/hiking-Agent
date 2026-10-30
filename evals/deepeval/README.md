# DeepEval suite

Live, trace-based evaluation of the real Chat v2 ASGI/SSE path. The seed
inputs are mechanically converted from `evals/promptfoo/promptfooconfig.yaml`
(`build_seed.py` → `promptfoo-seed.json` → committed `.dataset.json`).

The suite shares the TRACKED stable package `evals/hiking_evals` with the
Promptfoo provider — there is no scratch fallback.

## Status semantics (honest by construction)

- Only a normal `done` terminal with a non-empty answer is scored; error
  terminals, empty answers, and transport failures RAISE
  (`hiking_evals.status.TargetNotGradable`) instead of being swallowed
  into an "empty but valid" result.
- `JevJudge` returns a score ONLY for a numeric, finite value inside the
  judge's declared range (0..2, normalized to 0..1). Missing / null /
  wrong-type / non-finite / out-of-range judge answers raise
  `JevJudgeResponseError` — never a default zero. A legal 0.0 score is
  kept and participates in the threshold decision.
- Judge/service failure, a missing score, or an undecidable outcome is
  **not evaluated** — never a pass. Do NOT run with `--ignore-errors`;
  that flag previously turned judge failures into passes.
- DeepEval 4.2.9's final banner can exclude an errored case from its total
  (observed: pytest `1 failed, 39 passed` while the CLI printed
  `39 total tests / 100%`). The suite's `conftest.py` prints
  `LIVE EVAL STATUS: expected=... passed=... failed=... errors=...`; require
  `failed=0`, `errors=0`, `passed == dataset size`, and exit code 0. A missing
  case is **not evaluated**, not a pass.

## Dataset and metric calibration

`.dataset.json` contains the four Promptfoo-derived seeds plus 36 inputs
generated with `deepeval generate --method goldens` (24 Chinese, 12 English).
Generated inputs have no `expected_output`: both live metrics consume
`input`/`actual_output`, and the generated questions are intentionally
input-only. This is a first-pass regression/smoke set, not a benchmark.

`TaskCompletionMetric` uses a product-level task description so that a
grounded refusal of unsupported route facts, or a controlled checklist with
explicit assumptions, counts as completion. `JevEval` asks two narrow
TypeSafe questions instead of one compound question: grounding (no fabricated
route facts) and actionability (useful guidance or authoritative verification
channels). This keeps a grounded refusal with verification channels separate
from fabricated advice, and accepts the product's controlled gear-checklist
rendering when it marks unspecified season/experience and states its limits.
Metric names and the 0.7 thresholds are unchanged. The trace is evaluated once
at the golden level; do not attach the same metrics to `@observe`, or the span
and golden judgments can disagree on a stochastic judge.

## Offline default vs live opt-in

The default offline entry is `evals/test_evaluation_contract.py` (run from
the repo root; no external model or judge calls). This is the entry the
validation job (`.github/workflows/deploy.yml`) runs on every PR, with the
suite dependencies from `evals/deepeval/requirements.txt`:

```bash
python -m pytest -q evals/test_evaluation_contract.py
```

This suite below is live and therefore opt-in. Run it from the repository
root with the project venv (py3.12), which has both the app dependencies and
`typesafe-sdk`. Do not use the system miniconda py3.14: its `mcp` package
currently fails with a circular import (`StdioServerParameters`), so the live
app cannot start. The fixture loads `ai-service/.env` itself with `override=True`, so the app
model credentials do not depend on the shell cwd and the root compose `.env`
cannot shadow them.

```bash
HIKING_EVAL_LIVE=1 TYPESAFE_API_KEY="..." \
  DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=180 \
  ./ai-service/.venv/Scripts/deepeval.exe test run evals/deepeval/test_hiking_agent.py \
  --identifier "hiking-agent-jev-round-1"
```

The default per-attempt judge timeout is short enough to retry on slow
TypeSafe calls; the override above keeps a single slow attempt from looking
like a judge failure. Do not pass `--ignore-errors`.

Without `HIKING_EVAL_LIVE=1` the tests are skipped (they are never reported
as passed). The suite does not send results to Confident AI unless the user
explicitly configures that workflow.

Judge confidence is stochastic: a TaskCompletion score near the 0.7 threshold
can flip between runs. Repeat low-confidence failures before treating them as
product defects; use human review (or Confident AI annotations) when the judge
is undecided.

## Last recorded live run (2026-10-29)

40 goldens; offline contract 78/78; clean live run 38/40 passed, with no
missing cases (`LIVE EVAL STATUS: expected=40 passed=38 failed=2`). Both
failures were TaskCompletion-only borderline cases (0.698 and 0.553) with low
judge confidence; a targeted repeat flipped both directions (0.63 and 0.844),
so they are human-review candidates rather than stable defects. Safety metrics
passed on both.

## Historical records (kept, annotated as limited)

`history/round0-no-metric-pass.json` preserves the 2026-10-14 round-0
result verbatim in shape: **4 passed / 0 failed with `metricsData=[]` and
empty `actualOutput` for every golden — a false green**. It is annotated
with its limitation, classified `not_evaluated`, and excluded from any
semantic-pass count; the regression in `evals/test_evaluation_contract.py`
proves the completed-but-no-metric run is no longer counted as a semantic
pass.
