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

## Offline default vs live opt-in

The default offline entry is `evals/test_evaluation_contract.py` (from the
repo root; no external model or judge calls):

```powershell
E:/miniconda3/python.exe -B -m pytest -q evals/test_evaluation_contract.py
```

This suite is live and therefore opt-in. Run it from the repository root
with a working model provider AND a Jev credential:

```powershell
$env:HIKING_EVAL_LIVE = "1"
$env:TYPESAFE_API_KEY = "..."
E:/miniconda3/Scripts/deepeval.exe test run evals/deepeval/test_hiking_agent.py --identifier "hiking-agent-jev-round-1"
```

Without `HIKING_EVAL_LIVE=1` the tests are skipped (they are never reported
as passed). The suite does not send results to Confident AI unless the user
explicitly configures that workflow.

## Historical records (kept, annotated as limited)

`history/round0-no-metric-pass.json` preserves the 2026-10-14 round-0
result verbatim in shape: **4 passed / 0 failed with `metricsData=[]` and
empty `actualOutput` for every golden — a false green**. It is annotated
with its limitation, classified `not_evaluated`, and excluded from any
semantic-pass count; the regression in `evals/test_evaluation_contract.py`
proves the completed-but-no-metric run is no longer counted as a semantic
pass.
