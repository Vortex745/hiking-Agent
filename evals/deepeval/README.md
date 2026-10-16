# DeepEval suite

This suite evaluates the real Chat v2 ASGI/SSE path with DeepEval tracing. The
seed inputs are mechanically converted from `evals/promptfoo/promptfooconfig.yaml`;
the committed `.dataset.json` is the golden dataset used by pytest. Synthetic
multi-turn augmentation requires a reachable generation model endpoint.

Run from the repository root with a Jev credential:

```powershell
$env:TYPESAFE_API_KEY = "..."
E:/miniconda3/Scripts/deepeval.exe test run evals/deepeval/test_hiking_agent.py --identifier "hiking-agent-jev-round-1" --ignore-errors --skip-on-missing-params
```

The suite does not send results to Confident AI unless the user explicitly
configures that workflow.
