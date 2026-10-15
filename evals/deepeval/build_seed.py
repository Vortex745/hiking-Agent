"""Convert the committed Promptfoo cases into DeepEval golden seeds."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "evals" / "promptfoo" / "promptfooconfig.yaml"
OUTPUT = Path(__file__).resolve().parent / "promptfoo-seed.json"


def main() -> None:
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    seeds = []
    for case in config.get("tests", []):
        variables = case.get("vars", {})
        prompt = str(variables.get("prompt", "")).strip()
        if not prompt:
            continue
        assertions = case.get("assert", [])
        requirements = []
        for assertion in assertions:
            kind = assertion.get("type", "")
            value = assertion.get("value", "")
            if kind in {"regex", "contains"}:
                requirements.append(f"必须满足：{value}")
            elif kind in {"not-regex", "not-contains"}:
                requirements.append(f"不得出现：{value}")
        seeds.append(
            {
                "input": prompt,
                "expected_output": "；".join(requirements),
                "name": str(variables.get("case_id") or case.get("description") or "promptfoo"),
                "additional_metadata": {"source": "promptfoo", "description": case.get("description", "")},
            }
        )
    OUTPUT.write_text(json.dumps(seeds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(seeds)} seeds to {OUTPUT}")


if __name__ == "__main__":
    main()
