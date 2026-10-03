"""Repeatable provider parity regression. No network, credentials or wording assertions."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from route_compare.data import collect_candidates
from route_compare.fixtures import fixture_candidates
from route_compare.models import RouteQuery
from route_compare.providers import ProviderConfig, select_provider
from route_compare.service import compare_routes


class FakeLlm:
    async def ainvoke(self, messages):
        request = json.loads(messages[-1].content)
        return SimpleNamespace(content=json.dumps({"ranking": request["candidate_ids"], "confidence": 0.8}))


class FakeExternal:
    async def evaluate(self, body):
        return {"model": "jev-fixture", "answers": {
            cid: {"type": "score", "score": 4.0 - i, "confidence": 0.85}
            for i, cid in enumerate(body["questions"])
        }}


async def evaluate() -> dict:
    checks = []
    for name in ("llm", "external", "deterministic"):
        config = ProviderConfig(provider=name, data_mode="offline", external_key="fixture")
        provider = select_provider(config, llm=FakeLlm(), client=FakeExternal())
        for scenario, expected in [("complete", "recommended"), ("partial", "insufficient_evidence"),
                                   ("empty", "insufficient_evidence"), ("severe_weather", "no_go"),
                                   ("caution", "confirmation_required")]:
            result = await compare_routes(fixture_candidates(scenario), provider)
            passed = result.status == expected and result.report_allowed == (expected == "recommended")
            checks.append({"id": f"{name}-{scenario}", "passed": passed,
                           "status": result.status, "report_allowed": result.report_allowed,
                           "confidence_kind": result.decision.confidence.kind})
    for name in ("llm", "external"):
        result = await compare_routes(fixture_candidates(), select_provider(ProviderConfig(provider=name)))
        checks.append({"id": f"{name}-unavailable", "passed": result.status == "provider_unavailable"
                       and result.recommended_id is None, "status": result.status})
    class MissingTools:
        async def call(self, *args):
            return {"unavailable": "mcp_unavailable"}
    data = await collect_candidates([RouteQuery(name="未知路线", city="城市", origin="起点", destination="终点")], MissingTools())
    result = await compare_routes(data, select_provider(ProviderConfig()))
    checks.append({"id": "mcp-unavailable", "passed": result.status == "insufficient_evidence"
                   and not result.report_allowed, "status": result.status})
    return {"suite": "route_compare_provider_parity_v1", "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": sum(check["passed"] for check in checks), "total": len(checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
