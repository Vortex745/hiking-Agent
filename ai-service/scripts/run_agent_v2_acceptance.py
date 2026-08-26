"""Evaluate Chat v2 hard gates from a frozen corpus and real staging evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit


REQUIRED_CATEGORIES = frozenset(
    {
        "chat",
        "route_plan",
        "location_weather",
        "mcp_degradation",
        "artifact",
        "approval",
        "effect",
        "cowork",
        "memory",
        "cancellation",
        "rollback",
    }
)
FROZEN_CORPUS_SHA256 = "211bde7435509553e8c7d36b837ee96c61600452c22cef01815e51b01e6f9987"
REQUIRED_DEPENDENCIES = ("model", "e2b", "mcp", "amap", "tavily", "pexels")
REQUIRED_OPERATION_CHECKS = (
    "dependency_health",
    "restart_recovery",
    "duplicate_side_effects",
    "backup_restore",
    "capacity_protection",
    "sqlite_migration_metrics",
    "force_v1_rollback",
)
CATEGORY_EXPECTATIONS = {
    "chat": ("chat", "done", ("text",)),
    "route_plan": ("route_plan", "done", ("text",)),
    "location_weather": ("route_plan", "done", ("text",)),
    "mcp_degradation": ("route_plan", "done", ("text",)),
    "artifact": ("artifact", "run.paused", ("approval.required",)),
    "approval": ("artifact", "run.paused", ("approval.required",)),
    "effect": ("artifact", "run.paused", ("approval.required",)),
    "cowork": ("cowork", "run.paused", ("approval.required",)),
    "memory": ("chat", "done", ("text",)),
    "cancellation": ("chat", "done", ("text",)),
    "rollback": ("chat", "done", ("text",)),
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    category: str
    critical: bool
    computer: bool
    prompt: str
    requires: tuple[str, ...] = ()
    profile_id: str = "chat"
    expected_v2_terminal: str = "done"
    required_v2_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: Literal["passed", "failed", "skipped"]
    evidence: str
    reason: str = ""


@dataclass(frozen=True)
class AcceptanceReport:
    timestamp: str
    environment: dict[str, Any]
    gates: list[GateResult]
    metrics: dict[str, Any]
    overall_pass: bool
    executed_commands: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    residual_risks: list[str] = field(default_factory=list)


def load_corpus(path: Path) -> list[Scenario]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_CORPUS_SHA256:
        raise ValueError(
            f"frozen corpus digest mismatch: expected {FROZEN_CORPUS_SHA256}, got {digest}"
        )
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            scenario_id = str(item.get("id", "")).strip()
            category = str(item.get("category", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            if not scenario_id or not category or not prompt:
                raise ValueError(f"invalid corpus entry at line {line_number}")
            if scenario_id in seen:
                raise ValueError(f"duplicate scenario id: {scenario_id}")
            if category not in CATEGORY_EXPECTATIONS:
                raise ValueError(f"unknown scenario category at line {line_number}: {category}")
            profile_id, expected_terminal, required_events = CATEGORY_EXPECTATIONS[category]
            critical = item.get("critical", False)
            computer = item.get("computer", False)
            if not isinstance(critical, bool) or not isinstance(computer, bool):
                raise ValueError(f"scenario flags must be booleans at line {line_number}")
            seen.add(scenario_id)
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    category=category,
                    critical=critical,
                    computer=computer,
                    prompt=prompt,
                    requires=tuple(str(value) for value in item.get("requires", [])),
                    profile_id=profile_id,
                    expected_v2_terminal=expected_terminal,
                    required_v2_events=required_events,
                )
            )
    return scenarios


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _finite_number(value: Any, field_name: str, *, allow_zero: bool) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{field_name} must be a finite positive number")
    return number


def _safe_reference(value: Any) -> str:
    reference = str(value or "")
    parsed = urlsplit(reference)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return Path(reference).name


def _gate(
    gate_id: str,
    passed: bool,
    evidence: str,
    *,
    skipped: bool = False,
    reason: str = "",
) -> GateResult:
    status = "skipped" if skipped else ("passed" if passed else "failed")
    return GateResult(gate_id=gate_id, status=status, evidence=evidence, reason=reason)


def _skipped_measurement_gates(reason: str) -> list[GateResult]:
    return [
        _gate(gate_id, False, "", skipped=True, reason=reason)
        for gate_id in ("safety", "quality", "performance", "cost", "operations")
    ]


def _failed_measurement_gates(reason: str) -> list[GateResult]:
    return [
        _gate(gate_id, False, "", reason=reason)
        for gate_id in ("safety", "quality", "performance", "cost", "operations")
    ]


def _collect_evidence_paths(evidence: dict[str, Any], evidence_path: Path) -> list[str]:
    paths = {_safe_reference(evidence_path)}
    for section in ("dependencies", "operations"):
        for item in evidence.get(section, {}).values():
            if isinstance(item, dict) and item.get("evidence"):
                paths.add(_safe_reference(item["evidence"]))
    for item in evidence.get("scenarios", []):
        if isinstance(item, dict) and item.get("evidence_path"):
            paths.add(_safe_reference(item["evidence_path"]))
    return sorted(paths)


def _consume_sse(response: Any) -> tuple[list[dict[str, Any]], float | None]:
    events: list[dict[str, Any]] = []
    first_text_ms: float | None = None
    started = time.perf_counter()
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data = "\n".join(
                line[5:].lstrip()
                for line in frame.splitlines()
                if line.startswith("data:")
            )
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == "text" and first_text_ms is None:
                first_text_ms = (time.perf_counter() - started) * 1000
    return events, first_text_ms


def _run_live_protocol(
    client: Any,
    *,
    scenario: Scenario,
    protocol: Literal["v1", "v2"],
) -> dict[str, Any]:
    request_id = f"acceptance-{scenario.scenario_id}-{protocol}"
    if protocol == "v1":
        path = "/api/v1/chat/sse"
        payload = {
            "message": scenario.prompt,
            "chat_id": request_id,
        }
        expected_terminal = "done"
        required_events: tuple[str, ...] = ()
    else:
        path = "/api/v2/chat/sse"
        payload = {
            "message": scenario.prompt,
            "request_id": request_id,
            "conversation_id": f"acceptance-{scenario.scenario_id}",
            "profile_id": scenario.profile_id,
        }
        expected_terminal = scenario.expected_v2_terminal
        required_events = scenario.required_v2_events

    started = time.perf_counter()
    try:
        with client.stream("POST", path, json=payload) as response:
            status_code = response.status_code
            events, first_text_ms = _consume_sse(response)
    except Exception as exc:
        return {
            "success": False,
            "status_code": 0,
            "first_text_ms": None,
            "total_ms": (time.perf_counter() - started) * 1000,
            "terminal": "",
            "event_types": [],
            "error": type(exc).__name__,
        }

    event_types = [str(event.get("type", "")) for event in events]
    terminal_types = [
        event_type
        for event_type in event_types
        if event_type in {"done", "error", "cancelled", "run.paused"}
    ]
    terminal = terminal_types[-1] if terminal_types else ""
    success = (
        status_code == 200
        and len(terminal_types) == 1
        and terminal == expected_terminal
        and not (protocol == "v1" and "error" in event_types)
        and all(event_type in event_types for event_type in required_events)
    )
    return {
        "success": success,
        "status_code": status_code,
        "first_text_ms": first_text_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
        "terminal": terminal,
        "event_types": sorted(set(event_types)),
        "error": "",
    }


def collect_live_staging_evidence(
    *,
    base_url: str,
    corpus_path: Path,
    operations_evidence_path: Path | None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Execute the frozen corpus against live v1/v2 Gateway endpoints."""
    if not base_url.lower().startswith("https://"):
        raise ValueError("staging base URL must use HTTPS")
    import httpx

    scenarios = load_corpus(corpus_path)
    operations_evidence: dict[str, Any] = {}
    if operations_evidence_path and operations_evidence_path.exists():
        operations_evidence = json.loads(
            operations_evidence_path.read_text(encoding="utf-8")
        )

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
        follow_redirects=False,
    ) as client:
        health_response = client.get("/health")
        tools_response = client.get("/api/v1/tools/health")
        health = health_response.json() if health_response.status_code == 200 else {}
        tools_health = tools_response.json() if tools_response.status_code == 200 else {}

        observations = []
        for scenario in scenarios:
            v1 = _run_live_protocol(client, scenario=scenario, protocol="v1")
            v2 = _run_live_protocol(client, scenario=scenario, protocol="v2")
            costs = operations_evidence.get("scenario_costs", {}).get(
                scenario.scenario_id, {}
            )
            observations.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "v1_success": v1["success"],
                    "v2_success": v2["success"],
                    "v1_first_text_ms": v1["first_text_ms"],
                    "v2_first_text_ms": v2["first_text_ms"],
                    "v1_total_ms": v1["total_ms"],
                    "v2_total_ms": v2["total_ms"],
                    "sandbox_start_ms": costs.get("sandbox_start_ms"),
                    "model_cost": costs.get("model_cost"),
                    "e2b_cost": costs.get("e2b_cost"),
                    "evidence_path": f"{base_url.rstrip('/')}/acceptance/{scenario.scenario_id}",
                    "v1_terminal": v1["terminal"],
                    "v2_terminal": v2["terminal"],
                    "v2_event_types": v2["event_types"],
                }
            )

    v2_health = health.get("agent_v2", {})
    v2_dependencies = v2_health.get("dependencies", {})
    live_dependency_checks = operations_evidence.get("dependencies", {})
    dependencies: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_DEPENDENCIES:
        if name in {"model", "e2b", "mcp"}:
            raw_status = v2_dependencies.get(name, {}).get("status", "unavailable")
            passed = raw_status == "ready"
            dependencies[name] = {
                "status": "passed" if passed else "skipped",
                "reason": "" if passed else f"live health status is {raw_status}",
                "evidence": f"{base_url.rstrip('/')}/health",
            }
        else:
            item = live_dependency_checks.get(name, {})
            dependencies[name] = {
                "status": item.get("status", "skipped"),
                "reason": item.get("reason", f"live {name} check missing"),
                "evidence": item.get("evidence", ""),
            }

    return {
        "provenance": {
            "mode": "live_staging_http",
            "base_url": base_url.rstrip("/"),
            "corpus_sha256": FROZEN_CORPUS_SHA256,
        },
        "environment": {
            "name": "staging",
            "versions": {
                "uniharness": v2_health.get("uniharness_pin", "unknown"),
                "tools": tools_health.get("tools_total", "unknown"),
            },
        },
        "dependencies": dependencies,
        "approved_cost_baseline_per_success": operations_evidence.get(
            "approved_cost_baseline_per_success"
        ),
        "operations": operations_evidence.get("operations", {}),
        "scenarios": observations,
    }


def evaluate_acceptance(
    *,
    corpus_path: Path,
    evidence_path: Path,
    executed_commands: list[str] | None = None,
) -> AcceptanceReport:
    scenarios = load_corpus(corpus_path)
    categories = {scenario.category for scenario in scenarios}
    corpus_ok = len(scenarios) >= 50 and REQUIRED_CATEGORIES <= categories
    gates = [
        _gate(
            "corpus",
            corpus_ok,
            f"scenarios={len(scenarios)} categories={sorted(categories)}",
            reason="frozen corpus must contain at least 50 scenarios and every required category",
        )
    ]
    commands = list(executed_commands or [])
    metrics: dict[str, Any] = {"scenario_count": len(scenarios)}
    residual_risks: list[str] = []

    if not evidence_path.exists():
        reason = "real staging evidence was not supplied; model/E2B/MCP checks are unverified"
        gates.append(_gate("dependencies", False, "", skipped=True, reason=reason))
        gates.extend(_skipped_measurement_gates(reason))
        gates.append(
            _gate(
                "reporting",
                False,
                f"commands={len(commands)} versions={{}}",
                skipped=True,
                reason="dependency versions require real staging evidence",
            )
        )
        residual_risks.append(reason)
        return AcceptanceReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            environment={},
            gates=gates,
            metrics=metrics,
            overall_pass=False,
            executed_commands=commands,
            evidence_paths=[corpus_path.name],
            residual_risks=residual_risks,
        )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    provenance = evidence.get("provenance", {})
    raw_environment = evidence.get("environment", {})
    raw_versions = raw_environment.get("versions", {})
    environment = {
        "name": str(raw_environment.get("name", ""))[:64],
        "versions": {
            str(key)[:64]: str(value)[:128]
            for key, value in raw_versions.items()
        } if isinstance(raw_versions, dict) else {},
    }
    dependencies = evidence.get("dependencies", {})
    dependency_states = {
        name: dependencies.get(name, {}).get("status", "skipped")
        for name in REQUIRED_DEPENDENCIES
    }
    dependency_reasons = [
        str(dependencies.get(name, {}).get("reason", f"{name} evidence missing"))
        for name, status in dependency_states.items()
        if status != "passed"
    ]
    dependency_statuses_valid = all(
        status in {"passed", "failed", "skipped"}
        for status in dependency_states.values()
    )
    real_staging = (
        environment.get("name") == "staging"
        and provenance.get("mode") == "live_staging_http"
        and provenance.get("corpus_sha256") == FROZEN_CORPUS_SHA256
    )
    dependency_failed = (
        not dependency_statuses_valid
        or any(status == "failed" for status in dependency_states.values())
    )
    dependency_ok = real_staging and all(
        status == "passed" for status in dependency_states.values()
    )
    dependency_reason = "; ".join(dependency_reasons)
    if not real_staging:
        dependency_reason = "; ".join(
            part
            for part in (
                dependency_reason,
                "evidence was not collected by the live staging HTTP runner",
            )
            if part
        )
    gates.append(
        _gate(
            "dependencies",
            dependency_ok,
            f"environment={environment.get('name', '')} states={dependency_states}",
            skipped=not dependency_ok and not dependency_failed,
            reason=dependency_reason,
        )
    )

    raw_observations = evidence.get("scenarios", [])
    observation_ids = [
        str(item.get("scenario_id"))
        for item in raw_observations
        if isinstance(item, dict) and item.get("scenario_id")
    ]
    duplicate_ids = sorted(
        {scenario_id for scenario_id in observation_ids if observation_ids.count(scenario_id) > 1}
    )
    observations = {
        str(item.get("scenario_id")): item
        for item in raw_observations
        if isinstance(item, dict) and item.get("scenario_id")
    }
    expected_ids = {scenario.scenario_id for scenario in scenarios}
    complete = (
        isinstance(raw_observations, list)
        and not duplicate_ids
        and set(observations) == expected_ids
        and len(raw_observations) == len(expected_ids)
    )
    observation_validation_error = ""
    v1_values: dict[str, bool] = {}
    v2_values: dict[str, bool] = {}
    if complete:
        try:
            v1_values = {
                scenario_id: _strict_bool(item.get("v1_success"), "v1_success")
                for scenario_id, item in observations.items()
            }
            v2_values = {
                scenario_id: _strict_bool(item.get("v2_success"), "v2_success")
                for scenario_id, item in observations.items()
            }
        except ValueError as exc:
            complete = False
            observation_validation_error = str(exc)
    if not complete:
        missing = sorted(expected_ids - set(observations))
        reason = (
            f"staging observations invalid; missing={missing[:10]} "
            f"duplicates={duplicate_ids[:10]} error={observation_validation_error}"
        )
        gates.extend(_failed_measurement_gates(reason))
        residual_risks.append(reason)
    else:
        critical = [scenario for scenario in scenarios if scenario.critical]
        critical_successes = sum(
            v2_values[scenario.scenario_id]
            for scenario in critical
        )
        critical_rate = critical_successes / len(critical) if critical else 0.0
        metrics["critical_safety_rate"] = critical_rate
        gates.append(
            _gate(
                "safety",
                bool(critical) and critical_rate == 1.0,
                f"critical_success={critical_successes}/{len(critical)} rate={critical_rate:.4f}",
                reason="critical safety scenarios require 100% success",
            )
        )

        total = len(scenarios)
        v1_successes = sum(v1_values.values())
        v2_successes = sum(v2_values.values())
        v1_rate = v1_successes / total
        v2_rate = v2_successes / total
        metrics.update(v1_success_rate=v1_rate, v2_success_rate=v2_rate)
        gates.append(
            _gate(
                "quality",
                v2_rate >= 0.90 and v2_rate >= v1_rate - 0.03,
                f"v2={v2_rate:.4f} v1={v1_rate:.4f} floor=0.9000 allowed_delta=0.0300",
                reason="v2 success must be at least 90% and within 3 points of v1",
            )
        )

        non_computer = [
            observations[scenario.scenario_id]
            for scenario in scenarios
            if not scenario.computer
        ]
        computer = [
            observations[scenario.scenario_id]
            for scenario in scenarios
            if scenario.computer
        ]
        try:
            v1_first_p95 = _percentile(
                [
                    _finite_number(item["v1_first_text_ms"], "v1_first_text_ms", allow_zero=False)
                    for item in non_computer
                ],
                0.95,
            )
            v2_first_p95 = _percentile(
                [
                    _finite_number(item["v2_first_text_ms"], "v2_first_text_ms", allow_zero=False)
                    for item in non_computer
                ],
                0.95,
            )
            v1_total_p95 = _percentile(
                [
                    _finite_number(item["v1_total_ms"], "v1_total_ms", allow_zero=False)
                    for item in non_computer
                ],
                0.95,
            )
            v2_total_p95 = _percentile(
                [
                    _finite_number(item["v2_total_ms"], "v2_total_ms", allow_zero=False)
                    for item in non_computer
                ],
                0.95,
            )
            sandbox_values = [
                _finite_number(item["sandbox_start_ms"], "sandbox_start_ms", allow_zero=False)
                for item in computer
            ]
            first_ratio = v2_first_p95 / v1_first_p95
            total_ratio = v2_total_p95 / v1_total_p95
            sandbox_p50 = _percentile(sandbox_values, 0.50)
            sandbox_p95 = _percentile(sandbox_values, 0.95)
            performance_ok = first_ratio <= 1.25 and total_ratio <= 1.30
            metrics.update(
                non_computer_first_text_p95_ratio=first_ratio,
                non_computer_total_p95_ratio=total_ratio,
                sandbox_start_p50_ms=sandbox_p50,
                sandbox_start_p95_ms=sandbox_p95,
            )
            performance_evidence = (
                f"first_text_ratio={first_ratio:.4f} total_ratio={total_ratio:.4f} "
                f"sandbox_p50_ms={sandbox_p50:.2f} sandbox_p95_ms={sandbox_p95:.2f}"
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            performance_ok = False
            performance_evidence = "invalid or non-finite latency evidence"
        gates.append(
            _gate(
                "performance",
                performance_ok,
                performance_evidence,
                reason="p95 first text <=1.25x v1, total <=1.30x v1, sandbox p50/p95 required",
            )
        )

        baseline = evidence.get("approved_cost_baseline_per_success")
        try:
            total_cost = sum(
                _finite_number(item.get("model_cost"), "model_cost", allow_zero=True)
                + _finite_number(item.get("e2b_cost"), "e2b_cost", allow_zero=True)
                for item in observations.values()
            )
            cost_per_success = total_cost / v2_successes
            baseline_value = _finite_number(
                baseline, "approved_cost_baseline_per_success", allow_zero=False
            )
            cost_ok = baseline_value > 0 and cost_per_success <= baseline_value * 1.20
            metrics["cost_per_success"] = cost_per_success
            cost_evidence = (
                f"actual={cost_per_success:.6f} baseline={baseline_value:.6f} "
                f"limit={baseline_value * 1.20:.6f}"
            )
        except (TypeError, ValueError, ZeroDivisionError):
            cost_ok = False
            cost_evidence = "invalid or non-finite cost evidence"
        gates.append(
            _gate(
                "cost",
                cost_ok,
                cost_evidence,
                reason="model plus E2B cost per successful task must be <=1.20x baseline",
            )
        )

        operations = evidence.get("operations", {})
        operation_states = {
            name: operations.get(name, {}).get("status", "skipped")
            for name in REQUIRED_OPERATION_CHECKS
        }
        operation_failed = any(status == "failed" for status in operation_states.values())
        operation_ok = all(status == "passed" for status in operation_states.values())
        operation_reason = "; ".join(
            str(operations.get(name, {}).get("reason", f"{name} evidence missing"))
            for name, status in operation_states.items()
            if status != "passed"
        )
        gates.append(
            _gate(
                "operations",
                operation_ok,
                f"states={operation_states}",
                skipped=not operation_ok and not operation_failed,
                reason=operation_reason,
            )
        )

    reporting_ok = (
        bool(commands)
        and bool(environment.get("versions"))
        and provenance.get("mode") == "live_staging_http"
        and provenance.get("corpus_sha256") == FROZEN_CORPUS_SHA256
    )
    gates.append(
        _gate(
            "reporting",
            reporting_ok,
            f"commands={len(commands)} versions={environment.get('versions', {})}",
            reason="report must record its command and dependency versions",
        )
    )

    for gate in gates:
        if gate.status != "passed" and gate.reason:
            residual_risks.append(f"{gate.gate_id}: {gate.reason}")
    return AcceptanceReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        gates=gates,
        metrics=metrics,
        overall_pass=all(gate.status == "passed" for gate in gates),
        executed_commands=commands,
        evidence_paths=[str(corpus_path), *_collect_evidence_paths(evidence, evidence_path)],
        residual_risks=residual_risks,
    )


def main(argv: list[str] | None = None) -> int:
    ai_service_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate Chat v2 staging hard gates")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ai_service_dir / "tests" / "fixtures" / "agent_v2_cases.jsonl",
    )
    parser.add_argument(
        "--staging-base-url",
        help="HTTPS Gateway origin used to execute the frozen v1/v2 corpus",
    )
    parser.add_argument(
        "--operations-evidence",
        type=Path,
        help="Machine-readable cost, live dependency, recovery, backup, capacity, and rollback evidence",
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=ai_service_dir / "data" / "agent-v2-staging-evidence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ai_service_dir / "data" / "agent-v2-acceptance-report.json",
    )
    args = parser.parse_args(argv)
    if args.staging_base_url:
        evidence = collect_live_staging_evidence(
            base_url=args.staging_base_url,
            corpus_path=args.corpus,
            operations_evidence_path=args.operations_evidence,
        )
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        evidence_path = args.evidence_output
    else:
        evidence_path = ai_service_dir / "data" / "missing-staging-evidence.json"
    command = subprocess.list2cmdline(
        [sys.executable, "-m", "scripts.run_agent_v2_acceptance", *(argv or sys.argv[1:])]
    )
    report = evaluate_acceptance(
        corpus_path=args.corpus,
        evidence_path=evidence_path,
        executed_commands=[command],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"Chat v2 acceptance report: {args.output}")
    print(f"overall_pass: {report.overall_pass}")
    for gate in report.gates:
        print(f"  {gate.gate_id}: {gate.status}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
