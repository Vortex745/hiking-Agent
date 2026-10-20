"""Ticket 06（harness-p0-p1）— 评估入口可重现、状态分类如实。

主边界（execution-map T09/T10）：
- 稳定评估包 ``evals/hiking_evals``（被跟踪、无 scratch 后备）可导入；
- Promptfoo provider 与 DeepEval suite 消费同一稳定包；
- fixture 驱动真实 Chat v2 ASGI/SSE + 编排接口，用临时 SQLite 与受控
  内核/工具/沙箱替身，区分资料充分、不足、受控工具回复与多轮场景；
- 执行契约结果与语义质量分开记录：passed / failed / not_evaluated；
- 缺失、null、非法类型、非有限或越界的 score 不默认成有效零分；
  合法零分参与判定；部分必需 metric 缺失不能汇总为语义通过；
- 历史无分通过记录保留并注明限制，且回归证明它不再计入语义通过；
- 默认离线入口不调用外部模型或 judge（在线质量另行显式启用）。

Run（仓库根目录）:
    E:/miniconda3/python.exe -B -m pytest -q evals/test_evaluation_contract.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

EVALS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALS_ROOT.parent
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

from hiking_evals.fixtures import FakeArtifactComputer, FixtureEnv
from hiking_evals.models import ActionSpec, ResponseRecord
from hiking_evals.status import (
    ContractViolation,
    MetricObservation,
    ReportCredentialError,
    TargetNotGradable,
    assert_gradable,
    build_report,
    classify_case,
    dataset_identity,
    legal_score,
    load_historical_entries,
)
from hiking_evals.target import ASGITarget, ChatTurnOutcome, extract_chat_outcome

# ---------------------------------------------------------------------------
# 固定场景数据（全部离线；内核/工具/沙箱均为确定性替身）
# ---------------------------------------------------------------------------

GROUNDED_TEXT = "武功山金顶海拔约1918米，反穿线全程约25公里（来源：wugongshan-guide.md）。"
FABRICATED_TEXT = "鳌太线冬季穿越全程约120公里，营地水源可靠。"
MODEL_FREEFORM_WITH_ROUTE = "【模型自由发挥】全程约12公里，可当日往返。"

_EVIDENCE_OK_AUDIT = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "code": None,
    "duration_ms": 3,
    "evidence": {
        "has_chunks": True,
        "sources": ["wugongshan-guide.md"],
        "chunks": [
            {
                "title": "武功山指南",
                "source": "wugongshan-guide.md",
                "preview": "金顶海拔约1918米。",
            }
        ],
    },
}

_EVIDENCE_EMPTY_AUDIT = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "code": None,
    "duration_ms": 3,
    "evidence": {"has_chunks": False, "sources": [], "chunks": []},
}

SUFFICIENT_FIXTURE: dict[str, Any] = {
    "kernel": {
        "mode": "script",
        "text": GROUNDED_TEXT,
        "audits": [_EVIDENCE_OK_AUDIT],
    }
}

INSUFFICIENT_FIXTURE: dict[str, Any] = {
    "kernel": {
        "mode": "script",
        "text": FABRICATED_TEXT,
        "audits": [_EVIDENCE_EMPTY_AUDIT],
    }
}

# 受控工具回复：真实 gear_checklist 适配器 + 确定性委托结果（无宿主命令）。
CONTROLLED_TOOL_FIXTURE: dict[str, Any] = {
    "kernel": {
        "mode": "agent_policy",
        "rules": [
            {
                "when_message_contains": ["装备"],
                "tool": "gear_checklist",
                "args": {"days": 2, "season": "夏季", "experience": "新手", "camping": False},
            }
        ],
        "template": MODEL_FREEFORM_WITH_ROUTE,
    },
    "delegate_overrides": [
        {
            "tool": "gear_checklist",
            "behavior": "incomplete",
            "data": {
                "ok": True,
                "days": 2,
                "season": "夏季",
                "experience": "新手",
                "camping": False,
                "checklist": {"服装": ["速干衣", "雨衣"], "装备": ["登山杖"]},
                "message": "按天气与经验调整。",
            },
        }
    ],
}

MULTI_TURN_FIXTURE: dict[str, Any] = {
    "kernel": {
        "mode": "script",
        "text": GROUNDED_TEXT,
        "audits": [_EVIDENCE_OK_AUDIT],
    }
}

ERROR_TERMINAL_FIXTURE: dict[str, Any] = {
    "kernel": {
        "mode": "script",
        "error": {"code": "model_unavailable", "message": "主模型不可用（评测注入）"},
    }
}


# ---------------------------------------------------------------------------
# 场景执行辅助：真实 ASGI/SSE（离线替身），绝不访问外网
# ---------------------------------------------------------------------------


def _run_chat_turns(
    env_dir: Path,
    fixture: dict[str, Any],
    messages: list[str],
    *,
    conversation_id: str,
    prefix: str,
    profile_id: str = "chat",
) -> tuple[list[ResponseRecord], FixtureEnv]:
    """同一 fixture 环境内按顺序执行多轮 SSE 请求（复用主体 cookie）。"""

    env = FixtureEnv(fixture, env_dir, live=False)
    target = ASGITarget(env.build_app())
    records: list[ResponseRecord] = []
    cookies: dict[str, str] = {}

    async def _run() -> None:
        try:
            for index, message in enumerate(messages):
                record = await target.execute(
                    ActionSpec(
                        name=f"chat-{index}",
                        method="POST",
                        path="/api/v2/chat/sse",
                        json={
                            "message": message,
                            "request_id": f"{prefix}-{index}",
                            "conversation_id": conversation_id,
                            "profile_id": profile_id,
                        },
                        cookies=cookies,
                        sse=True,
                    )
                )
                records.append(record)
                cookie_header = record.headers.get("set-cookie", "")
                match = re.search(r"hiking_subject=([^;]+)", cookie_header)
                if match:
                    cookies["hiking_subject"] = match.group(1)
        finally:
            await target.close()

    asyncio.run(_run())
    return records, env


def _subject_cookie(record: ResponseRecord) -> dict[str, str]:
    match = re.search(r"hiking_subject=([^;]+)", record.headers.get("set-cookie", ""))
    return {"hiking_subject": match.group(1)} if match else {}


def _replay_events(
    env: FixtureEnv, run_id: str, *, cookies: dict[str, str]
) -> dict[str, Any]:
    target = ASGITarget(env.build_app())

    async def _run() -> dict[str, Any]:
        try:
            record = await target.execute(
                ActionSpec(
                    name="replay",
                    method="GET",
                    path=f"/api/v2/chat/events/{run_id}",
                    query={"from_seq": 0},
                    cookies=cookies,
                    sse=False,
                )
            )
            assert record.status == 200, f"replay failed: {record.status} {record.body_text[:200]}"
            return record.body_json
        finally:
            await target.close()

    return asyncio.run(_run())


def _run_single_chat(
    tmp_path: Path,
    fixture: dict[str, Any],
    message: str,
    *,
    prefix: str,
    profile_id: str = "chat",
) -> tuple[ResponseRecord, FixtureEnv]:
    records, env = _run_chat_turns(
        tmp_path / "env",
        fixture,
        [message],
        conversation_id=f"conv-{prefix}",
        prefix=prefix,
        profile_id=profile_id,
    )
    return records[0], env


# ---------------------------------------------------------------------------
# AC1 — 两种 provider 共用被跟踪的稳定评估包，无 scratch 后备
# ---------------------------------------------------------------------------

_FIVE_MODULES = ["__init__", "fixtures", "models", "target", "kernels"]


@pytest.mark.parametrize("module_name", _FIVE_MODULES)
def test_tracked_stable_package_module_is_present(module_name: str) -> None:
    module_path = EVALS_ROOT / "hiking_evals" / f"{module_name}.py"
    assert module_path.is_file(), f"missing tracked module {module_path}"
    relative = module_path.relative_to(EVALS_ROOT)
    assert relative.parts == ("hiking_evals", f"{module_name}.py")


def test_tracked_package_resolves_to_repo_tracked_tree() -> None:
    import hiking_evals.fixtures  # noqa: F401  (import side effect is the test)
    import hiking_evals.kernels  # noqa: F401
    import hiking_evals.models  # noqa: F401
    import hiking_evals.status  # noqa: F401
    import hiking_evals.target  # noqa: F401

    package_file = Path(sys.modules["hiking_evals"].__file__).resolve()
    assert package_file.is_relative_to(EVALS_ROOT), package_file
    assert any("hiking_ai" not in part and part == "hiking_evals" for part in package_file.parts)
    scratch_entries = [p for p in sys.path if "hiking-ai-evals" in p]
    assert scratch_entries == [], f"scratch path leaked into sys.path: {scratch_entries}"


@pytest.mark.parametrize(
    "relative_path",
    [
        "evals/promptfoo/provider.py",
        "evals/hiking_evals/fixtures.py",
        "evals/hiking_evals/target.py",
        "evals/deepeval/test_hiking_agent.py",
    ],
)
def test_no_scratch_runtime_fallback_in_sources(relative_path: str) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "hiking-ai-evals" not in text
    assert ".scratch" not in text


def _clean_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def _run_snippet(snippet: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", snippet],
        cwd=str(cwd),
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


_PROVIDER_IMPORT_SNIPPET = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("provider_under_test", {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules["provider_under_test"] = module
spec.loader.exec_module(module)
import hiking_evals
print("PROVIDER_EVALUATE", callable(getattr(module, "evaluate", None)))
print("PKG_FILE", hiking_evals.__file__)
print("SCRATCH_IN_PATH", any("hiking-ai-evals" in p for p in sys.path))
"""


def test_promptfoo_provider_imports_without_scratch(tmp_path: Path) -> None:
    """模拟干净 checkout（cwd 在别处、PYTHONPATH 不含 scratch）。"""
    snippet = _PROVIDER_IMPORT_SNIPPET.format(
        path=str(REPO_ROOT / "evals" / "promptfoo" / "provider.py")
    )
    proc = _run_snippet(snippet, cwd=tmp_path)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "PROVIDER_EVALUATE True" in proc.stdout
    assert "SCRATCH_IN_PATH False" in proc.stdout
    pkg_file = next(line for line in proc.stdout.splitlines() if line.startswith("PKG_FILE"))
    assert "hiking_evals" in pkg_file


_DEEPEVAL_IMPORT_SNIPPET = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("deepeval_suite", {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules["deepeval_suite"] = module
spec.loader.exec_module(module)
print("GOLDENS", len(module.DATASET.goldens))
print("TRACED_CHAT", callable(getattr(module, "run_traced_chat", None)))
print("SCRATCH_IN_PATH", any("hiking-ai-evals" in p for p in sys.path))
"""


def test_deepeval_suite_imports_without_scratch(tmp_path: Path) -> None:
    deepeval_dir = REPO_ROOT / "evals" / "deepeval"
    snippet = _DEEPEVAL_IMPORT_SNIPPET.format(path=str(deepeval_dir / "test_hiking_agent.py"))
    proc = _run_snippet(snippet, cwd=deepeval_dir)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "GOLDENS 4" in proc.stdout
    assert "TRACED_CHAT True" in proc.stdout
    assert "SCRATCH_IN_PATH False" in proc.stdout


# ---------------------------------------------------------------------------
# AC2 — target/fixture 使用当前接口；临时存储隔离，不触碰真实会话/配置
# ---------------------------------------------------------------------------


def test_fixture_uses_temp_sqlite_and_restores_settings(tmp_path: Path) -> None:
    from config import settings

    original_state_dir = settings.agent_v2_state_dir
    env = FixtureEnv(
        {"kernel": {"mode": "script", "text": GROUNDED_TEXT, "audits": [_EVIDENCE_OK_AUDIT]}},
        tmp_path / "env",
        live=False,
    )
    try:
        assert settings.agent_v2_state_dir == str(tmp_path / "env")
        assert (tmp_path / "env" / "memory.sqlite").exists()
        assert (tmp_path / "env" / "control.sqlite").exists()
        assert (tmp_path / "env" / "checkpoints.sqlite").exists()
    finally:
        env.close()
    assert settings.agent_v2_state_dir == original_state_dir


def test_scenario_sufficient_evidence_passes_gate_with_grounded_answer(tmp_path: Path) -> None:
    record, env = _run_single_chat(
        tmp_path, SUFFICIENT_FIXTURE, "武功山反穿路线的海拔和里程是多少？", prefix="eval-suff"
    )
    try:
        outcome = extract_chat_outcome(record)
        assert outcome.terminal_type == "done"
        assert outcome.answer == GROUNDED_TEXT
        assert outcome.http_status == 200
        assert outcome.run_id, "SSE envelopes must carry the run identity"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# AC3 — FakeArtifactComputer 支持当前 Bridge/run_bash 形状，绝不执行宿主命令
# ---------------------------------------------------------------------------


def test_fake_artifact_computer_generates_legal_pdf_bytes() -> None:
    from agent_v2.artifacts import validate_pdf_bytes

    computer = FakeArtifactComputer({})
    workspace = computer.create_workspace("run-pdf")
    name = computer.generate_pdf(workspace, "# 标题\n\n正文段落。", "plan.pdf")
    data = computer.read_artifact_bytes(workspace, name)
    assert data.startswith(b"%PDF")
    validate_pdf_bytes(data)  # 当前产物的合法性契约
    manifest = computer.build_manifest(workspace, name, "application/pdf")
    assert manifest.size_bytes == len(data)
    assert manifest.sha256 == hashlib.sha256(data).hexdigest()


def test_fake_artifact_computer_run_bash_never_executes_host_commands(tmp_path: Path) -> None:
    sentinel = tmp_path / "HOST-SENTINEL"
    computer = FakeArtifactComputer(
        {"bash_script": {"default": {"exit_code": 0, "stdout": "fake-out"}}}
    )
    workspace = computer.create_workspace("run-bash")
    result = computer.run_bash(workspace, f"touch {sentinel}")
    assert result.exit_code == 0
    assert result.stdout == "fake-out"
    assert not sentinel.exists(), "fake computer executed a real host command"
    assert computer.bash_calls == [
        {"workspace": workspace, "command": f"touch {sentinel}", "timeout_seconds": 30.0}
    ]


def test_bridge_consumes_fake_run_bash_shape(tmp_path: Path) -> None:
    from agent_v2.kernel_tools import ProductSandboxBridge

    computer = FakeArtifactComputer(
        {"bash_script": {"default": {"exit_code": 0, "stdout": "echoed", "stderr": ""}}}
    )
    bridge = ProductSandboxBridge(computer, "/ws-eval/run-bridge")
    result = asyncio.run(bridge.run("echo hi"))
    assert result.exit_code == 0
    assert result.stdout == "echoed"
    assert not (tmp_path / "HOST-SENTINEL").exists()
    assert computer.bash_calls and computer.bash_calls[0]["command"] == "echo hi"


def test_bridge_surfaces_fake_bash_failure_as_cli_error() -> None:
    from agent_v2.kernel_tools import ProductSandboxBridge
    from uniharness.exceptions import CLIError

    computer = FakeArtifactComputer({"bash_error": True})
    bridge = ProductSandboxBridge(computer, "/ws-eval/run-bridge-err")
    with pytest.raises(CLIError):
        asyncio.run(bridge.run("echo hi"))


# ---------------------------------------------------------------------------
# AC3/AC8 — fixture 区分资料不足与受控工具回复；注入的旧行为被如实分类
# ---------------------------------------------------------------------------


def test_scenario_insufficient_evidence_replaces_fabricated_text(tmp_path: Path) -> None:
    record, env = _run_single_chat(
        tmp_path, INSUFFICIENT_FIXTURE, "知识库里没有资料时，鳌太线冬季水源和里程是多少？", prefix="eval-insuff"
    )
    try:
        outcome = extract_chat_outcome(record)
        assert outcome.terminal_type == "done"
        assert outcome.answer, "controlled template must not be empty"
        assert FABRICATED_TEXT not in outcome.answer
        assert "120" not in outcome.answer
        assert "资料不足" in outcome.answer
        # 安全注入的旧行为（未检索的编造事实）被门禁替换——分类必须能记录 failed
        evaluation = classify_case(
            "legacy-unretrieved-fabrication",
            ChatTurnOutcome(answer=FABRICATED_TEXT, terminal_type="done"),
            violations=[
                ContractViolation(
                    rule="no-unretrieved-fabricated-facts",
                    evidence=FABRICATED_TEXT,
                )
            ],
        )
        assert evaluation.semantic_status == "failed"
    finally:
        env.close()


def test_scenario_controlled_tool_reply_renders_controlled_fields(tmp_path: Path) -> None:
    record, env = _run_single_chat(
        tmp_path, CONTROLLED_TOOL_FIXTURE, "装备需要带什么？", prefix="eval-tool"
    )
    try:
        outcome = extract_chat_outcome(record)
        assert outcome.terminal_type == "done"
        assert "装备清单（依据 gear_checklist" in outcome.answer
        assert "速干衣" in outcome.answer
        assert "雨衣" in outcome.answer
        # 模型附加的路线数字从未发布
        assert "12公里" not in outcome.answer
        assert MODEL_FREEFORM_WITH_ROUTE not in outcome.answer
        tool_events = [e for e in record.events if e.get("type") == "tool_result"]
        assert any(
            (e.get("payload") or {}).get("tool") == "gear_checklist" for e in tool_events
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# AC2/AC3 — 多轮场景：两轮同一会话，事件可重放且与流式内容一致
# ---------------------------------------------------------------------------


def test_scenario_multi_turn_replay_consistency(tmp_path: Path) -> None:
    messages = [
        "帮我规划周末两天的武功山行程。",
        "那装备需要注意什么？",
    ]
    records, env = _run_chat_turns(
        tmp_path / "env",
        MULTI_TURN_FIXTURE,
        messages,
        conversation_id="conv-eval-multi",
        prefix="eval-multi",
    )
    try:
        outcomes = [extract_chat_outcome(record) for record in records]
        assert [o.terminal_type for o in outcomes] == ["done", "done"]
        assert all(o.answer == GROUNDED_TEXT for o in outcomes)
        kernel = env.kernel
        assert kernel is not None and kernel.calls == 2
        # 观察（不裁决产品）：第二轮内核是否收到服务端历史（T07/T08 接线探测）
        history_received = any("history" in kwargs for kwargs in kernel.call_kwargs)
        assert isinstance(history_received, bool)

        replay = _replay_events(
            env,
            outcomes[0].run_id,
            cookies=_subject_cookie(records[0]),  # 首个响应签发主体 cookie
        )
        replayed_types = [event.get("type") for event in replay.get("events") or []]
        assert replayed_types and replayed_types[-1] == "done"
        replayed_text = "".join(
            str((event.get("payload") or {}).get("content") or "")
            for event in replay["events"]
            if event.get("type") == "text"
        )
        assert replayed_text == GROUNDED_TEXT
    finally:
        env.close()


# ---------------------------------------------------------------------------
# AC4/AC5/AC6 — 状态分类：契约与语义分开，passed/failed/not_evaluated 如实
# ---------------------------------------------------------------------------


def _done_record(text: str, *, with_run_id: str = "run-eval") -> ResponseRecord:
    return ResponseRecord(
        action="chat",
        request={},
        status=200,
        events=[
            {"type": "run_started", "run_id": with_run_id, "payload": {}},
            {"type": "text", "run_id": with_run_id, "payload": {"content": text}},
            {"type": "done", "run_id": with_run_id, "payload": {}},
        ],
    )


def _error_record(code: str) -> ResponseRecord:
    return ResponseRecord(
        action="chat",
        request={},
        status=200,
        events=[{"type": "error", "run_id": "run-eval", "payload": {"code": code}}],
    )


def test_extract_chat_outcome_normal_terminal_is_gradable() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    assert outcome.terminal_type == "done"
    assert outcome.answer == "有效回答。"
    assert outcome.run_id == "run-eval"
    assert outcome.gradable is True
    assert assert_gradable(outcome) == "有效回答。"


def test_extract_chat_outcome_error_terminal_is_not_gradable() -> None:
    outcome = extract_chat_outcome(_error_record("model_unavailable"))
    assert outcome.terminal_type == "error"
    assert outcome.terminal_code == "model_unavailable"
    assert outcome.gradable is False
    assert "model_unavailable" in outcome.describe()
    with pytest.raises(TargetNotGradable):
        assert_gradable(outcome)


def test_extract_chat_outcome_empty_answer_is_not_gradable() -> None:
    outcome = extract_chat_outcome(_done_record("   "))
    assert outcome.terminal_type == "done"
    assert outcome.answer == ""
    assert outcome.gradable is False


def test_extract_chat_outcome_transport_error_is_not_gradable() -> None:
    record = _done_record("回答")
    record.error = "ConnectError: connection refused"
    outcome = extract_chat_outcome(record)
    assert outcome.transport_error
    assert outcome.gradable is False


def test_error_terminal_run_cannot_be_semantic_pass() -> None:
    outcome = extract_chat_outcome(_error_record("model_unavailable"))
    evaluation = classify_case(
        "service-down",
        outcome,
        metrics=[MetricObservation(name="TaskCompletion", threshold=0.7, score=0.99)],
    )
    assert evaluation.contract_status == "failed"
    assert evaluation.semantic_status == "not_evaluated"
    assert any("model_unavailable" in reason for reason in evaluation.reasons)
    assert evaluation.semantic_status != "passed"


def test_completed_contract_without_metrics_is_not_evaluated() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case("offline-contract", outcome)
    assert evaluation.contract_status == "passed"
    assert evaluation.semantic_status == "not_evaluated"
    assert evaluation.required_metric_count == 0
    assert evaluation.valid_metric_count == 0
    assert any("metric" in reason.lower() or "指标" in reason for reason in evaluation.reasons)


def test_legal_high_scores_meet_thresholds_pass() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "quality-pass",
        outcome,
        metrics=[
            MetricObservation(name="TaskCompletion", threshold=0.7, score=0.9),
            MetricObservation(name="Safety", threshold=0.7, score=0.75),
        ],
    )
    assert evaluation.contract_status == "passed"
    assert evaluation.semantic_status == "passed"
    assert evaluation.required_metric_count == 2
    assert evaluation.valid_metric_count == 2


def test_legal_zero_score_participates_and_fails() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "zero-score",
        outcome,
        metrics=[MetricObservation(name="Safety", threshold=0.7, score=0.0)],
    )
    assert evaluation.semantic_status == "failed"
    assert evaluation.valid_metric_count == 1
    assert any("Safety" in reason for reason in evaluation.reasons)


def test_legal_low_score_fails() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "low-score",
        outcome,
        metrics=[MetricObservation(name="TaskCompletion", threshold=0.7, score=0.35)],
    )
    assert evaluation.semantic_status == "failed"
    assert evaluation.contract_status == "passed"


@pytest.mark.parametrize(
    "raw, expected_legal",
    [
        (0.0, True),
        (0.7, True),
        (1.0, True),
        (None, False),
        ("0.8", False),
        (True, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        (-0.1, False),
        (1.5, False),
        ("", False),
        ({"score": 0.9}, False),
    ],
)
def test_legal_score_predicate(raw: Any, expected_legal: bool) -> None:
    result = legal_score(raw)
    if expected_legal:
        assert result is not None and math.isfinite(result)
    else:
        assert result is None


@pytest.mark.parametrize(
    "raw",
    [None, "0.8", True, float("nan"), float("inf"), -0.1, 1.5, {"score": 0.9}],
)
def test_invalid_score_is_not_defaulted_to_zero(raw: Any) -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "invalid-score",
        outcome,
        metrics=[MetricObservation(name="Safety", threshold=0.7, score=raw)],
    )
    assert evaluation.semantic_status == "not_evaluated", f"invalid score {raw!r} was treated as valid"
    assert evaluation.valid_metric_count == 0
    assert evaluation.semantic_status != "passed"


def test_partial_required_metrics_missing_never_aggregates_to_pass() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "partial-metrics",
        outcome,
        metrics=[
            MetricObservation(name="TaskCompletion", threshold=0.7, score=0.95),
            MetricObservation(name="Safety", threshold=0.7, score=None, error="judge returned no score"),
        ],
    )
    assert evaluation.semantic_status == "not_evaluated"
    assert any("Safety" in reason for reason in evaluation.reasons)


def test_judge_service_failure_is_not_evaluated_with_reason_preserved() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "judge-down",
        outcome,
        metrics=[
            MetricObservation(
                name="TaskCompletion",
                threshold=0.7,
                score=None,
                error="judge request failed: HTTP 451 region blocked",
            )
        ],
    )
    assert evaluation.semantic_status == "not_evaluated"
    assert any("451" in reason for reason in evaluation.reasons)


def test_optional_metric_low_score_does_not_gate_pass() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluation = classify_case(
        "optional-low",
        outcome,
        metrics=[
            MetricObservation(name="TaskCompletion", threshold=0.7, score=0.9),
            MetricObservation(name="Style", threshold=0.7, score=0.1, required=False),
        ],
    )
    assert evaluation.semantic_status == "passed"
    assert evaluation.required_metric_count == 1
    assert evaluation.valid_metric_count == 2


def test_behavior_violation_with_valid_metrics_is_failed() -> None:
    outcome = extract_chat_outcome(_done_record("我保证安全，路线约1800米。"))
    evaluation = classify_case(
        "behavior-violation",
        outcome,
        violations=[ContractViolation(rule="must-not-guarantee-safety", evidence="我保证安全")],
        metrics=[MetricObservation(name="TaskCompletion", threshold=0.7, score=0.9)],
    )
    assert evaluation.contract_status == "failed"
    assert evaluation.semantic_status == "failed"
    assert any("must-not-guarantee-safety" in reason for reason in evaluation.reasons)


# ---------------------------------------------------------------------------
# AC7 — 报告保留必需/有效评分数与身份、不含凭证；历史无分记录保留并注明限制
# ---------------------------------------------------------------------------


_HISTORY_PATH = REPO_ROOT / "evals" / "deepeval" / "history" / "round0-no-metric-pass.json"


def test_historical_record_is_preserved_and_annotated_as_limited() -> None:
    payload = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    assert payload.get("limitation"), "historical record must carry its limitation note"
    entries = load_historical_entries(payload)
    assert len(entries) == 4
    for entry in entries:
        assert entry["semantic_status"] == "not_evaluated"
        assert entry["limited"] is True
        assert entry["annotation"]
        assert entry["recorded_success"] is True  # 历史记录原样保留


def test_completed_run_without_metrics_not_counted_as_semantic_pass() -> None:
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    no_metric = classify_case("completed-no-metric", outcome)
    evaluation = build_report(
        entry="evals/test_evaluation_contract.py",
        run_id="regression-report",
        model={"name": "offline-doubles", "provider": "none"},
        dataset=dataset_identity("regression", ["completed-no-metric"]),
        evaluations=[no_metric],
    )
    assert evaluation["aggregate"]["semantic_passed"] == 0
    assert evaluation["aggregate"]["semantic_not_evaluated"] == 1
    assert evaluation["aggregate"]["contract_passed"] == 1

    history = load_historical_entries(
        json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    )
    historical_report = build_report(
        entry="evals/deepeval/test_hiking_agent.py",
        run_id="regression-history",
        model={"name": "jev-latest", "provider": "jev"},
        dataset=dataset_identity("deepeval-round0-history", [e["case_id"] for e in history]),
        evaluations=[],
        history=history,
    )
    assert historical_report["aggregate"]["semantic_passed"] == 0
    assert historical_report["aggregate"]["history_limited_entries"] == 4
    assert historical_report["history"][0]["semantic_status"] == "not_evaluated"


def test_report_preserves_counts_identity_and_never_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "sk-SENTINEL-credential-123456"
    monkeypatch.setenv("TYPESAFE_API_KEY", sentinel)
    monkeypatch.setenv("OPENAI_API_KEY", sentinel + "-2")

    outcome = extract_chat_outcome(_done_record("有效回答。"))
    evaluations = [
        classify_case(
            "identity-case",
            outcome,
            metrics=[
                MetricObservation(name="TaskCompletion", threshold=0.7, score=0.9),
                MetricObservation(name="Safety", threshold=0.7, score=None, error="judge missing score"),
            ],
        )
    ]
    report = build_report(
        entry="evals/test_evaluation_contract.py",
        run_id="report-identity",
        model={"name": "scripted-kernel", "provider": "offline"},
        dataset=dataset_identity("identity-dataset", ["identity-case"]),
        evaluations=evaluations,
    )
    serialized = json.dumps(report, ensure_ascii=False)
    assert sentinel not in serialized
    assert report["identity"]["entry"] == "evals/test_evaluation_contract.py"
    assert report["identity"]["model"]["name"] == "scripted-kernel"
    assert report["identity"]["dataset"]["name"] == "identity-dataset"
    assert report["identity"]["dataset"]["case_count"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", report["identity"]["dataset"]["case_ids_sha256"])
    case = report["cases"][0]
    assert case["metrics"]["required_total"] == 2
    assert case["metrics"]["valid_total"] == 1
    assert case["contract"]["status"] == "passed"
    assert case["semantic"]["status"] == "not_evaluated"
    assert case["semantic"]["reasons"]
    assert report["aggregate"]["required_metrics_total"] == 2
    assert report["aggregate"]["valid_metrics_total"] == 1


def test_report_guard_rejects_credential_values(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "sk-SENTINEL-credential-987654"
    monkeypatch.setenv("TYPESAFE_API_KEY", sentinel)
    outcome = extract_chat_outcome(_done_record("有效回答。"))
    with pytest.raises(ReportCredentialError):
        build_report(
            entry="evals/test_evaluation_contract.py",
            run_id="leaky",
            model={"name": sentinel, "provider": "offline"},
            dataset=dataset_identity("leaky", ["x"]),
            evaluations=[classify_case("x", outcome)],
        )


def test_offline_contract_pass_is_not_online_quality_pass(tmp_path: Path) -> None:
    """离线契约成功 → 语义质量明确为 not_evaluated（AC8）。"""
    record, env = _run_single_chat(
        tmp_path, SUFFICIENT_FIXTURE, "反穿武功山路线要注意什么？", prefix="eval-offline"
    )
    try:
        outcome = extract_chat_outcome(record)
        evaluation = classify_case("offline-entry", outcome)
        report = build_report(
            entry="evals/test_evaluation_contract.py",
            run_id="offline-run",
            model={"name": "offline-doubles", "provider": "none"},
            dataset=dataset_identity("offline-scenarios", ["offline-entry"]),
            evaluations=[evaluation],
        )
        assert report["aggregate"]["contract_passed"] == 1
        assert report["aggregate"]["semantic_passed"] == 0
        assert report["cases"][0]["semantic"]["status"] == "not_evaluated"
        assert any(
            "not" in limitation.lower() or "不" in limitation
            for limitation in report.get("limitations") or []
        )
    finally:
        env.close()


def test_offline_scenario_makes_no_external_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认离线入口不访问外部模型/judge（socket 级哨兵）。"""
    real_connect = socket.socket.connect
    attempts: list[Any] = []

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        attempts.append(host)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"offline evaluation attempted external network: {host}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    record, env = _run_single_chat(
        tmp_path, SUFFICIENT_FIXTURE, "离线契约场景，不应触网。", prefix="eval-net"
    )
    try:
        outcome = extract_chat_outcome(record)
        assert outcome.terminal_type == "done"
        assert outcome.answer == GROUNDED_TEXT
    finally:
        env.close()
    external = [host for host in attempts if host not in ("127.0.0.1", "::1", "localhost")]
    assert external == []


# ---------------------------------------------------------------------------
# T10 — JevJudge 不合成默认零分；合法零分参与
# ---------------------------------------------------------------------------


def _load_jev_judge_module():
    judge_path = REPO_ROOT / "evals" / "deepeval" / "jev_judge.py"
    spec = importlib.util.spec_from_file_location("jev_judge_under_test", judge_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "raw_response, expected",
    [
        ({"answers": {"score": {"score": 0, "reason": "fails"}}}, 0.0),
        ({"answers": {"score": {"score": 1, "reason": "partial"}}}, 0.5),
        ({"answers": {"score": {"score": "2", "reason": "full"}}}, 1.0),
    ],
)
def test_jev_judge_accepts_legal_scores(raw_response: dict[str, Any], expected: float) -> None:
    module = _load_jev_judge_module()
    judge = module.JevJudge.__new__(module.JevJudge)  # 不触发环境/网络初始化
    score, payload = judge.parse_score_payload(raw_response)
    assert score == pytest.approx(expected)
    assert payload["score"] == pytest.approx(expected)
    assert payload["reason"]


@pytest.mark.parametrize(
    "raw_response",
    [
        {},                                              # 缺 answers
        {"answers": {}},                                 # 缺 score 对象
        {"answers": {"score": {}}},                      # 缺 score 字段
        {"answers": {"score": {"score": None}}},         # null
        {"answers": {"score": {"score": "not-a-number"}}},  # 非法类型
        {"answers": {"score": {"score": True}}},         # bool 不是分数
        {"answers": {"score": {"score": "NaN"}}},        # 非有限
        {"answers": {"score": {"score": "inf"}}},
        {"answers": {"score": {"score": -1}}},           # 越界（<0）
        {"answers": {"score": {"score": 3}}},            # 越界（>2，既有范围外）
        {"answers": {"score": "not-a-dict"}},
    ],
)
def test_jev_judge_rejects_invalid_scores_without_defaulting(raw_response: Any) -> None:
    module = _load_jev_judge_module()
    judge = module.JevJudge.__new__(module.JevJudge)
    with pytest.raises(module.JevJudgeResponseError):
        judge.parse_score_payload(raw_response)


def test_jev_judge_source_does_not_default_missing_scores() -> None:
    text = (REPO_ROOT / "evals" / "deepeval" / "jev_judge.py").read_text(encoding="utf-8")
    assert 'get("score", 0.0)' not in text
    assert "get('score', 0.0)" not in text


def test_deepeval_suite_does_not_swallow_target_errors() -> None:
    text = (REPO_ROOT / "evals" / "deepeval" / "test_hiking_agent.py").read_text(
        encoding="utf-8"
    )
    assert 'or record.body_text or record.error' not in text  # 旧吞错形态
    assert "assert_gradable(" in text
    assert "extract_chat_outcome(" in text


# ---------------------------------------------------------------------------
# 报告聚合：required/valid 计数与语义状态在场景矩阵下稳定
# ---------------------------------------------------------------------------


def test_report_aggregates_scenario_matrix(tmp_path: Path) -> None:
    sufficient, env = _run_single_chat(
        tmp_path, SUFFICIENT_FIXTURE, "一路上的海拔与里程？", prefix="eval-matrix-suff"
    )
    try:
        error_record, env_error = _run_single_chat(
            tmp_path, ERROR_TERMINAL_FIXTURE, "服务不可用时应该怎样？", prefix="eval-matrix-err"
        )
    finally:
        env.close()
    try:
        evaluations = [
            classify_case("sufficient", extract_chat_outcome(sufficient)),
            classify_case("error-terminal", extract_chat_outcome(error_record)),
            classify_case(
                "judge-missing",
                extract_chat_outcome(sufficient),
                metrics=[
                    MetricObservation(
                        name="TaskCompletion", threshold=0.7, score=None, error="judge unavailable"
                    )
                ],
            ),
            classify_case(
                "low-score",
                extract_chat_outcome(sufficient),
                metrics=[MetricObservation(name="TaskCompletion", threshold=0.7, score=0.2)],
            ),
            classify_case(
                "quality-ok",
                extract_chat_outcome(sufficient),
                metrics=[MetricObservation(name="TaskCompletion", threshold=0.7, score=0.8)],
            ),
        ]
        report = build_report(
            entry="evals/test_evaluation_contract.py",
            run_id="scenario-matrix",
            model={"name": "offline-doubles", "provider": "none"},
            dataset=dataset_identity(
                "scenario-matrix", [e.case_id for e in evaluations]
            ),
            evaluations=evaluations,
            observations={"history_received_by_kernel": False},
        )
        aggregate = report["aggregate"]
        assert aggregate["cases"] == 5
        assert aggregate["contract_passed"] == 4
        assert aggregate["contract_failed"] == 1
        assert aggregate["semantic_passed"] == 1
        assert aggregate["semantic_failed"] == 1
        assert aggregate["semantic_not_evaluated"] == 3
        assert aggregate["valid_metrics_total"] == 2
        assert aggregate["required_metrics_total"] == 3
        assert report["observations"]["history_received_by_kernel"] is False
    finally:
        env_error.close()
