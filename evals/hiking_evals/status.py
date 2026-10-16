"""执行契约与语义质量的分离分类 + 报告构建（spec T10）。

规则（与 06 号票验收一致）：

- **执行契约**（contract）：目标正常终态（``done``）+ 非空回答 + 无传输
  错误 = passed；错误终态 / 空回答 / 传输失败 / 显式行为契约违反 = failed。
- **语义质量**（semantic）只有同时满足以下条件才是 passed：
  正常终态、回答有效、全部必需 metric 具有**合法分数**且达到既有阈值。
- 已完成评估但行为违反契约，或合法分数低于阈值 → failed；
  评分缺失 / judge 非法返回 / 服务不可用导致无法判定 → not_evaluated，
  原因原样保留。
- 缺失、null、非法类型、非有限或越界 score **不默认成有效零分**；
  合法零分参与真实判定。部分必需 metric 缺失不能汇总为语义通过。
- 报告保留必需/有效评分数与入口、模型、数据集身份；凭证类值进入报告
  一律拒绝（``ReportCredentialError``）。
- 历史"执行完成但无 metric"的记录原样保留并注明证据限制，不计入
  语义通过数。

本模块是 Promptfoo 与 DeepEval 两种入口共用的稳定分类边界。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from .target import ChatTurnOutcome

CONTRACT_PASSED = "passed"
CONTRACT_FAILED = "failed"

SEMANTIC_PASSED = "passed"
SEMANTIC_FAILED = "failed"
SEMANTIC_NOT_EVALUATED = "not_evaluated"

REPORT_SCHEMA = "hiking-evals/status-v1"

SCORE_MIN = 0.0
SCORE_MAX = 1.0

DEFAULT_LIMITATIONS = (
    "离线契约运行不调用外部模型或 judge：语义质量维度为 not_evaluated，"
    "离线契约通过不代表在线质量通过。",
    "缺失/null/非法类型/非有限或越界的 score 不合成默认零分，只记 not_evaluated；"
    "合法的零分参与真实判定。",
)


class TargetNotGradable(RuntimeError):
    """目标终态不可评分：错误终态 / 空回答 / 传输失败（不可包装成通过）。"""


class ReportCredentialError(ValueError):
    """报告将包含疑似凭证值——拒绝生成。"""


# ---- 分数合法性 -------------------------------------------------------------


def legal_score(value: Any) -> float | None:
    """返回合法分数（[0,1] 内有限数值），否则 ``None``。

    bool 不是分数；字符串、dict、None 是非法类型；NaN/±Inf 非有限；
    越界（<0 或 >1）同样非法。合法零分必须保留并参与判定。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number < SCORE_MIN or number > SCORE_MAX:
        return None
    return number


def assert_gradable(outcome: ChatTurnOutcome) -> str:
    """返回可评估回答；不可评估时抛 :class:`TargetNotGradable`。

    供 DeepEval/Promptfoo 入口共用：错误 SSE 终态或空回答绝不能吞成
    "空但有效"的通过。
    """
    if not outcome.gradable:
        raise TargetNotGradable(outcome.describe())
    return outcome.answer


# ---- 分类模型 ---------------------------------------------------------------


@dataclass(frozen=True)
class ContractViolation:
    """显式行为契约违反（例如旧的无检索编造事实行为被注入并检出）。"""

    rule: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "evidence": self.evidence}


@dataclass(frozen=True)
class MetricObservation:
    """一条必需/可选 metric 的观测。

    ``score`` 为原始判定器返回值（可能缺失或非法）；``error`` 记录
    judge/服务故障原因，缺失评分时原样进入报告，绝不换算成 0 分。
    """

    name: str
    threshold: float
    score: Any = None
    error: str = ""
    required: bool = True

    def legal(self) -> float | None:
        return legal_score(self.score)

    def to_dict(self) -> dict[str, Any]:
        legal = self.legal()
        return {
            "name": self.name,
            "required": self.required,
            "threshold": self.threshold,
            "valid": legal is not None,
            "score": legal,  # 非法/缺失一律记 None，不合成零分
            "score_reported": self.score is not None,
            "passed": legal is not None and legal >= self.threshold,
            "error": self.error,
        }


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    outcome: ChatTurnOutcome
    contract_status: str
    semantic_status: str
    reasons: tuple[str, ...]
    violations: tuple[ContractViolation, ...] = ()
    metrics: tuple[MetricObservation, ...] = ()

    @property
    def required_metric_count(self) -> int:
        return sum(1 for m in self.metrics if m.required)

    @property
    def valid_metric_count(self) -> int:
        return sum(1 for m in self.metrics if m.legal() is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "contract": {
                "status": self.contract_status,
                "terminal_type": self.outcome.terminal_type,
                "terminal_code": self.outcome.terminal_code,
                "answer_chars": len(self.outcome.answer),
                "transport_error": self.outcome.transport_error or None,
                "run_id": self.outcome.run_id,
                "violations": [v.to_dict() for v in self.violations],
            },
            "semantic": {
                "status": self.semantic_status,
                "reasons": list(self.reasons),
            },
            "metrics": {
                "required_total": self.required_metric_count,
                "valid_total": self.valid_metric_count,
                "observations": [m.to_dict() for m in self.metrics],
            },
        }


def classify_case(
    case_id: str,
    outcome: ChatTurnOutcome,
    *,
    violations: Sequence[ContractViolation] = (),
    metrics: Sequence[MetricObservation] = (),
) -> CaseEvaluation:
    """把一个回合的终态 + 行为违反 + 指标观测分类为契约/语义双状态。"""
    violations = tuple(violations)
    metrics = tuple(metrics)

    # 1) 不可评分的目标：契约失败 + 语义 not_evaluated（原因保留）。
    if not outcome.gradable:
        return CaseEvaluation(
            case_id=case_id,
            outcome=outcome,
            contract_status=CONTRACT_FAILED,
            semantic_status=SEMANTIC_NOT_EVALUATED,
            reasons=(
                outcome.describe(),
                "语义质量无法判定：目标未产生可评估的有效回答（not_evaluated）。",
            ),
            violations=violations,
            metrics=metrics,
        )

    # 2) 完成评估但行为违反契约：语义 failed（无需等待判定器）。
    if violations:
        reasons = [
            f"行为契约违反：{v.rule}" + (f"（证据：{v.evidence}）" if v.evidence else "")
            for v in violations
        ]
        reasons.append("评估已完成但行为违反契约：语义质量判为 failed。")
        return CaseEvaluation(
            case_id=case_id,
            outcome=outcome,
            contract_status=CONTRACT_FAILED,
            semantic_status=SEMANTIC_FAILED,
            reasons=tuple(reasons),
            violations=violations,
            metrics=metrics,
        )

    # 3) 契约通过：按必需指标判定语义质量。
    required = [m for m in metrics if m.required]
    if not metrics:
        return CaseEvaluation(
            case_id=case_id,
            outcome=outcome,
            contract_status=CONTRACT_PASSED,
            semantic_status=SEMANTIC_NOT_EVALUATED,
            reasons=(
                "离线契约运行未产生任何指标观测：未调用外部模型/judge，"
                "离线契约通过不构成语义质量通过。",
            ),
            violations=violations,
            metrics=metrics,
        )

    invalid_required = [m for m in required if m.legal() is None]
    low_required = [
        m for m in required if m.legal() is not None and m.legal() < m.threshold
    ]
    reasons: list[str] = []
    for metric in low_required:
        reasons.append(
            f"{metric.name}: 合法低分 {metric.legal():.3f} < 阈值 {metric.threshold}"
        )
    for metric in invalid_required:
        detail = metric.error or (
            "缺少有效 score（缺失/null/非法类型/非有限/越界均不算零分）"
        )
        reasons.append(f"{metric.name}: {detail}（不合成默认零分）")

    if low_required:
        # 合法低分是决定性否定证据：即使另有必需指标缺失，也已是 failed。
        return CaseEvaluation(
            case_id=case_id,
            outcome=outcome,
            contract_status=CONTRACT_PASSED,
            semantic_status=SEMANTIC_FAILED,
            reasons=tuple(reasons),
            violations=violations,
            metrics=metrics,
        )
    if invalid_required:
        return CaseEvaluation(
            case_id=case_id,
            outcome=outcome,
            contract_status=CONTRACT_PASSED,
            semantic_status=SEMANTIC_NOT_EVALUATED,
            reasons=tuple(reasons),
            violations=violations,
            metrics=metrics,
        )
    return CaseEvaluation(
        case_id=case_id,
        outcome=outcome,
        contract_status=CONTRACT_PASSED,
        semantic_status=SEMANTIC_PASSED,
        reasons=("全部必需指标具备合法分数且达到既有阈值。",),
        violations=violations,
        metrics=metrics,
    )


# ---- 数据集身份 -------------------------------------------------------------


def dataset_identity(name: str, case_ids: Sequence[str]) -> dict[str, Any]:
    """稳定、可复现的数据集身份（名称 + 用例清单 + 清单哈希）。"""
    ids = [str(case_id) for case_id in case_ids]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    return {
        "name": str(name),
        "case_count": len(ids),
        "case_ids_sha256": digest,
        "cases": ids,
    }


# ---- 历史记录：保留 + 注明限制 ---------------------------------------------


def load_historical_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """把历史"执行完成但无有效分数"的通过记录标注为 limited。

    历史记录原样保留（不重写为通过或失败），但语义维度明确为
    ``not_evaluated`` 并附限制说明：它们不再计入语义通过数。
    """
    entries: list[dict[str, Any]] = []
    for test in payload.get("tests") or []:
        valid_count = int(test.get("valid_metric_count") or 0)
        entries.append(
            {
                "case_id": str(test.get("name") or ""),
                "recorded_success": bool(test.get("success")),
                "recorded_metric_count": int(
                    test.get("metric_count", valid_count) or 0
                ),
                "valid_metric_count": valid_count,
                "semantic_status": SEMANTIC_NOT_EVALUATED,
                "limited": True,
                "annotation": (
                    "历史记录原样保留（当时按执行完成计通过）；无有效 metric 分数，"
                    "不构成语义质量证据，不计入语义通过数。"
                ),
            }
        )
    return entries


# ---- 报告 -------------------------------------------------------------------


_CREDENTIAL_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _credential_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        upper = name.upper()
        if not any(hint in upper for hint in _CREDENTIAL_NAME_HINTS):
            continue
        if isinstance(value, str) and len(value) >= 8:
            values.append(value)
    return values


def _assert_no_credentials(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    for value in _credential_values():
        if value in text:
            raise ReportCredentialError(
                "报告将包含疑似凭证值（与 *_KEY/_TOKEN/_SECRET 环境变量匹配），已拒绝生成。"
            )


def build_report(
    *,
    entry: str,
    run_id: str,
    model: dict[str, Any],
    dataset: dict[str, Any],
    evaluations: Sequence[CaseEvaluation],
    history: Sequence[dict[str, Any]] = (),
    observations: dict[str, Any] | None = None,
    limitations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """构建评估报告（契约与语义分列聚合；无凭证）。"""
    cases = [evaluation.to_dict() for evaluation in evaluations]
    history_entries = [dict(entry_item) for entry_item in history]
    aggregate = {
        "cases": len(evaluations),
        "contract_passed": sum(
            1 for evaluation in evaluations if evaluation.contract_status == CONTRACT_PASSED
        ),
        "contract_failed": sum(
            1 for evaluation in evaluations if evaluation.contract_status == CONTRACT_FAILED
        ),
        "semantic_passed": sum(
            1 for evaluation in evaluations if evaluation.semantic_status == SEMANTIC_PASSED
        ),
        "semantic_failed": sum(
            1 for evaluation in evaluations if evaluation.semantic_status == SEMANTIC_FAILED
        ),
        "semantic_not_evaluated": sum(
            1
            for evaluation in evaluations
            if evaluation.semantic_status == SEMANTIC_NOT_EVALUATED
        ),
        "required_metrics_total": sum(
            evaluation.required_metric_count for evaluation in evaluations
        ),
        "valid_metrics_total": sum(
            evaluation.valid_metric_count for evaluation in evaluations
        ),
        "history_limited_entries": len(history_entries),
        "history_semantic_passed": 0,  # 历史无分记录永不折算为语义通过
    }
    report = {
        "schema": REPORT_SCHEMA,
        "run_id": str(run_id),
        "identity": {
            "entry": str(entry),
            "model": {
                "name": str((model or {}).get("name", "")),
                "provider": str((model or {}).get("provider", "")),
            },
            "dataset": dict(dataset),
        },
        "cases": cases,
        "history": history_entries,
        "observations": dict(observations or {}),
        "limitations": list(limitations) if limitations is not None else list(DEFAULT_LIMITATIONS),
        "aggregate": aggregate,
    }
    _assert_no_credentials(report)
    return report
