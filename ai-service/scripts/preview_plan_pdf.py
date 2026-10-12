"""Generate an offline plan-book sample and page previews without external services."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_v2.computer import _render_pdf_bytes
from route_compare.fixtures import fixture_candidates
from route_compare.models import BudgetItem, PlanDetails, PlanStage
from route_compare.providers import DeterministicProvider, ProviderConfig
from route_compare.report_pdf import build_report_document
from route_compare.service import compare_routes


def main() -> None:
    result = asyncio.run(compare_routes(fixture_candidates(),
        DeterministicProvider(ProviderConfig(data_mode="offline")), comparison_id="offline-plan-preview"))
    result.plan_details = PlanDetails(
        activity_date="2026-10-18（演示）", meeting_time="08:00", meeting_point="演示游客中心",
        participants=4, leader="演示领队", transport="演示：游客中心集合，结束后原车返回",
        accommodation="演示：当日往返，无住宿", retreat_plan="待领队核实后填写撤退点与接应方式",
        itinerary=[PlanStage(time="08:00", activity="集合、点名与装备检查"),
                   PlanStage(time="时间待确认", activity="按核验路线结伴徒步，适时补给与休息"),
                   PlanStage(time="返程时间待确认", activity="清点人员、带离垃圾并报平安")],
        budget=[BudgetItem(item="演示交通预算（全队）", amount_yuan=200),
                BudgetItem(item="演示餐食补给预算（全队）", amount_yuan=160)],
    )
    output = Path(__file__).resolve().parents[2] / "gui-test-screenshots"
    output.mkdir(exist_ok=True)
    path = output / "hiking-plan-book-preview.pdf"
    path.write_bytes(_render_pdf_bytes(json.dumps(build_report_document(result), ensure_ascii=False)))
    print(path)
    import fitz
    with fitz.open(path) as pdf:
        print(f"pages={len(pdf)}")
        for index, page in enumerate(pdf, 1):
            preview = output / f"hiking-plan-book-preview-page-{index}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(preview)
            print(preview)


if __name__ == "__main__":
    main()
