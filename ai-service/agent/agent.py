import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from agent.intake import AgentIntent, AgentRequestContext, CurrentLocation, understand_request
from agent.hiking_manus import HikingManus
from agent.prompts import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from agent.react_guard import RepeatCallDetector, StuckDetector
from agent.runtime import AgentRunRecord, AgentState, ExecutionLane, ExitReason
from agent.task_exit import AgentTaskExitController
from agent.tool_names import OPENMANUS_TOOL_NAMES
from api.models import RuntimeLlmConfig
from config import settings
from memory import MemoryManager
from rag.text_processing import clean_display_text, emphasize_display_terms
from tools.file_operation import file_operation
from tools.hiking_domain import (
    gear_checklist,
    risk_assessment,
    route_research,
    trip_report_export,
)
from tools.hiking_knowledge import hiking_knowledge_search
from tools.pdf_generation import generate_pdf
from tools.resource_download import resource_download
from tools.risk_classifier import TOOL_RISK_MAP, RiskLevel
from tools.terminal import terminal
from tools.terminate import terminate
from tools.tool_registry import ToolMetadata, ToolRegistry
from tools.web_scraping import web_scraping
from tools.web_search import web_search

logger = logging.getLogger("ai-service.agent")

MAX_STEPS = 20
MAX_EVENT_CHARS = 1200
MAX_ARGS_CHARS = 500
ROUTE_RECOMMENDATION_PROMPT = "要不要我继续给你推荐附近的户外徒步路线？"
ROUTE_FOLLOWUP_MARKERS = (
    ROUTE_RECOMMENDATION_PROMPT,
    "要不要我继续给你推荐",
    "是否要推荐",
    "推荐附近的户外徒步路线",
)
AFFIRMATIVE_ROUTE_FOLLOWUP = (
    "需要",
    "要",
    "好的",
    "好",
    "可以",
    "行",
    "安排",
    "推荐",
    "继续",
)


AVAILABLE_TOOL_MAP = {
    "web_search": web_search,
    "web_scraping": web_scraping,
    "file_operation": file_operation,
    "resource_download": resource_download,
    "terminal": terminal,
    "generate_pdf": generate_pdf,
    "terminate": terminate,
    "route_research": route_research,
    "hiking_knowledge_search": hiking_knowledge_search,
    "gear_checklist": gear_checklist,
    "risk_assessment": risk_assessment,
    "trip_report_export": trip_report_export,
}

AVAILABLE_TOOLS = [AVAILABLE_TOOL_MAP[name] for name in OPENMANUS_TOOL_NAMES]

def _unique_tool_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen and name in AVAILABLE_TOOL_MAP:
            seen.add(name)
            result.append(name)
    return result


def validate_tool_configuration() -> dict[str, Any]:
    """Validate the HikingManus tool surface, registry, and risk map."""
    available_names = set(AVAILABLE_TOOL_MAP)
    registered_names = {md.name for md in tool_registry.list_all_tools()}
    risk_names = set(TOOL_RISK_MAP)
    openmanus_names = set(OPENMANUS_TOOL_NAMES)

    issues: list[dict[str, Any]] = []

    def add_issue(kind: str, names: set[str]) -> None:
        if names:
            issues.append({"kind": kind, "tools": sorted(names)})

    add_issue("available_not_registered", available_names - registered_names)
    add_issue("registered_not_available", registered_names - available_names)
    add_issue("available_missing_risk", available_names - risk_names)
    add_issue("openmanus_not_available", openmanus_names - available_names)

    return {
        "ok": not issues,
        "issues": issues,
        "available_count": len(available_names),
        "registered_count": len(registered_names),
        "risk_count": len(risk_names & available_names),
        "openmanus_tool_count": len(openmanus_names),
    }


def _approval_required_payload(tool_name: str, args: dict[str, Any]) -> str:
    payload = {
        "type": "approval_required",
        "tool": tool_name,
        "args": args,
        "message": f"工具 {tool_name} 需要用户确认后才能执行。",
    }
    return json.dumps(payload, ensure_ascii=False)


def _guard_tool_for_confirmation(tool):
    tool_name = getattr(tool, "name", None)
    if not tool_name:
        return tool
    md = tool_registry.get(tool_name)
    if md is None or not tool_registry.needs_confirmation(tool_name):
        return tool

    async def guarded_runner(**kwargs) -> str:
        return _approval_required_payload(tool_name, kwargs)

    return StructuredTool.from_function(
        name=tool_name,
        description=getattr(tool, "description", ""),
        args_schema=getattr(tool, "args_schema", None),
        coroutine=guarded_runner,
    )


def apply_tool_confirmation_guards(tools: list) -> list:
    """Wrap high-risk tools so the model cannot execute side effects before approval."""
    return [_guard_tool_for_confirmation(tool) for tool in tools]


tool_registry = ToolRegistry()
tool_registry.register_many([
    ToolMetadata(
        name="web_search",
        description="Search the web for information relevant to the user's query or task.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=30,
        domain="web",
        scenarios=("general_chat", "route_plan"),
        result_policy="compact",
    ),
    ToolMetadata(
        name="web_scraping",
        description="Scrape and extract main content from a web page URL.",
        parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        risk_level=RiskLevel.MEDIUM,
        rate_limit_per_minute=20,
        domain="web",
        scenarios=("general_chat", "route_plan"),
        result_policy="compact",
    ),
    ToolMetadata(
        name="file_operation",
        description="Create, read, update, delete, or list files/directories in the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["create", "read", "update", "delete", "list", "write", "mkdir"]},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["operation", "path"],
        },
        risk_level=RiskLevel.HIGH,
        rate_limit_per_minute=20,
        domain="filesystem",
        scenarios=("report_export",),
        auto_allowed=False,
        result_policy="artifact",
    ),
    ToolMetadata(
        name="resource_download",
        description="Download a resource from a URL to local workspace.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "save_path": {"type": "string"},
            },
            "required": ["url"],
        },
        risk_level=RiskLevel.MEDIUM,
        rate_limit_per_minute=10,
        domain="filesystem",
        auto_allowed=False,
        result_policy="artifact",
    ),
    ToolMetadata(
        name="terminal",
        description="Execute a command in the local terminal/shell environment.",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        risk_level=RiskLevel.CRITICAL,
        rate_limit_per_minute=10,
        domain="system",
        auto_allowed=False,
        result_policy="raw",
    ),
    ToolMetadata(
        name="generate_pdf",
        description="Generate a PDF document from text content.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["title", "content"],
        },
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=10,
        domain="artifact",
        scenarios=("report_export",),
        result_policy="artifact",
    ),
    ToolMetadata(
        name="terminate",
        description="Call this when the task is complete or cannot continue further.",
        parameters={"type": "object", "properties": {"reason": {"type": "string"}}},
        risk_level=RiskLevel.HIGH,
        needs_confirmation=False,
        rate_limit_per_minute=60,
        domain="control",
        result_policy="compact",
    ),
    ToolMetadata(
        name="route_research",
        description="Collect hiking route facts from search queries, web reading, and RAG evidence.",
        parameters={
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "date": {"type": "string"},
                "days": {"type": "integer"},
                "focus": {"type": "string"},
            },
            "required": ["destination"],
        },
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=20,
        domain="hiking",
        scenarios=("route_plan", "risk_assessment", "report_export"),
        result_policy="compact",
        hidden=True,
    ),
    ToolMetadata(
        name="hiking_knowledge_search",
        description="Search the hiking RAG knowledge base and return traceable evidence chunks.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=30,
        domain="hiking",
        scenarios=("knowledge_qa", "route_plan", "gear_check", "risk_assessment", "report_export"),
        result_policy="compact",
        hidden=True,
    ),
    ToolMetadata(
        name="gear_checklist",
        description="Generate a conservative hiking gear checklist.",
        parameters={
            "type": "object",
            "properties": {
                "days": {"type": "integer"},
                "season": {"type": "string"},
                "experience": {"type": "string"},
                "camping": {"type": "boolean"},
                "gear_level": {"type": "string"},
            },
        },
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=30,
        domain="hiking",
        scenarios=("gear_check", "route_plan", "report_export"),
        result_policy="compact",
        hidden=True,
    ),
    ToolMetadata(
        name="risk_assessment",
        description="Assess hiking risk with conservative rule-based guardrails.",
        parameters={
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "weather": {"type": "string"},
                "route": {"type": "string"},
                "experience": {"type": "string"},
                "days": {"type": "integer"},
            },
        },
        risk_level=RiskLevel.LOW,
        rate_limit_per_minute=30,
        domain="hiking",
        scenarios=("gear_check", "route_plan", "risk_assessment", "report_export"),
        result_policy="compact",
        hidden=True,
    ),
    ToolMetadata(
        name="trip_report_export",
        description="Export a hiking trip report to Markdown and optionally PDF.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "format": {"type": "string", "enum": ["markdown", "md", "pdf", "both"]},
                "file_name": {"type": "string"},
            },
            "required": ["title", "content"],
        },
        risk_level=RiskLevel.HIGH,
        rate_limit_per_minute=10,
        domain="artifact",
        scenarios=("report_export",),
        auto_allowed=False,
        result_policy="artifact",
        hidden=True,
    ),
])

# Register tool instances into the registry so execute_tool works without tool_map
for _tool_name, _tool_instance in AVAILABLE_TOOL_MAP.items():
    if _tool_name in tool_registry:
        tool_registry._instances[_tool_name] = _tool_instance


def _compact_text(value: Any, limit: int = MAX_EVENT_CHARS) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _strip_leading_numbered_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    stripped = []
    for line in lines:
        stripped.append(re.sub(r"^\d+[.)、]\s*", "", line).strip())
    return "\n".join(line for line in stripped if line).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response that should contain only JSON."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _artifact_label(kind: str) -> str:
    labels = {
        "pdf": "PDF",
        "markdown": "Markdown",
    }
    return labels.get(kind.lower(), "文件")


def _artifact_events_from_tool_result(tool_name: str, content: str, *, step: int) -> list[dict[str, Any]]:
    payload = _parse_json_object(content)
    if not payload:
        return []

    artifacts: list[dict[str, Any]] = []
    raw_artifacts = payload.get("artifacts")
    if isinstance(raw_artifacts, list):
        artifacts.extend(item for item in raw_artifacts if isinstance(item, dict))

    raw_artifact = payload.get("artifact")
    if isinstance(raw_artifact, dict):
        artifacts.append(raw_artifact)

    raw_pdf_result = payload.get("pdf_result")
    if isinstance(raw_pdf_result, dict) and isinstance(raw_pdf_result.get("artifact"), dict):
        artifacts.append(raw_pdf_result["artifact"])

    seen_urls: set[str] = set()
    events: list[dict[str, Any]] = []
    for artifact in artifacts:
        download_url = artifact.get("download_url")
        filename = artifact.get("filename")
        if not isinstance(download_url, str) or not isinstance(filename, str):
            continue
        if download_url in seen_urls:
            continue
        seen_urls.add(download_url)
        kind = str(artifact.get("kind") or "file")
        metadata = {
            "step": step,
            "tool": tool_name,
            "runtime": "openmanus",
            "kind": kind,
            "title": artifact.get("title"),
            "filename": filename,
            "download_url": download_url,
            "mime_type": artifact.get("mime_type"),
            "size_bytes": artifact.get("size_bytes"),
        }
        events.append({
            "type": "artifact",
            "content": f"{_artifact_label(kind)} 已生成：{filename}",
            "metadata": metadata,
        })
    return events


def _pexels_markdown_previews_from_tool_result(tool_name: str, content: str, *, limit: int = 3) -> list[str]:
    if "pexels" not in (tool_name or "").lower():
        return []

    payload = _parse_json_object(content)
    if not payload:
        return []

    photos = payload.get("photos")
    if not isinstance(photos, list):
        return []

    previews: list[str] = []
    seen: set[str] = set()
    for photo in photos:
        if not isinstance(photo, dict):
            continue
        preview = photo.get("markdown_preview")
        if not isinstance(preview, str):
            continue
        preview = preview.strip()
        if not preview.startswith("![") or "](" not in preview or preview in seen:
            continue
        previews.append(preview)
        seen.add(preview)
        if len(previews) >= limit:
            break
    return previews


def _append_markdown_previews(text: str, previews: list[str]) -> str:
    if not previews:
        return text
    missing = [preview for preview in previews if preview and preview not in text]
    if not missing:
        return text
    return f"{text.rstrip()}\n\n" + "\n".join(missing)


class AIAgent:
    """HikingManus-style Agent with hiking-aware prompts and tools."""

    def __init__(self, memory_manager=None, llm_config: RuntimeLlmConfig | None = None):
        self.max_steps = MAX_STEPS
        self.exit_controller = AgentTaskExitController(max_steps=self.max_steps)
        base_url = llm_config.base_url if llm_config and llm_config.base_url else settings.openai_base_url
        api_key = llm_config.api_key if llm_config and llm_config.api_key else settings.openai_api_key
        model = llm_config.model if llm_config and llm_config.model else settings.openai_model
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.7,
        )
        if memory_manager is None and settings.memory_enabled:
            from memory import MemoryConfig

            memory_model = model if llm_config and llm_config.model else None
            config = MemoryConfig(
                compressor_model=memory_model or settings.memory_compressor_model,
                extractor_model=memory_model or settings.memory_extractor_model,
                llm_base_url=base_url,
                llm_api_key=api_key,
                vector_store_path=settings.memory_store_path,
                top_k=settings.memory_top_k,
            )
            self.memory_manager = MemoryManager(config=config)
        else:
            self.memory_manager = memory_manager
        self._memory_context: dict[str, str] = {
            "session_context": "",
            "knowledge_context": "",
        }

    @staticmethod
    def get_tool_registry() -> ToolRegistry:
        return tool_registry

    def _merge_run_record_metadata(self, base_metadata: dict[str, Any], run_record: AgentRunRecord) -> dict[str, Any]:
        """Merge run_record.to_metadata() into base_metadata without overwriting existing keys."""
        record_meta = run_record.to_metadata()
        merged = dict(base_metadata)
        for key, value in record_meta.items():
            if key not in merged:
                merged[key] = value
        for key, value in run_record.metadata.items():
            if key in {"status", "reason"}:
                continue
            if key not in merged:
                merged[key] = value
        merged.setdefault(
            "openmanus_runtime",
            {
                "base": "BaseAgent",
                "react": "ReActAgent",
                "tool_call": "ToolCallAgent",
                "manus": "HikingManus",
            },
        )
        logger.info(
            "AgentRunRecord: lane=%s exit_reason=%s step_count=%s tool_count=%s selected_tools=%s duration_ms=%.1f",
            record_meta.get("lane"),
            record_meta.get("exit_reason"),
            record_meta.get("step_count"),
            record_meta.get("tool_count"),
            record_meta.get("selected_tools"),
            record_meta.get("duration_ms", 0),
        )
        return merged

    def _build_system_prompt(self, context: AgentRequestContext, selected_tools: list) -> str:
        slot_lines = []
        for key, value in context.slots.to_dict().items():
            if value is not None:
                slot_lines.append(f"- {key}: {value}")
        slots_text = "\n".join(slot_lines) if slot_lines else "- 未抽取到稳定槽位"
        tool_names = "、".join(getattr(tool, "name", "unknown_tool") for tool in selected_tools) or "无"
        location_text = "<Location>\n- 未提供当前位置\n</Location>"
        if context.current_location:
            location_items = [
                f"- {key}: {value}"
                for key, value in context.current_location.to_dict().items()
                if value is not None and value != ""
            ]
            location_text = "<Location>\n" + ("\n".join(location_items) or "- 已授权但坐标为空") + "\n</Location>"
        rewritten_text = context.rewritten_query or context.raw_query
        prefetch_text = "<PrefetchedEvidence>\n- 无\n</PrefetchedEvidence>"
        if context.prefetched_tool_results:
            result_lines = [
                json.dumps(result, ensure_ascii=False, default=str)
                for result in context.prefetched_tool_results
            ]
            prefetch_text = "<PrefetchedEvidence>\n" + "\n".join(result_lines) + "\n</PrefetchedEvidence>"

        system_msg = (
            f"{SYSTEM_PROMPT}\n\n"
            "<RuntimeContext>\n"
            f"- raw_query: {context.raw_query}\n"
            f"- rewritten_query: {rewritten_text}\n"
            f"- intent: {context.intent.value}\n"
            f"- scenario: {context.scenario or '未显式指定'}\n"
            f"- selected_tools: {tool_names}\n"
            f"- missing_slots: {', '.join(context.missing_slots) if context.missing_slots else '无'}\n"
            f"- clarifying_question: {context.clarifying_question or '无'}\n"
            "</RuntimeContext>\n\n"
            "<Slots>\n"
            f"{slots_text}\n"
            "</Slots>\n\n"
            f"{location_text}\n\n"
            f"{prefetch_text}\n\n"
            "<ExecutionRules>\n"
            "- 缺少目的地、日期等关键条件时先追问，不要编造路线或天气。\n"
            "- 当用户询问今天/当前位置/附近天气是否适合徒步，且当前定位存在时，不要追问城市；如未预取，从 selected_tools 中选择最合适的 raw MCP 地理或天气工具。\n"
            "- 如当前定位只有 latitude/longitude，调用 raw MCP 地理工具时直接传 latitude 和 longitude；天气工具可使用地理工具返回的 adcode、city 或坐标。\n"
            "- 如 PrefetchedEvidence 已包含定位和天气结果，本轮不要重复查询；可直接回答或最多调用一次 risk_assessment。\n"
            "- 户外安全建议优先引用已预取证据、raw MCP 天气结果、hiking_knowledge_search、route_research 等证据。\n"
            "- 最终回答使用自然中文，不要 Markdown 标题或编号格式；但对关键信息使用 **加粗** 标记（每段 2-5 处）。\n"
            "- 普通徒步场景不要要求终端、下载或任意文件操作。\n"
            "- 生成文档时先组织 Markdown，再按需生成 PDF，并说明路径和影响范围。\n"
            "</ExecutionRules>\n\n"
            f"<NextStep>\n{NEXT_STEP_PROMPT}\n</NextStep>"
        )

        if session_ctx := self._memory_context.get("session_context", ""):
            system_msg += f"\n\n<SessionSummary>\n{session_ctx}\n</SessionSummary>"
        location_text = "<Location>\n- 未提供当前位置\n</Location>"
        if context.current_location:
            location_items = [
                f"- {key}: {value}"
                for key, value in context.current_location.to_dict().items()
                if value is not None and value != ""
            ]
            location_text = "<Location>\n" + ("\n".join(location_items) or "- 已授权但坐标为空") + "\n</Location>"
        rewritten_text = context.rewritten_query or context.raw_query
        prefetch_text = "<PrefetchedEvidence>\n- 无\n</PrefetchedEvidence>"
        if context.prefetched_tool_results:
            result_lines = [
                json.dumps(result, ensure_ascii=False, default=str)
                for result in context.prefetched_tool_results
            ]
            prefetch_text = "<PrefetchedEvidence>\n" + "\n".join(result_lines) + "\n</PrefetchedEvidence>"

        system_msg = (
            f"{SYSTEM_PROMPT}\n\n"
            "<RuntimeContext>\n"
            f"- raw_query: {context.raw_query}\n"
            f"- rewritten_query: {rewritten_text}\n"
            f"- intent: {context.intent.value}\n"
            f"- scenario: {context.scenario or '未显式指定'}\n"
            f"- selected_tools: {tool_names}\n"
            f"- missing_slots: {', '.join(context.missing_slots) if context.missing_slots else '无'}\n"
            f"- clarifying_question: {context.clarifying_question or '无'}\n"
            "</RuntimeContext>\n\n"
            "<Slots>\n"
            f"{slots_text}\n"
            "</Slots>\n\n"
            f"{location_text}\n\n"
            f"{prefetch_text}\n\n"
            "<ExecutionRules>\n"
            "- 缺少目的地、日期等关键条件时先追问，不要编造路线或天气。\n"
            "- 当用户询问今天/当前位置/附近天气是否适合徒步，且当前定位存在时，不要追问城市；如未预取，从 selected_tools 中选择最合适的 raw MCP 地理或天气工具。\n"
            "- 如当前定位只有 latitude/longitude，调用 raw MCP 地理工具时直接传 latitude 和 longitude；天气工具可使用地理工具返回的 adcode、city 或坐标。\n"
            "- 如 PrefetchedEvidence 已包含定位和天气结果，本轮不要重复查询；可直接回答或最多调用一次 risk_assessment。\n"
            "- 户外安全建议优先引用已预取证据、raw MCP 天气结果、hiking_knowledge_search、route_research 等证据。\n"
            "- 最终回答使用自然中文，不要 Markdown 标题或编号格式；但对关键信息使用 **加粗** 标记（每段 2-5 处）。\n"
            "- 普通徒步场景不要要求终端、下载或任意文件操作。\n"
            "- 生成文档时先组织 Markdown，再按需生成 PDF，并说明路径和影响范围。\n"
            "</ExecutionRules>\n\n"
            f"<NextStep>\n{NEXT_STEP_PROMPT}\n</NextStep>"
        )

        if session_ctx := self._memory_context.get("session_context", ""):
            system_msg += f"\n\n<SessionSummary>\n{session_ctx}\n</SessionSummary>"
        if knowledge_ctx := self._memory_context.get("knowledge_context", ""):
            system_msg += f"\n\n<LongTermMemory>\n{knowledge_ctx}\n</LongTermMemory>"
        return system_msg

    def _clean_final_answer(self, text: str) -> str:
        cleaned = clean_display_text(
            text,
            preserve_lines=True,
            keep_list_markers=True,
            keep_markdown_emphasis=True,
        )
        return emphasize_display_terms(cleaned)

    def _infer_followup_scenario(self, message: str, history: list | None, scenario: str | None) -> str | None:
        if scenario:
            return scenario
        if self._is_route_recommendation_followup(message, history):
            return AgentIntent.ROUTE_PLAN.value
        return None

    def _is_route_recommendation_followup(self, message: str, history: list | None) -> bool:
        text = " ".join((message or "").strip().split())
        if not text or not any(word == text or word in text for word in AFFIRMATIVE_ROUTE_FOLLOWUP):
            return False
        for msg in reversed(history or []):
            if msg.get("role") != "assistant":
                continue
            content = str(msg.get("content", ""))
            return any(marker in content for marker in ROUTE_FOLLOWUP_MARKERS)
        return False

    async def _build_hiking_manus(self, context: AgentRequestContext) -> HikingManus:
        selected_tools = self._openmanus_tools()
        runtime = await HikingManus.create(
            llm=self.llm,
            tools=apply_tool_confirmation_guards(selected_tools),
            max_steps=self.max_steps,
            mcp_server_configs=getattr(settings, "mcp_servers", {}) or {},
        )
        runtime.system_prompt = self._build_system_prompt(context, runtime.available_tools.to_langchain_tools())
        return runtime

    def _openmanus_tool_names(self, context: AgentRequestContext | None = None) -> list[str]:
        return _unique_tool_names(OPENMANUS_TOOL_NAMES)

    def _openmanus_tools(self, context: AgentRequestContext | None = None) -> list:
        return [AVAILABLE_TOOL_MAP[name] for name in self._openmanus_tool_names(context)]

    def _load_openmanus_history(self, runtime: HikingManus, history: list | None) -> None:
        if not history:
            return
        for item in history[-10:]:
            role = item.get("role", "") if isinstance(item, dict) else ""
            content = item.get("content", "") if isinstance(item, dict) else ""
            if not content:
                continue
            if role == "user":
                runtime.memory.add_message(HumanMessage(content=content))
            elif role == "assistant":
                runtime.memory.add_message(AIMessage(content=content))

    def _approval_event_from_openmanus_result(
        self,
        tool_name: str,
        content: str,
        *,
        step: int,
        call_id: str,
        args: dict[str, Any],
    ) -> dict | None:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            return None
        payload = _parse_json_object(content[start : end + 1])
        if not payload or payload.get("type") != "approval_required":
            return None
        req = tool_registry.get_call_request(tool_name, args)
        risk_level = req.risk_level.value if req else RiskLevel.MEDIUM.value
        return {
            "type": "approval_required",
            "content": payload.get("message") or f"工具 {tool_name} 需要用户确认后才能执行。",
            "metadata": {
                "step": step,
                "tool": tool_name,
                "args_raw": args,
                "tool_call_id": call_id,
                "risk_level": risk_level,
                "needs_confirmation": True,
                "runtime": "openmanus",
            },
        }

    def _approval_event_for_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        step: int,
        call_id: str,
    ) -> dict:
        req = tool_registry.get_call_request(tool_name, args)
        risk_level = req.risk_level.value if req else RiskLevel.MEDIUM.value
        return {
            "type": "approval_required",
            "content": f"工具 {tool_name} 需要用户确认后才能执行。",
            "metadata": {
                "step": step,
                "tool": tool_name,
                "args_raw": args,
                "tool_call_id": call_id,
                "risk_level": risk_level,
                "needs_confirmation": True,
                "runtime": "openmanus",
            },
        }

    async def _aexecute_openmanus(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: CurrentLocation | dict | None = None,
    ) -> dict:
        events = [
            event
            async for event in self._aexecute_stream_openmanus(
                message,
                history=history,
                scenario=scenario,
                current_location=current_location,
            )
        ]
        text = "".join(event.get("content", "") for event in events if event.get("type") == "text").strip()
        done = next((event for event in reversed(events) if event.get("type") == "done"), {})
        tool_calls = [
            {
                "tool": event.get("metadata", {}).get("tool"),
                "args": event.get("metadata", {}).get("args_raw") or event.get("metadata", {}).get("args"),
            }
            for event in events
            if event.get("type") == "tool_call"
        ]
        metadata = done.get("metadata", {}) if isinstance(done, dict) else {}
        return {
            "output": text or "任务已完成。",
            "intermediate_steps": tool_calls,
            "exit_status": metadata.get("status", "completed"),
            "exit_reason": metadata.get("reason", "openmanus_completed"),
            "metadata": metadata,
        }

    async def _aexecute_stream_openmanus(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: CurrentLocation | dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        run_record = AgentRunRecord()
        run_record.transition(AgentState.EXECUTING)
        run_record.lane = ExecutionLane.REACT

        effective_scenario = self._infer_followup_scenario(message, history, scenario)
        context = understand_request(message, scenario=effective_scenario, current_location=current_location)
        await self._inject_memory_context(history, message)

        runtime = await self._build_hiking_manus(context)
        run_record.selected_tools = [
            getattr(tool, "name", "unknown_tool")
            for tool in runtime.available_tools.tools
        ]
        self._load_openmanus_history(runtime, history)
        runtime.update_memory("user", message)

        yield {
            "type": "thought",
            "content": "HikingManus 第 0 步：初始化 BaseAgent/ReActAgent/ToolCallAgent/HikingManus 运行时。",
            "metadata": {
                "phase": "openmanus_start",
                "runtime": "openmanus",
                "max_steps": runtime.max_steps,
                "tools": run_record.selected_tools,
            },
        }

        assistant_parts: list[str] = []
        image_markdown_previews: list[str] = []
        repeat_detector = RepeatCallDetector()
        stuck_detector = StuckDetector()

        try:
            while runtime.current_step < runtime.max_steps and runtime.state.value != "FINISHED":
                runtime.current_step += 1
                step = runtime.current_step
                yield {
                    "type": "thought",
                    "content": f"HikingManus 第 {step} 步：think() 判断下一步动作。",
                    "metadata": {"step": step, "phase": "think", "runtime": "openmanus"},
                }

                should_act = await runtime.think()
                if not should_act or not runtime.tool_calls:
                    content = runtime._message_text(runtime.messages[-1]) if runtime.messages else ""
                    final_text = _append_markdown_previews(
                        self._clean_final_answer(content or "任务已完成。"),
                        image_markdown_previews,
                    )
                    assistant_parts.append(final_text)
                    stuck_detector.record_assistant_content(final_text)
                    run_record.complete(ExitReason.NATURAL_END, current_step=step)
                    yield {
                        "type": "text",
                        "content": final_text,
                        "metadata": {"phase": "final_answer", "runtime": "openmanus"},
                    }
                    await self._commit_memory(history, message, "".join(assistant_parts).strip(), context)
                    yield self.exit_controller.done_event(
                        "completed",
                        "openmanus_completed",
                        **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
                    )
                    return

                normalized_calls: list[tuple[str, dict[str, Any], str]] = []
                for call in runtime.tool_calls:
                    name, args, call_id = runtime._normalize_tool_call(call)
                    if repeat_detector.is_repeat(name, args):
                        continue
                    repeat_detector.record(name, args)
                    normalized_calls.append((name, args, call_id))

                if not normalized_calls:
                    run_record.complete(ExitReason.STUCK, current_step=step, reason="repeat_call_detected")
                    yield {"type": "text", "content": "检测到重复调用相同工具，已按 HikingManus 运行时停止。"}
                    yield self.exit_controller.done_event(
                        "stuck",
                        "repeat_call_detected",
                        **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
                    )
                    return

                yield {
                    "type": "thought",
                    "content": f"HikingManus 第 {step} 步：act() 执行工具：{'、'.join(name for name, _, _ in normalized_calls)}。",
                    "metadata": {
                        "step": step,
                        "phase": "act",
                        "runtime": "openmanus",
                        "tools": [name for name, _, _ in normalized_calls],
                    },
                }

                for name, args, call_id in normalized_calls:
                    req = tool_registry.get_call_request(name, args)
                    risk_level = req.risk_level.value if req else RiskLevel.MEDIUM.value
                    needs_confirmation = req.needs_confirmation if req else False
                    yield {
                        "type": "tool_call",
                        "content": f"HikingManus 第 {step} 步：调用 {name}，参数：{_compact_text(args, MAX_ARGS_CHARS)}",
                        "metadata": {
                            "step": step,
                            "tool": name,
                            "args": _compact_text(args, MAX_ARGS_CHARS),
                            "args_raw": args,
                            "tool_call_id": call_id,
                            "risk_level": risk_level,
                            "needs_confirmation": needs_confirmation,
                            "runtime": "openmanus",
                        },
                    }
                    if needs_confirmation:
                        yield self._approval_event_for_tool_call(name, args, step=step, call_id=call_id)
                        run_record.complete(ExitReason.CLARIFICATION_NEEDED, tool=name, step=step)
                        yield self.exit_controller.done_event(
                            "waiting_for_user",
                            "tool_confirmation_required",
                            **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
                        )
                        return

                    tool_started_at = time.monotonic()
                    result_text = await runtime.execute_tool(name, args)
                    run_record.record_tool_result({
                        "tool_name": name,
                        "args": args,
                        "result": result_text,
                        "error": None if "Error:" not in str(result_text) else str(result_text),
                        "duration_ms": (time.monotonic() - tool_started_at) * 1000,
                        "step_index": step,
                    })
                    for preview in _pexels_markdown_previews_from_tool_result(name, result_text):
                        if preview not in image_markdown_previews:
                            image_markdown_previews.append(preview)
                    runtime.memory.add_message(ToolMessage(content=result_text, name=name, tool_call_id=call_id or name))
                    stuck_detector.record_observation(name, result_text)
                    if stuck_detector.is_stuck():
                        reason = stuck_detector.stuck_reason()
                        run_record.complete(ExitReason.STUCK, current_step=step, reason=reason)
                        yield {"type": "text", "content": f"检测到循环卡住（{reason}），已自动停止。"}
                        yield self.exit_controller.done_event(
                            "stuck",
                            "openmanus_stuck",
                            **self._merge_run_record_metadata({"runtime": "openmanus", "stuck_reason": reason}, run_record),
                        )
                        return

                    approval_event = self._approval_event_from_openmanus_result(
                        name,
                        result_text,
                        step=step,
                        call_id=call_id,
                        args=args,
                    )
                    if approval_event:
                        # 与 _approval_event_for_tool_call 前置拦截路径（needs_confirmation 分支）
                        # 行为对齐：emit approval_required 后停止流程，等待用户确认。
                        # 否则前端看到 approval_required 事件但 SSE 流继续执行下一轮
                        # LLM 响应，语义不一致。
                        yield approval_event
                        run_record.complete(ExitReason.CLARIFICATION_NEEDED, tool=name, step=step)
                        yield self.exit_controller.done_event(
                            "waiting_for_user",
                            "tool_confirmation_required",
                            **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
                        )
                        return
                    else:
                        yield {
                            "type": "tool_result",
                            "content": f"HikingManus 第 {step} 步：{name} 返回：{_compact_text(result_text)}",
                            "metadata": {"step": step, "tool": name, "runtime": "openmanus"},
                        }
                        for artifact_event in _artifact_events_from_tool_result(name, result_text, step=step):
                            yield artifact_event

                    exit_result = self.exit_controller.from_tool_result(
                        name,
                        result_text,
                        current_step=step,
                    )
                    if exit_result:
                        exit_reason = ExitReason.TERMINATE_TOOL if name == "terminate" else ExitReason.NATURAL_END
                        if exit_result.status.value == "waiting_for_user":
                            exit_reason = ExitReason.CLARIFICATION_NEEDED
                        run_record.complete(exit_reason, tool=name, step=step)
                        text_event = exit_result.text_event()
                        if text_event:
                            assistant_parts.append(text_event.get("content", ""))
                            yield text_event
                        await self._commit_memory(history, message, "".join(assistant_parts).strip(), context)
                        yield self.exit_controller.done_event(
                            exit_result.status.value,
                            exit_result.reason,
                            **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
                        )
                        return

            run_record.complete(ExitReason.BUDGET_EXHAUSTED, current_step=runtime.current_step)
            yield {"type": "text", "content": self.exit_controller._budget_exhausted_message(context)}
            yield self.exit_controller.done_event(
                "budget_exhausted",
                "step_budget_exhausted",
                **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
            )
        except Exception as e:
            logger.exception("HikingManus Agent stream error")
            run_record.complete(ExitReason.ERROR, error=str(e))
            yield {"type": "error", "content": f"执行出错: {str(e)}"}
            yield self.exit_controller.done_event(
                "error",
                "openmanus_error",
                **self._merge_run_record_metadata({"runtime": "openmanus"}, run_record),
            )
        finally:
            try:
                await runtime.cleanup()
            except Exception:
                logger.warning("HikingManus cleanup failed", exc_info=True)

    async def _inject_memory_context(self, history: list | None, message: str) -> None:
        if self.memory_manager and settings.memory_enabled:
            try:
                ctx = await asyncio.to_thread(self.memory_manager.build_runtime_context, history or [], message)
                self._memory_context["session_context"] = ctx.get("session_context", "")
                self._memory_context["knowledge_context"] = ctx.get("knowledge_context", "")
            except Exception:
                logger.warning("Memory context injection failed", exc_info=True)

    async def _commit_memory(
        self,
        history: list | None,
        message: str,
        final_response: str,
        context: AgentRequestContext,
    ) -> None:
        if not (self.memory_manager and settings.memory_enabled and final_response):
            return
        commit_history = [*(history or []), {"role": "user", "content": message}]
        commit_history.append({"role": "assistant", "content": final_response})
        task_state = {"slots": context.slots.to_dict(), "intent": context.intent.value}
        try:
            await asyncio.to_thread(
                self.memory_manager.commit_interaction,
                commit_history,
                message,
                final_response,
                task_state,
            )
        except Exception:
            logger.warning("Memory commit failed", exc_info=True)

    async def aexecute(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: CurrentLocation | dict | None = None,
    ) -> dict:
        return await self._aexecute_openmanus(
            message,
            history=history,
            scenario=scenario,
            current_location=current_location,
        )

    async def aexecute_stream(
        self,
        message: str,
        history: list | None = None,
        scenario: str | None = None,
        current_location: CurrentLocation | dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        async for event in self._aexecute_stream_openmanus(
            message,
            history=history,
            scenario=scenario,
            current_location=current_location,
        ):
            yield event
