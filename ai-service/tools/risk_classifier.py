"""工具风险分类模块。

定义 RiskLevel 枚举和工具到风险级别的映射。
用于在工具执行前进行风险评估和用户确认门控。
"""

from enum import Enum


class RiskLevel(str, Enum):
    """工具风险级别枚举。

    按执行风险从低到高排列：
      LOW      — 只读、轻量、无副作用的操作（搜索、PDF生成）
      MEDIUM   — 对外部资源有轻度副作用（网页抓取、下载）
      HIGH     — 对本地系统有显著副作用（文件读写、终止）
      CRITICAL — 可能造成不可逆损害的系统级操作（终端执行）
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 工具名称到风险级别的静态映射
TOOL_RISK_MAP: dict[str, RiskLevel] = {
    "web_search": RiskLevel.LOW,
    "web_scraping": RiskLevel.MEDIUM,
    "generate_pdf": RiskLevel.LOW,
    "route_research": RiskLevel.LOW,
    "hiking_knowledge_search": RiskLevel.LOW,
    "gear_checklist": RiskLevel.LOW,
    "risk_assessment": RiskLevel.LOW,
    # Ticket 29: server-approved AMap MCP tools (amap_weather / amap_map)
    # — read-only evidence lookups. Registered explicitly so the
    # fail-closed unclassified rule does not reject the profile's own
    # allowlisted surface.
    "get_current_weather": RiskLevel.LOW,
    "get_weather_forecast": RiskLevel.LOW,
    "search_poi": RiskLevel.LOW,
    "plan_route": RiskLevel.LOW,
    "geocode": RiskLevel.LOW,
}


def classify_tool(tool_name: str, default: RiskLevel = RiskLevel.MEDIUM) -> RiskLevel:
    """返回指定工具的风险级别。

    Args:
        tool_name: 工具名称（与 tools/ 中各模块的 key 一致）。
        default: 未在映射表中找到时的默认风险级别。

    Returns:
        对应的 RiskLevel 枚举值。
    """
    return TOOL_RISK_MAP.get(tool_name, default)


def requires_confirmation(level: RiskLevel) -> bool:
    """判断指定风险级别是否需要用户确认。

    HIGH 和 CRITICAL 级别在执行前需要用户显式确认。
    """
    return level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
