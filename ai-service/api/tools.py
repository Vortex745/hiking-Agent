from fastapi import APIRouter

from config import settings
from mcp.runtime import get_mcp_runtime

tools_router = APIRouter(prefix="/tools")

AIAgent = None
validate_tool_configuration = None


def _get_agent_class():
    global AIAgent
    if AIAgent is None:
        from agent.agent import AIAgent as loaded_agent

        AIAgent = loaded_agent
    return AIAgent


def _validate_tool_configuration():
    global validate_tool_configuration
    if validate_tool_configuration is None:
        from agent.agent import validate_tool_configuration as loaded_validator

        validate_tool_configuration = loaded_validator
    return validate_tool_configuration()


def _mcp_configured() -> bool:
    return bool(getattr(settings, "mcp_servers", None))


def _external_keys_status() -> dict[str, bool]:
    return {
        "openai_api_key": bool(settings.openai_api_key),
        "embedding_api_key": bool(settings.embedding_api_key),
        "rerank_api_key": bool(settings.rerank_api_key),
    }


@tools_router.get("")
async def list_tools(include_hidden: bool = False):
    agent_cls = _get_agent_class()
    registry = agent_cls.get_tool_registry()
    tools = registry.tools_api_response(include_hidden=include_hidden)
    return {
        "count": len(tools),
        "include_hidden": include_hidden,
        "tools": tools,
    }


@tools_router.get("/health")
async def tools_health():
    agent_cls = _get_agent_class()
    registry = agent_cls.get_tool_registry()
    visible = registry.list_tools(include_hidden=False)
    all_tools = registry.list_all_tools()
    configured = _mcp_configured()
    runtime_health = get_mcp_runtime().health()
    return {
        "status": "ok",
        "tools_total": len(all_tools),
        "visible_tools": len(visible),
        "hidden_tools": len(all_tools) - len(visible),
        "configuration": _validate_tool_configuration(),
        "mcp": {
            "configured": configured,
            "loaded": runtime_health["loaded"],
            "servers": runtime_health["servers"],
            "errors": runtime_health["errors"],
        },
        "external_keys": _external_keys_status(),
    }
