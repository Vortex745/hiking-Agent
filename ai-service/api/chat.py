import json
import logging
import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
    ConfirmResponse,
    PendingConfirmationItem,
    PendingConfirmationsResponse,
)
from api.confirmation_store import get_store
from agent.tool_names import OPENMANUS_TOOL_NAMES
from memory.base import ChatMemory
from memory.factory import chat_memory_status, get_chat_memory
from config import settings

logger = logging.getLogger("ai-service.chat")
chat_router = APIRouter(prefix="/chat")

AIAgent = None
tool_registry = None


def _get_agent_class():
    global AIAgent
    if AIAgent is None:
        from agent.agent import AIAgent as loaded_agent

        AIAgent = loaded_agent
    return AIAgent


def _get_tool_registry():
    global tool_registry
    if tool_registry is None:
        from agent.agent import tool_registry as loaded_registry

        tool_registry = loaded_registry
    return tool_registry


def _missing_llm_config_message() -> str:
    return "Agent 主模型未配置 API Key：请在 LLM 配置页保存大模型配置，或在 ai-service/.env 中配置 OPENAI_API_KEY。"


def _request_llm_config(req: ChatRequest):
    return req.model_settings.llm if req.model_settings and req.model_settings.llm else None


def _request_current_location(req: ChatRequest) -> dict | None:
    if not req.current_location:
        return None
    return req.current_location.model_dump(exclude_none=True)


def _has_llm_api_key(req: ChatRequest) -> bool:
    runtime_llm = _request_llm_config(req)
    return bool((runtime_llm and runtime_llm.api_key) or settings.openai_api_key)


def _get_memory(chat_id: str) -> ChatMemory:
    return get_chat_memory(chat_id=chat_id)


def _attach_confirmation_if_needed(event: dict, store, chat_id: str, step: int) -> None:
    """Attach confirmation_id for high-risk tool calls using the unified key."""
    metadata = event.get("metadata")
    if not (
        event.get("type") == "tool_call"
        and isinstance(metadata, dict)
        and metadata.get("needs_confirmation") is True
    ):
        return

    args = metadata.get("args_raw")
    if not isinstance(args, dict):
        args = metadata.get("args") if isinstance(metadata.get("args"), dict) else {}

    cid = store.add(
        tool_name=metadata.get("tool") or event.get("content", ""),
        args=args,
        chat_id=chat_id,
        step=step,
    )
    metadata["confirmation_id"] = cid


@chat_router.post("/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest):
    """Synchronous chat endpoint — returns the full response at once."""
    try:
        if not _has_llm_api_key(req):
            raise HTTPException(status_code=503, detail=_missing_llm_config_message())

        memory = _get_memory(req.chat_id)
        history = memory.get_messages()
        memory.add_message("user", req.message)

        agent_cls = _get_agent_class()
        agent = agent_cls(llm_config=_request_llm_config(req))
        kwargs = {}
        if req.scenario:
            kwargs["scenario"] = req.scenario
        if current_location := _request_current_location(req):
            kwargs["current_location"] = current_location
        result = await agent.aexecute(req.message, history, **kwargs)

        reply = result.get("output", str(result))
        memory.add_message("assistant", reply)

        return ChatResponse(content=reply, chat_id=req.chat_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat sync error")
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/health")
async def chat_health():
    status = "ok" if settings.openai_api_key else "unconfigured"
    return {
        "status": status,
        "module": "agent",
        "service": "ai-service",
        "tools": len(OPENMANUS_TOOL_NAMES),
        "memory": chat_memory_status(),
    }


@chat_router.post("/sse")
async def chat_sse(req: ChatRequest):
    """SSE streaming chat endpoint."""
    memory = _get_memory(req.chat_id)
    history = memory.get_messages()
    memory.add_message("user", req.message)

    async def event_stream():
        assistant_parts: list[str] = []
        done_sent = False
        step = 0
        store = get_store()
        try:
            if not _has_llm_api_key(req):
                yield f"data: {json.dumps({'type': 'error', 'content': _missing_llm_config_message()}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
                return

            agent_cls = _get_agent_class()
            agent = agent_cls(llm_config=_request_llm_config(req))
            kwargs = {}
            if req.scenario:
                kwargs["scenario"] = req.scenario
            if current_location := _request_current_location(req):
                kwargs["current_location"] = current_location
            stream = agent.aexecute_stream(req.message, history, **kwargs)
            async for event in stream:
                event_type = event.get("type")
                _attach_confirmation_if_needed(event, store=store, chat_id=req.chat_id, step=step)

                if event_type == "text":
                    assistant_parts.append(event.get("content", ""))
                if event_type == "done":
                    done_sent = True

                step += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            assistant_reply = "".join(assistant_parts).strip()
            if assistant_reply:
                memory.add_message("assistant", assistant_reply)
            if not done_sent:
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
        except Exception as e:
            logger.exception("SSE stream error")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            if not done_sent:
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.get("/history/{chat_id}")
async def chat_history(chat_id: str):
    """Get chat history for a given chat_id."""
    try:
        memory = _get_memory(chat_id)
        messages = memory.get_messages()
        return {"chat_id": chat_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.delete("/history/{chat_id}")
async def chat_clear_history(chat_id: str):
    """Clear persisted chat history for a given chat_id."""
    try:
        memory = _get_memory(chat_id)
        memory.clear()
        return {"chat_id": chat_id, "status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/confirm", response_model=ConfirmResponse)
async def chat_confirm(req: ConfirmRequest):
    """用户确认或拒绝某条高风险工具调用。"""
    store = get_store()
    action = req.action.strip().lower()

    if action not in ("confirm", "reject"):
        raise HTTPException(
            status_code=400,
            detail=f"无效操作 '{req.action}'，请使用 'confirm' 或 'reject'。",
        )

    rec = store.get(req.confirmation_id)
    if rec is None:
        return ConfirmResponse(status="not_found", confirmation_id=req.confirmation_id)

    if rec.status != "pending":
        return ConfirmResponse(
            status="already_resolved", confirmation_id=req.confirmation_id
        )

    if action == "confirm":
        store.confirm(req.confirmation_id)
        registry = _get_tool_registry()
        tool_result = await registry.execute_tool(rec.tool_name, rec.args)
        return ConfirmResponse(
            status="confirmed",
            confirmation_id=req.confirmation_id,
            tool_name=rec.tool_name,
            tool_result=tool_result,
        )
    else:
        store.reject(req.confirmation_id)
        return ConfirmResponse(status="rejected", confirmation_id=req.confirmation_id)


@chat_router.get("/pending/{chat_id}", response_model=PendingConfirmationsResponse)
async def chat_pending(chat_id: str):
    """获取某次对话中全部待确认的记录。"""
    store = get_store()
    pending = store.get_pending_by_chat(chat_id)
    items = [
        PendingConfirmationItem(
            confirmation_id=rec.confirmation_id,
            tool_name=rec.tool_name,
            args=rec.args,
            step=rec.step,
            status=rec.status,
            created_at=rec.created_at,
        )
        for rec in pending
    ]
    return PendingConfirmationsResponse(chat_id=chat_id, pending=items)
