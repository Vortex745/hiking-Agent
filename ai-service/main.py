import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_v2.repositories.memory import MemoryStore
from auth.deps import get_current_user
from auth.router import auth_router
from auth.security import load_or_create_secret
from auth.service import AuthService
from auth.store import AuthStore
from config import settings
from observability.logging_setup import RequestContextMiddleware, configure_logging

configure_logging()
logger = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ai-hiking AI Service...")
    logger.info(f"OpenAI Base URL: {settings.openai_base_url}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    if settings.rag_docs_api_url:
        logger.info(f"RAG Docs API: {settings.rag_docs_api_url}")

    # 可观测性初始化：追踪关闭或凭据缺失时保持 no-op，不阻止启动。
    from observability import tracing

    tracing.get_tracer()

    # 账号认证：独立 SQLite 与 JWT secret；secret 默认自动生成并落盘。
    auth_secret = settings.auth_jwt_secret or load_or_create_secret(
        Path(settings.auth_secret_path)
    )
    auth_store = AuthStore(settings.auth_db_path)

    app.state.auth_service = AuthService(
        auth_store,
        MemoryStore(str(Path(settings.agent_v2_state_dir) / "agent_memory.sqlite")),
        secret=auth_secret,
        ttl_seconds=settings.auth_token_ttl_seconds,
    )
    logger.info("✓ Auth service assembled")

    # Ticket 03：Chat v2 启动守护 —— Vercel + v2 enabled 必须失败，不使用临时文件系统
    # 守护在所有其他装配之前执行，确保 v2 永远不会在不支持它的环境中启动
    from agent_v2.startup import V2StartupError, assemble_v2_runtime, check_v2_startup_guard

    try:
        check_v2_startup_guard(settings)
    except V2StartupError as exc:
        logger.error(f"✗ Chat v2 startup guard failed: {exc}")
        raise

    # Ticket 03：装配 v2 运行时（disabled 时返回 None，不创建 checkpoint/sandbox/MCP）
    app.state.agent_v2_runtime = None
    if settings.agent_v2_enabled:
        try:
            app.state.agent_v2_runtime = assemble_v2_runtime(settings)
            if app.state.agent_v2_runtime is not None:
                logger.info("✓ Chat v2 runtime assembled (Ticket 06: persistence + writer lock active)")
                # 暂停超时回收：弃用的审批暂停 run 会一直占住 CapacityGuard
                # 沙箱槽位，周期 sweep 到期释放。Ticket 41：janitor 实现归属
                # 编排 module（agent_v2.run_orchestration），装配只负责接线。
                from agent_v2.run_orchestration import start_paused_run_sweeper

                app.state.paused_run_sweeper = start_paused_run_sweeper(
                    app.state.agent_v2_runtime
                )
            else:
                logger.error("✗ Chat v2 enabled but assembly returned None")
        except V2StartupError as exc:
            logger.error(f"✗ Chat v2 assembly blocked by guard: {exc}")
            raise
        except Exception as exc:
            logger.error(f"✗ Chat v2 assembly failed: {exc}", exc_info=True)
            # 失败时 runtime 保持 None，/health 会报告 failed
    else:
        logger.warning("Chat v2 disabled — v1 已移除，chat 链路将不可用")

    # 知识源：本地单进程调度。Vercel 没有可写持久化，管理面也会被本机守卫拒绝。
    app.state.datasource_service = None
    if not os.getenv("VERCEL"):
        try:
            from datasource.runtime import build_datasource_service

            app.state.datasource_service = build_datasource_service(settings)
            app.state.datasource_service.start_scheduler()
            logger.info("✓ Datasource service assembled")
        except Exception as exc:
            logger.error(f"✗ Datasource service failed: {exc}", exc_info=True)

    yield

    datasource_service = getattr(app.state, "datasource_service", None)
    if datasource_service is not None:
        try:
            await datasource_service.stop_scheduler()
        except Exception as exc:
            logger.warning(f"datasource scheduler stop failed: {exc}")
        try:
            datasource_service.close()
        except Exception as exc:
            logger.warning(f"datasource service close failed: {exc}")

    # Ticket 06：释放 v2 runtime（SQLite stores + writer lock）
    from agent_v2.startup import release_v2_runtime

    paused_run_sweeper = getattr(app.state, "paused_run_sweeper", None)
    if paused_run_sweeper is not None:
        paused_run_sweeper.cancel()

    v2_runtime = getattr(app.state, "agent_v2_runtime", None)
    try:
        release_v2_runtime(v2_runtime)
    except Exception as exc:
        logger.warning(f"v2 runtime release failed: {exc}")

    auth_service = getattr(app.state, "auth_service", None)
    if auth_service is not None:
        try:
            auth_service.close()
        except Exception as exc:
            logger.warning(f"auth service close failed: {exc}")

    # 退出前刷新已产生的 span；重复调用安全，追踪关闭时为零操作。
    from observability import tracing

    tracing.shutdown_tracing()

    logger.info("Shutting down ai-hiking AI Service...")


app = FastAPI(
    title="ai-hiking AI Service",
    description="Multi-modal AI Agent system with LangChain",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=bool(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    # Spec #13 phase 2 (T06): the RAG query route runs on the kernel
    # (profile=rag); the HTTP RAG surface was removed in A11 — RAG is a
    # chat tool (hiking_knowledge_search) on the kernel. No separate RAG
    # runtime assembly exists anymore.
    from agent_v2.health import v2_health_status

    v2_status = v2_health_status(app.state, settings)
    return {
        "status": "ok",
        "service": "ai-service",
        "agent_v2": v2_status,
    }


@app.get("/ready")
async def ready():
    """Report whether the configured chat runtime can accept traffic.

    ``/health`` is a liveness endpoint and remains successful while a
    dependency is being assembled. ``/ready`` is used by production probes
    and fails when v2 is enabled but its persistent runtime did not assemble.
    """
    from agent_v2.health import v2_health_status

    v2_status = v2_health_status(app.state, settings)
    if settings.agent_v2_enabled and v2_status.get("status") != "ready":
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "ai-service",
                "agent_v2": v2_status,
            },
        )
    return {
        "status": "ready",
        "service": "ai-service",
        "agent_v2": v2_status,
    }


# 注册路由——按依赖可用性逐个加载
# 除 /health、/api/v2/chat/health 与 /api/auth/* 外，所有数据面接口都要求登录。
_AUTH = [Depends(get_current_user)]
app.include_router(auth_router, prefix="/api")

# Chat v2 路由（/api/v2/chat/sse）—— v1 chat 已移除，v2 是唯一 chat 链路
# 路由始终注册；v2 关闭时端点返回 error terminal event（保持 SSE 契约统一）
# Ticket 10: 同时注册 artifacts_v2_router，提供 /api/v2/artifacts/{artifact_id} 下载
# Ticket 11: 同时注册 approvals_v2_router，提供 /api/v2/approvals/{approval_id}/decision
try:
    from api.chat_v2 import (
        approvals_v2_router,
        artifacts_v2_router,
        chat_v2_public_router,
        chat_v2_router,
    )
    app.include_router(chat_v2_public_router, prefix="/api/v2")
    app.include_router(chat_v2_router, prefix="/api/v2", dependencies=_AUTH)
    app.include_router(artifacts_v2_router, prefix="/api/v2", dependencies=_AUTH)
    app.include_router(approvals_v2_router, prefix="/api/v2", dependencies=_AUTH)
    from api.memory_v2 import memory_v2_router
    app.include_router(memory_v2_router, prefix="/api/v2", dependencies=_AUTH)
    from api.mcp_router import mcp_router
    app.include_router(mcp_router, prefix="/api/v2", dependencies=_AUTH)
    logger.info("✓ Chat v2 router loaded")
except ImportError as e:
    logger.warning(f"✗ Chat v2 router unavailable: {e}")

# 会话生命周期审计路由：只记日志，不落库
try:
    from api.session_lifecycle import session_lifecycle_router
    app.include_router(session_lifecycle_router, prefix="/api/v2", dependencies=_AUTH)
    logger.info("✓ Session lifecycle router loaded")
except ImportError as e:
    logger.warning(f"✗ Session lifecycle router unavailable: {e}")

# models 路由: 需要 openai, httpx
try:
    from api.models_router import models_router
    app.include_router(models_router, prefix="/api/v1", dependencies=_AUTH)
    logger.info("✓ Models router loaded")
except ImportError as e:
    logger.warning(f"✗ Models router unavailable: {e}")

# LLM 配置路由: 本机 + 登录账号
try:
    from api.llm_config_router import llm_config_router
    app.include_router(llm_config_router, prefix="/api/v1", dependencies=_AUTH)
    logger.info("✓ LLM config router loaded")
except ImportError as e:
    logger.warning(f"✗ LLM config router unavailable: {e}")

# 知识源管理路由：本机来源守卫；服务缺失时返回 503
try:
    from api.datasource_router import datasource_router
    app.include_router(datasource_router, prefix="/api/v1", dependencies=_AUTH)
    logger.info("✓ Datasource router loaded")
except ImportError as e:
    logger.warning(f"✗ Datasource router unavailable: {e}")



if __name__ == "__main__":
    # Uvicorn's Windows reload subprocess policy uses SelectorEventLoop, which
    # cannot start stdio MCP subprocesses.
    # log_config=None keeps uvicorn from installing its own plain-text handlers:
    # its error tracebacks must flow through the product JSON formatter, or one
    # request becomes N unkeyed lines in the aggregated stream.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=(sys.platform != "win32"),
        log_config=None,
    )
