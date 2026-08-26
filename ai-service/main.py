import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ai-hiking AI Service...")
    logger.info(f"OpenAI Base URL: {settings.openai_base_url}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    if settings.rag_docs_api_url:
        logger.info(f"RAG Docs API: {settings.rag_docs_api_url}")

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
            else:
                logger.error("✗ Chat v2 enabled but assembly returned None")
        except V2StartupError as exc:
            logger.error(f"✗ Chat v2 assembly blocked by guard: {exc}")
            raise
        except Exception as exc:
            logger.error(f"✗ Chat v2 assembly failed: {exc}", exc_info=True)
            # 失败时 runtime 保持 None，/health 会报告 failed
    else:
        logger.info("Chat v2 disabled; v1 path unchanged")

    # 票据 01：注入 RAG checkpointer（AsyncSqliteSaver + data/rag_checkpoints.sqlite）
    # 单元测试用 FakeCheckpointer；生产 lifespan 用真实 SQLite saver
    rag_checkpointer_cm = None
    try:
        from rag.checkpointer import create_checkpointer

        rag_checkpointer_cm = create_checkpointer()
        saver = await rag_checkpointer_cm.__aenter__()
        try:
            await saver.setup()
        except Exception as exc:
            logger.warning(f"RAG checkpointer setup failed (continuing without): {exc}")
            await rag_checkpointer_cm.__aexit__(None, None, None)
            rag_checkpointer_cm = None
            saver = None
        if saver is not None:
            app.state.rag_checkpointer = saver
            logger.info("✓ RAG checkpointer loaded (AsyncSqliteSaver)")
    except ImportError as exc:
        logger.warning(f"✗ RAG checkpointer unavailable: {exc}")
        app.state.rag_checkpointer = None

    # 票据 01：装配生产 graph runtime（受控切换，默认关闭=零回归）
    # 票据 02：注入生产 retriever/augmenter/rewriter 依赖，使清晰单问题走完整检索生命周期
    # 当 settings.rag_use_graph_runtime=True 且 checkpointer 可用时创建 runtime；
    # 失败时不静默：runtime=None 让调用方明确检测，并日志记录错误。
    app.state.rag_runtime = None
    if settings.rag_use_graph_runtime:
        checkpointer = getattr(app.state, "rag_checkpointer", None)
        if checkpointer is None:
            logger.error(
                "✗ RAG runtime enabled (RAG_USE_GRAPH_RUNTIME=true) but "
                "checkpointer unavailable; runtime will not be assembled. "
                "Callers will see rag_runtime=None."
            )
        else:
            try:
                from rag.runtime import RagRuntimeError, build_rag_runtime

                # 票据 02：注入生产依赖（用默认 settings 创建，不支持 req.model_settings 自定义）
                # 创建失败时不静默，让 runtime 装配失败被 /health 暴露
                runtime_retriever = None
                runtime_augmenter = None
                runtime_rewriter = None
                try:
                    from rag.retriever import VectorStoreRetriever
                    runtime_retriever = VectorStoreRetriever()
                except Exception as exc:
                    logger.error(f"✗ RAG runtime: retriever init failed: {exc}")
                try:
                    from rag.augmenter import ContextAugmenter
                    runtime_augmenter = ContextAugmenter()
                except Exception as exc:
                    logger.error(f"✗ RAG runtime: augmenter init failed: {exc}")
                try:
                    from rag.rewriter import QueryRewriter
                    runtime_rewriter = QueryRewriter()
                except Exception as exc:
                    logger.error(f"✗ RAG runtime: rewriter init failed: {exc}")

                runtime = build_rag_runtime(
                    settings=settings,
                    checkpointer=checkpointer,
                    retriever=runtime_retriever,
                    augmenter=runtime_augmenter,
                    rewriter=runtime_rewriter,
                )
                app.state.rag_runtime = runtime
                logger.info("✓ RAG runtime assembled (graph runtime mode)")
            except RagRuntimeError as exc:
                logger.error(
                    f"✗ RAG runtime assembly failed (explicit failure): {exc}"
                )
            except Exception as exc:
                logger.error(
                    f"✗ RAG runtime assembly unexpected error: {exc}",
                    exc_info=True,
                )

    yield

    # 关闭 runtime（票据 01 验收 1：runtime 可关闭）
    runtime = getattr(app.state, "rag_runtime", None)
    if runtime is not None:
        try:
            await runtime.aclose()
        except Exception as exc:
            logger.warning(f"RAG runtime cleanup failed: {exc}")

    # Ticket 06：释放 v2 runtime（SQLite stores + writer lock）
    from agent_v2.startup import release_v2_runtime

    v2_runtime = getattr(app.state, "agent_v2_runtime", None)
    try:
        release_v2_runtime(v2_runtime)
    except Exception as exc:
        logger.warning(f"v2 runtime release failed: {exc}")

    if rag_checkpointer_cm is not None:
        try:
            await rag_checkpointer_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning(f"RAG checkpointer cleanup failed: {exc}")
    logger.info("Shutting down ai-hiking AI Service...")


app = FastAPI(
    title="ai-hiking AI Service",
    description="Multi-modal AI Agent system with LangChain",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    # 票据 01：暴露 rag_runtime 装配状态，避免开关打开但装配失败时静默走 legacy
    runtime_enabled = bool(getattr(settings, "rag_use_graph_runtime", False))
    runtime_assembled = getattr(app.state, "rag_runtime", None) is not None
    if runtime_enabled and not runtime_assembled:
        # 装配失败：显式标记，监控可察觉（不静默伪装生产成功）
        rag_runtime_status = "failed"
    elif runtime_enabled and runtime_assembled:
        rag_runtime_status = "ok"
    else:
        rag_runtime_status = "disabled"
    # Ticket 03：暴露 Chat v2 启用/装配状态 + UniHarness pin（可审计）
    from agent_v2.health import v2_health_status

    v2_status = v2_health_status(app.state, settings)
    return {
        "status": "ok",
        "service": "ai-service",
        "rag_runtime": {
            "enabled": runtime_enabled,
            "assembled": runtime_assembled,
            "status": rag_runtime_status,
        },
        "agent_v2": v2_status,
    }


# 注册路由——按依赖可用性逐个加载
# chat 路由: 需要 langgraph, langchain-core, openai
try:
    from api.chat import chat_router
    app.include_router(chat_router, prefix="/api/v1")
    logger.info("✓ Chat router loaded")
except ImportError as e:
    # 提供 fallback 端点
    from fastapi import APIRouter
    fallback = APIRouter()
    @fallback.get("/chat/health")
    async def chat_health():
        return {"status": "ok", "module": "chat", "fallback": "ai-service-health", "service": "fallback"}
    app.include_router(fallback, prefix="/api/v1")
    logger.warning(f"✗ Chat router unavailable: {e}")

# Ticket 04: Chat v2 路由（/api/v2/chat/sse）—— 默认关闭，灰度期由 Gateway 分桶控制
# 路由始终注册；v2 关闭时端点返回 error terminal event（保持 SSE 契约统一）
# Ticket 10: 同时注册 artifacts_v2_router，提供 /api/v2/artifacts/{artifact_id} 下载
# Ticket 11: 同时注册 approvals_v2_router，提供 /api/v2/approvals/{approval_id}/decision
try:
    from api.chat_v2 import (
        approvals_v2_router,
        artifacts_v2_router,
        chat_v2_router,
    )
    app.include_router(chat_v2_router, prefix="/api/v2")
    app.include_router(artifacts_v2_router, prefix="/api/v2")
    app.include_router(approvals_v2_router, prefix="/api/v2")
    from api.memory_v2 import memory_v2_router
    app.include_router(memory_v2_router, prefix="/api/v2")
    logger.info("✓ Chat v2 router loaded")
except ImportError as e:
    logger.warning(f"✗ Chat v2 router unavailable: {e}")

# models 路由: 需要 openai, httpx
try:
    from api.models_router import models_router
    app.include_router(models_router, prefix="/api/v1")
    logger.info("✓ Models router loaded")
except ImportError as e:
    logger.warning(f"✗ Models router unavailable: {e}")

# rag 路由: 需要 langchain, faiss-cpu, etc
try:
    from api.rag import rag_router
    app.include_router(rag_router, prefix="/api/v1")
    logger.info("✓ RAG router loaded")
except ImportError as e:
    from fastapi import APIRouter
    fallback = APIRouter()
    @fallback.get("/rag/health")
    async def rag_health():
        return {"status": "ok", "module": "rag", "fallback": "ai-service-health", "service": "fallback"}
    app.include_router(fallback, prefix="/api/v1")
    logger.warning(f"✗ RAG router unavailable: {e}")

# tools 路由: 需要 httpx
try:
    from api.tools import tools_router
    app.include_router(tools_router, prefix="/api/v1")
    logger.info("✓ Tools router loaded")
except ImportError as e:
    logger.warning(f"✗ Tools router unavailable: {e}")

# artifacts 路由: 下载 workspace 内由 Agent 生成的文件
try:
    from api.artifacts import artifacts_router
    app.include_router(artifacts_router, prefix="/api/v1")
    logger.info("✓ Artifacts router loaded")
except ImportError as e:
    logger.warning(f"✗ Artifacts router unavailable: {e}")


if __name__ == "__main__":
    # Uvicorn's Windows reload subprocess policy uses SelectorEventLoop, which
    # cannot start stdio MCP subprocesses.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=(sys.platform != "win32"))
