"""票据 01：生产查询生命周期 Runtime 基座。

提供可注入、可持久化的 graph runtime，统一装配查询图、checkpointer、
检索、改写、聚合及领域能力依赖。

设计要点（按 01 票据验收）：
1. 一次初始化、复用、关闭：RAGRuntime 在 app lifespan 内创建一次，
   后续请求复用同一编译图实例（runtime._graph）。
2. 显式依赖调用：FastAPI 通过 get_rag_runtime(request) 依赖注入获取 runtime。
3. 稳定会话标识：通过 thread_id_for(chat_id) 转换，沿用 08 票据决策（thread_id=chat_id）。
4. 装配失败显式：build_rag_graph 抛错时包装为 RagRuntimeError，不静默。
5. 受控切换：settings.rag_use_graph_runtime 开关（默认 False，零回归）。
6. 公共契约：测试通过 RAGRuntime 公开构造参数注入 fakes，
   arun() 返回 state dict 不暴露 graph 私有字段。

本模块只提供 runtime 基座；实际 /query 端点切换由 02-05 票据承接。
"""

from __future__ import annotations

import logging
from typing import Any

from rag.checkpointer import thread_id_for
from rag.graph import build_rag_graph, run_rag_graph

logger = logging.getLogger("ai-service.rag.runtime")


class RagRuntimeError(Exception):
    """RAG runtime 装配或运行失败。

    用于显式标记 runtime 层故障，避免静默回退到 legacy 路径。
    """


class RAGRuntime:
    """生产 RAG 查询生命周期 runtime。

    封装 build_rag_graph 编译产物 + 依赖（llm/retriever/augmenter/rewriter），
    应用生命周期内一次初始化、复用、关闭。

    构造参数对齐 build_rag_graph，保持公共契约稳定；测试可注入确定性 fakes。
    """

    def __init__(
        self,
        *,
        checkpointer: Any,
        llm: Any = None,
        retriever: Any = None,
        augmenter: Any = None,
        rewriter: Any = None,
        enable_clarification: bool = False,
        clarification_store: Any = None,
        recent_limit: int = 6,
        max_sub_questions: int = 3,
        retrieve_k: int = 4,
        thread_id_for_fn: Any = None,
    ) -> None:
        if checkpointer is None:
            raise RagRuntimeError(
                "checkpointer is required for RAGRuntime (不能为 None)"
            )
        self._checkpointer = checkpointer
        self._llm = llm
        self._retriever = retriever
        self._augmenter = augmenter
        self._rewriter = rewriter
        self._enable_clarification = enable_clarification
        self._clarification_store = clarification_store
        self._recent_limit = recent_limit
        self._max_sub_questions = max_sub_questions
        self._retrieve_k = retrieve_k
        # thread_id 适配函数（默认用 rag.checkpointer.thread_id_for）
        # 测试可注入 spy 用于行为契约验证（不暴露内部状态）
        self._thread_id_for_fn = thread_id_for_fn or thread_id_for

        try:
            self._graph = build_rag_graph(
                llm=llm,
                retriever=retriever,
                checkpointer=checkpointer,
                recent_limit=recent_limit,
                max_sub_questions=max_sub_questions,
                retrieve_k=retrieve_k,
                enable_clarification=enable_clarification,
                clarification_store=clarification_store,
                augmenter=augmenter,
                rewriter=rewriter,
            )
        except Exception as exc:
            raise RagRuntimeError(
                f"graph compile failed: {exc}"
            ) from exc

    async def arun(
        self,
        *,
        question: str,
        chat_id: str,
        messages: list[dict[str, str]] | None = None,
        clarification_answers: list[str] | None = None,
    ) -> dict[str, Any]:
        """运行 RAG 主图并返回最终 state。

        包装 run_rag_graph，将执行异常显式转为 RagRuntimeError。
        thread_id 通过 thread_id_for_fn(chat_id) 适配（默认 thread_id_for）。

        Args:
            question: 用户原始问题
            chat_id: 聊天会话标识（作为 checkpointer thread_id 来源）
            messages: 对话历史（可选）
            clarification_answers: 澄清 resume 时用户提交的答案（可选）

        Returns:
            RAGState dict，含 answer / retrieved_docs / process_messages 等
            公共契约字段（不暴露 graph 内部状态布局）。
        """
        try:
            result = await run_rag_graph(
                self._graph,
                question=question,
                chat_id=chat_id,
                messages=messages,
                thread_id_for_fn=self._thread_id_for_fn,
                clarification_answers=clarification_answers,
            )
        except Exception as exc:
            raise RagRuntimeError(
                f"arun failed for chat_id={chat_id!r}: {exc}"
            ) from exc

        return result

    async def aclose(self) -> None:
        """关闭 runtime 持有的资源。

        幂等：多次调用不抛异常。
        当前 runtime 不持有需显式释放的资源（checkpointer 由 main.py lifespan
        独立管理）；本方法为未来扩展预留 + 满足验收 1"关闭"语义。
        """
        # 当前无显式资源需释放；checkpointer 由 main.py lifespan 管理生命周期
        return None


def build_rag_runtime(
    *,
    settings: Any,
    checkpointer: Any,
    llm: Any = None,
    retriever: Any = None,
    augmenter: Any = None,
    rewriter: Any = None,
    clarification_store: Any = None,
) -> RAGRuntime:
    """从 settings + checkpointer 构造 RAGRuntime。

    本工厂仅做参数透传与配置映射；实际依赖（llm/retriever/augmenter/rewriter）
    在 02-05 票据接入生产时由调用方提供。当前 01 票据阶段均为 None，
    runtime 仅交付基座，arun() 走 graph fallback 路径（零回归）。

    Args:
        settings: config.settings 实例
        checkpointer: LangGraph checkpointer（必须非 None）
        llm: LLMProtocol 实例（可选，None 走 fallback）
        retriever: RetrieverProtocol 实例（可选，None 返回空文档）
        augmenter: ContextAugmenter 实例（可选）
        rewriter: QueryRewriter 实例（可选）
        clarification_store: ClarificationStore 实例（可选）

    Returns:
        RAGRuntime 实例

    Raises:
        RagRuntimeError: checkpointer 为 None 或图编译失败
    """
    if checkpointer is None:
        raise RagRuntimeError(
            "build_rag_runtime: checkpointer is required"
        )

    enable_clarification = bool(getattr(settings, "rag_enable_clarification", False))
    # 04 票据：从 settings 读取并行检索子问题上限（默认 3，零回归）
    max_sub_questions = int(getattr(settings, "rag_max_sub_questions", 3))

    return RAGRuntime(
        checkpointer=checkpointer,
        llm=llm,
        retriever=retriever,
        augmenter=augmenter,
        rewriter=rewriter,
        enable_clarification=enable_clarification,
        clarification_store=clarification_store,
        max_sub_questions=max_sub_questions,
    )


__all__ = [
    "RAGRuntime",
    "RagRuntimeError",
    "build_rag_runtime",
]
