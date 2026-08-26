"""票据 05：RAG 生命周期主图骨架。

LangGraph StateGraph 实现 4 阶段生命周期骨架：
1. summarize_conversation：滚动摘要 + 仅保留最近 N 条原始消息（上下文有界）
2. analyze_query：结构化查询分析（清晰判定 + 至多 3 个自包含子问题 + 澄清说明），
   融合既有 contextualize + rewrite，不存在双重改写
3. retrieve：经 Send 并行派发到检索子图（每个子问题一个 retrieve 节点实例）
4. aggregate_answer：聚合检索结果生成最终中文答案

本模块只提供图骨架与节点函数，不负责 SSE 事件发射（由 api/rag.py 在后续票据承接）。
LLM 调用通过 LLMProtocol 抽象，测试用确定性 FakeLLM 驱动。
"""

from __future__ import annotations

import hashlib
import logging
import operator
from typing import Annotated, Any, Callable, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Parent 检索回调类型（与 retrieve_subgraph.ParentRetrieverFn 同语义）
# 顶部定义避免循环 import，运行时仍为 Callable
ParentRetrieverFn = Callable[[list[str]], list[dict[str, Any]]]

logger = logging.getLogger("ai-service.rag.graph")

# 滚动摘要保留的最近原始消息条数
DEFAULT_RECENT_MESSAGE_LIMIT = 6
# 单次查询分析最多生成的子问题数
DEFAULT_MAX_SUB_QUESTIONS = 3


# 跨轮重置哨兵：summarize_conversation（每轮第一个节点）返回此字符串时，
# 自定义 reducer 会把累加器字段重置为空列表，避免上一轮的 retrieved_docs
# / process_messages 污染当前轮。用字符串是为了让 checkpointer 可序列化。
_RESET = "__RAG_LIFECYCLE_RESET__"


def _list_accumulator(old: list | None, new: Any) -> list:
    """支持重置的列表累加 reducer。

    - new == _RESET（字符串哨兵）：返回空列表（每轮第一个节点重置累加器）
    - new is list：append 到 old（并行 retrieve 节点累加）
    - new is None：保留 old（节点未返回该字段时）
    """
    if new == _RESET:
        return []
    if new is None:
        return list(old or [])
    return (old or []) + list(new)


@runtime_checkable
class LLMProtocol(Protocol):
    """LLM 调用协议，允许注入确定性 FakeLLM 驱动测试。"""

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """同步调用 LLM，返回文本响应。"""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """检索器协议，允许注入 FakeRetriever 驱动测试。"""

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """对单个查询做混合检索，返回文档列表。"""
        ...


class RAGState(TypedDict, total=False):
    """RAG 主图状态。

    使用 total=False 允许节点返回部分更新（LangGraph 会合并到现有 state）。
    retrieved_docs 与 process_messages 使用自定义 _list_accumulator reducer：
    - 并行 retrieve 节点（经 Send 派发）的返回值会累加（单轮内 fan-out）
    - summarize_conversation（每轮第一个节点）返回 _RESET 哨兵时重置为空列表
      避免跨轮 checkpointer 持久化导致的累加器污染
    """

    # 输入
    question: str                       # 用户原始问题
    chat_id: str                        # 08 票据：聊天会话标识（用于计算 clarification_id）
    messages: list[dict[str, str]]      # 完整对话历史（role/content）

    # summarize_conversation 输出
    summary: str                        # 滚动摘要（旧消息压缩）
    recent_messages: list[dict[str, str]]  # 最近 N 条原始消息

    # analyze_query 输出
    is_clear: bool                      # 查询是否清晰
    sub_questions: list[str]            # 自包含子问题列表（至多 3 个）
    clarification_reason: str           # 不清晰时的澄清说明

    # retrieve 输出（多 retrieve 实例并行写入 → 用 _list_accumulator 累加）
    retrieved_docs: Annotated[list[dict[str, Any]], _list_accumulator]

    # aggregate_answer 输出
    answer: str                         # 最终中文答案

    # 08 票据：澄清双请求
    clarification_needed: dict[str, Any]     # 澄清请求信息（clarification_id, questions, expires_at）
    clarification_answers: list[str]         # resume 时提交的澄清答案

    # 贯穿全流程（多节点追加 → 用 _list_accumulator 累加）
    process_messages: Annotated[list[str], _list_accumulator]


def _append_process(messages: list[str] | None, message: str) -> list[str]:
    """向进度消息列表追加（返回新列表，配合 _list_accumulator reducer）。"""
    if messages is None:
        return [message]
    return messages + [message]


def summarize_conversation(
    state: RAGState,
    *,
    llm: LLMProtocol | None = None,
    recent_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
) -> dict[str, Any]:
    """节点 1：会话摘要。

    滚动摘要策略：
    - 保留最近 recent_limit 条原始消息不动
    - 更早的消息并入 summary（若 LLM 可用，用 LLM 压缩；否则简单拼接）
    - 首次调用（无 summary）时，若消息数 <= recent_limit，summary 保持空
    - 跨轮调用时，依赖 checkpointer 持久化的 summary 做滚动累积

    Returns:
        {"summary": str, "recent_messages": list,
         "retrieved_docs": _RESET, "process_messages": _RESET}
        - _RESET 哨兵触发 _list_accumulator reducer 重置累加器，避免跨轮污染
    """
    messages = state.get("messages", [])
    # checkpointer 持久化的 summary（跨轮滚动累积）
    existing_summary = state.get("summary", "")

    if len(messages) <= recent_limit:
        # 消息不多，无需摘要
        return {
            "summary": existing_summary,
            "recent_messages": list(messages),
            # 跨轮重置累加器（P0-1 修复）
            "retrieved_docs": _RESET,
            "process_messages": _RESET,
        }

    # 分割：旧消息 → 摘要，最近 N 条 → 保留
    to_summarize = messages[:-recent_limit]
    recent = messages[-recent_limit:]

    # 构造待压缩文本（合并已有摘要 + 新旧消息）
    old_text = existing_summary + "\n" if existing_summary else ""
    for msg in to_summarize:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        old_text += f"{role}: {content}\n"

    if llm is not None:
        try:
            new_summary = llm.invoke(
                "你是对话摘要助手。将给定的对话历史压缩为简洁的中文摘要，"
                "保留关键事实、用户意图和已确定的信息，不超过 200 字。",
                f"对话历史：\n{old_text}",
            )
        except Exception as exc:
            logger.warning("summarize_conversation LLM failed: %s, using fallback", exc)
            new_summary = _fallback_summary(old_text)
    else:
        new_summary = _fallback_summary(old_text)

    return {
        "summary": new_summary,
        "recent_messages": list(recent),
        # 跨轮重置累加器（P0-1 修复）
        "retrieved_docs": _RESET,
        "process_messages": _RESET,
    }


def _fallback_summary(text: str, max_chars: int = 200) -> str:
    """无 LLM 时的简单摘要：截断到 max_chars。"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def analyze_query(
    state: RAGState,
    *,
    llm: LLMProtocol | None = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    recent_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
) -> dict[str, Any]:
    """节点 2：结构化查询分析。

    融合 contextualize（指代解析）+ rewrite（语义变体）+ 清晰判定为单一节点。
    输出结构化结果：
    - is_clear: 是否清晰（True → 进入检索；False → 返回澄清说明）
    - sub_questions: 至多 max_sub_questions 个自包含子问题
    - clarification_reason: 不清晰时的说明

    追问指代（"它怎么走"）结合 summary + recent_messages 解析为自包含问题。
    新主题、命名术语直接判定清晰，不被误判为不清晰。

    Args:
        recent_limit: 取近期对话的最后 N 条构造上下文，与 summarize_conversation 对齐

    Returns:
        {"is_clear": bool, "sub_questions": list, "clarification_reason": str, ...}
    """
    question = state.get("question", "").strip()
    summary = state.get("summary", "")
    recent = state.get("recent_messages", state.get("messages", []))

    if not question:
        return {
            "is_clear": False,
            "sub_questions": [],
            "clarification_reason": "问题为空，请提供具体的徒步相关问题。",
        }

    # 无 LLM 时用规则 fallback
    if llm is None:
        return _fallback_analyze(question, summary, recent, max_sub_questions)

    # 构造上下文（与 summarize_conversation 的 recent_limit 对齐）
    context_parts = []
    if summary:
        context_parts.append(f"对话摘要：{summary}")
    if recent:
        # P1-1 修复：用 recent_limit 替代硬编码 6
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in recent[-recent_limit:]
            if isinstance(m, dict)
        )
        if history_text:
            context_parts.append(f"近期对话：\n{history_text}")
    context = "\n".join(context_parts) if context_parts else "（无历史对话）"

    system_prompt = (
        "你是徒步领域的查询分析助手。分析用户问题并输出结构化 JSON。\n"
        "判定规则：\n"
        "1. 新主题、命名术语（如'西湖群山路线'）直接判定清晰\n"
        "2. 追问中的指代（'它怎么走'）结合历史解析为自包含问题\n"
        "3. 模糊到无法检索（如'那个东西'且无上下文）判定不清晰\n"
        "4. 至多生成 3 个自包含子问题（含原问题的改写）\n\n"
        "输出 JSON 格式（不要 markdown 代码块）：\n"
        '{"is_clear": true/false, "sub_questions": ["问题1", "问题2"], '
        '"clarification_reason": "不清晰时的说明，清晰时为空字符串"}'
    )
    user_prompt = f"上下文：\n{context}\n\n用户问题：{question}"

    try:
        response = llm.invoke(system_prompt, user_prompt)
        # P1-2 修复：把 summary/recent 传给 _parse_analyze_response，
        # JSON 解析失败时 fallback 能用真实上下文（而非空）
        result = _parse_analyze_response(
            response, question, max_sub_questions, summary, recent
        )
    except Exception as exc:
        logger.warning("analyze_query LLM failed: %s, using fallback", exc)
        result = _fallback_analyze(question, summary, recent, max_sub_questions)

    return result


def _parse_analyze_response(
    response: str,
    original_question: str,
    max_sub_questions: int,
    summary: str = "",
    recent: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """解析 LLM 返回的 JSON，容错处理。

    Args:
        summary/recent: JSON 解析失败时传给 _fallback_analyze 保留上下文（P1-2 修复）
    """
    import json

    text = response.strip()
    # 去除可能的 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # P1-2 修复：解析失败时用真实上下文降级，而非空上下文
        return _fallback_analyze(
            original_question, summary, recent or [], max_sub_questions
        )

    is_clear = bool(data.get("is_clear", True))
    sub_questions = data.get("sub_questions", [])
    clarification_reason = data.get("clarification_reason", "")

    if not isinstance(sub_questions, list):
        sub_questions = [original_question]

    # 截断到 max_sub_questions
    sub_questions = [str(q) for q in sub_questions[:max_sub_questions]]

    if is_clear:
        # 清晰时：保证至少有一个子问题（兜底用原问题）
        if not sub_questions:
            sub_questions = [original_question]
        # 确保原问题（或其改写）在列表中
        elif original_question not in sub_questions and len(sub_questions) < max_sub_questions:
            sub_questions.insert(0, original_question)
    else:
        # 不清晰时：不进入检索，sub_questions 强制为空（避免误派发 Send）
        sub_questions = []

    if not is_clear and not clarification_reason:
        clarification_reason = "问题不够明确，请补充更多细节。"

    return {
        "is_clear": is_clear,
        "sub_questions": sub_questions,
        "clarification_reason": clarification_reason,
    }


def _fallback_analyze(
    question: str,
    summary: str,
    recent: list[dict[str, str]],
    max_sub_questions: int,
) -> dict[str, Any]:
    """无 LLM 时的规则化查询分析。

    规则：
    1. 含指代词（它/这个/那个/这些/上面/前面）且有历史 → 结合计史生成自包含问题
    2. 新主题（无指代词）→ 直接判定清晰，子问题 = [原问题]
    3. 空或纯噪声 → 不清晰
    """
    # P1-4 修复：用 list 替代 set 保证迭代顺序确定性（更具体的指代词优先）
    # 按长度降序排列，先匹配更具体的（"这个" 优先于 "这"）
    reference_words = [
        "这个", "那个", "这些", "那些", "上面", "前面", "刚才", "之前", "它",
    ]
    has_reference = any(ref in question for ref in reference_words)

    if has_reference and recent:
        # 追问：结合上一条 user 消息生成自包含问题
        last_user_msgs = [m for m in recent if isinstance(m, dict) and m.get("role") == "user"]
        if last_user_msgs:
            last_topic = last_user_msgs[-1].get("content", "")
            # 按优先级顺序替换第一个匹配的指代词
            resolved = question
            for ref in reference_words:
                if ref in resolved:
                    resolved = resolved.replace(ref, last_topic)
                    break
            sub_questions = [resolved, question]
            if summary:
                sub_questions.append(f"结合{summary[:30]}，{question}")
            return {
                "is_clear": True,
                "sub_questions": sub_questions[:max_sub_questions],
                "clarification_reason": "",
            }

    # 新主题或无指代词 → 直接清晰
    if len(question) >= 2:
        return {
            "is_clear": True,
            "sub_questions": [question],
            "clarification_reason": "",
        }

    # 太短且无上下文
    return {
        "is_clear": False,
        "sub_questions": [],
        "clarification_reason": "问题太短，请提供更具体的徒步相关问题。",
    }


def retrieve(
    state: dict[str, Any],
    *,
    retriever: RetrieverProtocol | None = None,
    k: int = 4,
) -> dict[str, Any]:
    """节点 3：检索子图节点（由 Send 并行派发）。

    接收单个 sub_question（通过 Send 传入），执行混合检索。
    本节点是并行 fan-out 的工作节点，每个实例处理一个子问题。

    07 票据：返回的 docs 带 question_index 字段，供 aggregate_answer 保序聚合。

    注意：本函数签名接收的是 Send 传入的子图 state（非完整 RAGState），
    返回的 docs 会被 LangGraph 自动合并到父图的 retrieved_docs。
    """
    question = state.get("question", state.get("sub_question", ""))
    # 07 票据：question_index 由 Send 派发时传入，用于保序聚合
    question_index = state.get("question_index", 0)
    if not question:
        return {"retrieved_docs": []}

    if retriever is None:
        logger.warning("retrieve: no retriever, returning empty docs")
        return {"retrieved_docs": []}

    try:
        docs = retriever.search(question, k=k)
    except Exception as exc:
        logger.warning("retrieve failed for %r: %s", question, exc)
        docs = []

    # 07 票据：给每条 doc 标记 question_index（保序聚合依据）
    for doc in docs:
        if isinstance(doc, dict):
            doc["question_index"] = question_index

    return {"retrieved_docs": docs}


def aggregate_answer(
    state: RAGState,
    *,
    llm: LLMProtocol | None = None,
    augmenter: Any = None,
    rewriter: Any = None,
) -> dict[str, Any]:
    """节点 4：聚合检索结果生成最终答案。

    07 票据：按 question_index 保序聚合 + 冲突显式呈现。
    09 票据：领域能力融合 - 在最终边界应用 ContextAugmenter + humanize_for_answer + clean_display_text。

    聚合策略：
    1. 按 question_index 分组 docs（乱序完成的子图答案按序号排序）
    2. 组内按 content 哈希去重
    3. 检测冲突：同子问题不同来源对同一数字/版本有不同说法时显式标注
    4. 无证据时给出明确的查不到回答
    5. 聚合只使用子图答案及其证据，不用外部知识调和冲突

    09 票据领域能力融合（legacy-capabilities-placement 结论）：
    - ContextAugmenter.augment 在聚合节点调用（最终答案生成边界，保留+移动）
    - humanize_for_answer 在最终用户可见边界应用一次（移动）
    - _clean_display_text 在最终用户可见边界一次应用（CJK 规范化，移动）
    - 聚合后无第二次全局检索（双重检索消除）

    Args:
        state: RAG 状态
        llm: LLM 调用协议实例（None = 用 fallback 规则）
        augmenter: ContextAugmenter 实例（None = 不用 augmenter，走 llm.invoke 路径）
        rewriter: QueryRewriter 实例（None = 不应用 humanize_for_answer）

    Returns:
        {"answer": str, "process_messages": list}
    """
    docs = state.get("retrieved_docs", [])
    question = state.get("question", "")
    sub_questions = state.get("sub_questions", [])
    process: list[str] = []

    # 07 票据：按 question_index 分组（保序聚合依据）
    # 没有标记 question_index 的 doc（如 05 票据遗留路径）归入 index=0
    grouped: dict[int, list[dict[str, Any]]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        idx = doc.get("question_index", 0)
        grouped.setdefault(idx, []).append(doc)

    # 按 question_index 排序（保序聚合）
    sorted_indices = sorted(grouped.keys())

    # 组内去重（按 content 哈希）—— 每个 question_index 维护独立 seen_hashes
    # 07 票据审查修复：跨子问题全局去重会让"子问题1的唯一证据与子问题0某条doc
    # 内容相同"时被静默丢弃，导致子问题1误报"未找到相关文档"（违反验收标准3）。
    # 改为按组内去重，相同内容可分别归属到各自子问题下展示。
    unique_grouped: list[tuple[int, list[dict[str, Any]]]] = []
    total_unique = 0
    for idx in sorted_indices:
        group_docs: list[dict[str, Any]] = []
        seen_in_group: set[str] = set()
        for doc in grouped[idx]:
            content = doc.get("content", "")
            h = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
            if h not in seen_in_group:
                seen_in_group.add(h)
                group_docs.append(doc)
        unique_grouped.append((idx, group_docs))
        total_unique += len(group_docs)

    process.append(
        f"聚合 {len(docs)} 条检索结果，去重后 {total_unique} 条，"
        f"按 {len(sorted_indices)} 个子问题保序聚合"
    )

    if total_unique == 0:
        # 07 票据：无有用信息时给出明确的查不到回答
        answer = "未找到相关文档，无法回答该问题。请尝试换个方式提问或上传相关文档。"
        return {"answer": answer, "process_messages": process}

    # 07 票据：检测冲突（同子问题不同来源对同一数字/版本有不同说法）
    conflicts = _detect_conflicts(unique_grouped)
    if conflicts:
        process.append(f"检测到 {len(conflicts)} 处来源间冲突，将显式呈现")

    # 09 票据：领域能力融合 - 优先使用 ContextAugmenter 在最终边界生成答案
    # legacy-capabilities-placement：ContextAugmenter.augment 在聚合节点调用
    if augmenter is not None:
        # 09 票据：对 question 应用 humanize_for_answer（legacy 语义：humanize question 而非 answer）
        # legacy api/rag.py:1066-1071 先 humanize_for_answer(question) 再传给 augmenter
        augment_question = _humanize_question_for_augmenter(question, rewriter)

        # 将聚合后的 docs 转为 LangChain Document 供 augmenter 使用
        try:
            from langchain_core.documents import Document as _LCDocument

            all_docs_for_augment: list[_LCDocument] = []
            for _idx, _group_docs in unique_grouped:
                for _doc in _group_docs:
                    all_docs_for_augment.append(
                        _LCDocument(
                            page_content=_doc.get("content", ""),
                            metadata={
                                "source": _doc.get("source", "unknown"),
                                "title": _doc.get("file_name", _doc.get("title", "")),
                                "question_index": _idx,
                            },
                        )
                    )
            # augmenter.augment 内部已通过 _format_answer_text 应用 clean_display_text + emphasize_display_terms
            # 不再后处理 answer，避免 double-apply clean_display_text 剥掉安全术语加粗
            answer = augmenter.augment(augment_question, all_docs_for_augment)
        except Exception as exc:
            logger.warning("aggregate_answer augmenter failed: %s, using fallback", exc)
            answer = _fallback_answer_grouped(question, sub_questions, unique_grouped, conflicts)
    elif llm is None:
        # 零回归路径：llm=None + augmenter=None → fallback answer，不应用最终边界处理
        answer = _fallback_answer_grouped(question, sub_questions, unique_grouped, conflicts)
        return {"answer": answer, "process_messages": process}
    else:
        # 构造上下文（按子问题序号分组）
        context_parts = []
        for idx, group_docs in unique_grouped:
            sub_q = sub_questions[idx] if idx < len(sub_questions) else f"子问题{idx + 1}"
            context_parts.append(f"## 子问题 {idx + 1}：{sub_q}")
            for i, doc in enumerate(group_docs, 1):
                source = doc.get("file_name", doc.get("source", f"文档{i}"))
                content = doc.get("content", "")
                context_parts.append(f"[{i}] 来源：{source}\n{content}")
        context = "\n\n".join(context_parts)

        # 冲突提示注入 system prompt
        conflict_hint = ""
        if conflicts:
            conflict_hint = (
                "\n5. 以下来源间存在冲突，请如实呈现不同说法，不要用外部知识调和：\n"
                + "\n".join(f"  - {c}" for c in conflicts)
            )

        system_prompt = (
            "你是徒步领域问答助手。根据检索到的文档回答用户问题。\n"
            "要求：\n"
            "1. 中文短句，无套话\n"
            "2. 关键信息用**加粗**\n"
            "3. 文档没说清就说不确定\n"
            "4. 不要说'根据文档'等元叙述"
            f"{conflict_hint}"
        )
        user_prompt = f"检索到的文档（按子问题分组）：\n{context}\n\n用户问题：{question}"

        try:
            answer = llm.invoke(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning("aggregate_answer LLM failed: %s, using fallback", exc)
            answer = _fallback_answer_grouped(question, sub_questions, unique_grouped, conflicts)

        # 09 票据：llm 路径最终边界处理
        # augmenter 路径已通过 _format_answer_text 处理；llm 路径需补充安全术语强调 + CJK 规范化
        # legacy-capabilities-placement：emphasize_display_terms + clean_display_text 在最终边界应用一次
        answer = _finalize_llm_path_answer(answer)

    return {"answer": answer, "process_messages": process}


def _humanize_question_for_augmenter(question: str, rewriter: Any = None) -> str:
    """09 票据：对 question 应用 humanize_for_answer（legacy 语义）。

    legacy api/rag.py:1066-1071 的语义是 humanize_for_answer(question) 后传给 augmenter，
    而非对最终 answer 调用 humanize_for_answer。本函数对齐 legacy 语义。

    legacy-capabilities-placement 结论：
    - humanize_for_answer 在最终边界应用一次（对 question，用于最终答案生成）

    零回归保障：rewriter 为 None 时直接返回原 question，保持 05/07 票据行为不变。

    Args:
        question: 原始用户问题
        rewriter: QueryRewriter 实例（None = 跳过 humanize，零回归）

    Returns:
        humanized question（用于传给 augmenter.augment）
    """
    if not question or rewriter is None:
        return question

    if hasattr(rewriter, "humanize_for_answer"):
        try:
            return rewriter.humanize_for_answer(question)
        except Exception as exc:
            logger.warning("aggregate_answer humanize_for_answer failed: %s", exc)

    return question


def _finalize_llm_path_answer(answer: str) -> str:
    """09 票据：llm 路径的最终边界处理（augmenter 路径不需要此处理）。

    augmenter.augment 内部已通过 _format_answer_text 应用 clean_display_text + emphasize_display_terms，
    所以 augmenter 路径不需要此处理。llm 路径直接调 llm.invoke，需补充：
    1. emphasize_display_terms - 安全术语强调（危险/时间/费用/地点/安全指引加粗）
    2. clean_display_text(keep_markdown_emphasis=True) - CJK 规范化，保留加粗以保留安全术语强调

    legacy-capabilities-placement 结论：
    - emphasize_display_terms + clean_display_text 仅在最终用户可见边界应用一次

    Args:
        answer: llm.invoke 返回的原始答案

    Returns:
        处理后的答案
    """
    if not answer:
        return answer

    # 1. emphasize_display_terms（安全术语强调，最终边界应用一次）
    try:
        from rag.text_processing import emphasize_display_terms

        answer = emphasize_display_terms(answer)
    except Exception as exc:
        logger.warning("aggregate_answer emphasize_display_terms failed: %s", exc)

    # 2. clean_display_text（保留 markdown emphasis 以保留安全术语加粗）
    try:
        from rag.text_processing import clean_display_text

        answer = clean_display_text(answer, keep_markdown_emphasis=True)
    except Exception as exc:
        logger.warning("aggregate_answer clean_display_text failed: %s", exc)

    return answer


def _detect_conflicts(
    grouped: list[tuple[int, list[dict[str, Any]]]],
) -> list[str]:
    """检测同子问题不同来源间的数字/版本冲突。

    策略：对每个子问题组，提取所有"数字+单位"对，按单位维度分组后比较。
    只有同一单位维度下不同来源给出不同数字，才判定为冲突。

    07 票据审查修复：
    1. 单位清单加入中文（米/公里/千米/厘米/毫米/级/天/小时/分钟/元/人/度/°），
       避免"1900米"与"1900m"被误判为冲突。
    2. 按"数字+单位"分组比较，避免跨维度误报（如"海拔1900m"与"距离5km"
       不再被误判为冲突）。

    Returns:
        冲突描述列表（如"子问题1：来源A说海拔1900m，来源B说海拔2000m"）
    """
    import re

    # 单位清单（中英文兼容）
    unit_pattern = r"m|km|cm|mm|米|公里|千米|厘米|毫米|级|天|小时|分钟|元|人|°|度"

    conflicts: list[str] = []
    for idx, docs in grouped:
        if len(docs) < 2:
            continue
        # 提取每条 doc 的 (数字, 单位) 对
        # source_values: list[(source, dict[unit, set[value]])]
        source_values: list[tuple[str, dict[str, set[str]]]] = []
        for doc in docs:
            source = doc.get("file_name", doc.get("source", "未知"))
            content = doc.get("content", "")
            # 数字 + 可选单位（捕获组）
            matches = re.findall(
                r"(\d+(?:\.\d+)?)\s*(" + unit_pattern + r")?", content
            )
            unit_map: dict[str, set[str]] = {}
            for num, unit in matches:
                # 无单位的归入"裸数字"维度，版本号也归入此维度
                u = unit if unit else "裸数字"
                unit_map.setdefault(u, set()).add(num)
            # 版本号（如 1.2.3）单独归入"版本"维度
            versions = re.findall(r"\d+(?:\.\d+){1,}", content)
            if versions:
                unit_map.setdefault("版本", set()).update(versions)
            source_values.append((source, unit_map))

        # 按单位维度分组比较：同单位下不同来源有不同数字 → 冲突
        for i, (src_a, units_a) in enumerate(source_values):
            for j, (src_b, units_b) in enumerate(source_values):
                if i >= j:
                    continue
                # 找两来源共有的单位维度
                common_units = set(units_a.keys()) & set(units_b.keys())
                for u in common_units:
                    diff_a = units_a[u] - units_b[u]
                    diff_b = units_b[u] - units_a[u]
                    if diff_a and diff_b:
                        # 同单位下双向都有差异 → 冲突
                        conflicts.append(
                            f"子问题{idx + 1}：{src_a} 说 {sorted(diff_a)[:2]}{u}，"
                            f"{src_b} 说 {sorted(diff_b)[:2]}{u}"
                        )
                        break  # 同一对来源只报一次冲突
                else:
                    continue
                break  # 同一子问题只报一对冲突

    return conflicts


def _fallback_answer_grouped(
    question: str,
    sub_questions: list[str],
    grouped: list[tuple[int, list[dict[str, Any]]]],
    conflicts: list[str],
) -> str:
    """无 LLM 时的分组 fallback 答案：按子问题序号呈现证据 + 冲突标注。

    07 票据：无 docs 的子问题也显式说明"未找到相关文档"，
    保证用户能看到每个子问题的回答状态。
    """
    if not grouped:
        return "未找到相关文档。"

    # 构造 index → docs 映射，便于按 sub_questions 顺序遍历
    grouped_dict = dict(grouped)
    parts: list[str] = []

    # 若有 sub_questions，按其顺序遍历；否则按 grouped 顺序
    indices_to_render = (
        list(range(len(sub_questions)))
        if sub_questions
        else sorted(grouped_dict.keys())
    )

    for idx in indices_to_render:
        sub_q = sub_questions[idx] if idx < len(sub_questions) else f"子问题{idx + 1}"
        docs = grouped_dict.get(idx, [])
        if not docs:
            # 07 票据：无证据的子问题显式说明查不到
            parts.append(f"### {idx + 1}. {sub_q}\n未找到相关文档。")
            continue
        evidence_lines = []
        for doc in docs[:3]:  # 每个子问题最多展示 3 条证据
            source = doc.get("file_name", doc.get("source", ""))
            content = doc.get("content", "")
            snippet = content[:100] + ("..." if len(content) > 100 else "")
            evidence_lines.append(f"- **{source}**：{snippet}")
        parts.append(f"### {idx + 1}. {sub_q}\n" + "\n".join(evidence_lines))

    # 冲突显式呈现
    if conflicts:
        parts.append("### 来源冲突（如实呈现，未用外部知识调和）")
        for c in conflicts:
            parts.append(f"- {c}")

    return "\n\n".join(parts)


def _route_after_analysis(state: RAGState) -> list[Send] | str:
    """条件边：查询分析后的路由（无澄清模式）。

    - is_clear=True → 返回 [Send("retrieve", {...}) for each sub_question]（并行 fan-out）
    - is_clear=False → 返回 "aggregate_answer"（生成 fallback 澄清回复）

    07 票据：Send 派发时传入 question_index，供 aggregate_answer 保序聚合。
    每个 retrieve 实例有独立的子图状态（LangGraph Send 语义保证互不共享）。

    注意：08 票据的澄清模式（enable_clarification=True）使用
    _route_after_analysis_with_clarification，不走此函数。
    """
    if state.get("is_clear", False):
        sub_questions = state.get("sub_questions", [])
        if not sub_questions:
            return "aggregate_answer"
        # Send 并行派发：每个子问题一个 retrieve 实例，带 question_index
        return [
            Send("retrieve", {
                "question": q,
                "sub_question": q,
                "question_index": i,
            })
            for i, q in enumerate(sub_questions)
        ]
    return "aggregate_answer"


def _route_after_analysis_with_clarification(state: RAGState) -> list[Send] | str:
    """条件边：查询分析后的路由（08 票据澄清模式）。

    - is_clear=True → 返回 [Send("retrieve", {...}) for each sub_question]（并行 fan-out）
    - is_clear=False → 返回 "request_clarification"（用 interrupt 触发澄清交互）

    与 _route_after_analysis 的差异：不清晰问题进入 request_clarification 节点
    而非 aggregate_answer，由 interrupt 暂停图执行等待用户提交澄清答案。
    """
    if state.get("is_clear", False):
        sub_questions = state.get("sub_questions", [])
        if not sub_questions:
            return "aggregate_answer"
        return [
            Send("retrieve", {
                "question": q,
                "sub_question": q,
                "question_index": i,
            })
            for i, q in enumerate(sub_questions)
        ]
    return "request_clarification"


def _route_after_clarification(state: RAGState) -> list[Send] | str:
    """条件边：澄清节点恢复后的路由（08 票据）。

    - is_clear=True（resume 后由 request_clarification 设置）→ Send fan-out 到 retrieve
    - is_clear=False（防御性，不应发生）→ aggregate_answer

    与 _route_after_analysis 共用 fan-out 逻辑，独立函数以便未来扩展。
    """
    if state.get("is_clear", False):
        sub_questions = state.get("sub_questions", [])
        if not sub_questions:
            return "aggregate_answer"
        return [
            Send("retrieve", {
                "question": q,
                "sub_question": q,
                "question_index": i,
            })
            for i, q in enumerate(sub_questions)
        ]
    return "aggregate_answer"


def _build_clarification_questions(question: str, clarification_reason: str) -> list[str]:
    """根据原问题和澄清说明构造可操作的澄清问题列表。

    策略：把模糊的 clarification_reason 转为可操作的选择/补充题。
    返回 1-2 个澄清问题，避免给用户造成认知负担。
    """
    # 默认问"想查哪条路线/哪个方面"
    questions: list[str] = []
    if clarification_reason:
        questions.append(clarification_reason)
    if not questions:
        questions.append("你想了解哪条路线或哪个具体方面？")
    return questions[:2]  # 至多 2 个问题


def _merge_clarification_with_question(
    question: str, answers: list[str] | None
) -> list[str]:
    """把澄清答案与原问题合并为自包含检索输入（08 票据）。

    按冻结契约：澄清答案与原问题合并为自包含检索输入，
    用户无需重述问题。

    策略：把答案拼到原问题后形成自包含子问题，避免指代丢失。
    """
    if not answers:
        return [question] if question else []
    # 把答案用"；"拼接，附加到原问题后形成自包含检索输入
    answer_text = "；".join(a.strip() for a in answers if a and a.strip())
    if not answer_text:
        return [question] if question else []
    combined = f"{question}（补充：{answer_text}）" if question else answer_text
    # 同时保留原问题作为第二个子问题（提高召回率）
    return [combined] + ([question] if question else [])


def request_clarification(
    state: RAGState,
    *,
    clarification_store: Any = None,
) -> dict[str, Any]:
    """节点：触发澄清交互（08 票据）。

    使用 LangGraph interrupt() 暂停图执行，等待用户提交澄清答案。

    流程：
    1. 首次进入（无 clarification_answers）：
       - 创建 ClarificationStatus（幂等，存入 store）
       - 调用 interrupt(clarification_info) 暂停图
       - API 层检测到 interrupt 后构造 clarification_needed SSE 事件 + done
    2. 恢复（用户提交 Command(resume=answers)）：
       - interrupt() 返回 answers（list[str]）
       - 标记 clarification 为 resumed
       - 把 answers + 原问题合并为自包含检索输入
       - 设置 is_clear=True，供后续路由 fan-out 到 retrieve

    Args:
        clarification_store: ClarificationStore 实例；None 时用全局单例

    Returns:
        - 首次（interrupt 前，实际不会执行到 return）：
          {"clarification_needed": dict}
        - 恢复后：
          {"clarification_needed": dict, "clarification_answers": list,
           "is_clear": True, "sub_questions": list}
    """
    from langgraph.types import interrupt

    if clarification_store is None:
        from rag.clarification import get_clarification_store
        clarification_store = get_clarification_store()

    question = state.get("question", "")
    chat_id = state.get("chat_id", "")
    clarification_reason = state.get("clarification_reason", "")

    # 构造可操作的澄清问题
    clarification_questions = _build_clarification_questions(
        question, clarification_reason
    )

    # 创建澄清状态（幂等：同 chat_id + 同 question 返回同 clarification_id）
    status = clarification_store.create(
        chat_id=chat_id,
        question=question,
        questions=clarification_questions,
    )

    # 票据 03：契约§5 - 澄清期间发新问题，旧 clarification 自动过期
    # 同 chat_id 下其他 active clarification 标记为 expired（除刚创建的之外）
    # 与 legacy 路径行为对齐（api/rag.py legacy 分支也调用 expire_active_for_chat）
    clarification_store.expire_active_for_chat(
        chat_id=chat_id,
        except_clarification_id=status.clarification_id,
    )

    # 准备 SSE 事件需要的 metadata
    clarification_info = {
        "clarification_id": status.clarification_id,
        "questions": list(status.questions),
        "expires_at": status.expires_at,
    }

    # 触发 interrupt：值是 clarification_info
    # API 层检测到 interrupt 后构造 clarification_needed SSE 事件
    # 用户提交答案后，Command(resume=answers) 恢复，answers 进入此节点返回值
    answers = interrupt(clarification_info)

    # 恢复后：answers 应为 list[str]（防御性：单值时包装为 list）
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        answers = list(answers) if answers else []

    # 标记 clarification 为 resumed（幂等：多次 resume 由 checkpoint 去重）
    clarification_store.mark_resumed(status.clarification_id)

    # 把澄清答案 + 原问题合并为自包含检索输入
    combined_questions = _merge_clarification_with_question(question, answers)

    return {
        "clarification_needed": clarification_info,
        "clarification_answers": answers,
        "is_clear": True,  # 恢复后视为清晰，进入 retrieve
        "sub_questions": combined_questions,
    }


def build_rag_graph(
    *,
    llm: LLMProtocol | None = None,
    retriever: RetrieverProtocol | None = None,
    checkpointer: Any = None,
    recent_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    retrieve_k: int = 4,
    use_retrieve_subgraph: bool = False,
    parent_retriever_fn: ParentRetrieverFn | None = None,
    subgraph_max_iterations: int = 5,
    subgraph_max_tool_calls: int = 4,
    subgraph_max_token_estimate: int = 4000,
    enable_clarification: bool = False,
    clarification_store: Any = None,
    augmenter: Any = None,
    rewriter: Any = None,
) -> Any:
    """构建 RAG 生命周期主图。

    图结构（enable_clarification=False，默认，零回归）：
        START → summarize_conversation → analyze_query → [conditional]
                                                           ├─ is_clear=True
                                                           │  → Send("retrieve") × N
                                                           │  → aggregate_answer → END
                                                           └─ is_clear=False
                                                              → aggregate_answer → END

    图结构（enable_clarification=True，08 票据澄清模式）：
        START → summarize_conversation → analyze_query → [conditional]
                                                           ├─ is_clear=True
                                                           │  → Send("retrieve") × N
                                                           │  → aggregate_answer → END
                                                           └─ is_clear=False
                                                              → request_clarification (interrupt)
                                                              → [resume Command(resume=answers)]
                                                              → [conditional]
                                                                 ├─ is_clear=True
                                                                 │  → Send("retrieve") × N
                                                                 │  → aggregate_answer → END
                                                                 └─ is_clear=False
                                                                    → aggregate_answer → END

    Args:
        llm: LLM 调用协议实例（None = 用 fallback 规则）
        retriever: 检索器协议实例（None = 返回空文档）
        checkpointer: LangGraph checkpointer 实例（None = 不持久化）。
            08 票据：澄清模式必须传入 checkpointer，否则 interrupt 无法恢复。
        recent_limit: 摘要保留的最近消息条数
        max_sub_questions: 单次分析最多子问题数
        retrieve_k: 每个子问题检索的文档数
        use_retrieve_subgraph: 是否启用 06 票据的检索子图（预算+压缩+fallback）。
            默认 False，用 05 票据的简单 retrieve 节点（零回归）。
            True 时 retrieve 节点调用 build_retrieve_subgraph 编译实例，
            子图观测消息合并到主图 process_messages。
        parent_retriever_fn: parent 回溯回调（仅 use_retrieve_subgraph=True 时生效）
        subgraph_max_iterations: 子图最大迭代次数
        subgraph_max_tool_calls: 子图最大工具调用次数
        subgraph_max_token_estimate: 子图工作记忆 token 上限
        enable_clarification: 是否启用 08 票据澄清双请求模式。
            默认 False，保持 05/07 票据行为零回归。
            True 时不清晰问题进入 request_clarification 节点（interrupt 触发澄清），
            用户提交答案后经 Command(resume=) 恢复执行。
        clarification_store: ClarificationStore 实例（仅 enable_clarification=True 生效）。
            None 时用全局单例 get_clarification_store()。
        augmenter: ContextAugmenter 实例（09 票据领域能力融合）。
            None = 走 llm.invoke 路径（零回归）；非 None 时聚合节点调用 augmenter.augment。
        rewriter: QueryRewriter 实例（09 票据领域能力融合）。
            None = 不应用 humanize_for_answer（零回归）；非 None 时在最终边界应用一次。

    Returns:
        编译后的 LangGraph 可执行图
    """
    graph = StateGraph(RAGState)

    # 绑定依赖（闭包捕获）
    def _summarize(state: RAGState) -> dict[str, Any]:
        return summarize_conversation(state, llm=llm, recent_limit=recent_limit)

    def _analyze(state: RAGState) -> dict[str, Any]:
        return analyze_query(
            state,
            llm=llm,
            max_sub_questions=max_sub_questions,
            recent_limit=recent_limit,
        )

    def _retrieve(state: dict[str, Any]) -> dict[str, Any]:
        return retrieve(state, retriever=retriever, k=retrieve_k)

    def _aggregate(state: RAGState) -> dict[str, Any]:
        # 09 票据：领域能力融合 - 传递 augmenter/rewriter 给 aggregate_answer
        return aggregate_answer(state, llm=llm, augmenter=augmenter, rewriter=rewriter)

    def _request_clarification(state: RAGState) -> dict[str, Any]:
        return request_clarification(state, clarification_store=clarification_store)

    # 根据 use_retrieve_subgraph 开关选择 retrieve 节点实现
    if use_retrieve_subgraph:
        # 编译一次子图，闭包捕获复用（避免每次 retrieve 调用都重新编译）
        from rag.retrieve_subgraph import build_retrieve_subgraph as _build_subgraph

        _compiled_subgraph = _build_subgraph(
            llm=llm,
            retriever=retriever,
            parent_retriever_fn=parent_retriever_fn,
            max_iterations=subgraph_max_iterations,
            max_tool_calls=subgraph_max_tool_calls,
            max_token_estimate=subgraph_max_token_estimate,
            retrieve_k=retrieve_k,
        )

        def _retrieve_via_compiled_subgraph(state: dict[str, Any]) -> dict[str, Any]:
            """通过预编译的子图执行 retrieve（06 票据，编译复用版）。

            07 票据：返回的 final_docs 带 question_index，供 aggregate_answer 保序聚合。
            """
            sub_question = state.get("question", state.get("sub_question", ""))
            question_index = state.get("question_index", 0)
            result = _compiled_subgraph.invoke({"sub_question": sub_question})
            subgraph_process = result.get("subgraph_process", [])
            # 07 票据：给每条 final_doc 标记 question_index
            final_docs = result.get("final_docs", [])
            for doc in final_docs:
                if isinstance(doc, dict):
                    doc["question_index"] = question_index
            return {
                "retrieved_docs": final_docs,
                "process_messages": subgraph_process,
            }

        retrieve_node = _retrieve_via_compiled_subgraph
    else:
        retrieve_node = _retrieve

    graph.add_node("summarize_conversation", _summarize)
    graph.add_node("analyze_query", _analyze)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("aggregate_answer", _aggregate)

    graph.add_edge(START, "summarize_conversation")
    graph.add_edge("summarize_conversation", "analyze_query")

    if enable_clarification:
        # 08 票据：澄清模式
        # 不清晰问题 → request_clarification（interrupt 触发）
        # resume 后 → 条件路由（is_clear=True 时 fan-out 到 retrieve）
        graph.add_node("request_clarification", _request_clarification)
        graph.add_conditional_edges(
            "analyze_query",
            _route_after_analysis_with_clarification,
            {
                "aggregate_answer": "aggregate_answer",
                "request_clarification": "request_clarification",
                # Send 路径会自动路由到 retrieve 节点
            },
        )
        graph.add_conditional_edges(
            "request_clarification",
            _route_after_clarification,
            {
                "aggregate_answer": "aggregate_answer",
                # Send 路径会自动路由到 retrieve 节点
            },
        )
    else:
        # 默认模式（零回归）：不清晰问题直接进 aggregate_answer
        graph.add_conditional_edges(
            "analyze_query",
            _route_after_analysis,
            {
                "aggregate_answer": "aggregate_answer",
                # Send 路径会自动路由到 retrieve 节点
            },
        )

    graph.add_edge("retrieve", "aggregate_answer")
    graph.add_edge("aggregate_answer", END)

    return graph.compile(checkpointer=checkpointer)


async def run_rag_graph(
    graph: Any,
    question: str,
    chat_id: str,
    messages: list[dict[str, str]] | None = None,
    *,
    thread_id_for_fn: Any = None,
    clarification_answers: list[str] | None = None,
) -> dict[str, Any]:
    """运行 RAG 主图并返回最终状态。

    08 票据：支持澄清双请求
    - 首请求：clarification_answers=None，图执行到 request_clarification 时
      interrupt 暂停，返回部分 state（含 clarification_needed）
    - resume 请求：clarification_answers 非空，用 Command(resume=answers) 恢复执行，
      从 request_clarification 节点继续，最终完成全图

    Args:
        graph: build_rag_graph 编译的图
        question: 用户问题
        chat_id: 聊天 ID（用作 LangGraph thread_id）
        messages: 对话历史
        thread_id_for_fn: chat_id → thread_id 适配函数（默认用 chat_id 本身）
        clarification_answers: 澄清答案列表（08 票据 resume 请求时传入）。
            None 表示首请求；非空 list 表示 resume 请求。

    Returns:
        最终 RAGState（含 answer, retrieved_docs, process_messages 等）。
        首请求触发 interrupt 时返回部分 state（含 clarification_needed）。

    Note:
        不在 initial_state 中传 summary 字段，让 checkpointer 持久化的 summary
        跨轮滚动累积（P0-3 修复）。累加器（retrieved_docs/process_messages）
        由 summarize_conversation 节点返回 _RESET 哨兵重置（P0-1 修复）。
    """
    if thread_id_for_fn is None:
        thread_id = chat_id
    else:
        thread_id = thread_id_for_fn(chat_id)

    config = {"configurable": {"thread_id": thread_id}}

    if clarification_answers is not None:
        # 08 票据：resume 请求，用 Command(resume=answers) 恢复
        from langgraph.types import Command
        result = await graph.ainvoke(
            Command(resume=list(clarification_answers)), config=config
        )
        return result

    # 首请求：传 initial_state，图执行到 interrupt 时暂停
    # 不传 summary / retrieved_docs / process_messages：
    # - summary：依赖 checkpointer 跨轮持久化（滚动累积）
    # - retrieved_docs / process_messages：由 summarize_conversation 节点重置
    initial_state: dict[str, Any] = {
        "question": question,
        "chat_id": chat_id,
        "messages": messages or [],
    }

    result = await graph.ainvoke(initial_state, config=config)
    return result


def is_clarification_needed(state: dict[str, Any]) -> bool:
    """检查 state 是否处于澄清中断状态（08 票据，API 层辅助函数）。

    当图执行到 request_clarification 节点并调用 interrupt 后，
    LangGraph 返回的 state 包含 __interrupt__ 字段（Interrupt 对象列表）。
    每个 Interrupt.value 是 clarification_info dict。

    Args:
        state: graph.ainvoke() 的返回值或 aget_state().values

    Returns:
        True 表示当前处于澄清中断状态，需向用户推送 clarification_needed SSE 事件
    """
    # 优先检查 __interrupt__ 字段（LangGraph interrupt 暂停时的标准格式）
    interrupts = state.get("__interrupt__")
    if interrupts:
        for intr in interrupts:
            # Interrupt 对象有 value 属性
            value = getattr(intr, "value", None) if not isinstance(intr, dict) else intr.get("value")
            if isinstance(value, dict) and "clarification_id" in value:
                return True
    # 兼容性检查：state 直接含 clarification_needed 字段
    return bool(state.get("clarification_needed"))


def get_clarification_info(state: dict[str, Any]) -> dict[str, Any] | None:
    """从 state 提取澄清信息（08 票据，API 层辅助函数）。

    Returns:
        澄清信息 dict（含 clarification_id, questions, expires_at）或 None
    """
    # 优先从 __interrupt__ 提取（LangGraph interrupt 暂停时的标准格式）
    interrupts = state.get("__interrupt__")
    if interrupts:
        for intr in interrupts:
            value = getattr(intr, "value", None) if not isinstance(intr, dict) else intr.get("value")
            if isinstance(value, dict) and "clarification_id" in value:
                return value
    # 兼容性：state 直接含 clarification_needed 字段
    info = state.get("clarification_needed")
    if isinstance(info, dict):
        return info
    return None


__all__ = [
    "DEFAULT_MAX_SUB_QUESTIONS",
    "DEFAULT_RECENT_MESSAGE_LIMIT",
    "LLMProtocol",
    "RAGState",
    "RetrieverProtocol",
    "aggregate_answer",
    "analyze_query",
    "build_rag_graph",
    "get_clarification_info",
    "is_clarification_needed",
    "request_clarification",
    "run_rag_graph",
    "retrieve",
    "summarize_conversation",
]
