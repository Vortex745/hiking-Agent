"""票据 06：检索子图 — 预算控制 + 上下文压缩 + fallback。

单个检索子图在硬预算内自我修正：
1. orchestrator 决策下一步（search_child / search_parent / compress / finalize）
2. search_tool 执行 child 或 parent 检索（去重已执行查询/已见 parent_id）
3. compress 压缩工作记忆，保留来源名/数字/版本/证据缺口，跨压缩持久查询与 parent_id
4. finalizer 产出 fallback 答案 + 观测记录（压缩计数/工具调用数/token 估算）

预算边界：
- max_iterations：迭代达上限 → 强制 finalize（接受已积累证据）
- max_tool_calls：工具调用达上限 → orchestrator 拒绝再 search，强制 compress 或 finalize
- max_token_estimate：token 超阈值 → 强制 compress 或 finalize
- 会超支的工具调用：orchestrator 在判定阶段直接拒绝（不执行）

本模块不修改 05 票据的 graph.py 主图骨架；通过 build_rag_graph(use_retrieve_subgraph=True)
开关式启用，默认关闭以保持 05 票据零回归。

LLM 调用通过 LLMProtocol 抽象，测试用确定性 FakeLLM 驱动。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Annotated, Any, Callable, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger("ai-service.rag.retrieve_subgraph")

# === 预算默认值 ===
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_TOOL_CALLS = 4
DEFAULT_MAX_TOKEN_ESTIMATE = 4000
# 触发压缩的 token 阈值（达 max_token_estimate 的 80% 时压缩）
DEFAULT_COMPRESS_THRESHOLD_RATIO = 0.8
# 单条文档压缩后保留的最大字符数
DEFAULT_DOC_SNIPPET_CHARS = 200


# === 自定义 reducer ===
def _list_accumulator(old: list | None, new: Any) -> list:
    """支持 list 累加的 reducer（与 graph.py 同语义，但不支持 _RESET）。

    - new is list：append 到 old
    - new is None：保留 old
    """
    if new is None:
        return list(old or [])
    return (old or []) + list(new)


# === 协议 ===
@runtime_checkable
class LLMProtocol(Protocol):
    """LLM 调用协议，与 graph.py 同契约。"""

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """检索器协议，与 graph.py 同契约。"""

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        ...


# Parent 检索回调：parent_ids → parent docs 列表
# 独立 callable 注入，避免扩展 05 票据的 RetrieverProtocol 契约
ParentRetrieverFn = Callable[[list[str]], list[dict[str, Any]]]


# === 子图状态 ===
class RetrieveSubgraphState(TypedDict, total=False):
    """检索子图状态。

    设计要点：
    - accumulated_docs / final_docs / subgraph_process 用 _list_accumulator 累加
    - executed_queries / seen_parent_ids 用普通 list（节点返回完整列表覆盖）
      因为它们要做去重判定，需要完整视图
    - 跨压缩持久：executed_queries / seen_parent_ids / compressed_summary 不被压缩清除
    """

    # 输入
    sub_question: str                       # 子问题（Send 派发时传入）

    # 累积检索结果（含 parent_id/child_id/content/source/file_name）
    accumulated_docs: Annotated[list[dict[str, Any]], _list_accumulator]

    # 已执行查询（规范化后），跨压缩持久，去重用
    executed_queries: list[str]
    # 已见 parent_id（从 child docs 提取），跨压缩持久，避免重复回溯
    seen_parent_ids: list[str]

    # 工作记忆（待压缩的原始文本拼接）
    work_memory: str
    # 压缩后摘要（跨压缩持久累积）
    compressed_summary: str
    # 已压缩过的 doc content hash（跨压缩持久，避免重复处理同一 doc）
    compressed_doc_hashes: list[str]
    # 证据缺口（压缩时识别，用于 fallback 说明）
    evidence_gaps: Annotated[list[str], _list_accumulator]

    # 预算计数
    iteration_count: int
    tool_call_count: int
    token_estimate: int
    compression_count: int

    # orchestrator 决策结果（条件边路由依据）
    next_action: str

    # 最终输出
    final_docs: Annotated[list[dict[str, Any]], _list_accumulator]
    # 子图观测消息（写入主图 process_messages）
    subgraph_process: Annotated[list[str], _list_accumulator]


# === 工具函数 ===
def _normalize_query(query: str) -> str:
    """查询规范化：strip + lowercase + 折叠连续空白。

    用于 executed_queries 去重，避免"  q  "和"q"被误判为不同查询。
    """
    return re.sub(r"\s+", " ", query.strip().lower())


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按字符数计，英文按空格分词数计。

    简化策略：len(text) // 2 + 词数（启发式，无需 tokenizer 依赖）。
    """
    if not text:
        return 0
    char_count = len(text)
    word_count = len(text.split())
    # 启发式：中文字符 ≈ 1 token，英文词 ≈ 1.3 token
    return char_count // 2 + int(word_count * 1.3)


def _doc_content_hash(doc: dict[str, Any]) -> str:
    """文档内容哈希（去重用）。"""
    content = doc.get("content", "")
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def _extract_numbers(text: str) -> list[str]:
    """提取文本中的数字与版本号（保留安全事实）。

    覆盖：整数、小数、版本号（如 1.2.3）、带单位的数字（如 500m）。
    """
    # 版本号优先（含两个以上点）
    version_pattern = r"\d+(?:\.\d+){1,}"
    # 数字 + 可选单位
    number_pattern = r"\d+(?:\.\d+)?\s*(?:m|km|cm|mm|级|天|小时|分钟|元|人|°|度)?"

    versions = re.findall(version_pattern, text)
    numbers = re.findall(number_pattern, text)
    # 合并去重保序
    seen: set[str] = set()
    result: list[str] = []
    for n in versions + numbers:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _extract_safe_facts(text: str) -> list[str]:
    """提取含安全关键词的句子（保留安全事实）。

    安全关键词：不、未、禁止、危险、注意、警告、必须、不可、避免。
    """
    if not text:
        return []
    # 按句号/问号/叹号切分
    sentences = re.split(r"[。！？!?]", text)
    safe_keywords = ["不", "未", "禁止", "危险", "注意", "警告", "必须", "不可", "避免"]
    return [
        s.strip()
        for s in sentences
        if s.strip() and any(kw in s for kw in safe_keywords)
    ]


# === 节点函数 ===
def orchestrator(
    state: RetrieveSubgraphState,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_token_estimate: int = DEFAULT_MAX_TOKEN_ESTIMATE,
    compress_threshold_ratio: float = DEFAULT_COMPRESS_THRESHOLD_RATIO,
    llm: LLMProtocol | None = None,
) -> dict[str, Any]:
    """orchestrator 节点：决策下一步动作。

    决策规则（按优先级）：
    1. 首轮强制 search_child：iteration_count==0 且无 compressed_summary → search_child
    2. 迭代达上限：iteration_count >= max_iterations → finalize
    3. 工具调用达上限：tool_call_count >= max_tool_calls → compress（若 token 超）或 finalize
    4. token 超阈值：token_estimate > max_token_estimate * compress_threshold_ratio → compress
    5. 会超支的工具调用：tool_call_count + 1 > max_tool_calls → finalize
    6. 否则用规则决策（无 LLM）或 LLM 决策

    Returns:
        {"next_action": str, "iteration_count": int, "subgraph_process": [str]}
    """
    iteration = state.get("iteration_count", 0)
    tool_calls = state.get("tool_call_count", 0)
    token_est = state.get("token_estimate", 0)
    compressed_summary = state.get("compressed_summary", "")
    accumulated_docs = state.get("accumulated_docs", [])

    new_iteration = iteration + 1
    process_msgs: list[str] = []

    # 规则 1：首轮强制 search_child（除非已有压缩摘要，即压缩上下文已充分）
    if iteration == 0 and not compressed_summary:
        process_msgs.append(
            f"orchestrator[iter={new_iteration}]: 首轮强制 search_child"
        )
        return {
            "next_action": "search_child",
            "iteration_count": new_iteration,
            "subgraph_process": process_msgs,
        }

    # 规则 2：迭代达上限 → finalize
    if new_iteration > max_iterations:
        process_msgs.append(
            f"orchestrator[iter={new_iteration}]: finalize（迭代计数 {new_iteration}）"
        )
        return {
            "next_action": "finalize",
            "iteration_count": new_iteration,
            "subgraph_process": process_msgs,
        }

    # 规则 3：工具调用达上限 → 不能再 search
    tool_budget_exhausted = tool_calls >= max_tool_calls

    # 规则 4：token 超阈值 → 压缩（但 work_memory 为空时无新内容可压缩，直接 finalize）
    compress_threshold = int(max_token_estimate * compress_threshold_ratio)
    work_memory = state.get("work_memory", "")
    if token_est > compress_threshold and iteration > 0:
        if tool_budget_exhausted or not work_memory:
            # 工具耗尽 或 无新内容可压缩 → 直接 finalize
            process_msgs.append(
                f"orchestrator[iter={new_iteration}]: finalize（token={token_est}，"
                f"tool_calls={tool_calls}，work_memory={'空' if not work_memory else '非空'}）"
            )
            return {
                "next_action": "finalize",
                "iteration_count": new_iteration,
                "subgraph_process": process_msgs,
            }
        process_msgs.append(
            f"orchestrator[iter={new_iteration}]: compress（token={token_est}）"
        )
        return {
            "next_action": "compress",
            "iteration_count": new_iteration,
            "subgraph_process": process_msgs,
        }

    # 规则 5：会超支的工具调用（tool_calls + 1 > max_tool_calls）→ finalize
    if tool_budget_exhausted:
        # 工具耗尽但有累积证据 → finalize
        if accumulated_docs:
            process_msgs.append(
                f"orchestrator[iter={new_iteration}]: finalize（tool_calls={tool_calls}，"
                f"docs={len(accumulated_docs)}）"
            )
            return {
                "next_action": "finalize",
                "iteration_count": new_iteration,
                "subgraph_process": process_msgs,
            }
        # 工具耗尽但无证据 → finalize（fallback 会显式说明无证据）
        process_msgs.append(
            f"orchestrator[iter={new_iteration}]: finalize（tool_calls={tool_calls}，无累积证据）"
        )
        return {
            "next_action": "finalize",
            "iteration_count": new_iteration,
            "subgraph_process": process_msgs,
        }

    # 规则 6：默认决策（无 LLM 用规则，有 LLM 用 LLM）
    next_action = _decide_next_action(
        state, llm=llm, tool_calls=tool_calls, max_tool_calls=max_tool_calls
    )
    process_msgs.append(
        f"orchestrator[iter={new_iteration}]: 决策 {next_action}"
    )
    return {
        "next_action": next_action,
        "iteration_count": new_iteration,
        "subgraph_process": process_msgs,
    }


def _decide_next_action(
    state: RetrieveSubgraphState,
    *,
    llm: LLMProtocol | None,
    tool_calls: int,
    max_tool_calls: int,
) -> str:
    """默认决策逻辑（无 LLM 或 LLM 失败时用规则）。

    规则策略：
    - 已有 child docs 但未回溯 parent → search_parent（扩充上下文）
    - 已回溯 parent 或无 parent 可回溯 → finalize（最简策略，不过度迭代）

    LLM 策略（可选）：
    - 让 LLM 判断证据是否充分，不充分时给新 query
    - 当前简化版不实现 LLM 决策（避免 LLM 调用复杂化），保留接口
    """
    accumulated_docs = state.get("accumulated_docs", [])
    seen_parent_ids = state.get("seen_parent_ids", [])

    # 提取已检索 docs 中的 parent_id（未回溯的）
    all_parent_ids = {
        d.get("parent_id")
        for d in accumulated_docs
        if isinstance(d, dict) and d.get("parent_id")
    }
    untraced_parents = all_parent_ids - set(seen_parent_ids)

    # 有未回溯的 parent → search_parent（一次只回溯一批）
    if untraced_parents and tool_calls + 1 <= max_tool_calls:
        return "search_parent"

    # 否则 finalize（最简策略：单次 child 检索 + 必要时一次 parent 回溯）
    return "finalize"


def search_tool(
    state: RetrieveSubgraphState,
    *,
    action: str,
    retriever: RetrieverProtocol | None = None,
    parent_retriever_fn: ParentRetrieverFn | None = None,
    k: int = 4,
) -> dict[str, Any]:
    """search_tool 节点：执行 child 或 parent 检索。

    根据 action 参数执行不同检索：
    - "search_child"：retriever.search(sub_question, k)
    - "search_parent"：parent_retriever_fn(untraced_parent_ids)

    去重逻辑：
    - child 检索：sub_question 规范化后若已执行则跳过（返回空 docs）
    - parent 检索：仅回溯 seen_parent_ids 之外的 parent_id

    Args:
        action: 由 orchestrator 决定，通过 add_node 闭包绑定
        retriever: child 检索器
        parent_retriever_fn: parent 回溯回调
        k: child 检索 top-K

    Returns:
        {"accumulated_docs": list, "executed_queries": list, "seen_parent_ids": list,
         "tool_call_count": int, "work_memory": str, "token_estimate": int,
         "subgraph_process": [str]}
    """
    sub_question = state.get("sub_question", "")
    executed_queries = list(state.get("executed_queries", []))
    seen_parent_ids = list(state.get("seen_parent_ids", []))
    accumulated_docs = list(state.get("accumulated_docs", []))
    work_memory = state.get("work_memory", "")
    token_est = state.get("token_estimate", 0)
    tool_calls = state.get("tool_call_count", 0)
    process_msgs: list[str] = []

    if action == "search_child":
        # 去重：规范化后已执行则跳过
        normalized = _normalize_query(sub_question)
        if normalized in executed_queries:
            process_msgs.append(
                f"search_tool[child]: 查询已执行过，跳过 (q={sub_question!r})"
            )
            return {
                "tool_call_count": tool_calls,  # 不计 tool_call（未实际调用）
                "executed_queries": executed_queries,
                "seen_parent_ids": seen_parent_ids,
                "subgraph_process": process_msgs,
            }

        executed_queries.append(normalized)

        if retriever is None:
            process_msgs.append("search_tool[child]: 无 retriever，返回空 docs")
            return {
                "tool_call_count": tool_calls,  # 无 retriever 未实际调用，不计 tool_call
                "executed_queries": executed_queries,
                "seen_parent_ids": seen_parent_ids,
                "subgraph_process": process_msgs,
            }

        try:
            docs = retriever.search(sub_question, k=k)
        except Exception as exc:
            logger.warning("search_tool[child] failed for %r: %s", sub_question, exc)
            docs = []
            process_msgs.append(f"search_tool[child]: 检索失败 {exc}")

        # 注意：不把 child docs 的 parent_id 加入 seen_parent_ids。
        # seen_parent_ids 只记录"已回溯"的 parent_id；
        # "从 child 提取但未回溯"的 parent_id 由 search_parent 节点
        # 通过 accumulated_docs - seen_parent_ids 动态计算 untraced。
        # 这样 search_parent 才能找到待回溯的 parent。

        # 拼接工作记忆
        new_memory = _docs_to_memory_text(docs)
        work_memory = (work_memory + "\n" + new_memory).strip() if work_memory else new_memory
        token_est = _estimate_tokens(work_memory)

        process_msgs.append(
            f"search_tool[child]: 检索到 {len(docs)} 条，"
            f"token_est={token_est}"
        )

        return {
            "accumulated_docs": docs,
            "executed_queries": executed_queries,
            "seen_parent_ids": seen_parent_ids,  # 不修改
            "tool_call_count": tool_calls + 1,
            "work_memory": work_memory,
            "token_estimate": token_est,
            "subgraph_process": process_msgs,
        }

    if action == "search_parent":
        # 找未回溯的 parent_id
        all_parent_ids = {
            d.get("parent_id")
            for d in accumulated_docs
            if isinstance(d, dict) and d.get("parent_id")
        }
        untraced = [pid for pid in all_parent_ids if pid not in seen_parent_ids]

        if not untraced:
            process_msgs.append("search_tool[parent]: 无未回溯 parent，跳过")
            return {
                "tool_call_count": tool_calls,
                "subgraph_process": process_msgs,
            }

        if parent_retriever_fn is None:
            # 无 parent 检索器，标记为已见（避免下次再尝试）
            process_msgs.append(
                f"search_tool[parent]: 无 parent_retriever_fn，标记 {len(untraced)} 个 parent 为已见"
            )
            return {
                "tool_call_count": tool_calls,
                "seen_parent_ids": seen_parent_ids + untraced,
                "subgraph_process": process_msgs,
            }

        try:
            parent_docs = parent_retriever_fn(untraced)
        except Exception as exc:
            logger.warning("search_tool[parent] failed: %s", exc)
            parent_docs = []
            process_msgs.append(f"search_tool[parent]: 回溯失败 {exc}")

        # 标记为已见
        for pid in untraced:
            if pid not in seen_parent_ids:
                seen_parent_ids.append(pid)

        new_memory = _docs_to_memory_text(parent_docs)
        work_memory = (work_memory + "\n" + new_memory).strip() if work_memory else new_memory
        token_est = _estimate_tokens(work_memory)

        process_msgs.append(
            f"search_tool[parent]: 回溯 {len(untraced)} 个 parent，"
            f"获得 {len(parent_docs)} 条，token_est={token_est}"
        )

        return {
            "accumulated_docs": parent_docs,
            "seen_parent_ids": seen_parent_ids,
            "tool_call_count": tool_calls + 1,
            "work_memory": work_memory,
            "token_estimate": token_est,
            "subgraph_process": process_msgs,
        }

    # 未知 action，不执行
    process_msgs.append(f"search_tool: 未知 action={action!r}，跳过")
    return {
        "subgraph_process": process_msgs,
    }


def _docs_to_memory_text(docs: list[dict[str, Any]]) -> str:
    """把 docs 拼接为工作记忆文本（保留来源名 + 内容）。"""
    parts: list[str] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        source = d.get("file_name", d.get("source", "未知"))
        content = d.get("content", "")
        parts.append(f"[来源:{source}]\n{content}")
    return "\n---\n".join(parts)


def compress(
    state: RetrieveSubgraphState,
    *,
    max_token_estimate: int = DEFAULT_MAX_TOKEN_ESTIMATE,
    doc_snippet_chars: int = DEFAULT_DOC_SNIPPET_CHARS,
    llm: LLMProtocol | None = None,
) -> dict[str, Any]:
    """compress 节点：压缩工作记忆。

    压缩策略：
    1. 去重：按 content 哈希去除重复文档
    2. 保留关键信息：
       - 来源名（file_name/source）
       - 数字与版本号（_extract_numbers）
       - 安全事实句（_extract_safe_facts）
       - 证据缺口（识别"不确定"/"未提及"等表述）
    3. 截断：每条文档保留前 doc_snippet_chars 字符
    4. 跨压缩持久：executed_queries / seen_parent_ids / compressed_summary 不丢失

    Args:
        max_token_estimate: 压缩后目标 token 上限
        doc_snippet_chars: 单条文档 snippet 字符数
        llm: 可选 LLM 做更智能压缩（当前简化版用规则）

    Returns:
        {"compressed_summary": str, "work_memory": str, "compression_count": int,
         "token_estimate": int, "evidence_gaps": list, "subgraph_process": [str]}
    """
    work_memory = state.get("work_memory", "")
    existing_summary = state.get("compressed_summary", "")
    compression_count = state.get("compression_count", 0)
    accumulated_docs = list(state.get("accumulated_docs", []))
    # 跨压缩持久的 doc hash 集合：避免同一 doc 被多次并入摘要
    compressed_doc_hashes = list(state.get("compressed_doc_hashes", []))
    already_compressed: set[str] = set(compressed_doc_hashes)
    process_msgs: list[str] = []

    # 守卫：work_memory 为空时无新内容可压缩，直接跳过（避免空转循环）
    if not work_memory:
        process_msgs.append("compress: 工作记忆为空（无新内容），跳过")
        return {
            "compression_count": compression_count,
            "subgraph_process": process_msgs,
        }

    # 按内容哈希去重 docs，并跳过已压缩过的 doc（避免跨压缩重复处理）
    seen_hashes: set[str] = set()
    unique_docs: list[dict[str, Any]] = []
    new_hashes: list[str] = []  # 本次新压缩的 doc hash
    for d in accumulated_docs:
        h = _doc_content_hash(d)
        if h in already_compressed:
            # 已被上次 compress 处理过，不重复处理
            continue
        if h in seen_hashes:
            # 本次内重复（同 content 多条 doc），只处理一次
            continue
        seen_hashes.add(h)
        unique_docs.append(d)
        new_hashes.append(h)

    # 提取每条文档的关键信息
    compressed_parts: list[str] = []
    evidence_gaps: list[str] = []
    # 已识别的 gap 集合（跨 doc 去重，避免同一 gap 被多次追加）
    existing_gaps = set(state.get("evidence_gaps", []))
    if existing_summary:
        compressed_parts.append(f"[摘要]\n{existing_summary}")

    for d in unique_docs:
        source = d.get("file_name", d.get("source", "未知"))
        content = d.get("content", "")

        # 提取数字与版本
        numbers = _extract_numbers(content)
        # 提取安全事实
        safe_facts = _extract_safe_facts(content)
        # 截断 snippet
        snippet = content[:doc_snippet_chars] + ("..." if len(content) > doc_snippet_chars else "")

        # 识别证据缺口（去重：同一 gap 描述只追加一次）
        gap_keywords = ["不确定", "未提及", "未知", "暂无", "不清楚", "缺乏"]
        gaps = [kw for kw in gap_keywords if kw in content]
        for gap in gaps:
            gap_desc = f"{source}: 含'{gap}'"
            if gap_desc not in existing_gaps:
                evidence_gaps.append(gap_desc)
                existing_gaps.add(gap_desc)

        part = f"[来源:{source}|数字:{','.join(numbers) if numbers else '无'}]"
        if safe_facts:
            part += f"\n安全事实:{'; '.join(safe_facts[:3])}"
        part += f"\n{snippet}"
        compressed_parts.append(part)

    new_summary = "\n---\n".join(compressed_parts) if compressed_parts else existing_summary
    new_token_est = _estimate_tokens(new_summary)

    # 若压缩后 token 仍高，循环截断到 80% 阈值以下（避免边界震荡导致空转）
    if new_token_est > max_token_estimate:
        target = int(max_token_estimate * 0.8)
        max_chars = len(new_summary)
        while _estimate_tokens(new_summary) > target and max_chars > 100:
            max_chars = int(max_chars * 0.8)
            new_summary = new_summary[:max_chars] + "\n[已截断]"
        new_token_est = _estimate_tokens(new_summary)
        process_msgs.append(
            f"compress: 压缩后 token={new_token_est}，已截断到 {len(new_summary)} 字符"
        )

    skipped = len(accumulated_docs) - len(unique_docs)
    process_msgs.append(
        f"compress: 处理 {len(unique_docs)} 条新文档（跳过 {skipped} 条已压缩/重复），"
        f"token {state.get('token_estimate', 0)}→{new_token_est}，"
        f"证据缺口 {len(evidence_gaps)} 处"
    )

    # 更新 compressed_doc_hashes（跨压缩持久）
    updated_hashes = already_compressed | set(new_hashes)

    return {
        "compressed_summary": new_summary,
        "work_memory": "",  # 工作记忆已并入摘要，清空
        "compressed_doc_hashes": list(updated_hashes),
        "compression_count": compression_count + 1,
        "token_estimate": new_token_est,
        "evidence_gaps": evidence_gaps,
        "subgraph_process": process_msgs,
    }


def finalizer(
    state: RetrieveSubgraphState,
    *,
    llm: LLMProtocol | None = None,
    sub_question: str = "",
) -> dict[str, Any]:
    """finalizer 节点：产出最终文档集 + 观测记录。

    输出：
    - final_docs：去重后的累积文档（合并到主图 retrieved_docs）
    - subgraph_process：观测消息（含压缩计数/工具调用数/token 估算）

    fallback 行为：
    - 无证据 → 显式说明 "未找到相关文档"
    - 有证据但缺口 → 在 subgraph_process 中记录（不向用户描述内部限制）
    - 不向用户描述内部预算/迭代限制

    Args:
        llm: 当前简化版不用 LLM（保留接口）
        sub_question: 用于观测记录（可选）
    """
    accumulated_docs = list(state.get("accumulated_docs", []))
    compressed_summary = state.get("compressed_summary", "")
    compression_count = state.get("compression_count", 0)
    tool_call_count = state.get("tool_call_count", 0)
    token_estimate = state.get("token_estimate", 0)
    evidence_gaps = list(state.get("evidence_gaps", []))
    process_msgs: list[str] = []

    # 按内容哈希去重
    seen_hashes: set[str] = set()
    unique_docs: list[dict[str, Any]] = []
    for d in accumulated_docs:
        h = _doc_content_hash(d)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)

    # 兜底：无原始 docs 但有压缩摘要时，将摘要作为一条 doc 加入 final_docs
    # 这样 compressed_summary 的丰富信息（来源/数字/安全事实）能被主图 aggregate_answer 消费
    used_summary_fallback = False
    if not unique_docs and compressed_summary:
        unique_docs = [{
            "content": compressed_summary,
            "file_name": "compressed_summary.md",
            "source": "compressed_summary.md",
            "parent_id": "",
            "child_id": "summary",
            "is_summary_fallback": True,
        }]
        used_summary_fallback = True

    # 观测记录（写入 subgraph_process，由主图合并到 process_messages）
    process_msgs.append(
        f"finalizer: 最终文档 {len(unique_docs)} 条（去重前 {len(accumulated_docs)}），"
        f"压缩 {compression_count} 次，工具调用 {tool_call_count} 次，"
        f"token 估算 {token_estimate}"
    )

    if not unique_docs:
        # 无证据 → 显式说明（不描述内部限制）
        process_msgs.append("finalizer: 无证据，fallback 将显式说明查不到")
    elif used_summary_fallback:
        # 用压缩摘要兜底 → 记录到 subgraph_process（不向用户描述内部限制）
        process_msgs.append("finalizer: 用压缩摘要兜底产出 final_docs")
    elif evidence_gaps:
        # 有证据但缺口 → 记录到 subgraph_process（不向用户描述内部限制）
        process_msgs.append(
            f"finalizer: 识别到 {len(evidence_gaps)} 处证据缺口，"
            f"缺口影响作答时 fallback 会说明"
        )

    return {
        "final_docs": unique_docs,
        "subgraph_process": process_msgs,
    }


# === 条件边路由 ===
def _route_after_orchestrator(state: RetrieveSubgraphState) -> str:
    """条件边：根据 next_action 路由。

    返回值必须匹配 add_conditional_edges 的 path_map key。
    """
    action = state.get("next_action", "finalize")
    if action in ("search_child", "search_parent", "compress", "finalize"):
        return action
    # 未知 action → 兜底 finalize（保证可预测终止）
    return "finalize"


# === 子图构建 ===
def build_retrieve_subgraph(
    *,
    llm: LLMProtocol | None = None,
    retriever: RetrieverProtocol | None = None,
    parent_retriever_fn: ParentRetrieverFn | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_token_estimate: int = DEFAULT_MAX_TOKEN_ESTIMATE,
    compress_threshold_ratio: float = DEFAULT_COMPRESS_THRESHOLD_RATIO,
    retrieve_k: int = 4,
    doc_snippet_chars: int = DEFAULT_DOC_SNIPPET_CHARS,
) -> Any:
    """构建检索子图。

    图结构：
        START → orchestrator ─(search_child)→ search_child_node → orchestrator
                          ─(search_parent)→ search_parent_node → orchestrator
                          ─(compress)→ compress → orchestrator
                          ─(finalize)→ finalizer → END

    Args:
        llm: LLM 调用协议实例（None = 用 fallback 规则）
        retriever: child 检索器（None = 返回空 docs）
        parent_retriever_fn: parent 回溯回调（None = 跳过 parent 回溯）
        max_iterations: 最大迭代次数（orchestrator 执行次数）
        max_tool_calls: 最大工具调用次数（search_child + search_parent 总和）
        max_token_estimate: 工作记忆 token 上限
        compress_threshold_ratio: 触发压缩的阈值比例
        retrieve_k: child 检索 top-K
        doc_snippet_chars: 压缩时单条文档 snippet 字符数

    Returns:
        编译后的 LangGraph 子图
    """
    graph = StateGraph(RetrieveSubgraphState)

    # 闭包绑定参数
    def _orchestrator(state: RetrieveSubgraphState) -> dict[str, Any]:
        return orchestrator(
            state,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_token_estimate=max_token_estimate,
            compress_threshold_ratio=compress_threshold_ratio,
            llm=llm,
        )

    def _search_child(state: RetrieveSubgraphState) -> dict[str, Any]:
        return search_tool(
            state,
            action="search_child",
            retriever=retriever,
            parent_retriever_fn=parent_retriever_fn,
            k=retrieve_k,
        )

    def _search_parent(state: RetrieveSubgraphState) -> dict[str, Any]:
        return search_tool(
            state,
            action="search_parent",
            retriever=retriever,
            parent_retriever_fn=parent_retriever_fn,
            k=retrieve_k,
        )

    def _compress(state: RetrieveSubgraphState) -> dict[str, Any]:
        return compress(
            state,
            max_token_estimate=max_token_estimate,
            doc_snippet_chars=doc_snippet_chars,
            llm=llm,
        )

    def _finalizer(state: RetrieveSubgraphState) -> dict[str, Any]:
        return finalizer(state, llm=llm)

    graph.add_node("orchestrator", _orchestrator)
    graph.add_node("search_child", _search_child)
    graph.add_node("search_parent", _search_parent)
    graph.add_node("compress", _compress)
    graph.add_node("finalizer", _finalizer)

    graph.add_edge(START, "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "search_child": "search_child",
            "search_parent": "search_parent",
            "compress": "compress",
            "finalize": "finalizer",
        },
    )

    # 执行节点回到 orchestrator（循环）
    graph.add_edge("search_child", "orchestrator")
    graph.add_edge("search_parent", "orchestrator")
    graph.add_edge("compress", "orchestrator")
    graph.add_edge("finalizer", END)

    return graph.compile()


# === 子图运行入口 ===
def run_retrieve_subgraph(
    subgraph: Any,
    sub_question: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行检索子图并返回最终状态。

    Args:
        subgraph: build_retrieve_subgraph 编译的子图
        sub_question: 子问题
        config: LangGraph 配置（可选，用于 checkpointer）

    Returns:
        最终 RetrieveSubgraphState（含 final_docs, subgraph_process, 观测字段）
    """
    initial_state: dict[str, Any] = {
        "sub_question": sub_question,
    }
    if config is None:
        result = subgraph.invoke(initial_state)
    else:
        result = subgraph.invoke(initial_state, config=config)
    return result


__all__ = [
    "DEFAULT_COMPRESS_THRESHOLD_RATIO",
    "DEFAULT_DOC_SNIPPET_CHARS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOKEN_ESTIMATE",
    "DEFAULT_MAX_TOOL_CALLS",
    "LLMProtocol",
    "ParentRetrieverFn",
    "RetrieverProtocol",
    "RetrieveSubgraphState",
    "build_retrieve_subgraph",
    "compress",
    "finalizer",
    "orchestrator",
    "run_retrieve_subgraph",
    "search_tool",
]
