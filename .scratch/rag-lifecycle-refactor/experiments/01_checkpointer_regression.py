"""01 票据最小回归实验：AsyncSqliteSaver + LangGraph 1.2.6 + Command(resume=)。

按 `.wayfinder/maps/rag-lifecycle-refactor/tickets/checkpoint-adapter.md` 第 8 节
最小回归实验定义改写为 SQLite saver 版本（用户决策覆盖 Postgres 推荐）。

5 项验收：
1. import 成功（1.2.6 + langgraph-checkpoint-sqlite 安装正确）
2. setup() 成功创建 checkpoint 表
3. 第一次 ainvoke 停在 interrupt 点
4. 第二次 ainvoke(Command(resume=...)) 恢复并完成
5. 跨进程恢复：重启 Python 进程后用同 thread_id 仍能 resume

使用方法：
    python 01_checkpointer_regression.py setup_and_interrupt
    python 01_checkpointer_regression.py resume
    python 01_checkpointer_regression.py full
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict

# 确保 ai-service 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ai-service"))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

DB_PATH = Path(__file__).resolve().parent / "rag_checkpoints_test.sqlite"
THREAD_ID = "exp-01-1"


class State(TypedDict):
    messages: list[str]


def ask_node(state: State) -> State:
    answer = interrupt("question?")
    return {"messages": state.get("messages", []) + [f"answered: {answer}"]}


async def setup_and_interrupt() -> dict:
    """第 1-3 项验收：import + setup + 第一次 invoke 触发 interrupt。"""
    if DB_PATH.exists():
        DB_PATH.unlink()
    result: dict = {"imports_ok": False, "setup_ok": False, "interrupt_ok": False}
    try:
        # 验收 1：import
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401

        result["imports_ok"] = True
    except Exception as exc:
        result["import_error"] = str(exc)
        return result

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        # 验收 2：setup()
        try:
            await saver.setup()
            result["setup_ok"] = True
        except Exception as exc:
            result["setup_error"] = str(exc)
            return result

        builder = StateGraph(State)
        builder.add_node("ask", ask_node)
        builder.set_entry_point("ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": THREAD_ID}}
        # 验收 3：第一次 ainvoke 应停在 interrupt 点
        # langgraph 1.2：interrupt 时 ainvoke 正常返回（部分 state），不抛异常
        try:
            await graph.ainvoke({"messages": []}, config=config)
        except Exception as exc:
            # 异常也接受，只要后续 state.next 表明停在 interrupt
            result["interrupt_invoke_error"] = str(exc)
        state = await graph.aget_state(config)
        result["interrupt_state_next"] = list(state.next) if state.next else []
        result["interrupt_ok"] = bool(state.next)
        if not result["interrupt_ok"]:
            result["interrupt_error"] = (
                f"no interrupt detected; state.next={state.next}, "
                f"values={state.values}"
            )
    return result


async def resume_only() -> dict:
    """第 4-5 项验收：跨进程 resume 恢复。"""
    result: dict = {"resume_ok": False}
    if not DB_PATH.exists():
        result["resume_error"] = f"db not found: {DB_PATH}"
        return result

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        # 不调用 setup()——表已存在；模拟跨进程恢复
        builder = StateGraph(State)
        builder.add_node("ask", ask_node)
        builder.set_entry_point("ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": THREAD_ID}}
        # 验收 4-5：第二次 ainvoke(Command(resume=...)) 恢复并完成
        try:
            final_state = await graph.ainvoke(Command(resume="yes"), config=config)
            messages = (final_state or {}).get("messages", [])
            if messages == ["answered: yes"]:
                result["resume_ok"] = True
                result["final_messages"] = messages
            else:
                result["resume_error"] = f"unexpected final state: {final_state}"
        except Exception as exc:
            result["resume_error"] = str(exc)
    return result


async def full_in_process() -> dict:
    """单进程完整 5 项验收（不含跨进程，但含 setup+interrupt+resume）。"""
    if DB_PATH.exists():
        DB_PATH.unlink()
    result: dict = {}

    # 1. import
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401

        result["imports_ok"] = True
    except Exception as exc:
        result["imports_ok"] = False
        result["import_error"] = str(exc)
        return result

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        # 2. setup
        try:
            await saver.setup()
            result["setup_ok"] = True
        except Exception as exc:
            result["setup_ok"] = False
            result["setup_error"] = str(exc)
            return result

        builder = StateGraph(State)
        builder.add_node("ask", ask_node)
        builder.set_entry_point("ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": THREAD_ID}}

        # 3. 第一次 ainvoke 应停在 interrupt
        try:
            await graph.ainvoke({"messages": []}, config=config)
            state = await graph.aget_state(config)
            result["interrupt_ok"] = bool(state.next)
            result["interrupt_state_next"] = list(state.next) if state.next else []
        except Exception as exc:
            result["interrupt_ok"] = False
            result["interrupt_error"] = str(exc)
            return result

        # 4. resume
        try:
            final_state = await graph.ainvoke(Command(resume="yes"), config=config)
            messages = (final_state or {}).get("messages", [])
            result["resume_ok"] = messages == ["answered: yes"]
            result["final_messages"] = messages
        except Exception as exc:
            result["resume_ok"] = False
            result["resume_error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["setup_and_interrupt", "resume", "full"])
    args = parser.parse_args()

    if args.mode == "setup_and_interrupt":
        result = asyncio.run(setup_and_interrupt())
    elif args.mode == "resume":
        result = asyncio.run(resume_only())
    else:
        result = asyncio.run(full_in_process())

    print("=== 01 Checkpointer Regression Experiment ===")
    print(f"mode: {args.mode}")
    print(f"db: {DB_PATH}")
    print(f"thread_id: {THREAD_ID}")
    print(f"result: {result}")

    # 退出码：所有验收通过=0
    if args.mode == "setup_and_interrupt":
        ok = result.get("imports_ok") and result.get("setup_ok") and result.get("interrupt_ok")
    elif args.mode == "resume":
        ok = result.get("resume_ok")
    else:
        ok = (
            result.get("imports_ok")
            and result.get("setup_ok")
            and result.get("interrupt_ok")
            and result.get("resume_ok")
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
