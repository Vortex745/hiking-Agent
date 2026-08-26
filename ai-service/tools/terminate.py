from langchain_core.tools import tool


@tool
async def terminate(reason: str = "") -> str:
    """Call this when the task is complete or cannot continue further.

    Args:
        reason: Optional reason for terminating
    """
    msg = "任务已被 Agent 终止"
    if reason:
        msg += f"（原因: {reason}）"
    return msg
