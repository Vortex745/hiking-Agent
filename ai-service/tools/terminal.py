import asyncio
from langchain_core.tools import tool

# Whitelist of allowed commands
ALLOWED_COMMANDS = {
    "ls", "cat", "pwd", "echo", "python", "pip", "mkdir", "cp", "mv", "whoami", "date", "head", "tail", "wc", "sort", "grep", "find"
}

BLOCKED_PATTERNS = ["rm", "sudo", "curl", "wget", "chmod", "chown", "kill", "dd", ">"]


def _is_safe(command: str) -> tuple[bool, str]:
    """Check if command is safe to execute."""
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False, "命令为空"

    base_cmd = cmd_parts[0]

    # Check for blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_parts:
            return False, f"命令包含禁止使用的指令: {pattern}"

    # Check whitelist
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"命令不在白名单中: {base_cmd}（允许: {', '.join(sorted(ALLOWED_COMMANDS))}）"

    return True, ""


@tool
async def terminal(command: str) -> str:
    """Execute a shell command in a sandboxed environment. Only whitelisted commands are allowed.

    Allowed: ls, cat, pwd, echo, python, pip, mkdir, cp, mv, whoami, date, head, tail, wc, sort, grep, find
    Blocked: rm, sudo, curl, wget, chmod, chown, kill, dd
    """
    safe, msg = _is_safe(command)
    if not safe:
        return f"[安全限制] {msg}"

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="./workspace",
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            process.kill()
            return "命令执行超时（30 秒）"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            if output:
                output += "\n"
            output += stderr.decode("utf-8", errors="replace")

        return output.strip() or "命令执行完成（无输出）"
    except Exception as e:
        return f"执行错误: {str(e)}"
