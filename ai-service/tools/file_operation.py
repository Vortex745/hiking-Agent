from pathlib import Path
from langchain_core.tools import tool
from runtime_paths import runtime_dir

WORKSPACE_DIR = runtime_dir("WORKSPACE_DIR", "workspace")


def _resolve_path(path: str) -> Path:
    """Resolve and validate file path within workspace."""
    workspace = WORKSPACE_DIR.resolve()
    full_path = (workspace / path).resolve()
    try:
        full_path.relative_to(workspace)
    except ValueError:
        raise ValueError("文件路径超出工作目录范围")
    return full_path


@tool
async def file_operation(operation: str, path: str, content: str | None = None) -> str:
    """Perform file operations: read, write, list, or mkdir.

    Args:
        operation: One of 'read', 'write', 'create', 'update', 'delete', 'list', 'mkdir'
        path: File or directory path (relative to workspace)
        content: File content (required for 'write' operation)
    """
    try:
        resolved = _resolve_path(path)

        if operation == "read":
            if not resolved.exists():
                return f"文件不存在: {path}"
            return resolved.read_text(encoding="utf-8")

        elif operation in {"write", "create", "update"}:
            if content is None:
                return "写入操作需要提供 content 参数"
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return f"文件已写入: {path} ({len(content)} 字符)"
        
        elif operation == "delete":
            if not resolved.exists():
                return f"文件不存在: {path}"
            if resolved.is_dir():
                return "不支持通过 file_operation 删除目录"
            resolved.unlink()
            return f"文件已删除: {path}"

        elif operation == "list":
            target = resolved if resolved.exists() else WORKSPACE_DIR
            items = []
            for item in target.iterdir():
                suffix = "/" if item.is_dir() else ""
                items.append(f"{item.name}{suffix}")
            return "\n".join(sorted(items)) if items else "目录为空"

        elif operation == "mkdir":
            resolved.mkdir(parents=True, exist_ok=True)
            return f"目录已创建: {path}"

        else:
            return f"不支持的操作: {operation}（支持: read, write, create, update, delete, list, mkdir）"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"文件操作出错: {str(e)}"
