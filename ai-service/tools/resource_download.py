import httpx
from pathlib import Path
from langchain_core.tools import tool
from runtime_paths import runtime_dir

WORKSPACE_DIR = runtime_dir("WORKSPACE_DIR", "workspace")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".csv", ".json", ".xml", ".html", ".zip"}


@tool
async def resource_download(url: str, save_path: str | None = None) -> str:
    """Download a resource from a URL to local workspace.

    Args:
        url: The URL of the resource to download
        save_path: Optional filename to save as (default: auto-detect from URL)
    """
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.content
            if len(content) > MAX_FILE_SIZE:
                return f"文件过大（{len(content)} bytes），超过限制（{MAX_FILE_SIZE} bytes）"

            if not save_path:
                filename = url.split("/")[-1].split("?")[0]
                if not filename:
                    filename = "downloaded_file"
                save_path = filename

            ext = Path(save_path).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
                return f"不允许下载该文件类型: {ext or '无扩展名'}（允许: {allowed}）"

            workspace = WORKSPACE_DIR.resolve()
            file_path = (workspace / save_path).resolve()
            try:
                file_path.relative_to(workspace)
            except ValueError:
                return "保存路径超出 workspace 范围"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

            return f"文件已下载: {save_path} ({len(content)} bytes)"
    except httpx.HTTPStatusError as e:
        return f"下载失败，HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return f"网络错误: {str(e)}"
    except Exception as e:
        return f"下载出错: {str(e)}"
