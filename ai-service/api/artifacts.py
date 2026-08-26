from pathlib import Path
import logging
import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from runtime_paths import runtime_dir

logger = logging.getLogger("ai-service.artifacts")


artifacts_router = APIRouter(prefix="/artifacts")

WORKSPACE_DIR = runtime_dir("WORKSPACE_DIR", "workspace")
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".html",
    ".zip",
}


def _resolve_artifact_path(file_path: str) -> Path:
    if not file_path or Path(file_path).is_absolute():
        raise HTTPException(status_code=400, detail="无效的文件路径")

    workspace = WORKSPACE_DIR.resolve()
    target = (workspace / file_path).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise HTTPException(status_code=400, detail="文件路径超出 workspace 范围")

    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持下载该文件类型")
    if not target.is_file():
        logger.warning("Artifact not found: %s (resolved: %s)", file_path, target)
        raise HTTPException(status_code=404, detail="文件不存在或已过期")
    return target


@artifacts_router.get("/{file_path:path}")
async def download_artifact(file_path: str):
    logger.info("Artifact download request: %s", file_path)
    target = _resolve_artifact_path(file_path)
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        path=target,
        media_type=media_type,
        filename=target.name,
    )
