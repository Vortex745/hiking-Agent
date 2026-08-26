"""Minimal stdio MCP server for the Pexels REST API."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

API_BASE = "https://api.pexels.com/v1"
DEFAULT_PER_PAGE = 5
MAX_PER_PAGE = 10


def _text_response(request_id: Any, payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


def _error_response(request_id: Any, message: str, *, code: int = -32000) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _clamp_per_page(value: Any) -> int:
    try:
        per_page = int(value or DEFAULT_PER_PAGE)
    except (TypeError, ValueError):
        per_page = DEFAULT_PER_PAGE
    return max(1, min(per_page, MAX_PER_PAGE))


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def _optional_params(arguments: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: arguments[name]
        for name in names
        if arguments.get(name) not in (None, "")
    }


def _pexels_get(path: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    api_key = os.getenv("PEXELS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY is not configured")

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        response = client.get(
            f"{API_BASE}{path}",
            params=params,
            headers={"Authorization": api_key},
        )
        response.raise_for_status()
        headers = {
            "limit": response.headers.get("X-Ratelimit-Limit", ""),
            "remaining": response.headers.get("X-Ratelimit-Remaining", ""),
            "reset": response.headers.get("X-Ratelimit-Reset", ""),
        }
        return response.json(), headers


def _photo_summary(photo: dict[str, Any]) -> dict[str, Any]:
    src = photo.get("src") if isinstance(photo.get("src"), dict) else {}
    photographer = photo.get("photographer")
    alt = photo.get("alt") or ""
    medium_url = src.get("medium") or src.get("large") or src.get("original") or ""
    return {
        "id": photo.get("id"),
        "url": photo.get("url"),
        "photographer": photographer,
        "photographer_url": photo.get("photographer_url"),
        "alt": alt,
        "avg_color": photo.get("avg_color"),
        "width": photo.get("width"),
        "height": photo.get("height"),
        "src": {
            "original": src.get("original"),
            "large": src.get("large"),
            "medium": src.get("medium"),
            "landscape": src.get("landscape"),
            "portrait": src.get("portrait"),
        },
        "markdown_preview": f"![{alt}]({medium_url})" if medium_url else "",
        "attribution": f"Photo by {photographer} on Pexels" if photographer else "Photo on Pexels",
    }


def _video_file_summary(video: dict[str, Any]) -> list[dict[str, Any]]:
    files = video.get("video_files")
    if not isinstance(files, list):
        return []
    summarized = []
    for file_info in files[:5]:
        if not isinstance(file_info, dict):
            continue
        summarized.append({
            "quality": file_info.get("quality"),
            "file_type": file_info.get("file_type"),
            "width": file_info.get("width"),
            "height": file_info.get("height"),
            "fps": file_info.get("fps"),
            "link": file_info.get("link"),
        })
    return summarized


def _video_summary(video: dict[str, Any]) -> dict[str, Any]:
    user = video.get("user") if isinstance(video.get("user"), dict) else {}
    return {
        "id": video.get("id"),
        "url": video.get("url"),
        "image": video.get("image"),
        "duration": video.get("duration"),
        "width": video.get("width"),
        "height": video.get("height"),
        "user": {
            "name": user.get("name"),
            "url": user.get("url"),
        },
        "video_files": _video_file_summary(video),
        "attribution": f"Video by {user.get('name')} on Pexels" if user.get("name") else "Video on Pexels",
    }


def search_photos(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    params = {
        "query": query,
        "per_page": _clamp_per_page(arguments.get("per_page")),
        "page": _positive_int(arguments.get("page")),
        **_optional_params(arguments, ("orientation", "size", "color", "locale")),
    }
    data, rate_limit = _pexels_get("/search", params)
    photos = data.get("photos") if isinstance(data.get("photos"), list) else []
    return {
        "provider": "Pexels",
        "query": query,
        "page": data.get("page"),
        "per_page": data.get("per_page"),
        "total_results": data.get("total_results"),
        "next_page": data.get("next_page"),
        "photos": [_photo_summary(photo) for photo in photos],
        "rate_limit": rate_limit,
        "usage_note": "Credit photographers and link back to Pexels when using these assets.",
    }


def search_videos(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    params = {
        "query": query,
        "per_page": _clamp_per_page(arguments.get("per_page")),
        "page": _positive_int(arguments.get("page")),
        **_optional_params(arguments, ("orientation", "size", "locale")),
    }
    data, rate_limit = _pexels_get("/videos/search", params)
    videos = data.get("videos") if isinstance(data.get("videos"), list) else []
    return {
        "provider": "Pexels",
        "query": query,
        "url": data.get("url"),
        "page": data.get("page"),
        "per_page": data.get("per_page"),
        "total_results": data.get("total_results"),
        "next_page": data.get("next_page"),
        "videos": [_video_summary(video) for video in videos],
        "rate_limit": rate_limit,
        "usage_note": "Credit creators and link back to Pexels when using these assets.",
    }


TOOLS: dict[str, dict[str, Any]] = {
    "search_photos": {
        "name": "search_photos",
        "description": "搜索 Pexels 徒步目的地或户外场景的照片。当用户请求图片、照片、风景照时调用此工具，用目的地名称或场景描述作为 query。返回结果包含 markdown_preview 字段（Markdown 图片语法），请在回答中直接使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，如目的地名称或户外场景，例如：白云山、梧桐山徒步、mountain hiking"},
                "per_page": {"type": "integer", "description": "Number of results, capped at 10."},
                "page": {"type": "integer", "description": "Page number, starting at 1."},
                "orientation": {"type": "string", "description": "landscape, portrait, or square."},
                "size": {"type": "string", "description": "large, medium, or small."},
                "color": {"type": "string", "description": "Named color or hex color."},
                "locale": {"type": "string", "description": "Locale such as en-US or zh-CN."},
            },
            "required": ["query"],
        },
    },
    "search_videos": {
        "name": "search_videos",
        "description": "Search Pexels videos by topic and return video URLs, preview images, creator attribution, and source links.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. hiking trail, mountain tent."},
                "per_page": {"type": "integer", "description": "Number of results, capped at 10."},
                "page": {"type": "integer", "description": "Page number, starting at 1."},
                "orientation": {"type": "string", "description": "landscape, portrait, or square."},
                "size": {"type": "string", "description": "large, medium, or small."},
                "locale": {"type": "string", "description": "Locale such as en-US or zh-CN."},
            },
            "required": ["query"],
        },
    },
}


def list_tools() -> list[dict[str, Any]]:
    return list(TOOLS.values())


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "search_photos":
        return search_photos(arguments)
    if name == "search_videos":
        return search_videos(arguments)
    raise ValueError(f"Unknown Pexels tool: {name}")


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": request.get("params", {}).get("protocolVersion") or "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ai-hiking/pexels-mcp", "version": "1.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list_tools()}}
    if method == "tools/call":
        params = request.get("params") or {}
        try:
            result = call_tool(str(params.get("name") or ""), params.get("arguments") or {})
        except Exception as exc:
            return _text_response(request_id, {"error": str(exc)}, is_error=True)
        return _text_response(request_id, result)
    return _error_response(request_id, f"Unsupported method: {method}", code=-32601)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = handle_request(json.loads(line))
        except json.JSONDecodeError as exc:
            response = _error_response(None, f"Invalid JSON: {exc}", code=-32700)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
