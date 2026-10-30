import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

import config
from api.models import ModelsFetchRequest, ModelsFetchResponse
from auth.deps import current_user_id
from auth.user_config import effective_settings
from config import settings

logger = logging.getLogger("ai-service.models")
models_router = APIRouter(prefix="/models")


@models_router.post("/fetch", response_model=ModelsFetchResponse)
async def models_fetch(req: ModelsFetchRequest, request: Request):
    base_url = req.base_url.rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    api_key = (req.api_key or "").strip()
    if not api_key:
        # The config page intentionally never echoes stored keys. A section
        # request may therefore ask the backend to use its stored key, but
        # only for the same host and only from a loopback page; otherwise the
        # endpoint would be a stored-credential exfiltration primitive.
        if req.section is None:
            raise HTTPException(status_code=400, detail="api_key is required")
        from api.llm_config_router import require_local_request

        require_local_request(request)
        attrs = config.section_settings_attrs().get(req.section)
        if attrs is None:
            raise HTTPException(status_code=400, detail=f"unknown section: {req.section}")
        owner_id = current_user_id(request)
        effective = effective_settings(settings, owner_id) if owner_id else settings
        stored_base_url = str(getattr(effective, attrs[0], "") or "").rstrip("/")
        stored_api_key = str(getattr(effective, attrs[1], "") or "")
        if not stored_api_key:
            raise HTTPException(
                status_code=400, detail=f"{req.section}.api_key 未配置"
            )
        if base_url != stored_base_url:
            raise HTTPException(
                status_code=400,
                detail=(
                    "请求的 base_url 与已存配置不同且未提供 api_key；"
                    "为避免把已存密钥发送到新主机，请求被拒绝"
                ),
            )
        api_key = stored_api_key
    url = f"{base_url}/models"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream returned {resp.status_code}",
                )

        data = resp.json()
        raw = data.get("data", []) if isinstance(data, dict) else []
        model_ids = sorted(
            entry["id"].strip()
            for entry in raw
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and entry["id"].strip()
        )
        return ModelsFetchResponse(models=model_ids)
    except httpx.HTTPStatusError as e:
        logger.warning("Models fetch HTTP error: %s %s", e.response.status_code, url)
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Upstream returned {e.response.status_code}",
        )
    except httpx.RequestError as e:
        logger.warning("Models fetch request error: %s %s", url, e)
        raise HTTPException(status_code=502, detail=f"Cannot connect to {url}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Models fetch error")
        raise HTTPException(status_code=500, detail=str(e))
