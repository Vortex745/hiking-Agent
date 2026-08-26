import logging

import httpx
from fastapi import APIRouter, HTTPException

from api.models import ModelsFetchRequest, ModelsFetchResponse

logger = logging.getLogger("ai-service.models")
models_router = APIRouter(prefix="/models")


@models_router.post("/fetch", response_model=ModelsFetchResponse)
async def models_fetch(req: ModelsFetchRequest):
    base_url = req.base_url.rstrip("/")
    url = f"{base_url}/models"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {req.api_key}"},
            )
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream returned {resp.status_code}",
                )

        data = resp.json()
        raw = data.get("data", [])
        model_ids = sorted(
            m.get("id", "") for m in raw if isinstance(m, dict) and m.get("id")
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
