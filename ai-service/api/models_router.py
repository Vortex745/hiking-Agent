import ipaddress
import logging
import socket
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException

from api.models import ModelsFetchRequest, ModelsFetchResponse

logger = logging.getLogger("ai-service.models")
models_router = APIRouter(prefix="/models")


_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
}


def _validate_public_base_url(raw_url: str) -> str:
    """Validate a model provider URL before making a server-side request.

    The endpoint is intentionally limited to public HTTP(S) origins. This
    prevents a browser user from turning the AI service into a proxy for
    loopback, cloud metadata, or private network addresses.
    """
    candidate = raw_url.strip().rstrip("/")
    try:
        parsed = urlsplit(candidate)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url is malformed") from exc

    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError("base_url must use http or https and include a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query, or fragment")

    normalized_host = host.rstrip(".").lower()
    if normalized_host in _BLOCKED_HOSTNAMES:
        raise ValueError("base_url host is not allowed")

    try:
        resolved = [ipaddress.ip_address(normalized_host)]
    except ValueError:
        try:
            addresses = socket.getaddrinfo(
                normalized_host,
                port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError("base_url host could not be resolved") from exc
        resolved = [ipaddress.ip_address(item[4][0]) for item in addresses]

    if not resolved or any(not address.is_global for address in resolved):
        raise ValueError("base_url must resolve to a public address")

    return candidate


@models_router.post("/fetch", response_model=ModelsFetchResponse)
async def models_fetch(req: ModelsFetchRequest):
    try:
        base_url = _validate_public_base_url(req.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    url = f"{base_url}/models"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {req.api_key}"},
            )
            if 300 <= resp.status_code < 400:
                raise HTTPException(
                    status_code=502,
                    detail="Model provider redirects are not followed",
                )
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream returned {resp.status_code}",
                )

        try:
            data = resp.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Model provider returned invalid JSON",
            ) from exc
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
