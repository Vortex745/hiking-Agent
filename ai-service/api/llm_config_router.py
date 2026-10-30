"""LLM/embedding/rerank 配置读写与连通性测试。

路由只负责：本机来源校验、参数校验、调用 ``config``/``rag`` 助手、组装响应。
它不知道 .env 键名，也不 import ``tools.hiking_knowledge``。
"""

import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

import config
from agent_v2.decisions import DecisionCallError, probe_systemone
from api.local_guard import require_local_request
from api.models import (
    LlmConfigDecisionSectionResponse,
    LlmConfigEmbeddingSectionResponse,
    LlmConfigResponse,
    LlmConfigSectionResponse,
    LlmConfigTestRequest,
    LlmConfigTestResponse,
    LlmConfigUpdateRequest,
    LlmConfigUpdateResponse,
)
from auth.deps import current_user_id
from auth.paths import user_rag_path
from config import settings
from rag.sqlite_schema import read_vec0_dimensions

logger = logging.getLogger("ai-service.llm_config")

llm_config_router = APIRouter(
    prefix="/llm-config", dependencies=[Depends(require_local_request)]
)


def _mask_api_key(value: str) -> tuple[str, bool]:
    """返回 (打码值, 是否已设置)；永不返回明文。"""
    if not value:
        return "", False
    if len(value) < 10:
        return "已设置", True
    return f"{value[:4]}...{value[-4:]}", True


def _index_path(owner_id: str) -> str:
    return (
        str(user_rag_path(settings, owner_id))
        if owner_id
        else settings.rag_sqlite_db_path
    )


def _build_snapshot(owner_id: str = "") -> LlmConfigResponse:
    from auth.user_config import effective_settings, section_sources

    effective = effective_settings(settings, owner_id) if owner_id else settings
    sources = (
        section_sources(settings, owner_id)
        if owner_id
        else {section: "system" for section in ("llm", "embedding", "rerank", "decision")}
    )
    llm_masked, llm_set = _mask_api_key(effective.openai_api_key)
    embedding_masked, embedding_set = _mask_api_key(effective.embedding_api_key)
    rerank_masked, rerank_set = _mask_api_key(effective.rerank_api_key)
    decision_masked, decision_set = _mask_api_key(effective.decision_model_api_key)

    return LlmConfigResponse(
        llm=LlmConfigSectionResponse(
            base_url=effective.openai_base_url,
            model=effective.openai_model,
            api_key_masked=llm_masked,
            api_key_set=llm_set,
        ),
        embedding=LlmConfigEmbeddingSectionResponse(
            base_url=effective.embedding_base_url,
            model=effective.embedding_model,
            dimensions=effective.embedding_dimensions,
            api_key_masked=embedding_masked,
            api_key_set=embedding_set,
        ),
        rerank=LlmConfigSectionResponse(
            base_url=effective.rerank_base_url,
            model=effective.rerank_model,
            api_key_masked=rerank_masked,
            api_key_set=rerank_set,
        ),
        decision=LlmConfigDecisionSectionResponse(
            base_url=effective.decision_model_base_url,
            model=effective.decision_model_name,
            api_key_masked=decision_masked,
            api_key_set=decision_set,
            mode=effective.decision_model_mode,
        ),
        updated_at=config.env_file_updated_at(),
        index_dimensions=read_vec0_dimensions(_index_path(owner_id)),
        sources=sources,
    )


@llm_config_router.get("", response_model=LlmConfigResponse)
async def get_llm_config(request: Request) -> LlmConfigResponse:
    return _build_snapshot(current_user_id(request))


_REQUIRED_SECTIONS = ("llm", "embedding", "rerank")
_DIMENSIONS_MIN = 64
_DIMENSIONS_MAX = 4096


def _require_non_empty(section_name: str, field: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=400, detail=f"{section_name}.{field} 不能为空"
        )
    return value.strip()


def _validate_update(payload: LlmConfigUpdateRequest) -> dict[str, dict[str, object]]:
    """把线上请求收敛成 config/user_config 接受的字段字典。"""
    sections: dict[str, dict[str, object]] = {}
    for section_name in _REQUIRED_SECTIONS:
        section = getattr(payload, section_name)
        if section is None:
            raise HTTPException(
                status_code=400, detail=f"缺少 {section_name} 配置"
            )
        validated: dict[str, object] = {
            "base_url": _require_non_empty(section_name, "base_url", section.base_url),
            "model": _require_non_empty(section_name, "model", section.model),
            "api_key": section.api_key,
        }
        if section_name == "embedding":
            dimensions = section.dimensions
            if dimensions is None:
                raise HTTPException(
                    status_code=400, detail="embedding.dimensions 不能为空"
                )
            if not _DIMENSIONS_MIN <= dimensions <= _DIMENSIONS_MAX:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"embedding.dimensions 必须在 "
                        f"{_DIMENSIONS_MIN}..{_DIMENSIONS_MAX} 之间"
                    ),
                )
            validated["dimensions"] = dimensions
        sections[section_name] = validated
    if payload.decision is not None:
        decision = payload.decision
        validated = {
            "base_url": _require_non_empty(
                "decision", "base_url", decision.base_url
            ),
            "model": _require_non_empty("decision", "model", decision.model),
            "api_key": decision.api_key,
        }
        if decision.mode is not None:
            validated["mode"] = decision.mode
        sections["decision"] = validated
    return sections


@llm_config_router.put("", response_model=LlmConfigUpdateResponse)
async def update_llm_config(
    payload: LlmConfigUpdateRequest,
    request: Request,
) -> LlmConfigUpdateResponse:
    owner_id = current_user_id(request)
    sections = _validate_update(payload)
    requested_dimensions = sections["embedding"]["dimensions"]

    # 维度冲突必须在任何写入之前拒绝，避免配置与索引永久不一致。
    index_dimensions = read_vec0_dimensions(_index_path(owner_id))
    if index_dimensions is not None and index_dimensions != requested_dimensions:
        raise HTTPException(
            status_code=409,
            detail=(
                f"embedding.dimensions={requested_dimensions} 与索引实际维度 "
                f"{index_dimensions} 不一致，需重建索引"
            ),
        )

    if owner_id:
        from auth.user_config import effective_settings, save_overrides

        before = effective_settings(settings, owner_id)
        try:
            changed = save_overrides(settings, owner_id, sections)
        except OSError:
            logger.exception("user llm-config write failed user=%s", owner_id)
            raise HTTPException(status_code=500, detail="账号配置写入失败，配置未保存")
        after = effective_settings(settings, owner_id)
        logger.info(
            "user llm-config updated fields=%s user=%s",
            ", ".join(changed) or "(none)",
            owner_id,
        )
        snapshot = _build_snapshot(owner_id)
        return LlmConfigUpdateResponse(
            **snapshot.model_dump(),
            reindex_recommended=(
                after.embedding_base_url != before.embedding_base_url
                or after.embedding_model != before.embedding_model
                or after.embedding_dimensions != before.embedding_dimensions
            ),
        )

    previous_embedding_base_url = settings.embedding_base_url
    previous_embedding_model = settings.embedding_model
    try:
        changed = config.update_runtime_config(
            sections, env_path=config.ENV_PATH
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError:
        logger.exception("llm-config write failed")
        raise HTTPException(status_code=500, detail=".env 写入失败，配置未保存")

    # 只记录变更字段名，绝不记录字段值。
    logger.info("llm-config updated fields: %s", ", ".join(changed) or "(none)")

    snapshot = _build_snapshot()
    return LlmConfigUpdateResponse(
        **snapshot.model_dump(),
        reindex_recommended=(
            settings.embedding_base_url != previous_embedding_base_url
            or settings.embedding_model != previous_embedding_model
        ),
    )


@llm_config_router.delete("", response_model=LlmConfigUpdateResponse)
async def reset_llm_config(request: Request) -> LlmConfigUpdateResponse:
    owner_id = current_user_id(request)
    if not owner_id:
        raise HTTPException(status_code=400, detail="仅登录账号支持恢复系统默认")
    from auth.user_config import reset_overrides

    reset_overrides(settings, owner_id)
    snapshot = _build_snapshot(owner_id)
    return LlmConfigUpdateResponse(
        **snapshot.model_dump(), reindex_recommended=False
    )


_SECTION_SETTINGS_ATTRS = config.section_settings_attrs()
_PROBE_TIMEOUT_SECONDS = 15.0


@llm_config_router.post("/test", response_model=LlmConfigTestResponse)
async def test_llm_config(
    payload: LlmConfigTestRequest,
    request: Request,
) -> LlmConfigTestResponse:
    """探测上游 /models；空 base_url/api_key 回退账号/系统已存配置，不落盘。"""
    owner_id = current_user_id(request)
    if owner_id:
        from auth.user_config import effective_settings

        effective = effective_settings(settings, owner_id)
    else:
        effective = settings
    base_url_attr, api_key_attr = _SECTION_SETTINGS_ATTRS[payload.section]
    requested_base_url = (payload.base_url or "").strip()
    requested_api_key = (payload.api_key or "").strip()
    stored_base_url = (getattr(effective, base_url_attr) or "").strip()
    stored_api_key = (getattr(effective, api_key_attr) or "").strip()

    # 禁止把已存密钥转发到请求指定的新主机：base_url 变了就必须自带 key。
    if (
        requested_base_url
        and not requested_api_key
        and requested_base_url.rstrip("/") != stored_base_url.rstrip("/")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "请求的 base_url 与已存配置不同且未提供 api_key；"
                "为避免把已存密钥发送到新主机，测试被拒绝"
            ),
        )

    base_url = requested_base_url or stored_base_url
    if not base_url:
        raise HTTPException(
            status_code=400, detail=f"{payload.section}.base_url 未配置"
        )
    api_key = requested_api_key or stored_api_key
    if payload.section == "decision":
        if not api_key:
            raise HTTPException(status_code=400, detail="decision.api_key 未配置")
        model = (payload.model or "").strip() or (
            getattr(effective, "decision_model_name", "") or "Kev-4b"
        )
        try:
            latency_ms, choice = await probe_systemone(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_seconds=_PROBE_TIMEOUT_SECONDS,
            )
        except DecisionCallError as exc:
            if exc.code == "upstream_auth":
                raise HTTPException(status_code=401, detail="上游拒绝访问") from exc
            logger.warning(
                "llm-config decision probe failed code=%s", exc.code
            )
            raise HTTPException(status_code=502, detail="决策模型探测失败") from exc
        return LlmConfigTestResponse(
            latency_ms=latency_ms, model_count=1, models=[choice]
        )

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as upstream:
            response = await upstream.get(url, headers=headers)
    except httpx.RequestError:
        logger.warning("llm-config probe cannot connect section=%s", payload.section)
        raise HTTPException(status_code=502, detail="无法连接上游模型服务")
    latency_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code in (401, 403):
        raise HTTPException(status_code=response.status_code, detail="上游拒绝访问")
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"上游返回 {response.status_code}"
        )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="上游响应不是有效 JSON")
    entries = data.get("data") if isinstance(data, dict) else None
    entries = entries if isinstance(entries, list) else []
    model_ids = [
        entry["id"].strip()
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and entry["id"].strip()
    ]
    return LlmConfigTestResponse(
        latency_ms=latency_ms,
        model_count=len(entries),
        models=model_ids,
    )
