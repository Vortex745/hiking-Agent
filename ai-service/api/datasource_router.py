"""Local-only datasource management API."""

from __future__ import annotations

from typing import Any, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.datasource_schemas import (
    DatasourceCreateRequest,
    DatasourceResponse,
    DatasourceUpdateRequest,
    DocumentResponse,
    OperationResponse,
    PreviewAncestorsRequest,
    PreviewResourcesRequest,
    ResourceNodeResponse,
    SyncLogResponse,
    SyncStartResponse,
    TypeMetadataResponse,
    ValidateCredentialsRequest,
)
from api.local_guard import require_local_request
from auth.deps import current_user_id
from datasource.errors import (
    ConflictError,
    DatasourceError,
    NotFoundError,
    ValidationError as DatasourceValidationError,
)
from datasource.service import DatasourceService
from datasource.types import ConnectorMetadata, ResourceNode

datasource_router = APIRouter(
    prefix="/datasources", dependencies=[Depends(require_local_request)]
)


def get_service(request: Request) -> DatasourceService:
    service = getattr(request.app.state, "datasource_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="知识源功能未启用")
    return service


def _http_error(exc: DatasourceError) -> HTTPException:
    if isinstance(exc, NotFoundError):
        status = 404
    elif isinstance(exc, ConflictError):
        status = 409
    elif isinstance(exc, DatasourceValidationError):
        status = 422
    else:
        status = 400
    return HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message})


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except DatasourceError as exc:
        raise _http_error(exc) from exc


def _metadata_response(metadata: ConnectorMetadata) -> TypeMetadataResponse:
    return TypeMetadataResponse(
        type=metadata.type,
        label=metadata.label,
        auth_fields=[
            {
                "key": field.key,
                "label": field.label,
                "secret": field.secret,
                "required": field.required,
                "placeholder": field.placeholder,
                "help": field.help,
            }
            for field in metadata.auth_fields
        ],
        setting_fields=[
            {
                "key": field.key,
                "label": field.label,
                "default": field.default,
                "placeholder": field.placeholder,
                "help": field.help,
            }
            for field in metadata.setting_fields
        ],
        supports_tree=metadata.supports_tree,
        supports_incremental=metadata.supports_incremental,
        supports_deletion=metadata.supports_deletion,
    )


_LOG_COUNT_KEYS = (
    "discovered",
    "fetched",
    "created",
    "updated",
    "skipped",
    "skipped_by_type",
    "truncated",
    "failed",
    "deleted",
)


def _log_response(log: dict[str, Any]) -> dict[str, Any]:
    return {
        **log,
        "counts": {key: int(log.get(key, 0)) for key in _LOG_COUNT_KEYS},
    }


def _node_response(node: ResourceNode) -> ResourceNodeResponse:
    return ResourceNodeResponse(
        id=node.id,
        title=node.title,
        kind=node.kind,
        parent_id=node.parent_id,
        has_children=node.has_children,
        rev=node.rev,
        url=node.url,
    )


@datasource_router.get("/types", response_model=list[TypeMetadataResponse])
def list_types(service: DatasourceService = Depends(get_service)) -> list[TypeMetadataResponse]:
    return [_metadata_response(metadata) for metadata in service.list_types()]


@datasource_router.get("", response_model=list[DatasourceResponse])
def list_datasources(
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> list[dict[str, Any]]:
    return service.list_datasources(current_user_id(http_request))


@datasource_router.post("", response_model=DatasourceResponse, status_code=201)
def create_datasource(
    request: DatasourceCreateRequest,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> dict[str, Any]:
    return _call(
        service.create_datasource,
        type=request.type,
        name=request.name,
        credentials=request.credentials,
        settings=request.settings,
        selected_resources=request.selected_resources,
        schedule=request.schedule,
        sync_deletions=request.sync_deletions,
        owner_id=current_user_id(http_request),
    )


@datasource_router.post("/preview/validate", response_model=OperationResponse)
def validate_preview(
    request: ValidateCredentialsRequest,
    service: DatasourceService = Depends(get_service),
) -> OperationResponse:
    _call(service.validate, request.type, request.credentials, request.settings)
    return OperationResponse(ok=True, message="连接成功")


@datasource_router.post("/preview/resources", response_model=list[ResourceNodeResponse])
def preview_resources(
    request: PreviewResourcesRequest,
    service: DatasourceService = Depends(get_service),
) -> list[ResourceNodeResponse]:
    nodes = _call(
        service.list_resources,
        request.type,
        request.credentials,
        request.settings,
        request.parent_id,
    )
    return [_node_response(node) for node in nodes]


@datasource_router.post("/preview/ancestors", response_model=list[ResourceNodeResponse])
def preview_ancestors(
    request: PreviewAncestorsRequest,
    service: DatasourceService = Depends(get_service),
) -> list[ResourceNodeResponse]:
    nodes = _call(
        service.resolve_ancestors,
        request.type,
        request.credentials,
        request.settings,
        request.ids,
    )
    return [_node_response(node) for node in nodes]


@datasource_router.get("/{ds_id}", response_model=DatasourceResponse)
def get_datasource(
    ds_id: str,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> dict[str, Any]:
    return _call(service.get_datasource, ds_id, current_user_id(http_request))


@datasource_router.patch("/{ds_id}", response_model=DatasourceResponse)
def update_datasource(
    ds_id: str,
    request: DatasourceUpdateRequest,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> dict[str, Any]:
    fields = request.model_dump(exclude_unset=True)
    return _call(
        service.update_datasource,
        ds_id,
        owner_id=current_user_id(http_request),
        **fields,
    )


@datasource_router.delete("/{ds_id}", response_model=OperationResponse)
def delete_datasource(
    ds_id: str,
    http_request: Request,
    purge_docs: bool = Query(default=False),
    service: DatasourceService = Depends(get_service),
) -> OperationResponse:
    _call(
        service.delete_datasource,
        ds_id,
        purge_docs=purge_docs,
        owner_id=current_user_id(http_request),
    )
    return OperationResponse(ok=True, message="已删除" if purge_docs else "已删除（保留文档）")


@datasource_router.post("/{ds_id}/validate", response_model=OperationResponse)
def validate_saved(
    ds_id: str,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> OperationResponse:
    _call(service.validate_saved, ds_id, current_user_id(http_request))
    return OperationResponse(ok=True, message="连接成功")


@datasource_router.get("/{ds_id}/resources", response_model=list[ResourceNodeResponse])
def list_saved_resources(
    ds_id: str,
    http_request: Request,
    parent_id: str | None = Query(default=None),
    service: DatasourceService = Depends(get_service),
) -> list[ResourceNodeResponse]:
    nodes = _call(
        service.list_resources_for,
        ds_id,
        parent_id,
        owner_id=current_user_id(http_request),
    )
    return [_node_response(node) for node in nodes]


@datasource_router.get("/{ds_id}/ancestors", response_model=list[ResourceNodeResponse])
def list_saved_ancestors(
    ds_id: str,
    http_request: Request,
    ids: str = Query(default=""),
    service: DatasourceService = Depends(get_service),
) -> list[ResourceNodeResponse]:
    parsed = [item.strip() for item in ids.split(",") if item.strip()]
    if not parsed:
        return []
    nodes = _call(
        service.resolve_ancestors_for,
        ds_id,
        parsed,
        owner_id=current_user_id(http_request),
    )
    return [_node_response(node) for node in nodes]


@datasource_router.post("/{ds_id}/sync", response_model=SyncStartResponse, status_code=202)
def start_sync(
    ds_id: str,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> SyncStartResponse:
    sync_id = _call(
        service.start_sync,
        ds_id,
        "manual",
        owner_id=current_user_id(http_request),
    )
    return SyncStartResponse(sync_id=sync_id)


@datasource_router.get("/{ds_id}/syncs", response_model=list[SyncLogResponse])
def list_syncs(
    ds_id: str,
    http_request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    service: DatasourceService = Depends(get_service),
) -> list[dict[str, Any]]:
    logs = _call(
        service.list_sync_logs,
        ds_id,
        limit,
        owner_id=current_user_id(http_request),
    )
    return [_log_response(log) for log in logs]


@datasource_router.get("/{ds_id}/syncs/{sync_id}", response_model=SyncLogResponse)
def get_sync(
    ds_id: str,
    sync_id: str,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> dict[str, Any]:
    return _log_response(
        _call(
            service.get_sync_log,
            ds_id,
            sync_id,
            owner_id=current_user_id(http_request),
        )
    )


@datasource_router.get("/{ds_id}/documents", response_model=list[DocumentResponse])
def list_documents(
    ds_id: str,
    http_request: Request,
    service: DatasourceService = Depends(get_service),
) -> Sequence[dict[str, Any]]:
    return _call(service.list_documents, ds_id, current_user_id(http_request))
