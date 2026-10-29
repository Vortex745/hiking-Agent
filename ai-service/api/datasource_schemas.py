"""Pydantic schemas for the datasource management API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthFieldResponse(BaseModel):
    key: str
    label: str
    secret: bool
    required: bool
    placeholder: str
    help: str


class SettingFieldResponse(BaseModel):
    key: str
    label: str
    default: str
    placeholder: str
    help: str


class TypeMetadataResponse(BaseModel):
    type: str
    label: str
    auth_fields: list[AuthFieldResponse]
    setting_fields: list[SettingFieldResponse]
    supports_tree: bool
    supports_incremental: bool
    supports_deletion: bool


class DatasourceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    name: str
    credentials: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    selected_resources: list[str] = Field(default_factory=list)
    schedule: str = ""
    sync_deletions: bool = True


class DatasourceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    credentials: dict[str, str] | None = None
    settings: dict[str, Any] | None = None
    selected_resources: list[str] | None = None
    schedule: str | None = None
    sync_deletions: bool | None = None


class ValidateCredentialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    credentials: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class PreviewResourcesRequest(ValidateCredentialsRequest):
    parent_id: str | None = None


class PreviewAncestorsRequest(ValidateCredentialsRequest):
    ids: list[str] = Field(default_factory=list)


class ResourceNodeResponse(BaseModel):
    id: str
    title: str
    kind: str
    parent_id: str | None
    has_children: bool
    rev: str
    url: str


class DocumentResponse(BaseModel):
    external_id: str
    resource_id: str
    rev: str
    rag_source: str
    file_name: str
    last_seen_sync_id: str | None
    updated_at: str


class SyncLogResponse(BaseModel):
    id: str
    datasource_id: str
    trigger: str
    status: str
    started_at: str
    finished_at: str | None
    counts: dict[str, int]
    failures: list[dict[str, Any]]
    enumeration: dict[str, Any]
    error_code: str | None
    message: str | None


class DatasourceResponse(BaseModel):
    id: str
    type: str
    name: str
    credentials_configured: dict[str, bool]
    settings: dict[str, Any]
    selected_resources: list[str]
    schedule: str
    sync_deletions: bool
    sync_status: str
    running_since: str | None
    next_run_at: str | None
    last_sync_at: str | None
    last_sync_status: str | None
    last_sync: dict[str, Any] | None
    doc_count: int
    created_at: str
    updated_at: str


class SyncStartResponse(BaseModel):
    sync_id: str
    status: Literal["running"] = "running"


class OperationResponse(BaseModel):
    ok: bool = True
    message: str = ""
