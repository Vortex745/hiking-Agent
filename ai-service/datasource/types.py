"""Datasource domain types shared by connectors, engine, store and API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

SyncStatus = Literal["idle", "running"]
SyncLogStatus = Literal["running", "success", "partial", "failed"]
ContentType = Literal["markdown", "pdf"]


@dataclass(frozen=True)
class AuthField:
    key: str
    label: str
    secret: bool = False
    required: bool = True
    placeholder: str = ""
    help: str = ""


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    default: str = ""
    placeholder: str = ""
    help: str = ""


@dataclass(frozen=True)
class ConnectorMetadata:
    type: str
    label: str
    auth_fields: tuple[AuthField, ...] = ()
    setting_fields: tuple[SettingField, ...] = ()
    supports_tree: bool = True
    supports_incremental: bool = True
    supports_deletion: bool = True


@dataclass(frozen=True)
class ResourceNode:
    id: str
    title: str
    kind: str = "document"
    parent_id: str | None = None
    has_children: bool = False
    rev: str = ""
    url: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchedItem:
    external_id: str
    resource_id: str
    title: str
    content: str = ""
    content_type: ContentType = "markdown"
    data: bytes | None = None
    file_name: str = ""
    rev: str = ""
    url: str = ""
    updated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchOutcome:
    """Fetch result.

    ``skipped`` means a deterministic skip (the platform can never provide
    content for this item). Transient failures raise instead; the engine then
    refuses to advance that document's rev.
    """

    item: FetchedItem | None = None
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True)
class EnumerationResult:
    """One root's descendants.

    Deletion reconciliation only trusts ``complete`` roots. ``truncated`` is a
    vendor cap (for example Notion's 1000-block limit); it is also not enough
    evidence to delete anything.
    """

    nodes: tuple[ResourceNode, ...] = ()
    complete: bool = True
    truncated: bool = False
    error: str = ""


@dataclass
class SyncCounts:
    discovered: int = 0
    fetched: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    skipped_by_type: int = 0
    truncated: int = 0
    failed: int = 0
    deleted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "discovered": self.discovered,
            "fetched": self.fetched,
            "created": self.created,
            "updated": self.updated,
            "skipped": self.skipped,
            "skipped_by_type": self.skipped_by_type,
            "truncated": self.truncated,
            "failed": self.failed,
            "deleted": self.deleted,
        }
