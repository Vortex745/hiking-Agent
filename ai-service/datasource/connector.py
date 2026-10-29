"""Connector protocol shared by every platform."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

from datasource.types import (
    ConnectorMetadata,
    EnumerationResult,
    FetchOutcome,
    ResourceNode,
)


class Connector(Protocol):
    metadata: ConnectorMetadata

    def validate(self) -> None:
        """Raise DatasourceError when credentials or permissions fail."""

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        """Return direct children; ``parent_id=None`` returns roots."""

    def resolve_ancestors(self, ids: Sequence[str]) -> list[ResourceNode]:
        """Return the selected nodes and their parent chain, when the platform has one."""

    def enumerate(self, root: str) -> EnumerationResult:
        """Return every descendant under one selected root and whether listing was complete."""

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        """Fetch one item. Deterministic gaps return FetchOutcome(skipped=True); transient failures raise."""

    def close(self) -> None:
        """Release per-connection resources."""


ConnectorFactory = Callable[[Mapping[str, str], Mapping[str, Any]], Connector]
