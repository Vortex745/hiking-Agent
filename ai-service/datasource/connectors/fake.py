"""A controllable connector used only by tests.

It is not part of the production registry; tests build their own registry and
share a mutable FakeState between syncs to simulate edits, failures, partial
listings and empty listings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from datasource.registry import ConnectorRegistry
from datasource.errors import AuthError, NetworkError
from datasource.types import (
    AuthField,
    ConnectorMetadata,
    EnumerationResult,
    FetchedItem,
    FetchOutcome,
    ResourceNode,
)


@dataclass
class FakeDoc:
    external_id: str
    title: str
    content: str = "hello"
    rev: str = "1"
    resource_id: str = "root"
    skip_reason: str = ""
    fail: bool = False


@dataclass
class FakeState:
    docs: list[FakeDoc] = field(default_factory=list)
    roots: list[ResourceNode] = field(
        default_factory=lambda: [
            ResourceNode(id="root", title="测试空间", kind="space", has_children=True)
        ]
    )
    fail_enumerate: set[str] = field(default_factory=set)
    truncated_roots: set[str] = field(default_factory=set)
    validate_token: str = "ok"
    fetch_calls: list[str] = field(default_factory=list)


FAKE_METADATA = ConnectorMetadata(
    type="fake",
    label="测试来源",
    auth_fields=(AuthField(key="token", label="Token", secret=True),),
    setting_fields=(),
)


class FakeConnector:
    metadata = FAKE_METADATA

    def __init__(
        self,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        state: FakeState,
    ) -> None:
        self._credentials = credentials
        self._settings = settings
        self._state = state

    def validate(self) -> None:
        if self._state.validate_token and self._credentials.get("token") != self._state.validate_token:
            raise AuthError("测试 token 无效")

    def list_resources(self, parent_id: str | None) -> list[ResourceNode]:
        if parent_id is None:
            return list(self._state.roots)
        children = []
        for doc in self._state.docs:
            if doc.resource_id == parent_id:
                children.append(
                    ResourceNode(
                        id=doc.external_id,
                        title=doc.title,
                        kind="document",
                        parent_id=parent_id,
                        rev=doc.rev,
                    )
                )
        return children

    def resolve_ancestors(self, ids: Sequence[str]) -> list[ResourceNode]:
        nodes: list[ResourceNode] = []
        known = {root.id: root for root in self._state.roots}
        for external_id in ids:
            if external_id in known:
                nodes.append(known[external_id])
                continue
            for doc in self._state.docs:
                if doc.external_id == external_id:
                    nodes.append(
                        ResourceNode(
                            id=doc.external_id,
                            title=doc.title,
                            kind="document",
                            parent_id=doc.resource_id,
                            rev=doc.rev,
                        )
                    )
                    parent = known.get(doc.resource_id)
                    if parent is not None:
                        nodes.append(parent)
        return nodes

    def enumerate(self, root: str) -> EnumerationResult:
        if root in self._state.fail_enumerate:
            return EnumerationResult(nodes=(), complete=False, error="enumeration_incomplete")
        nodes = tuple(
            ResourceNode(
                id=doc.external_id,
                title=doc.title,
                kind="document",
                parent_id=root,
                rev=doc.rev,
            )
            for doc in self._state.docs
            if doc.resource_id == root
        )
        truncated = root in self._state.truncated_roots
        return EnumerationResult(nodes=nodes, complete=not truncated, truncated=truncated)

    def fetch(self, node: ResourceNode) -> FetchOutcome:
        self._state.fetch_calls.append(node.id)
        doc = next((d for d in self._state.docs if d.external_id == node.id), None)
        if doc is None:
            raise NetworkError(f"fake doc not found: {node.id}")
        if doc.fail:
            raise NetworkError(f"fake fetch failure: {node.id}")
        if doc.skip_reason:
            return FetchOutcome(skipped=True, skip_reason=doc.skip_reason)
        return FetchOutcome(
            item=FetchedItem(
                external_id=doc.external_id,
                resource_id=doc.resource_id,
                title=doc.title,
                content=doc.content,
                content_type="markdown",
                file_name=f"{doc.title}.md",
                rev=doc.rev,
            )
        )

    def close(self) -> None:
        return None


def register_fake(
    registry: ConnectorRegistry, state: FakeState | None = None
) -> FakeState:
    shared = state or FakeState()
    registry.register(
        FAKE_METADATA,
        lambda credentials, settings: FakeConnector(credentials, settings, shared),
    )
    return shared
