"""Type registry: metadata for the UI and factories for real connectors."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping

from datasource.connector import Connector, ConnectorFactory
from datasource.errors import ValidationError
from datasource.types import ConnectorMetadata

logger = logging.getLogger("ai-service.datasource.registry")

_CONNECTOR_MODULES = ("yuque", "feishu", "notion", "ima")


class ConnectorRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[ConnectorMetadata, ConnectorFactory]] = {}

    def register(self, metadata: ConnectorMetadata, factory: ConnectorFactory) -> None:
        self._entries[metadata.type] = (metadata, factory)

    def types(self) -> list[ConnectorMetadata]:
        return [metadata for metadata, _ in self._entries.values()]

    def metadata(self, type: str) -> ConnectorMetadata:
        entry = self._entries.get(type)
        if entry is None:
            raise ValidationError(f"不支持的知识源类型：{type}")
        return entry[0]

    def create(
        self, type: str, credentials: Mapping[str, str], settings: Mapping[str, Any]
    ) -> Connector:
        entry = self._entries.get(type)
        if entry is None:
            raise ValidationError(f"不支持的知识源类型：{type}")
        return entry[1](credentials, settings)


def build_default_registry() -> ConnectorRegistry:
    """Register every connector module that is present in this checkout.

    Modules are optional during the staged rollout, so a missing connector does
    not prevent the datasource feature from starting.
    """
    registry = ConnectorRegistry()
    for name in _CONNECTOR_MODULES:
        try:
            module = importlib.import_module(f"datasource.connectors.{name}")
        except ImportError:
            continue
        register = getattr(module, "register", None)
        if callable(register):
            register(registry)
        else:
            logger.warning("connector module %s has no register(registry)", name)
    return registry
