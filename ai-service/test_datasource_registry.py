"""Default registry should expose the four real connectors, never the fake one."""

from __future__ import annotations

from datasource.connectors.ima import ImaConnector
from datasource.connectors.notion import NotionConnector
from datasource.connectors.feishu import FeishuConnector
from datasource.connectors.yuque import YuqueConnector
from datasource.registry import build_default_registry


def test_default_registry_contains_real_connectors_only():
    registry = build_default_registry()
    types = {metadata.type for metadata in registry.types()}
    assert types == {"yuque", "feishu", "notion", "ima"}
    assert "fake" not in types


def test_real_connector_factories_accept_settings_without_network():
    registry = build_default_registry()
    assert isinstance(
        registry.create("yuque", {"token": "t"}, {"base_url": "https://www.yuque.test"}),
        YuqueConnector,
    )
    assert isinstance(
        registry.create(
            "feishu",
            {"app_id": "a", "app_secret": "s"},
            {"base_url": "https://open.feishu.test"},
        ),
        FeishuConnector,
    )
    assert isinstance(
        registry.create("notion", {"token": "t"}, {"base_url": "https://notion.test"}),
        NotionConnector,
    )
    assert isinstance(
        registry.create(
            "ima", {"client_id": "c", "api_key": "k"}, {"base_url": "https://ima.test"}
        ),
        ImaConnector,
    )
