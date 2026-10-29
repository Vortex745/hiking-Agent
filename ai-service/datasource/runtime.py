"""Assemble the datasource service from process settings."""

from __future__ import annotations

from pathlib import Path

from datasource.ingest import RagIngestor
from datasource.registry import build_default_registry
from datasource.secrets import SecretBox
from datasource.service import DatasourceService
from datasource.store import DatasourceStore


def build_datasource_service(settings) -> DatasourceService:
    db_path = settings.datasource_db_path or str(
        Path(settings.rag_sqlite_db_path).with_name("datasource.sqlite")
    )
    store = DatasourceStore(db_path)
    secret_box = SecretBox.from_settings(
        settings.datasource_secret_key,
        Path(db_path).with_name("datasource.key"),
    )
    ingestor = RagIngestor.from_settings(
        settings.rag_sqlite_db_path,
        settings.embedding_dimensions,
    )
    return DatasourceService(
        store,
        build_default_registry(),
        ingestor,
        secret_box,
    )
