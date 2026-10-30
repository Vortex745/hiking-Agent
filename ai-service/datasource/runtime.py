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

    def ingestor_factory(owner_id: str) -> RagIngestor:
        from auth.paths import user_rag_path
        from auth.user_config import effective_settings
        from datasource.errors import IngestFailed

        effective = effective_settings(settings, owner_id)
        if not str(getattr(effective, "embedding_api_key", "") or "").strip():
            # 先报错再建库：否则会用默认维度创建 vec0，之后正确维度反而被 409 拦住。
            raise IngestFailed("未配置 embedding 服务，请先在模型配置中填写 Embedding")
        db_path = user_rag_path(settings, owner_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return RagIngestor.from_settings(
            str(db_path),
            effective.embedding_dimensions,
            runtime_settings=effective,
        )

    return DatasourceService(
        store,
        build_default_registry(),
        ingestor,
        secret_box,
        ingestor_factory=ingestor_factory,
    )
