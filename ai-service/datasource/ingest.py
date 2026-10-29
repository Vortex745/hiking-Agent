"""Bridge between fetched datasource items and the existing RAG ingestion pipeline."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, Protocol

from datasource.errors import IngestFailed, UnsupportedType
from datasource.types import FetchedItem
from rag.ingestion import (
    EmbeddingProvider,
    IngestionError,
    IngestionPipeline,
    OpenAIEmbeddingAdapter,
    ScannedPDFError,
    extract_markdown,
)
from rag.sqlite_store import SQLiteStore

logger = logging.getLogger("ai-service.datasource.ingest")

_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_NAME = 80

ProviderFactory = Callable[[], EmbeddingProvider]


def sanitize_file_name(title: str, extension: str) -> str:
    name = _INVALID_FILENAME.sub(" ", title or "").strip()
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = "untitled"
    name = name[:_MAX_NAME].rstrip(" .")
    if not name:
        name = "untitled"
    ext = extension if extension.startswith(".") else f".{extension}" if extension else ""
    if ext and not name.lower().endswith(ext.lower()):
        name = f"{name}{ext}"
    return name


def build_embedding_provider_from_settings() -> EmbeddingProvider:
    from langchain_openai import OpenAIEmbeddings

    from config import settings

    if not settings.embedding_api_key:
        raise IngestFailed("未配置 embedding 服务，无法同步文档")
    return OpenAIEmbeddingAdapter(
        OpenAIEmbeddings(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            check_embedding_ctx_length=False,
        )
    )


class RagIngestor:
    def __init__(
        self,
        db_path: Path | str,
        dimensions: int,
        embedding_provider: EmbeddingProvider | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._store = SQLiteStore(db_path, embedding_dimensions=dimensions)
        self._provider = embedding_provider
        self._provider_factory = provider_factory
        self._pipeline: IngestionPipeline | None = None

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @classmethod
    def from_settings(cls, db_path: Path | str, dimensions: int) -> "RagIngestor":
        return cls(
            db_path,
            dimensions,
            provider_factory=build_embedding_provider_from_settings,
        )

    def _pipeline_obj(self) -> IngestionPipeline:
        if self._pipeline is None:
            provider = self._provider
            if provider is None and self._provider_factory is not None:
                provider = self._provider_factory()
            if provider is None:
                raise IngestFailed("未配置 embedding 服务，无法同步文档")
            self._pipeline = IngestionPipeline(self._store, provider)
        return self._pipeline

    def upsert(self, item: FetchedItem, source: str) -> str:
        file_name = item.file_name or sanitize_file_name(
            item.title, ".pdf" if item.content_type == "pdf" else ".md"
        )
        if item.content_type == "pdf":
            return self._upsert_pdf(item, source, file_name)

        markdown = item.content.strip()
        if not markdown:
            markdown = f"# {item.title or 'Untitled'}"
        try:
            report = self._pipeline_obj().ingest_content(
                markdown=markdown,
                source=source,
                file_name=file_name,
            )
        except IngestionError as exc:
            raise IngestFailed(str(exc)) from exc
        return _action_of(report.action)

    def _upsert_pdf(self, item: FetchedItem, source: str, file_name: str) -> str:
        if not item.data:
            raise IngestFailed("PDF 内容为空")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(item.data)
                temporary = Path(handle.name)
            try:
                markdown = extract_markdown(temporary)
            except ScannedPDFError as exc:
                raise UnsupportedType(str(exc)) from exc
            report = self._pipeline_obj().ingest_content(
                markdown=markdown,
                source=source,
                file_name=file_name,
            )
            return _action_of(report.action)
        except IngestionError as exc:
            raise IngestFailed(str(exc)) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("could not remove temporary PDF %s: %s", temporary, exc)

    def delete(self, source: str) -> bool:
        return self._store.delete_document(source)

    def close(self) -> None:
        self._store.close()


def _action_of(action: str) -> str:
    if action == "inserted":
        return "created"
    if action == "replaced":
        return "updated"
    return "skipped"


__all__ = [
    "ProviderFactory",
    "RagIngestor",
    "build_embedding_provider_from_settings",
    "sanitize_file_name",
]
