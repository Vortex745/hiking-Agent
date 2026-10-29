"""Datasource CRUD, validation, sync engine and scheduler."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from datasource.connector import Connector
from datasource.errors import (
    ConflictError,
    CredentialsUnreadable,
    DatasourceError,
    NotFoundError,
    UnsupportedType,
    ValidationError,
    VendorError,
)
from datasource.ingest import RagIngestor, sanitize_file_name
from datasource.registry import ConnectorRegistry
from datasource.scheduler import next_run_at
from datasource.secrets import SecretBox
from datasource.store import DatasourceStore
from datasource.timeutil import iso, now, now_iso
from datasource.types import (
    ConnectorMetadata,
    EnumerationResult,
    FetchedItem,
    FetchOutcome,
    ResourceNode,
    SyncCounts,
)

logger = logging.getLogger("ai-service.datasource.service")


@dataclass
class _SyncResult:
    status: str
    counts: SyncCounts
    failures: list[dict[str, Any]] = field(default_factory=list)
    enumeration: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    message: str = ""


class DatasourceService:
    def __init__(
        self,
        store: DatasourceStore,
        registry: ConnectorRegistry,
        ingestor: RagIngestor,
        secret_box: SecretBox,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.ingestor = ingestor
        self.secret_box = secret_box
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="datasource-sync"
        )
        self._owns_executor = executor is None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self.store.recover_interrupted()

    # ------------------------------------------------------------------ metadata / CRUD

    def list_types(self) -> list[ConnectorMetadata]:
        return self.registry.types()

    def create_datasource(
        self,
        *,
        type: str,
        name: str,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        selected_resources: Sequence[str],
        schedule: str = "",
        sync_deletions: bool = True,
    ) -> dict[str, Any]:
        metadata = self.registry.metadata(type)
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValidationError("知识源名称不能为空")
        clean_credentials = _filter_credentials(metadata, credentials)
        clean_settings = _filter_settings(metadata, settings)
        selected = _clean_selected(selected_resources)
        schedule = (schedule or "").strip()
        next_at = _next_at_or_raise(schedule)
        ds = self.store.create_datasource(
            type=type,
            name=clean_name,
            credentials_enc=self.secret_box.encrypt(clean_credentials),
            settings=clean_settings,
            selected_resources=selected,
            schedule=schedule,
            sync_deletions=sync_deletions,
            next_run_at=next_at,
        )
        return self.response(ds)

    def update_datasource(
        self,
        ds_id: str,
        *,
        name: str | None = None,
        credentials: Mapping[str, str] | None = None,
        settings: Mapping[str, Any] | None = None,
        selected_resources: Sequence[str] | None = None,
        schedule: str | None = None,
        sync_deletions: bool | None = None,
    ) -> dict[str, Any]:
        ds = self._require(ds_id)
        metadata = self.registry.metadata(ds["type"])
        fields: dict[str, Any] = {}
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValidationError("知识源名称不能为空")
            fields["name"] = clean_name
        if credentials is not None:
            fields["credentials_enc"] = self.secret_box.encrypt(
                _filter_credentials(metadata, credentials)
            )
        if settings is not None:
            fields["settings"] = _filter_settings(metadata, settings)
        if selected_resources is not None:
            fields["selected_resources"] = _clean_selected(selected_resources)
        if sync_deletions is not None:
            fields["sync_deletions"] = bool(sync_deletions)
        next_at: str | None = None
        schedule_changed = schedule is not None
        if schedule_changed:
            schedule = schedule.strip()
            next_at = _next_at_or_raise(schedule)
            fields["schedule"] = schedule
        updated = self.store.update_datasource(ds_id, **fields)
        assert updated is not None
        if schedule_changed:
            self.store.set_next_run(ds_id, next_at)
            updated = self._require(ds_id)
        return self.response(updated)

    def delete_datasource(self, ds_id: str, *, purge_docs: bool = False) -> bool:
        self._require(ds_id)
        if purge_docs:
            self.purge_docs(ds_id)
        return self.store.delete_datasource(ds_id)

    def purge_docs(self, ds_id: str) -> int:
        self._require(ds_id)
        deleted = 0
        for row in self.store.list_synced_docs(ds_id):
            if row["rag_source"]:
                try:
                    self.ingestor.delete(row["rag_source"])
                except Exception as exc:
                    logger.warning("purge rag doc failed ds=%s external_id=%s: %s", ds_id, row["external_id"], exc)
                    continue
            self.store.delete_synced_doc(ds_id, row["external_id"])
            deleted += 1
        return deleted

    def list_datasources(self) -> list[dict[str, Any]]:
        return [self.response(ds) for ds in self.store.list_datasources()]

    def get_datasource(self, ds_id: str) -> dict[str, Any]:
        return self.response(self._require(ds_id))

    def response(self, ds: Mapping[str, Any]) -> dict[str, Any]:
        configured: dict[str, bool] = {}
        try:
            metadata = self.registry.metadata(ds["type"])
        except ValidationError:
            metadata = None
        if metadata is not None:
            try:
                credentials = self.secret_box.decrypt(ds["credentials_enc"])
            except CredentialsUnreadable:
                credentials = {}
            configured = {
                field.key: bool(str(credentials.get(field.key, "")).strip())
                for field in metadata.auth_fields
            }
        logs = self.store.list_sync_logs(ds["id"], 1)
        last = _log_summary(logs[0]) if logs else None
        return {
            "id": ds["id"],
            "type": ds["type"],
            "name": ds["name"],
            "credentials_configured": configured,
            "settings": ds["settings"],
            "selected_resources": ds["selected_resources"],
            "schedule": ds["schedule"],
            "sync_deletions": ds["sync_deletions"],
            "sync_status": ds["sync_status"],
            "running_since": ds["running_since"],
            "next_run_at": ds["next_run_at"],
            "last_sync_at": ds["last_sync_at"],
            "last_sync_status": ds["last_sync_status"],
            "last_sync": last,
            "doc_count": self.store.count_synced_docs(ds["id"], only_with_source=True),
            "created_at": ds["created_at"],
            "updated_at": ds["updated_at"],
        }

    # ------------------------------------------------------------------ preview / validation

    def validate(
        self, type: str, credentials: Mapping[str, str], settings: Mapping[str, Any]
    ) -> None:
        metadata = self.registry.metadata(type)
        connector = self.registry.create(
            type, _filter_credentials(metadata, credentials), _filter_settings(metadata, settings)
        )
        try:
            connector.validate()
        finally:
            connector.close()

    def list_resources(
        self,
        type: str,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        parent_id: str | None,
    ) -> list[ResourceNode]:
        metadata = self.registry.metadata(type)
        connector = self.registry.create(
            type, _filter_credentials(metadata, credentials), _filter_settings(metadata, settings)
        )
        try:
            return connector.list_resources(parent_id)
        finally:
            connector.close()

    def resolve_ancestors(
        self,
        type: str,
        credentials: Mapping[str, str],
        settings: Mapping[str, Any],
        ids: Sequence[str],
    ) -> list[ResourceNode]:
        metadata = self.registry.metadata(type)
        connector = self.registry.create(
            type, _filter_credentials(metadata, credentials), _filter_settings(metadata, settings)
        )
        try:
            return connector.resolve_ancestors(ids)
        finally:
            connector.close()

    def list_resources_for(self, ds_id: str, parent_id: str | None) -> list[ResourceNode]:
        ds = self._require(ds_id)
        connector = self._connector(ds)
        try:
            return connector.list_resources(parent_id)
        finally:
            connector.close()

    def resolve_ancestors_for(self, ds_id: str, ids: Sequence[str]) -> list[ResourceNode]:
        ds = self._require(ds_id)
        connector = self._connector(ds)
        try:
            return connector.resolve_ancestors(ids)
        finally:
            connector.close()

    # ------------------------------------------------------------------ sync

    def start_sync(self, ds_id: str, trigger: str = "manual") -> str:
        ds = self._require(ds_id)
        if ds["sync_status"] == "running":
            raise ConflictError("该知识源正在同步，请等待本轮结束")
        log_id = self.store.start_sync_log(ds_id, trigger)
        self.store.set_sync_status(ds_id, "running", now_iso())
        if ds["schedule"]:
            self.store.set_next_run(ds_id, iso(next_run_at(ds["schedule"], now())))
        try:
            self._executor.submit(self._execute, ds_id, log_id, trigger)
        except Exception:
            self.store.set_sync_status(ds_id, "idle")
            self.store.finish_sync_log(
                log_id,
                status="failed",
                counts={},
                failures=[],
                enumeration={},
                error_code="interrupted",
                message="同步执行器不可用",
            )
            raise
        return log_id

    def run_sync_now(self, ds_id: str, trigger: str = "manual") -> str:
        """Synchronous sync used by tests and explicit maintenance calls."""
        ds = self._require(ds_id)
        if ds["sync_status"] == "running":
            raise ConflictError("该知识源正在同步，请等待本轮结束")
        log_id = self.store.start_sync_log(ds_id, trigger)
        self.store.set_sync_status(ds_id, "running", now_iso())
        if ds["schedule"]:
            self.store.set_next_run(ds_id, iso(next_run_at(ds["schedule"], now())))
        self._execute(ds_id, log_id, trigger)
        return log_id

    def _execute(self, ds_id: str, log_id: str, trigger: str) -> None:
        try:
            result = self._sync(ds_id, log_id, trigger)
        except Exception as exc:
            logger.warning("datasource sync failed ds=%s code=%s: %s", ds_id, _code(exc), _message(exc))
            result = _SyncResult(
                status="failed",
                counts=SyncCounts(),
                failures=[
                    {
                        "external_id": "",
                        "title": "",
                        "error_code": _code(exc),
                        "message": _message(exc),
                    }
                ],
                enumeration={},
                error_code=_code(exc),
                message=_message(exc),
            )
        self._finalize(ds_id, log_id, result)

    def _execute_and_return(self, ds_id: str, log_id: str, trigger: str) -> _SyncResult:
        try:
            result = self._sync(ds_id, log_id, trigger)
        except Exception as exc:
            result = _SyncResult(
                status="failed",
                counts=SyncCounts(),
                failures=[
                    {
                        "external_id": "",
                        "title": "",
                        "error_code": _code(exc),
                        "message": _message(exc),
                    }
                ],
                enumeration={},
                error_code=_code(exc),
                message=_message(exc),
            )
        self._finalize(ds_id, log_id, result)
        return result

    def _sync(self, ds_id: str, log_id: str, trigger: str) -> _SyncResult:
        ds = self._require(ds_id)
        selected = list(ds["selected_resources"])
        if not selected:
            raise ValidationError("未选择任何资源")
        credentials = self.secret_box.decrypt(ds["credentials_enc"])
        connector = self.registry.create(ds["type"], credentials, ds["settings"])
        counts = SyncCounts()
        failures: list[dict[str, Any]] = []
        enumeration: dict[str, Any] = {}
        seen: dict[str, set[str]] = {root: set() for root in selected}
        known = {row["external_id"]: row for row in self.store.list_synced_docs(ds_id)}
        enum_errors = 0
        try:
            for root in selected:
                try:
                    result = connector.enumerate(root)
                except Exception as exc:
                    enum_errors += 1
                    enumeration[root] = {
                        "complete": False,
                        "truncated": False,
                        "error": _code(exc),
                    }
                    failures.append(_failure("", root, exc))
                    continue
                enumeration[root] = {
                    "complete": result.complete,
                    "truncated": result.truncated,
                    "error": result.error,
                }
                if result.truncated:
                    counts.truncated += 1
                if result.error:
                    enum_errors += 1
                for node in result.nodes:
                    self._ingest_node(
                        ds_id,
                        log_id,
                        root,
                        node,
                        connector,
                        known,
                        seen,
                        counts,
                        failures,
                    )
            guard_tripped = False
            guard_message = ""
            if ds["sync_deletions"]:
                guard_tripped, guard_message = self._reconcile_deletions(
                    ds_id, selected, seen, known, enumeration, counts, failures
                )

            status = "success"
            error_code: str | None = None
            message = ""
            if enum_errors and enum_errors == len(selected) and counts.fetched == 0:
                status = "failed"
                error_code = "enumeration_incomplete"
                message = "所有选中资源的列举均失败"
            elif counts.discovered > 0 and counts.fetched == 0 and counts.failed > 0:
                status = "failed"
                error_code = "ingest_failed"
                message = "本轮所有文档均处理失败"
            elif counts.failed or counts.truncated or enum_errors or guard_tripped:
                status = "partial"
            if guard_tripped:
                status = "partial"
                error_code = "deletion_guard_tripped"
                message = guard_message
            return _SyncResult(
                status=status,
                counts=counts,
                failures=failures,
                enumeration=enumeration,
                error_code=error_code,
                message=message,
            )
        finally:
            connector.close()

    def _ingest_node(
        self,
        ds_id: str,
        log_id: str,
        root: str,
        node: ResourceNode,
        connector: Connector,
        known: Mapping[str, dict[str, Any]],
        seen: dict[str, set[str]],
        counts: SyncCounts,
        failures: list[dict[str, Any]],
    ) -> None:
        counts.discovered += 1
        seen[root].add(node.id)
        previous = known.get(node.id)
        if previous is not None and previous["rev"] == node.rev:
            counts.skipped += 1
            self.store.touch_synced_doc(ds_id, node.id, log_id)
            return
        try:
            outcome = connector.fetch(node)
            if outcome.truncated:
                counts.truncated += 1
            if outcome.skipped:
                counts.skipped += 1
                counts.skipped_by_type += 1
                self._record_skip(ds_id, log_id, root, node, previous, outcome.skip_reason)
                return
            item = outcome.item
            if item is None:
                raise VendorError("连接器返回空结果")
            counts.fetched += 1
            source = f"ds:{ds_id}:{node.id}"
            action = self.ingestor.upsert(item, source)
            if action == "created":
                counts.created += 1
            elif action == "updated":
                counts.updated += 1
            else:
                counts.skipped += 1
            self.store.upsert_synced_doc(
                ds_id,
                resource_id=root,
                external_id=node.id,
                rev=item.rev or node.rev,
                rag_source=source,
                file_name=item.file_name or sanitize_file_name(node.title, ".md"),
                last_seen_sync_id=log_id,
            )
        except UnsupportedType as exc:
            counts.skipped += 1
            counts.skipped_by_type += 1
            self._record_skip(ds_id, log_id, root, node, previous, str(exc))
        except Exception as exc:
            counts.failed += 1
            failures.append(_failure(node.id, node.title, exc))

    def _record_skip(
        self,
        ds_id: str,
        log_id: str,
        root: str,
        node: ResourceNode,
        previous: Mapping[str, Any] | None,
        reason: str,
    ) -> None:
        self.store.upsert_synced_doc(
            ds_id,
            resource_id=root,
            external_id=node.id,
            rev=node.rev,
            rag_source=(previous or {}).get("rag_source", ""),
            file_name=sanitize_file_name(node.title, ".md"),
            last_seen_sync_id=log_id,
        )

    def _reconcile_deletions(
        self,
        ds_id: str,
        selected: Sequence[str],
        seen: Mapping[str, set[str]],
        known: Mapping[str, dict[str, Any]],
        enumeration: Mapping[str, Any],
        counts: SyncCounts,
        failures: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        candidates: list[dict[str, Any]] = []
        known_total = 0
        for root in selected:
            info = enumeration.get(root) or {}
            if not info.get("complete"):
                continue
            root_rows = [row for row in known.values() if row["resource_id"] == root]
            known_total += len(root_rows)
            if not seen.get(root) and root_rows:
                return True, f"{root} 列举结果为空，已跳过本轮删除"
            candidates.extend(
                row
                for row in root_rows
                if row["external_id"] not in seen.get(root, set()) and row["rag_source"]
            )
        guard = max(5, math.ceil(known_total * 0.2))
        if len(candidates) > guard:
            return True, f"待删除 {len(candidates)} 篇超过警戒值 {guard}，本轮不删除"
        for row in candidates:
            try:
                self.ingestor.delete(row["rag_source"])
            except Exception as exc:
                failures.append(_failure(row["external_id"], row["file_name"], exc))
                counts.failed += 1
                continue
            self.store.delete_synced_doc(ds_id, row["external_id"])
            counts.deleted += 1
        return False, ""

    def _finalize(self, ds_id: str, log_id: str, result: _SyncResult) -> None:
        self.store.finish_sync_log(
            log_id,
            status=result.status,
            counts=result.counts.as_dict(),
            failures=result.failures,
            enumeration=result.enumeration,
            error_code=result.error_code,
            message=result.message,
        )
        self.store.set_last_sync(ds_id, now_iso(), result.status)
        self.store.set_sync_status(ds_id, "idle")
        ds = self.store.get_datasource(ds_id)
        if ds is not None and ds["schedule"]:
            try:
                self.store.set_next_run(ds_id, iso(next_run_at(ds["schedule"], now())))
            except ValueError:
                logger.warning("invalid schedule for datasource %s", ds_id)

    # ------------------------------------------------------------------ logs / documents

    def list_sync_logs(self, ds_id: str, limit: int = 20) -> list[dict[str, Any]]:
        self._require(ds_id)
        return self.store.list_sync_logs(ds_id, limit)

    def get_sync_log(self, ds_id: str, log_id: str) -> dict[str, Any]:
        self._require(ds_id)
        log = self.store.get_sync_log(log_id)
        if log is None or log["datasource_id"] != ds_id:
            raise NotFoundError("同步记录不存在")
        return log

    def list_documents(self, ds_id: str) -> list[dict[str, Any]]:
        self._require(ds_id)
        return self.store.list_synced_docs(ds_id)

    # ------------------------------------------------------------------ scheduler

    def start_scheduler(self) -> asyncio.Task[None]:
        if self._scheduler_task is not None:
            return self._scheduler_task
        self._stop_event = asyncio.Event()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        return self._scheduler_task

    async def stop_scheduler(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._scheduler_task
        self._scheduler_task = None
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def run_due_once(self, moment_iso: str | None = None) -> list[str]:
        """Start every due datasource once; used by the loop and by tests."""
        started: list[str] = []
        for ds in self.store.list_due(moment_iso):
            try:
                started.append(self.start_sync(ds["id"], "schedule"))
            except ConflictError:
                continue
            except Exception:
                logger.exception("scheduled datasource sync failed to start ds=%s", ds["id"])
        return started

    async def _scheduler_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                self.run_due_once()
            except Exception:
                logger.exception("datasource scheduler tick failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------ helpers

    def _require(self, ds_id: str) -> dict[str, Any]:
        ds = self.store.get_datasource(ds_id)
        if ds is None:
            raise NotFoundError("知识源不存在")
        return ds

    def _connector(self, ds: Mapping[str, Any]) -> Connector:
        credentials = self.secret_box.decrypt(ds["credentials_enc"])
        return self.registry.create(ds["type"], credentials, ds["settings"])

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self.ingestor.close()
        self.store.close()


def _filter_credentials(
    metadata: ConnectorMetadata, credentials: Mapping[str, str]
) -> dict[str, str]:
    filtered: dict[str, str] = {}
    missing: list[str] = []
    for field in metadata.auth_fields:
        value = str(credentials.get(field.key, "")).strip()
        if field.required and not value:
            missing.append(field.key)
        filtered[field.key] = value
    if missing:
        raise ValidationError(f"缺少凭据字段：{', '.join(missing)}")
    return filtered


def _filter_settings(metadata: ConnectorMetadata, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field.key: settings.get(field.key, field.default)
        for field in metadata.setting_fields
    }


def _clean_selected(selected: Sequence[str]) -> list[str]:
    cleaned = [str(item).strip() for item in selected if str(item).strip()]
    if not cleaned:
        raise ValidationError("请至少选择一个资源")
    return list(dict.fromkeys(cleaned))


def _next_at_or_raise(schedule: str) -> str | None:
    if not schedule:
        return None
    try:
        return iso(next_run_at(schedule, now()))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _code(exc: BaseException) -> str:
    return getattr(exc, "code", "vendor_error")


def _message(exc: BaseException) -> str:
    text = getattr(exc, "message", "") or str(exc)
    return text[:300]


def _failure(external_id: str, title: str, exc: BaseException) -> dict[str, str]:
    return {
        "external_id": external_id,
        "title": title,
        "error_code": _code(exc),
        "message": _message(exc),
    }


def _log_summary(log: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": log["id"],
        "status": log["status"],
        "trigger": log["trigger"],
        "started_at": log["started_at"],
        "finished_at": log["finished_at"],
        "counts": {key: log[key] for key in (
            "discovered",
            "fetched",
            "created",
            "updated",
            "skipped",
            "skipped_by_type",
            "truncated",
            "failed",
            "deleted",
        )},
        "error_code": log["error_code"],
        "message": log["message"],
    }
