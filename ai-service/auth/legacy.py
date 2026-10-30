"""首个/首个触发注册的账号认领旧版单用户数据。

根目录旧文件只复制不移动，回滚到旧版本仍然可用。每类数据独立标记，
这样后续阶段新增的认领类型不会被已经完成的注册挡住。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

from auth.paths import user_mcp_path, user_rag_path
from auth.store import AuthStore

logger = logging.getLogger("ai-service.auth.legacy")

_MCP_META_KEY = "legacy_mcp_claimed_by"
_DATASOURCE_META_KEY = "legacy_datasource_claimed_by"
_RAG_META_KEY = "legacy_rag_claimed_by"
_OWNER_META_KEY = "legacy_owner_id"


def claim_legacy_data(settings, auth_store: AuthStore, owner_id: str) -> None:
    # 旧单用户数据只能归第一个注册账号；即使某类复制失败，也不能让后注册账号接手。
    claimed_by = auth_store.get_meta(_OWNER_META_KEY)
    if not claimed_by:
        auth_store.set_meta(_OWNER_META_KEY, owner_id)
    elif claimed_by != owner_id:
        return
    _claim_mcp(settings, auth_store, owner_id)
    _claim_datasources(settings, auth_store, owner_id)
    _claim_rag(settings, auth_store, owner_id)


def _claim_mcp(settings, auth_store: AuthStore, owner_id: str) -> None:
    if auth_store.get_meta(_MCP_META_KEY):
        return
    source = Path(settings.agent_v2_state_dir) / "mcp_servers.json"
    target = user_mcp_path(settings, owner_id)
    try:
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            logger.info("legacy MCP config copied to user=%s", owner_id)
    except OSError:
        logger.exception("legacy MCP config claim failed user=%s", owner_id)
        return
    auth_store.set_meta(_MCP_META_KEY, owner_id)


def _claim_datasources(settings, auth_store: AuthStore, owner_id: str) -> None:
    if auth_store.get_meta(_DATASOURCE_META_KEY):
        return
    from datasource.store import DatasourceStore

    db_path = getattr(settings, "datasource_db_path", "") or str(
        Path(settings.rag_sqlite_db_path).with_name("datasource.sqlite")
    )
    store = DatasourceStore(db_path)
    try:
        claimed = store.claim_legacy(owner_id)
        if claimed:
            logger.info("legacy datasources claimed user=%s count=%s", owner_id, claimed)
    except Exception:
        logger.exception("legacy datasource claim failed user=%s", owner_id)
        return
    finally:
        store.close()
    auth_store.set_meta(_DATASOURCE_META_KEY, owner_id)


def _claim_rag(settings, auth_store: AuthStore, owner_id: str) -> None:
    if auth_store.get_meta(_RAG_META_KEY):
        return
    source = Path(settings.rag_sqlite_db_path)
    target = user_rag_path(settings, owner_id)
    try:
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source_conn = sqlite3.connect(str(source), timeout=5.0)
            target_conn = sqlite3.connect(str(target), timeout=5.0)
            try:
                source_conn.backup(target_conn)
            finally:
                target_conn.close()
                source_conn.close()
            logger.info("legacy RAG index copied to user=%s", owner_id)
    except (OSError, sqlite3.Error):
        logger.exception("legacy RAG claim failed user=%s", owner_id)
        return
    auth_store.set_meta(_RAG_META_KEY, owner_id)
