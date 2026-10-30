"""知识库按账号隔离：datasource 行归属 + 账号独立 RAG 索引。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from agent_v2.user_scope import owner_scope
from auth.paths import user_rag_path
from config import settings
from datasource.store import DatasourceStore
from rag.ingestion import FakeEmbedding, IngestionPipeline
from rag.sqlite_store import SQLiteStore

OWNER_A = "a" * 32
OWNER_B = "b" * 32


def _fixture_effective() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_base_url="http://127.0.0.1:1/v1",
        embedding_api_key="fixture-key",
        embedding_model="fixture-embedding",
        embedding_dimensions=8,
    )


def _create_datasource(store: DatasourceStore, name: str, owner_id: str) -> dict:
    return store.create_datasource(
        type="fake",
        name=name,
        credentials_enc="enc",
        owner_id=owner_id,
    )


def test_datasource_rows_are_visible_only_to_their_owner(tmp_path: Path) -> None:
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    try:
        a = _create_datasource(store, "A 的来源", OWNER_A)
        b = _create_datasource(store, "B 的来源", OWNER_B)

        assert [row["id"] for row in store.list_datasources(OWNER_A)] == [a["id"]]
        assert [row["id"] for row in store.list_datasources(OWNER_B)] == [b["id"]]
        assert store.get_datasource(a["id"], OWNER_B) is None
        assert store.update_datasource(a["id"], owner_id=OWNER_B, name="偷改") is None
        assert store.delete_datasource(a["id"], OWNER_B) is False
        assert store.get_datasource(a["id"], OWNER_A)["name"] == "A 的来源"
    finally:
        store.close()


def test_legacy_rows_are_claimed_once(tmp_path: Path) -> None:
    store = DatasourceStore(tmp_path / "datasource.sqlite")
    try:
        legacy = _create_datasource(store, "旧来源", "")
        assert store.claim_legacy(OWNER_A) == 1
        assert store.claim_legacy(OWNER_B) == 0
        assert store.get_datasource(legacy["id"], OWNER_A) is not None
        assert store.get_datasource(legacy["id"], OWNER_B) is None
    finally:
        store.close()


def _seed_user_index(settings, owner_id: str, source: str, content: str) -> None:
    path = user_rag_path(settings, owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(str(path), embedding_dimensions=8)
    try:
        pipeline = IngestionPipeline(
            store=store, embedding_provider=FakeEmbedding(dimensions=8)
        )
        report = pipeline.ingest_content(
            markdown=content,
            source=source,
            file_name=source,
            status="active",
        )
        assert report.child_count > 0
    finally:
        store.close()


def test_retrieval_never_crosses_account_indexes(tmp_path: Path, monkeypatch) -> None:
    from rag.retriever import VectorStoreRetriever
    import tools.hiking_knowledge as hk

    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(tmp_path / "data" / "rag.sqlite"))
    monkeypatch.setattr(settings, "embedding_api_key", "fixture-key")
    monkeypatch.setattr(settings, "embedding_base_url", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(settings, "embedding_model", "fixture-embedding")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)
    monkeypatch.setattr(
        VectorStoreRetriever,
        "__init__",
        _fake_retriever_init(VectorStoreRetriever.__init__),
    )
    hk._owner_retrieval_stacks.clear()

    _seed_user_index(settings, OWNER_A, "A-鳌太线.md", "# A 专属\n\nA 账号的营地水源资料：甲营地有季节性水源。")
    _seed_user_index(settings, OWNER_B, "B-雨崩.md", "# B 专属\n\nB 账号的雨崩路线资料：下雨崩有客栈补给。")

    with owner_scope(OWNER_A, _fixture_effective()):
        result_a = asyncio.run(hk.hiking_knowledge_search.ainvoke({"query": "营地水源补给"}))
    with owner_scope(OWNER_B, _fixture_effective()):
        result_b = asyncio.run(hk.hiking_knowledge_search.ainvoke({"query": "营地水源补给"}))

    sources_a = {chunk["source"] for chunk in result_a["chunks"]}
    sources_b = {chunk["source"] for chunk in result_b["chunks"]}
    assert sources_a and sources_a <= {"A-鳌太线.md"}, sources_a
    assert sources_b and sources_b <= {"B-雨崩.md"}, sources_b


def _fake_retriever_init(original):
    """用确定性 embedding 替换 OpenAI 客户端，保留 retriever 的其余构造逻辑。"""

    def init(self, *args, **kwargs):
        kwargs.setdefault("api_key", "fixture-key")
        kwargs.setdefault("base_url", "http://127.0.0.1:1/v1")
        kwargs.setdefault("model", "fixture-embedding")
        kwargs.setdefault("dimensions", 8)
        original(self, *args, **kwargs)
        self.embeddings = FakeEmbedding(dimensions=8)

    return init


def test_new_account_without_directory_gets_empty_knowledge_not_a_crash(
    tmp_path: Path, monkeypatch
) -> None:
    from rag.retriever import VectorStoreRetriever
    import tools.hiking_knowledge as hk

    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(tmp_path / "data" / "rag.sqlite"))
    monkeypatch.setattr(settings, "embedding_api_key", "fixture-key")
    monkeypatch.setattr(settings, "embedding_base_url", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(settings, "embedding_model", "fixture-embedding")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)
    monkeypatch.setattr(
        VectorStoreRetriever,
        "__init__",
        _fake_retriever_init(VectorStoreRetriever.__init__),
    )
    hk._owner_retrieval_stacks.clear()
    fresh_owner = "c" * 32
    assert not user_rag_path(settings, fresh_owner).parent.exists()

    with owner_scope(fresh_owner, _fixture_effective()):
        result = asyncio.run(hk.hiking_knowledge_search.ainvoke({"query": "鳌太线"}))

    assert result["chunks"] == []
    assert "资料不足" in result["message"]
    assert user_rag_path(settings, fresh_owner).parent.is_dir()


def test_unconfigured_account_embedding_returns_empty_without_creating_index(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.hiking_knowledge as hk

    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(tmp_path / "data" / "rag.sqlite"))
    hk._owner_retrieval_stacks.clear()
    fresh_owner = "d" * 32

    with owner_scope(fresh_owner, _fixture_effective_unset()):
        result = asyncio.run(hk.hiking_knowledge_search.ainvoke({"query": "鳌太线"}))

    assert result["chunks"] == []
    assert "尚未配置 Embedding" in result["message"]
    assert not user_rag_path(settings, fresh_owner).parent.exists()


def _fixture_effective_unset() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_base_url="",
        embedding_api_key="",
        embedding_model="",
        embedding_dimensions=1536,
    )


def test_explicit_empty_owner_fails_closed_instead_of_reading_legacy_index() -> None:
    import tools.hiking_knowledge as hk

    with owner_scope(""):
        result = asyncio.run(hk.hiking_knowledge_search.ainvoke({"query": "鳌太线"}))
    assert result["chunks"] == []
    assert "未绑定" in result["message"]
