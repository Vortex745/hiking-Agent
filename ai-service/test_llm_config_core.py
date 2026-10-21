"""Ticket 01: LLM runtime configuration core.

Covers atomic .env persistence, in-process settings/os.environ hot update,
vec0 dimension discovery, and fingerprint-based retrieval stack invalidation.

Hermetic: every persistence test writes its own temporary .env; process-wide
settings and os.environ mutations are restored after each test.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from dotenv import dotenv_values

import config
from config import settings
from rag import sqlite_schema

ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "RERANK_BASE_URL",
    "RERANK_API_KEY",
    "RERANK_MODEL",
)

SETTINGS_ATTRS = (
    "openai_base_url",
    "openai_api_key",
    "openai_model",
    "embedding_base_url",
    "embedding_api_key",
    "embedding_model",
    "embedding_dimensions",
    "rerank_base_url",
    "rerank_api_key",
    "rerank_model",
)


@pytest.fixture(autouse=True)
def _restore_process_state():
    """update_runtime_config mutates the global singleton and os.environ."""
    saved_settings = {name: getattr(settings, name) for name in SETTINGS_ATTRS}
    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    yield
    for name, value in saved_settings.items():
        setattr(settings, name, value)
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_env(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_update_runtime_config_persists_applies_and_preserves_other_lines(tmp_path):
    env_path = _write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "# llm settings",
                "OPENAI_BASE_URL='https://old.example/v1'",
                "OPENAI_API_KEY='sk-old-llm'",
                "OPENAI_MODEL=old-model",
                "",
                "# unrelated block",
                "UNRELATED_KEY=keep-me",
                "EMBEDDING_DIMENSIONS=1536",
                "",
            ]
        ),
    )
    setattr(settings, "embedding_dimensions", 1536)

    changed = config.update_runtime_config(
        {
            "llm": {
                "base_url": "https://new.example/v1",
                "model": "new-model",
                "api_key": "sk-new-llm",
            },
            "embedding": {"dimensions": "1024"},
            "rerank": {"model": "ranker-v2"},
        },
        env_path=env_path,
    )

    assert set(changed) == {
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "EMBEDDING_DIMENSIONS",
        "RERANK_MODEL",
    }

    text = env_path.read_text(encoding="utf-8")
    assert "# llm settings" in text
    assert "# unrelated block" in text
    assert "UNRELATED_KEY=keep-me" in text

    values = dotenv_values(env_path)
    assert values["OPENAI_BASE_URL"] == "https://new.example/v1"
    assert values["OPENAI_MODEL"] == "new-model"
    assert values["OPENAI_API_KEY"] == "sk-new-llm"
    assert values["EMBEDDING_DIMENSIONS"] == "1024"
    assert values["RERANK_MODEL"] == "ranker-v2"

    assert settings.openai_base_url == "https://new.example/v1"
    assert settings.openai_api_key == "sk-new-llm"
    assert settings.openai_model == "new-model"
    assert settings.embedding_dimensions == 1024
    assert isinstance(settings.embedding_dimensions, int)
    assert settings.rerank_model == "ranker-v2"

    assert os.environ["OPENAI_BASE_URL"] == "https://new.example/v1"
    assert os.environ["OPENAI_MODEL"] == "new-model"
    assert os.environ["EMBEDDING_DIMENSIONS"] == "1024"


def test_update_runtime_config_keeps_api_key_when_none_or_empty(tmp_path):
    env_path = _write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "OPENAI_API_KEY='sk-keep-llm'",
                "EMBEDDING_API_KEY='emb-keep'",
                "RERANK_API_KEY='rerank-keep'",
                "",
            ]
        ),
    )
    settings.openai_api_key = "settings-keep-llm"
    settings.embedding_api_key = "settings-keep-emb"
    settings.rerank_api_key = "settings-keep-rerank"

    changed = config.update_runtime_config(
        {
            "llm": {"api_key": None},
            "embedding": {"api_key": ""},
            "rerank": {"api_key": "   "},
        },
        env_path=env_path,
    )

    assert changed == []
    values = dotenv_values(env_path)
    assert values["OPENAI_API_KEY"] == "sk-keep-llm"
    assert values["EMBEDDING_API_KEY"] == "emb-keep"
    assert values["RERANK_API_KEY"] == "rerank-keep"
    assert settings.openai_api_key == "settings-keep-llm"
    assert settings.embedding_api_key == "settings-keep-emb"
    assert settings.rerank_api_key == "settings-keep-rerank"


def test_update_runtime_config_returns_empty_for_unchanged_values(tmp_path):
    env_path = _write_env(
        tmp_path / ".env",
        "OPENAI_MODEL=stable-model\nEMBEDDING_DIMENSIONS=1024\n",
    )
    before = env_path.read_text(encoding="utf-8")

    changed = config.update_runtime_config(
        {"llm": {"model": "stable-model"}, "embedding": {"dimensions": 1024}},
        env_path=env_path,
    )

    assert changed == []
    assert env_path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("bad_dimensions", ["not-a-number", 0, -1024, 1.5, True, None])
def test_update_runtime_config_rejects_invalid_dimensions_without_writing(
    tmp_path, bad_dimensions
):
    env_path = _write_env(
        tmp_path / ".env",
        "OPENAI_MODEL=before-model\nEMBEDDING_DIMENSIONS=1536\n",
    )
    before = env_path.read_text(encoding="utf-8")
    settings.openai_model = "settings-before"
    settings.embedding_dimensions = 1536

    with pytest.raises(ValueError):
        config.update_runtime_config(
            {
                "llm": {"model": "after-model"},
                "embedding": {"dimensions": bad_dimensions},
            },
            env_path=env_path,
        )

    assert env_path.read_text(encoding="utf-8") == before
    assert settings.openai_model == "settings-before"
    assert settings.embedding_dimensions == 1536


@pytest.mark.parametrize("field", ["base_url", "model"])
def test_update_runtime_config_rejects_blank_required_text_fields(tmp_path, field):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=before-model\n")
    before = env_path.read_text(encoding="utf-8")
    settings.openai_model = "settings-before"

    with pytest.raises(ValueError):
        config.update_runtime_config({"llm": {field: "   "}}, env_path=env_path)

    assert env_path.read_text(encoding="utf-8") == before
    assert settings.openai_model == "settings-before"


def test_update_runtime_config_creates_missing_env_file(tmp_path):
    env_path = tmp_path / ".env"
    assert not env_path.exists()

    changed = config.update_runtime_config(
        {
            "llm": {"base_url": "https://fresh.example/v1", "model": "fresh-model"},
            "embedding": {"model": "fresh-embed", "dimensions": 1024},
        },
        env_path=env_path,
    )

    assert set(changed) == {
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
    }
    assert env_path.exists()
    assert [p.name for p in tmp_path.iterdir()] == [".env"]
    values = dotenv_values(env_path)
    assert values["OPENAI_BASE_URL"] == "https://fresh.example/v1"
    assert values["EMBEDDING_DIMENSIONS"] == "1024"
    assert settings.embedding_model == "fresh-embed"


def test_update_runtime_config_defaults_to_module_env_path(tmp_path, monkeypatch):
    """Default target is config.ENV_PATH, not a freshly recomputed path."""
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=old-model\n")
    monkeypatch.setattr(config, "ENV_PATH", env_path)

    changed = config.update_runtime_config({"llm": {"model": "default-path-model"}})

    assert changed == ["OPENAI_MODEL"]
    assert dotenv_values(env_path)["OPENAI_MODEL"] == "default-path-model"
    assert settings.openai_model == "default-path-model"


def test_update_runtime_config_retries_transient_replace_permission_error(
    tmp_path, monkeypatch
):
    env_path = _write_env(tmp_path / ".env", "OPENAI_MODEL=old-model\n")
    real_set_key = config.set_key
    calls = {"count": 0}

    def flaky_set_key(path, key, value):
        calls["count"] += 1
        if calls["count"] == 1:
            # Windows: freshly written temp files can be briefly locked.
            raise PermissionError("transient lock")
        return real_set_key(path, key, value)

    monkeypatch.setattr(config, "set_key", flaky_set_key)

    changed = config.update_runtime_config(
        {"llm": {"model": "retried-model"}}, env_path=env_path
    )

    assert changed == ["OPENAI_MODEL"]
    assert calls["count"] == 2
    assert dotenv_values(env_path)["OPENAI_MODEL"] == "retried-model"
    assert settings.openai_model == "retried-model"


def test_read_vec0_dimensions_reads_temp_vec0_table(tmp_path):
    db_path = tmp_path / "rag.sqlite"
    conn = sqlite_schema.connect(db_path, embedding_dimensions=1024)
    conn.close()

    assert sqlite_schema.read_vec0_dimensions(db_path) == 1024


def test_read_vec0_dimensions_reads_repo_index_dimension():
    db_path = Path(__file__).resolve().parent / "data" / "rag.sqlite"
    if not db_path.is_file():
        pytest.skip("repo rag.sqlite not present")

    assert sqlite_schema.read_vec0_dimensions(db_path) == 1024


def test_read_vec0_dimensions_returns_none_for_missing_or_unparseable(tmp_path):
    assert sqlite_schema.read_vec0_dimensions(tmp_path / "missing.sqlite") is None

    plain_db = tmp_path / "plain.sqlite"
    conn = sqlite3.connect(plain_db)
    conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    assert sqlite_schema.read_vec0_dimensions(plain_db) is None

    no_dimensions_db = tmp_path / "no_dimensions.sqlite"
    conn = sqlite3.connect(no_dimensions_db)
    conn.execute("CREATE TABLE child_vec (child_id TEXT PRIMARY KEY, embedding TEXT)")
    conn.commit()
    conn.close()
    assert sqlite_schema.read_vec0_dimensions(no_dimensions_db) is None

    garbage = tmp_path / "garbage.sqlite"
    garbage.write_text("this is not a sqlite database", encoding="utf-8")
    assert sqlite_schema.read_vec0_dimensions(garbage) is None


class _FakeRetriever:
    def __init__(self) -> None:
        self.close_calls = 0
        self.embeddings = None
        self.sqlite_store = None

    def close(self) -> None:
        self.close_calls += 1


def _reset_hiking_knowledge_stack():
    import tools.hiking_knowledge as hk

    saved = (
        hk._retriever,
        hk._subgraph,
        getattr(hk, "_retrieval_stack_fingerprint", None),
    )
    hk._retriever = None
    hk._subgraph = None
    if hasattr(hk, "_retrieval_stack_fingerprint"):
        hk._retrieval_stack_fingerprint = None
    return hk, saved


def _restore_hiking_knowledge_stack(hk, saved):
    hk._retriever, hk._subgraph, fingerprint = saved
    hk._retrieval_stack_fingerprint = fingerprint


def test_retrieval_stack_rebuilds_on_config_change(monkeypatch):
    import rag.retrieve_subgraph as retrieve_subgraph

    hk, saved = _reset_hiking_knowledge_stack()
    created_retrievers: list[_FakeRetriever] = []
    built_retrievers: list[_FakeRetriever] = []

    def fake_retriever_factory() -> _FakeRetriever:
        retriever = _FakeRetriever()
        created_retrievers.append(retriever)
        return retriever

    def fake_build_retrieve_subgraph(*, retriever, parent_retriever_fn):
        built_retrievers.append(retriever)
        return {"retriever": retriever, "parent_retriever_fn": parent_retriever_fn}

    monkeypatch.setattr(hk, "VectorStoreRetriever", fake_retriever_factory)
    monkeypatch.setattr(
        retrieve_subgraph, "build_retrieve_subgraph", fake_build_retrieve_subgraph
    )
    monkeypatch.setattr(settings, "embedding_base_url", "https://embed-a.example/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "emb-key-a")
    monkeypatch.setattr(settings, "embedding_model", "embed-model-a")
    monkeypatch.setattr(settings, "embedding_dimensions", 1024)
    monkeypatch.setattr(settings, "rag_sqlite_db_path", "./data/rag-a.sqlite")

    try:
        first_retriever, first_subgraph = hk._get_retrieval_stack()
        again_retriever, again_subgraph = hk._get_retrieval_stack()
        assert again_retriever is first_retriever
        assert again_subgraph is first_subgraph
        assert len(created_retrievers) == 1

        changes = (
            ("embedding_base_url", "https://embed-b.example/v1"),
            ("embedding_api_key", "emb-key-b"),
            ("embedding_model", "embed-model-b"),
            ("embedding_dimensions", 1536),
            ("rag_sqlite_db_path", "./data/rag-b.sqlite"),
        )
        for index, (attr, value) in enumerate(changes, start=1):
            monkeypatch.setattr(settings, attr, value)
            current_retriever, _ = hk._get_retrieval_stack()
            assert current_retriever is created_retrievers[index]
            assert len(created_retrievers) == index + 1

        assert built_retrievers == created_retrievers
        # Stale stacks are only dereferenced. An immediate close would pull the
        # SQLite connection from under requests still using the old stack, so
        # the connection is left to the garbage collector.
        assert all(retriever.close_calls == 0 for retriever in created_retrievers)
    finally:
        _restore_hiking_knowledge_stack(hk, saved)


def test_hiking_knowledge_exposes_no_public_cache_reset():
    import tools.hiking_knowledge as hk

    public_names = [name for name in vars(hk) if not name.startswith("_")]
    assert not [name for name in public_names if "reset" in name.lower()]
