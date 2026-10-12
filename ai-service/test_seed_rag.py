"""Deterministic RAG seed smoke test."""

from __future__ import annotations

from scripts.seed_rag import DEMO_DOCUMENTS, _documents_from_directory


def test_seed_documents_are_stable_and_non_empty(tmp_path) -> None:
    assert len(DEMO_DOCUMENTS) >= 2
    assert all(source.startswith("demo/") and content.strip() for source, content in DEMO_DOCUMENTS)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "b.md").write_text("# B\ncontent", encoding="utf-8")
    (docs_dir / "a.md").write_text("# A\ncontent", encoding="utf-8")
    assert [source for source, _ in _documents_from_directory(docs_dir)] == ["a.md", "b.md"]
