"""Distance-value extraction on knowledge chunks; no retrieval stack.

The gate (run_orchestration) consumes audit-transported distance_values,
so the tool layer must extract them from the FULL doc content — the
preview is truncated at 220 chars and must never be the source.
"""
from tools.hiking_knowledge import _chunks_from_docs


def test_chunks_carry_distance_values_extracted_from_full_content():
    docs = [{"content": "全程 12.50 公里，累计爬升 800 米，支线约 3km。", "source": "trail.md"}]
    chunks = _chunks_from_docs(docs)
    assert chunks[0]["distance_values"] == [12.5, 3.0]


def test_chunk_without_distance_numbers_has_empty_distance_values():
    docs = [{"content": "注意防滑，冬季注意结冰。", "source": "safety.md"}]
    chunks = _chunks_from_docs(docs)
    assert chunks[0]["distance_values"] == []


def test_distance_extraction_uses_full_content_not_truncated_preview():
    tail = "尾段" + "细节" * 200 + "终点前 2 公里设补给。"
    docs = [{"content": "全程 9 公里。" + tail, "source": "t.md"}]
    chunks = _chunks_from_docs(docs)
    assert len(chunks[0]["preview"]) <= 223  # preview stays truncated...
    assert chunks[0]["distance_values"] == [9.0, 2.0]  # ...but extraction sees all of it.
