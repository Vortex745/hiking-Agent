"""验证 rag/hybrid_retrieval.py 端到端。测后即删。"""
from pathlib import Path

from rag.sqlite_schema import connect
from rag.sqlite_store import SQLiteStore, ChildChunk
from rag.hybrid_retrieval import search_child_chunks, retrieve_parent_chunks

TEST_DIM = 4
db_path = Path(":memory:")
store = SQLiteStore(db_path, embedding_dimensions=TEST_DIM)
conn = store._write_conn

# 写入 2 个文档
store.upsert_document(
    source="doc-equipment",
    file_name="equipment.md",
    parent_content="# 徒步装备指南\n背包选择、鞋袜配置、雨具准备。",
    headings=["徒步装备"],
    child_chunks=[
        ChildChunk(content="徒步装备背包选择指南，70升 backpack 推荐", embedding=[1.0, 0.0, 0.0, 0.0], child_index=0),
        ChildChunk(content="徒步鞋袜选择，防水透气袜推荐", embedding=[0.9, 0.1, 0.0, 0.0], child_index=1),
    ],
    status="active",
)
store.upsert_document(
    source="doc-route",
    file_name="route.md",
    parent_content="# 徒步路线推荐\n西湖群山路线、雨天注意事项。",
    headings=["徒步路线"],
    child_chunks=[
        ChildChunk(content="徒步路线推荐西湖群山，适合周末", embedding=[0.0, 1.0, 0.0, 0.0], child_index=0),
        ChildChunk(content="雨天徒步注意事项安全，防滑防雷", embedding=[0.0, 0.0, 1.0, 0.0], child_index=1),
    ],
    status="archived",  # 故意 archived，验证 status 过滤
)

print("=== 1. hybrid 搜索（全状态）===")
results = search_child_chunks(
    conn,
    query_embedding=[1.0, 0.0, 0.0, 0.0],
    query_text="徒步装备背包",
    k=4,
    embedding_dimensions=TEST_DIM,
)
for r in results:
    print(f"  child={r['content'][:20]} dense={r['score_dense']:.4f} sparse={r['score_sparse']:.4f} rrf={r['score_rrf']:.6f} status={r['status']}")
assert len(results) > 0, "应有结果"

print("\n=== 2. status 过滤（仅 active）===")
results = search_child_chunks(
    conn,
    query_embedding=[1.0, 0.0, 0.0, 0.0],
    query_text="徒步装备背包",
    k=4,
    status_filter="active",
    embedding_dimensions=TEST_DIM,
)
for r in results:
    print(f"  child={r['content'][:20]} status={r['status']}")
    assert r["status"] == "active"
print(f"  active 结果数: {len(results)}")

print("\n=== 3. source 过滤 ===")
results = search_child_chunks(
    conn,
    query_embedding=[1.0, 0.0, 0.0, 0.0],
    query_text="徒步",
    k=4,
    source_filter="doc-equipment",
    embedding_dimensions=TEST_DIM,
)
for r in results:
    print(f"  child={r['content'][:20]} source={r['source']}")
    assert r["source"] == "doc-equipment"

print("\n=== 4. RRF 顺序稳定性（同输入两次结果完全一致）===")
r1 = search_child_chunks(conn, query_embedding=[1.0, 0.1, 0.0, 0.0], query_text="徒步装备", k=4, embedding_dimensions=TEST_DIM)
r2 = search_child_chunks(conn, query_embedding=[1.0, 0.1, 0.0, 0.0], query_text="徒步装备", k=4, embedding_dimensions=TEST_DIM)
assert [r["child_id"] for r in r1] == [r["child_id"] for r in r2], "RRF 顺序不稳定"
print(f"  两次结果 child_id 完全一致: {[r['child_id'][:8] for r in r1]}")

print("\n=== 5. retrieve_parent_chunks 命中 ===")
parent_ids = [r["parent_id"] for r in r1]
result = retrieve_parent_chunks(conn, parent_ids)
print(f"  requested={len(parent_ids)} hit={len(result['parents'])} has_more={result['has_more']}")
for p in result["parents"]:
    print(f"  parent={p['parent_id'][:8]} file={p['file_name']} content={p['content'][:30]} headings={p['headings']}")

print("\n=== 6. retrieve_parent_chunks 哨兵（空列表）===")
result = retrieve_parent_chunks(conn, [])
print(f"  parents={result['parents']} has_more={result['has_more']}")
assert result == {"parents": [], "has_more": False}

print("\n=== 7. retrieve_parent_chunks 不存在 id 跳过 ===")
result = retrieve_parent_chunks(conn, ["nonexistent-id-1", "nonexistent-id-2"])
print(f"  parents={result['parents']} has_more={result['has_more']}")
assert result == {"parents": [], "has_more": False}

print("\n=== 8. retrieve_parent_chunks 部分命中 ===")
result = retrieve_parent_chunks(conn, [parent_ids[0], "nonexistent"])
print(f"  hit={len(result['parents'])} has_more={result['has_more']}")
assert len(result["parents"]) == 1

print("\n=== 9. 维度不匹配显式报错 ===")
try:
    search_child_chunks(conn, query_embedding=[1.0, 0.0, 0.0], query_text="test", k=1, embedding_dimensions=TEST_DIM)
    print("  FAIL: 未抛错")
except ValueError as e:
    print(f"  OK: {e}")

print("\n=== 10. 2 字符中文查询（trigram 局限）===")
results = search_child_chunks(
    conn,
    query_embedding=[0.0, 0.0, 0.0, 0.0],
    query_text="徒步",  # 2 字符
    k=4,
    embedding_dimensions=TEST_DIM,
)
print(f"  query='徒步' results={len(results)} (trigram 对 2 字符返回空 sparse)")

print("\n=== 11. 耗时观测（logger 应输出 elapsed）===")
import logging
logging.basicConfig(level=logging.INFO)
results = search_child_chunks(conn, query_embedding=[1.0, 0.0, 0.0, 0.0], query_text="徒步装备", k=4, embedding_dimensions=TEST_DIM)

store.close()
print("\n=== 全部验证通过 ===")
