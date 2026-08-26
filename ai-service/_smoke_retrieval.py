"""验证 sqlite-vec vec0 和 FTS5 查询语法。测后即删。"""
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec

from rag.sqlite_schema import connect
from rag.sqlite_store import SQLiteStore, ChildChunk

# 用 4 维向量测试
db_path = Path(":memory:")
store = SQLiteStore(db_path, embedding_dimensions=4)

# 写入 3 个 child（同一 parent）
chunks = [
    ChildChunk(content="徒步装备背包选择指南", embedding=[1.0, 0.0, 0.0, 0.0], child_index=0),
    ChildChunk(content="徒步路线推荐西湖群山", embedding=[0.0, 1.0, 0.0, 0.0], child_index=1),
    ChildChunk(content="雨天徒步注意事项安全", embedding=[0.0, 0.0, 1.0, 0.0], child_index=2),
]
store.upsert_document(
    source="test-doc",
    file_name="test.md",
    parent_content="完整文档内容",
    headings=["徒步"],
    child_chunks=chunks,
)

conn = store._write_conn

print("=== vec0 dense 查询 ===")
query_emb = struct.pack("4f", 1.0, 0.1, 0.0, 0.0)
rows = conn.execute(
    "SELECT child_id, parent_id, distance FROM child_vec WHERE embedding MATCH ? AND k = 3 ORDER BY distance",
    (query_emb,),
).fetchall()
for r in rows:
    print(f"  child_id={r[0][:8]} parent_id={r[1][:8]} distance={r[2]:.4f}")

print("\n=== FTS5 sparse 查询（trigram）===")
# trigram 查询：用 3 字符 n-gram
for query in ["徒步", "徒步装备", "背包", "西湖", "安全"]:
    rows = conn.execute(
        "SELECT rowid, content FROM child_fts WHERE child_fts MATCH ? ORDER BY rank LIMIT 5",
        (query,),
    ).fetchall()
    print(f"  query='{query}' → {len(rows)} hits: {[r[1][:20] for r in rows]}")

print("\n=== 联合查询（dense + parent 信息）===")
rows = conn.execute(
    """
    SELECT c.child_id, c.content, p.file_name, p.source, p.status,
           v.distance
    FROM child_vec v
    JOIN child c ON c.child_id = v.child_id
    JOIN parent p ON p.parent_id = v.parent_id
    WHERE v.embedding MATCH ? AND v.k = 3
    ORDER BY v.distance
    """,
    (query_emb,),
).fetchall()
for r in rows:
    print(f"  child={r[1][:15]} file={r[2]} source={r[3]} status={r[4]} dist={r[5]:.4f}")

print("\n=== source/status 过滤 ===")
rows = conn.execute(
    """
    SELECT c.child_id, c.content, p.source, p.status
    FROM child_vec v
    JOIN child c ON c.child_id = v.child_id
    JOIN parent p ON p.parent_id = v.parent_id
    WHERE v.embedding MATCH ? AND v.k = 10
      AND p.status = 'active'
    ORDER BY v.distance
    """,
    (query_emb,),
).fetchall()
print(f"  status=active: {len(rows)} hits")

rows = conn.execute(
    """
    SELECT c.child_id, c.content, p.source, p.status
    FROM child_vec v
    JOIN child c ON c.child_id = v.child_id
    JOIN parent p ON p.parent_id = v.parent_id
    WHERE v.embedding MATCH ? AND v.k = 10
      AND p.source = 'test-doc'
    ORDER BY v.distance
    """,
    (query_emb,),
).fetchall()
print(f"  source=test-doc: {len(rows)} hits")

print("\n=== 维度不匹配测试 ===")
try:
    wrong_emb = struct.pack("3f", 1.0, 0.0, 0.0)  # 3 维，期望 4 维
    rows = conn.execute(
        "SELECT child_id FROM child_vec WHERE embedding MATCH ? AND k = 1",
        (wrong_emb,),
    ).fetchall()
    print(f"  3维查询4维库: {len(rows)} hits (未报错)")
except Exception as exc:
    print(f"  3维查询4维库报错: {type(exc).__name__}: {exc}")

print("\n=== retrieve_parent 测试 ===")
parent_id = rows[0][2] if rows else store.get_parent("test-doc")["parent_id"]
print(f"  parent_id={parent_id[:8]}")
row = conn.execute(
    "SELECT parent_id, file_name, content, headings FROM parent WHERE parent_id = ?",
    (parent_id,),
).fetchone()
if row:
    print(f"  file_name={row[1]} content={row[2][:20]} headings={row[3]}")

# 不存在的 parent_id
row = conn.execute(
    "SELECT parent_id, file_name, content, headings FROM parent WHERE parent_id = ?",
    ("nonexistent-id",),
).fetchone()
print(f"  nonexistent parent: {row}")

store.close()
print("\n=== 语法验证完成 ===")
