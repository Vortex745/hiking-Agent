"""临时脚本：验证 pymupdf4llm 与 MarkdownHeaderTextSplitter 可用性。"""
import sys

# 1. pymupdf4llm
try:
    import pymupdf4llm
    print(f"pymupdf4llm OK: {pymupdf4llm.__version__ if hasattr(pymupdf4llm, '__version__') else 'no version attr'}")
except ImportError as e:
    print(f"FAIL pymupdf4llm: {e}")
    sys.exit(1)

# 2. MarkdownHeaderTextSplitter
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    print("MarkdownHeaderTextSplitter OK")
except ImportError as e:
    print(f"FAIL MarkdownHeaderTextSplitter: {e}")
    sys.exit(1)

# 3. 验证 MarkdownHeaderTextSplitter 行为
md = """
# H1 徒步装备

## H2 背包选择

### H3 容量

40L 背包适合 weekend hike。

### H3 材质

尼龙材质轻便。

## H2 鞋子

登山鞋需防水。

# H2 食物

能量棒便于携带。
"""

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
)
chunks = splitter.split_text(md)
print(f"\nMarkdownHeaderTextSplitter 产出 {len(chunks)} 个 chunks:")
for i, c in enumerate(chunks):
    print(f"  [{i}] H1={c.metadata.get('H1')!r} H2={c.metadata.get('H2')!r} H3={c.metadata.get('H3')!r}")
    print(f"      content={c.page_content!r}")

# 4. 验证 RecursiveCharacterTextSplitter
rcs = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
sub = rcs.split_text("a" * 200)
print(f"\nRecursiveCharacterTextSplitter chunk_size=50: 200字符 → {len(sub)} chunks, 每个 {len(sub[0])} 字符")
