"""Seed a deterministic demo knowledge base for local review and interviews.

The seed uses :class:`FakeEmbedding`, so it never calls an external embedding
provider. Re-running the command is idempotent because documents are upserted
by their stable source identifiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_SERVICE_DIR.parent
sys.path.insert(0, str(AI_SERVICE_DIR))

from config import settings  # noqa: E402
from rag.ingestion import FakeEmbedding, IngestionPipeline  # noqa: E402
from rag.sqlite_store import SQLiteStore  # noqa: E402


DEMO_DOCUMENTS: tuple[tuple[str, str], ...] = (
    (
        "demo/徒步安全.md",
        """# 徒步安全

## 出发前检查

出发前应确认天气、路线开放状态、个人体能和返程时间。高海拔路线需要准备保暖层、备用电源、头灯、地图和离线导航，并把行程告知联系人。

## 恶劣天气

雷暴、大风、暴雪或能见度快速下降时应停止前进，优先向安全、低风险位置撤退。不要把模型建议当作现场救援或官方封山信息。
""",
    ),
    (
        "demo/装备清单.md",
        """# 装备清单

## 基础装备

短途徒步至少准备合脚的鞋、饮水、能量食品、轻量雨具、急救包、手机和备用电源。路线超过一天时，还要根据营地和补给条件准备帐篷、睡袋和炉具。

## 高海拔补充

高海拔环境应增加保暖中间层、防风外层、手套、帽子和防晒用品。装备选择应结合季节、海拔、风力和个人经验复核。
""",
    ),
)


def _documents_from_directory(directory: Path) -> list[tuple[str, str]]:
    if not directory.is_dir():
        return []
    documents: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*.md")):
        documents.append((path.relative_to(directory).as_posix(), path.read_text(encoding="utf-8")))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=settings.rag_sqlite_db_path,
        help="SQLite path (default: RAG_SQLITE_DB_PATH or ./data/rag.sqlite)",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=REPO_ROOT / "rag_docs",
        help="Optional directory of Markdown documents to ingest",
    )
    args = parser.parse_args()

    documents = _documents_from_directory(args.docs_dir) or list(DEMO_DOCUMENTS)
    store = SQLiteStore(args.db, embedding_dimensions=settings.embedding_dimensions)
    try:
        pipeline = IngestionPipeline(
            store=store,
            embedding_provider=FakeEmbedding(dimensions=settings.embedding_dimensions),
        )
        reports = [
            pipeline.ingest_content(
                markdown=content,
                source=f"seed:{source}",
                file_name=source,
                status="active",
            )
            for source, content in documents
        ]
    finally:
        store.close()

    print(json.dumps([report.__dict__ for report in reports], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
