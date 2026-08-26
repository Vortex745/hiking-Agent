"""票据 04 验收 5：中文 FTS5 trigram vs trigram+jieba 召回对比评测。

评测目标：为 sqlite-storage-contract.md §6 的 FTS5 tokenizer 选择提供中文语料证据。

评测方法：
1. 构造 15 篇中文徒步语料（确定性，手写覆盖装备/路线/安全/天气等主题）
2. 定义 10 个查询 + 人工标注的相关文档（ground truth）
3. 建两套 FTS5 索引：
   - A: trigram tokenizer（当前 schema）
   - B: unicode61 tokenizer + jieba 预分词（写入前用 jieba.cut 分词，空格连接）
4. 对每个查询计算 recall@5 / precision@5 / MRR
5. 输出 markdown 报告到 .scratch/rag-lifecycle-refactor/eval/fts5-chinese-eval.md

运行：E:\\miniconda3\\python.exe eval_fts5_chinese.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import jieba
import sqlite_vec

# === 语料 ===

CORPUS: list[tuple[str, str]] = [
    # (doc_id, content)
    ("d01", "徒步装备背包选择指南，70升背包适合多日徒步，30升适合单日。"),
    ("d02", "徒步鞋袜搭配，防水透气袜推荐，防磨水泡处理方法。"),
    ("d03", "雨天徒步注意事项，防滑鞋底选择，雷暴天气严禁登山。"),
    ("d04", "西湖群山徒步路线推荐，适合周末单日往返。"),
    ("d05", "高海拔徒步装备清单，羽绒服、氧气瓶、防晒霜必备。"),
    ("d06", "夏季徒步中暑预防，每小时补水200毫升，避开正午时段。"),
    ("d07", "冬季徒步失温风险，三层穿衣法则，内层排汗中层保暖外层防风。"),
    ("d08", "徒步食物准备，高能量食品选择，坚果、巧克力、能量胶。"),
    ("d09", "山洪防范，暴雨后24小时不入山沟，注意水位上涨征兆。"),
    ("d10", "夜徒装备，头灯选择，反光条，哨子，备用电池。"),
    ("d11", "背包防水处理，防雨罩使用，内置防水袋分层收纳。"),
    ("d12", "徒步拐杖使用技巧，上下坡正确长度调节，减轻膝盖压力。"),
    ("d13", "迷路应急处理，STOP 原则：停下、思考、观察、计划。"),
    ("d14", "落石风险路段，佩戴头盔，靠山壁行走，避免停留。"),
    ("d15", "徒步后恢复，拉伸、补水、蛋白质补充，48小时内冰敷。"),
]

# === 查询 + ground truth ===

QUERIES: list[tuple[str, str, list[str]]] = [
    # (query_id, query_text, relevant_doc_ids)
    # 2 字符查询（trigram 理论上无法匹配）
    ("q01", "背包", ["d01", "d11"]),
    ("q02", "徒步", ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d10", "d12", "d13", "d15"]),
    ("q03", "雨", ["d03", "d11"]),
    # 3-4 字符查询（trigram 可匹配）
    ("q04", "徒步装备", ["d01", "d02", "d05", "d10"]),
    ("q05", "雨天徒步", ["d03"]),
    ("q06", "中暑预防", ["d06"]),
    # 长查询（自然语言问题）
    ("q07", "夏季徒步如何预防中暑", ["d06"]),
    ("q08", "高海拔需要带什么装备", ["d05"]),
    ("q09", "徒步迷路了怎么办", ["d13"]),
    ("q10", "背包防水怎么做", ["d11", "d01"]),
]


def build_trigram_db() -> sqlite3.Connection:
    """构建 trigram tokenizer 的 FTS5 索引。"""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.execute(
        """
        CREATE VIRTUAL TABLE child_fts USING fts5(
            doc_id,
            content,
            tokenize='trigram'
        )
        """
    )
    for doc_id, content in CORPUS:
        conn.execute("INSERT INTO child_fts(doc_id, content) VALUES (?, ?)", (doc_id, content))
    return conn


def build_jieba_db() -> sqlite3.Connection:
    """构建 unicode61 + jieba 预分词的 FTS5 索引。"""
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    # unicode61 按空白分词；jieba 预分词后用空格连接
    conn.execute(
        """
        CREATE VIRTUAL TABLE child_fts USING fts5(
            doc_id,
            content,
            tokenize='unicode61'
        )
        """
    )
    for doc_id, content in CORPUS:
        # jieba 预分词：用空格连接 tokens
        tokens = list(jieba.cut(content, cut_all=False))
        tokenized = " ".join(tokens)
        conn.execute("INSERT INTO child_fts(doc_id, content) VALUES (?, ?)", (doc_id, tokenized))
    return conn


def search_trigram(conn: sqlite3.Connection, query: str, k: int = 5) -> list[str]:
    """trigram 查询：短语匹配。返回 doc_id 列表。"""
    escaped = query.replace('"', '""')
    fts_query = f'"{escaped}"'
    try:
        rows = conn.execute(
            "SELECT doc_id FROM child_fts WHERE child_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, k),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def search_jieba(conn: sqlite3.Connection, query: str, k: int = 5) -> list[str]:
    """jieba 预分词查询：对 query 也 jieba 分词，用 OR 连接（更适合自然语言查询）。

    策略：对短查询（<=4 字符）用短语匹配（精确词项）；
    对长查询用 OR 连接（召回更广）。
    """
    tokens = [t for t in jieba.cut(query, cut_all=False) if t.strip()]
    if not tokens:
        return []

    if len(query) <= 4:
        # 短查询：短语匹配（所有 token 必须连续出现）
        fts_query = '"' + " ".join(tokens) + '"'
    else:
        # 长查询：OR 连接（任一 token 出现即命中）
        # 用 OR 时仍按 rank 排序，相关度高的排前
        fts_query = " OR ".join(tokens)

    try:
        rows = conn.execute(
            "SELECT doc_id FROM child_fts WHERE child_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, k),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def compute_metrics(
    retrieved: list[str], relevant: list[str], k: int = 5
) -> dict[str, float]:
    """计算 recall@k / precision@k / MRR。"""
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = [d for d in retrieved_k if d in relevant_set]
    recall = len(hits) / len(relevant) if relevant else 0.0
    precision = len(hits) / len(retrieved_k) if retrieved_k else 0.0
    # MRR：第一个相关文档的倒数排名
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break
    return {"recall@5": recall, "precision@5": precision, "mrr": mrr}


def run_eval() -> dict[str, Any]:
    """运行完整评测。"""
    trigram_conn = build_trigram_db()
    jieba_conn = build_jieba_db()

    results: list[dict[str, Any]] = []
    for qid, query, relevant in QUERIES:
        # trigram
        trig_retrieved = search_trigram(trigram_conn, query, k=5)
        trig_metrics = compute_metrics(trig_retrieved, relevant)

        # jieba
        jieba_retrieved = search_jieba(jieba_conn, query, k=5)
        jieba_metrics = compute_metrics(jieba_retrieved, relevant)

        results.append({
            "qid": qid,
            "query": query,
            "query_len": len(query),
            "relevant": relevant,
            "trigram": {
                "retrieved": trig_retrieved,
                **trig_metrics,
            },
            "jieba": {
                "retrieved": jieba_retrieved,
                **jieba_metrics,
            },
        })

    trigram_conn.close()
    jieba_conn.close()
    return {"results": results}


def write_report(eval_data: dict[str, Any], output_path: Path) -> None:
    """生成 markdown 评测报告。"""
    results = eval_data["results"]

    # 汇总指标
    n = len(results)
    trig_avg_recall = sum(r["trigram"]["recall@5"] for r in results) / n
    jieba_avg_recall = sum(r["jieba"]["recall@5"] for r in results) / n
    trig_avg_precision = sum(r["trigram"]["precision@5"] for r in results) / n
    jieba_avg_precision = sum(r["jieba"]["precision@5"] for r in results) / n
    trig_avg_mrr = sum(r["trigram"]["mrr"] for r in results) / n
    jieba_avg_mrr = sum(r["jieba"]["mrr"] for r in results) / n

    # 按查询长度分组
    short_queries = [r for r in results if r["query_len"] <= 2]
    long_queries = [r for r in results if r["query_len"] > 2]

    trig_short_recall = (
        sum(r["trigram"]["recall@5"] for r in short_queries) / len(short_queries)
        if short_queries else 0.0
    )
    jieba_short_recall = (
        sum(r["jieba"]["recall@5"] for r in short_queries) / len(short_queries)
        if short_queries else 0.0
    )
    trig_long_recall = (
        sum(r["trigram"]["recall@5"] for r in long_queries) / len(long_queries)
        if long_queries else 0.0
    )
    jieba_long_recall = (
        sum(r["jieba"]["recall@5"] for r in long_queries) / len(long_queries)
        if long_queries else 0.0
    )

    # 结论
    if jieba_avg_recall > trig_avg_recall + 0.1:
        conclusion = (
            "jieba 预分词在整体召回率上显著优于 trigram"
            f"（{jieba_avg_recall:.2%} vs {trig_avg_recall:.2%}），"
            f"尤其在短查询（<=2 字符）场景 trigram 召回为 0，jieba 仍可命中。"
            f"**推荐：将 FTS5 tokenizer 改为 unicode61 + jieba 预分词。**"
        )
    elif jieba_avg_recall > trig_avg_recall:
        conclusion = (
            f"jieba 预分词在整体召回率上略优于 trigram"
            f"（{jieba_avg_recall:.2%} vs {trig_avg_recall:.2%}）。"
            f"若短查询场景重要，推荐 jieba；若主要处理 3+ 字符查询，trigram 可接受。"
        )
    else:
        conclusion = (
            f"trigram 与 jieba 召回率相当（{trig_avg_recall:.2%} vs {jieba_avg_recall:.2%}），"
            f"trigram 实现更简单（无需 jieba 依赖），可保留。"
        )

    lines = [
        "# 中文 FTS5 评测报告：trigram vs trigram+jieba",
        "",
        f"**评测时间**：{Path(__file__).stat().st_mtime if Path(__file__).exists() else 'N/A'}",
        f"**语料规模**：{len(CORPUS)} 篇中文徒步文档",
        f"**查询数量**：{len(QUERIES)} 个（含 {len(short_queries)} 个短查询 <=2 字符）",
        "",
        "## 1. 评测设计",
        "",
        "### 1.1 语料",
        "",
        "15 篇手写中文徒步文档，覆盖装备（背包/鞋袜/食物）、路线（西湖群山/高海拔）、",
        "安全（中暑/失温/山洪/迷路/落石）、天气（雨天/夏季/冬季）等主题。",
        "",
        "### 1.2 查询",
        "",
        "10 个查询，按长度分两组：",
        f"- 短查询（<=2 字符）：{len(short_queries)} 个，trigram 理论上无法匹配",
        f"- 长查询（3+ 字符）：{len(long_queries)} 个，trigram 可匹配",
        "",
        "### 1.3 索引策略",
        "",
        "| 策略 | tokenizer | 文本处理 |",
        "|------|-----------|----------|",
        "| A: trigram | `tokenize='trigram'` | 原文直接写入 |",
        "| B: jieba | `tokenize='unicode61'` | 写入前 jieba.cut 分词，空格连接 |",
        "",
        "### 1.4 指标",
        "",
        "- **recall@5**：前 5 条结果中相关文档占所有相关文档的比例",
        "- **precision@5**：前 5 条结果中相关文档的比例",
        "- **MRR**：第一个相关文档的倒数排名",
        "",
        "## 2. 整体结果",
        "",
        "| 指标 | trigram | jieba | 差值 |",
        "|------|---------|-------|------|",
        f"| recall@5 | {trig_avg_recall:.2%} | {jieba_avg_recall:.2%} | {jieba_avg_recall - trig_avg_recall:+.2%} |",
        f"| precision@5 | {trig_avg_precision:.2%} | {jieba_avg_precision:.2%} | {jieba_avg_precision - trig_avg_precision:+.2%} |",
        f"| MRR | {trig_avg_mrr:.4f} | {jieba_avg_mrr:.4f} | {jieba_avg_mrr - trig_avg_mrr:+.4f} |",
        "",
        "## 3. 按查询长度分组",
        "",
        "| 查询长度 | trigram recall@5 | jieba recall@5 |",
        "|----------|------------------|----------------|",
        f"| 短（<=2 字符，{len(short_queries)} 个） | {trig_short_recall:.2%} | {jieba_short_recall:.2%} |",
        f"| 长（3+ 字符，{len(long_queries)} 个） | {trig_long_recall:.2%} | {jieba_long_recall:.2%} |",
        "",
        "## 4. 逐查询详情",
        "",
        "| qid | 查询 | 长度 | 相关文档 | trigram 召回 | jieba 召回 | trigram 命中 | jieba 命中 |",
        "|-----|------|------|----------|-------------|-----------|-------------|-----------|",
    ]

    for r in results:
        trig_hits = [d for d in r["trigram"]["retrieved"] if d in r["relevant"]]
        jieba_hits = [d for d in r["jieba"]["retrieved"] if d in r["relevant"]]
        lines.append(
            f"| {r['qid']} | {r['query']} | {r['query_len']} | "
            f"{','.join(r['relevant'])} | "
            f"{r['trigram']['recall@5']:.2%} | {r['jieba']['recall@5']:.2%} | "
            f"{','.join(trig_hits) or '-'} | {','.join(jieba_hits) or '-'} |"
        )

    lines.extend([
        "",
        "## 5. 关键观察",
        "",
    ])

    # 关键观察
    short_trig_zero = [r for r in short_queries if r["trigram"]["recall@5"] == 0.0]
    if short_trig_zero:
        lines.append(
            f"- **trigram 对 2 字符查询全部失败**：{len(short_trig_zero)}/{len(short_queries)} "
            f"个短查询 recall=0（trigram 最小 3 字符才能生成 n-gram）"
        )
    short_jieba_success = [r for r in short_queries if r["jieba"]["recall@5"] > 0.0]
    if short_jieba_success:
        lines.append(
            f"- **jieba 对短查询有效**：{len(short_jieba_success)}/{len(short_queries)} "
            f"个短查询 recall>0"
        )
    long_compare = sum(1 for r in long_queries if r["jieba"]["recall@5"] > r["trigram"]["recall@5"])
    if long_compare > 0:
        lines.append(
            f"- **长查询 jieba 也优**：{long_compare}/{len(long_queries)} 个长查询 jieba 召回更高"
            f"（jieba 分词后能匹配语义单元，trigram 仅匹配 3-gram 子串）"
        )

    lines.extend([
        "",
        "## 6. 结论",
        "",
        conclusion,
        "",
        "## 7. 对 sqlite-storage-contract.md §6 的影响",
        "",
        "若结论为采用 jieba：",
        "- FTS5 schema 改为 `tokenize='unicode61'`",
        "- 写入路径（`rag/sqlite_store.py` 的 `upsert_document`）在插入 child_fts 前",
        "  对 content 做 jieba.cut 分词（空格连接）",
        "- 查询路径（`rag/hybrid_retrieval.py` 的 `_sparse_search`）对 query_text 做",
        "  jieba.cut 分词后再送入 FTS5 MATCH",
        "- 触发器 `child_ai` 需改为调用 jieba 分词后的内容（或改为应用层同步）",
        "",
        "若结论为保留 trigram：",
        "- 当前 schema 不变",
        "- 接受 2 字符查询无 sparse 命中的局限（由 dense 检索兜底）",
        "",
        "---",
        "",
        "本评测为 sqlite-storage-contract.md 关闭前置条件的兑现（§关闭前置条件）。",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入：{output_path}")


def main() -> int:
    eval_data = run_eval()

    # 控制台打印汇总
    print("\n=== 评测汇总 ===")
    print(f"{'qid':<6} {'query':<20} {'len':<4} {'trig_recall':<14} {'jieba_recall':<14} {'trig_retrieved':<20} {'jieba_retrieved':<20}")
    for r in eval_data["results"]:
        print(
            f"{r['qid']:<6} {r['query']:<20} {r['query_len']:<4} "
            f"{r['trigram']['recall@5']:<14.2%} {r['jieba']['recall@5']:<14.2%} "
            f"{','.join(r['trigram']['retrieved']):<20} {','.join(r['jieba']['retrieved']):<20}"
        )

    n = len(eval_data["results"])
    trig_avg = sum(r["trigram"]["recall@5"] for r in eval_data["results"]) / n
    jieba_avg = sum(r["jieba"]["recall@5"] for r in eval_data["results"]) / n
    print(f"\n整体 recall@5: trigram={trig_avg:.2%}  jieba={jieba_avg:.2%}")

    # 写报告
    report_path = Path(__file__).resolve().parent.parent / ".scratch" / "rag-lifecycle-refactor" / "eval" / "fts5-chinese-eval.md"
    write_report(eval_data, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
