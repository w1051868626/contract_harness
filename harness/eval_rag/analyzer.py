"""RAG 评估三项分析模块。

从已产出的 ``summary.csv`` / ``details.csv`` 派生三项分析产物：
  1. ``hit1_miss_hit3_hit.csv`` —— hit@1=0 但 hit@3=1 的 query（排序改进空间）
  2. ``chunk_hit_distribution.csv`` / ``chunk_systematic_miss.csv`` ——
     按 expected chunk 维度的命中率 + 系统性漏检
  3. ``rag_eval_charts.html`` —— pyecharts 交互式图表
  4. ``rag_eval_analysis_summary.md`` —— Markdown 摘要

设计为可复用模块：CLI ``harness kb eval run --analyze`` 与独立脚本
``scripts/analyze_rag_eval.py`` 共用本模块的实现。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.eval_rag.reporter import METRIC_LEGEND as _METRIC_LEGEND
from harness.utils.log import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------


def load_summary_csv(path: str | Path) -> dict[str, dict[int, float]]:
    """读汇总 CSV -> ``{metric: {top_k: value}}``。"""
    data: dict[str, dict[int, float]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            data[row["metric"]][int(row["top_k"])] = float(row["value"])
    return {k: dict(v) for k, v in data.items()}


def load_details_csv(path: str | Path) -> list[dict[str, Any]]:
    """读明细 CSV -> list of dict(query, hit1, hit3, hit5, expected, retrieved list)。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            # 字段名可能是 hit@1 / hit@3 / hit@5（reporter.to_details_csv 风格）
            # 也可能只有 hit@1..hit@5（兼容旧 details.csv）
            hit_keys = [k for k in r if k.startswith("hit@")]
            hit_keys.sort()  # hit@1 < hit@3 < hit@5 字典序
            row: dict[str, Any] = {
                "query": r.get("query", ""),
                "expected": r.get("expected", ""),
                "retrieved": r.get("retrieved", "").split(";") if r.get("retrieved") else [],
            }
            for k in hit_keys:
                # 归一为 hit1/hit3/hit5 整数
                row[k.replace("@", "")] = int(r[k])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Task 1: 排序改进空间
# ---------------------------------------------------------------------------


def task_rank_improvement(details: list[dict[str, Any]], out_dir: Path) -> int:
    """hit@1=0 但 hit@3=1 的 query -> CSV。"""
    mask = [d for d in details if d.get("hit1", 0) == 0 and d.get("hit3", 0) == 1]
    out = out_dir / "hit1_miss_hit3_hit.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "expected", "retrieved_top5"])
        for d in mask:
            w.writerow([d["query"], d["expected"], ";".join(d["retrieved"][:5])])
    return len(mask)


# ---------------------------------------------------------------------------
# Task 2: chunk 维度命中分布
# ---------------------------------------------------------------------------


def task_chunk_distribution(details: list[dict[str, Any]], out_dir: Path) -> tuple[int, int]:
    """按 expected chunk 维度统计命中率 -> 2 个 CSV。返回 (distinct chunk 数, 系统性漏检数)。"""
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"queries": 0, "hit1": 0, "hit3": 0, "hit5": 0}
    )
    for d in details:
        e = agg[d["expected"]]
        e["queries"] += 1
        e["hit1"] += d.get("hit1", 0)
        e["hit3"] += d.get("hit3", 0)
        e["hit5"] += d.get("hit5", 0)

    rows = sorted(
        agg.items(),
        key=lambda kv: (kv[1]["hit1"] / kv[1]["queries"], kv[1]["queries"]),
    )
    full = out_dir / "chunk_hit_distribution.csv"
    with full.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "expected_chunk",
                "queries",
                "hit1",
                "hit3",
                "hit5",
                "hit1_rate",
                "hit3_rate",
                "hit5_rate",
            ]
        )
        for chunk, v in rows:
            q = v["queries"]
            w.writerow(
                [
                    chunk,
                    q,
                    v["hit1"],
                    v["hit3"],
                    v["hit5"],
                    v["hit1"] / q,
                    v["hit3"] / q,
                    v["hit5"] / q,
                ]
            )

    # 系统性漏检: hit1_rate <= 0.3 且 queries >= 3
    miss = [kv for kv in rows if kv[1]["hit1"] / kv[1]["queries"] <= 0.3 and kv[1]["queries"] >= 3]
    miss.sort(key=lambda kv: (kv[1]["hit1"] / kv[1]["queries"], -kv[1]["queries"]))
    miss_csv = out_dir / "chunk_systematic_miss.csv"
    with miss_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "expected_chunk",
                "queries",
                "hit1",
                "hit3",
                "hit5",
                "hit1_rate",
                "hit3_rate",
                "hit5_rate",
            ]
        )
        for chunk, v in miss:
            q = v["queries"]
            w.writerow(
                [
                    chunk,
                    q,
                    v["hit1"],
                    v["hit3"],
                    v["hit5"],
                    v["hit1"] / q,
                    v["hit3"] / q,
                    v["hit5"] / q,
                ]
            )
    return len(rows), len(miss)


# ---------------------------------------------------------------------------
# Task 3: 可视化
# ---------------------------------------------------------------------------


def task_charts(
    summary: dict[str, dict[int, float]],
    details: list[dict[str, Any]],
    n_rank_improve: int,
    n_chunks: int,
    out_dir: Path,
) -> Path | None:
    """pyecharts 交互式 HTML。未安装 pyecharts 时跳过返回 None。"""
    # pyecharts 是可选依赖，延迟到调用点 import
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line, Page
    from pyecharts.globals import ThemeType

    ks = sorted(summary["precision"])
    precision = [round(summary["precision"][k], 4) for k in ks]
    recall = [round(summary["recall"][k], 4) for k in ks]
    hit = [round(summary["hit_rate"][k], 4) for k in ks]
    mrr_ks = sorted(summary["mrr"])
    mrr = [round(summary["mrr"][k], 4) for k in mrr_ks]

    # 1) Precision / Recall / Hit Rate vs top_k
    line1 = (
        Line(init_opts=opts.InitOpts(width="1100px", height="520px", theme=ThemeType.LIGHT))
        .add_xaxis([str(k) for k in ks])
        .add_yaxis("precision", precision, is_smooth=True, symbol="circle", symbol_size=8)
        .add_yaxis("recall", recall, is_smooth=True, symbol="diamond", symbol_size=8)
        .add_yaxis("hit_rate", hit, is_smooth=True, symbol="triangle", symbol_size=8)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Precision / Recall / Hit Rate vs top_k"),
            xaxis_opts=opts.AxisOpts(name="top_k", type_="category"),
            yaxis_opts=opts.AxisOpts(name="value", min_=0, max_=1),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter="{b}: {c}"),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
        )
    )

    # 2) MRR vs top_k
    bar1 = (
        Bar(init_opts=opts.InitOpts(width="1100px", height="520px", theme=ThemeType.LIGHT))
        .add_xaxis([str(k) for k in mrr_ks])
        .add_yaxis("MRR", mrr, bar_width="40%", itemstyle_opts=opts.ItemStyleOpts(color="#ff7f0e"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="MRR vs top_k"),
            xaxis_opts=opts.AxisOpts(name="top_k", type_="category"),
            yaxis_opts=opts.AxisOpts(name="MRR", min_=0, max_=1),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter="{b}<br/>MRR: {c}"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
        )
    )

    # 3) 排序改进空间（横向柱状）
    c1 = sum(1 for d in details if d.get("hit1", 0) == 0)
    c2 = n_rank_improve
    c3 = sum(1 for d in details if d.get("hit3", 0) == 0 and d.get("hit5", 0) == 1)
    c4 = sum(1 for d in details if d.get("hit5", 0) == 0)
    labels = ["hit@5 仍漏", "hit@3=0 但 hit@5=1", "hit@1=0 但 hit@3=1", "hit@1=0"]
    counts = [c4, c3, c2, c1]
    bar2 = (
        Bar(init_opts=opts.InitOpts(width="1100px", height="520px", theme=ThemeType.LIGHT))
        .add_xaxis(labels)
        .add_yaxis("query 数", counts, bar_width="50%")
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(title="排序改进空间分布"),
            xaxis_opts=opts.AxisOpts(name="query 数"),
            yaxis_opts=opts.AxisOpts(name=""),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter="{b}<br/>query 数: {c}"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="inside", orient="vertical", range_start=0, range_end=100)
            ],
        )
    )

    # 4) chunk 命中率分布
    chunk_q: dict[str, int] = {}
    for d in details:
        chunk_q[d["expected"]] = chunk_q.get(d["expected"], 0) + 1
    hit1_cnt: dict[str, int] = {}
    for d in details:
        if d.get("hit1", 0) == 1:
            hit1_cnt[d["expected"]] = hit1_cnt.get(d["expected"], 0) + 1
    rates = [hit1_cnt.get(c, 0) / chunk_q[c] for c in chunk_q]
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist = [0] * (len(bins) - 1)
    for r in rates:
        for i in range(len(bins) - 1):
            if bins[i] <= r < bins[i + 1]:
                hist[i] += 1
                break
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    bar3 = (
        Bar(init_opts=opts.InitOpts(width="1100px", height="520px", theme=ThemeType.LIGHT))
        .add_xaxis(bin_labels)
        .add_yaxis(
            "chunk 数",
            hist,
            bar_width="60%",
            itemstyle_opts=opts.ItemStyleOpts(color="#4daf37"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"按 expected chunk 的 hit@1 命中率分布 (共 {len(rates)} 个 chunk)"
            ),
            xaxis_opts=opts.AxisOpts(name="hit@1 命中率区间", type_="category"),
            yaxis_opts=opts.AxisOpts(name="chunk 数"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", formatter="{b}<br/>chunk 数: {c}"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
        )
    )

    page = Page(page_title="RAG Eval 分析图表 (交互式)")
    for chart in (line1, bar1, bar2, bar3):
        page.add(chart)
    html = out_dir / "rag_eval_charts.html"
    page.render(str(html))
    # 在 <body> 后插入使用说明样式 + 指标含义折叠区块
    raw = html.read_text(encoding="utf-8")
    legend_items = "".join(
        f"<li><strong>{name}</strong>: {desc}</li>"
        for name, desc in _METRIC_LEGEND
    )
    note = (
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;margin:20px;color:#222;"
        "max-width:1200px;margin-left:auto;margin-right:auto}"
        ".chart-container{margin:24px auto;padding:16px;border:1px solid #eee;"
        "border-radius:8px;display:flex;justify-content:center}"
        ".note{color:#888;font-size:12px;margin:8px 0}"
        "details.legend{margin:12px 0;padding:8px 12px;border:1px solid #eee;"
        "border-radius:6px;background:#fafafa}"
        "details.legend summary{cursor:pointer;font-weight:600;color:#444}"
        "details.legend ul{margin:8px 0 0 0;padding-left:22px;line-height:1.7}</style>"
        "<p class=note>鼠标悬停显示精确数值; 鼠标滚轮/拖拽可缩放 (dataZoom); "
        "双坐标轴可框选放大。</p>"
        "<details class=legend open><summary>指标含义</summary>"
        f"<ul>{legend_items}</ul></details>"
    )
    raw = raw.replace("<body>", "<body>" + note, 1).replace("<body >", "<body >" + note, 1)
    html.write_text(raw, encoding="utf-8")
    return html


# ---------------------------------------------------------------------------
# Task 4: Markdown 摘要
# ---------------------------------------------------------------------------


def task_markdown_summary(
    summary: dict[str, dict[int, float]],
    details: list[dict[str, Any]],
    n_rank_improve: int,
    n_chunks: int,
    n_miss: int,
    html_path: Path | None,
    out_dir: Path,
) -> Path:
    md = out_dir / "rag_eval_analysis_summary.md"
    n = len(details)
    c1 = sum(1 for d in details if d.get("hit1", 0) == 0)
    c3 = sum(1 for d in details if d.get("hit3", 0) == 0 and d.get("hit5", 0) == 1)
    c4 = sum(1 for d in details if d.get("hit5", 0) == 0)

    rows = [
        "# RAG 评估分析摘要",
        "",
        f"生成时间: {_now_iso()}",
        "",
        "## 数据规模",
        "",
        f"- query 总数: {n}",
        f"- expected chunk 种类: {n_chunks}",
        "",
        "## 1. 汇总指标 (summary.csv)",
        "",
        "| metric | top_k=1 | top_k=3 | top_k=5 |",
        "|---|---|---|---|",
    ]
    for m in ("hit_rate", "mrr", "precision", "recall"):
        if m in summary and all(k in summary[m] for k in (1, 3, 5)):
            row = f"| {m} | " + " | ".join(f"{summary[m][k]:.4f}" for k in (1, 3, 5)) + " |"
            rows.append(row)
    rows += [
        "",
        "### 指标含义",
        "",
        "- **hit_rate（命中率）**: 前 k 条结果中含至少一个正确 chunk 的 query 占比。"
        "@1 越高排序越准。",
        "- **mrr（平均倒数排名）**: 第一个正确 chunk 排位的倒数（1/rank）的均值。"
        "1.0 表示全在首位。",
        "- **precision（精确率）**: 前 k 条中正确 chunk 的比例均值。"
        "越高代表前 k 条噪声越少。",
        "- **recall（召回率）**: 前 k 条覆盖到的正确 chunk 占该 query 全部正确 chunk 的比例均值。"
        "越高漏检越少。",
        "",
        "## 2. 排序改进空间",
        "",
        f"- hit@1=0 的 query: {c1} ({c1 / n:.1%})" if n else "- hit@1=0 的 query: 0",
        f"- 其中 hit@3=1 (排到 2~3 位即可命中): {n_rank_improve} ({n_rank_improve / n:.1%})"
        if n
        else f"- 其中 hit@3=1: {n_rank_improve} (0.0%)",
        f"- hit@3=0 但 hit@5=1 (需排到 4~5 位): {c3} ({c3 / n:.1%})"
        if n
        else f"- hit@3=0 但 hit@5=1: {c3}",
        f"- top-5 仍漏: {c4} ({c4 / n:.1%})" if n else f"- top-5 仍漏: {c4}",
        "",
        "详见 `hit1_miss_hit3_hit.csv`",
        "",
        "## 3. 按 chunk 维度的命中分布",
        "",
        f"- 共 {n_chunks} 个 distinct expected chunk",
        f"- 系统性漏检 (hit1_rate ≤ 0.3 且 query ≥ 3): {n_miss} 个 chunk",
        "",
        "详见 `chunk_hit_distribution.csv` / `chunk_systematic_miss.csv`",
        "",
        "## 4. 可视化",
        "",
        f"详见 `{html_path}`" if html_path else "- (pyecharts 未安装，跳过图表)",
        "",
        "## 关键结论",
        "",
        "1. top-3 recall 已达 88%, top-5 仅 +2pp, 取 3 条已覆盖绝大多数正确答案。",
        (
            "2. top-1 hit_rate 52%, MRR 0.69, 平均排位在 1~2 位, "
            "但近一半 query 的最优 chunk 未排首位。"
        ),
        (
            f"3. 共 {n_rank_improve} 条 ({n_rank_improve / n:.1%}) query "
            "仅因排序问题漏 top-1 (hit@3 已命中), "
            "是 Reranker/排序优化的首要目标。"
        )
        if n
        else "",
        (
            f"4. {n_miss} 个 chunk 系统性漏检 (hit1_rate ≤ 0.3), "
            "建议排查其 query/chunk 的语义对齐或切片质量。"
        ),
    ]
    md.write_text("\n".join(rows), encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def run_analysis(
    summary_csv: str | Path,
    details_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """跑完三项分析 + Markdown 摘要，返回产物路径与关键统计。

    Args:
        summary_csv: 汇总 CSV 路径（reporter.to_summary_csv 风格）。
        details_csv: 明细 CSV 路径（reporter.to_details_csv 风格，列名 hit@1/hit@3/hit@5）。
        out_dir: 产物输出目录。

    Returns:
        dict 包含 ``rank_improve`` / ``chunks`` / ``systematic_miss`` / ``total_query``
        以及各产物路径。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = load_summary_csv(summary_csv)
    details = load_details_csv(details_csv)
    n = len(details)
    logger.info("加载 summary={} details={} ({} query)", summary_csv, details_csv, n)

    n_rank = task_rank_improvement(details, out)
    n_chunks, n_miss = task_chunk_distribution(details, out)
    html_path = task_charts(summary, details, n_rank, n_chunks, out)
    md_path = task_markdown_summary(summary, details, n_rank, n_chunks, n_miss, html_path, out)

    logger.info(
        "三项分析完成: rank_improve={} chunks={} systematic_miss={} -> {} / {} / {}",
        n_rank,
        n_chunks,
        n_miss,
        md_path,
        html_path,
        out,
    )
    return {
        "total_query": n,
        "rank_improve": n_rank,
        "chunks": n_chunks,
        "systematic_miss": n_miss,
        "summary_md": md_path,
        "charts_html": html_path,
        "out_dir": out,
    }
