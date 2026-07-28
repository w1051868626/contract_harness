from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from harness.eval_rag.dataset import EvalRagResult

# 指标含义说明（CSV 注释行 / 配套 .md / HTML 区块共用）
METRIC_LEGEND: list[tuple[str, str]] = [
    (
        "hit_rate",
        "前 k 条结果中含至少一个正确 chunk 的 query 占比。@1 越高排序越准。",
    ),
    (
        "mrr",
        "第一个正确 chunk 排位的倒数（1/rank）的均值。1.0 表示全在首位。",
    ),
    (
        "precision",
        "前 k 条结果中正确 chunk 的比例均值。越高代表前 k 条噪声越少。",
    ),
    (
        "recall",
        "前 k 条覆盖到的正确 chunk 占该 query 全部正确 chunk 的比例均值。"
        "越高漏检越少。",
    ),
]


def metric_legend_lines() -> list[str]:
    """返回 CSV 注释风格的指标含义行列表。"""
    lines = ["# metrics_legend", "# metric,definition"]
    for name, desc in METRIC_LEGEND:
        lines.append(f"# {name},{desc}")
    return lines


def metric_legend_markdown() -> str:
    """返回 Markdown 风格的指标含义段落。"""
    lines = ["# RAG 评估指标含义", ""]
    for name, desc in METRIC_LEGEND:
        lines.append(f"- **{name}**: {desc}")
    lines.append("")
    return "\n".join(lines)


class RagEvalReporter:
    def to_markdown(self, result: EvalRagResult) -> str:
        lines = [f"# RAG 评估报告: {result.dataset_name}", ""]
        ks_str = " | ".join(f"@{k}" for k in result.top_ks)
        lines.append("| 指标 | " + ks_str + " |")
        lines.append("|---" * (len(result.top_ks) + 1) + "|")

        def fmt(val: float) -> str:
            return f"{val:.2%}"

        def row(name: str, source: dict[int, float]) -> str:
            cols = " | ".join(fmt(source.get(k, 0.0)) for k in result.top_ks)
            return f"| {name} | {cols} |"

        lines.append(row("Hit Rate", result.hit_rates))
        lines.append(row("MRR", result.mrr))
        lines.append(row("Precision", result.precisions))
        lines.append(row("Recall", result.recalls))
        lines.append("")
        lines.append("## 指标含义")
        lines.append("")
        lines.append(
            "- **Hit Rate（命中率）**: 前 k 条结果中包含至少一个正确 chunk 的 query 占比。"
            "@1 越高代表排序越准。"
        )
        lines.append(
            "- **MRR（平均倒数排名）**: 第一个正确 chunk 排位的倒数（1/rank）的均值。"
            "MRR@1=1.0 表示正确 chunk 全在首位。"
        )
        lines.append(
            "- **Precision（精确率）**: 前 k 条结果中正确 chunk 的比例均值。"
            "越高代表前 k 条噪声越少。"
        )
        lines.append(
            "- **Recall（召回率）**: 前 k 条覆盖到的正确 chunk "
            "占该 query 全部正确 chunk 的比例均值。越高代表漏检越少。"
        )
        lines.append("")
        lines.append(f"共 {len(result.details)} 条评估项")
        return "\n".join(lines)

    def to_json(self, result: EvalRagResult) -> str:
        return json.dumps(
            {
                "dataset_name": result.dataset_name,
                "top_ks": result.top_ks,
                "hit_rates": result.hit_rates,
                "mrr": result.mrr,
                "precisions": result.precisions,
                "recalls": result.recalls,
                "item_count": len(result.details),
            },
            ensure_ascii=False,
            indent=2,
        )

    def to_summary_csv(self, result: EvalRagResult) -> str:
        """汇总指标 CSV：每行一个指标，列为 dataset/top_k/value。"""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["dataset", "metric", "top_k", "value"])
        metric_map = [
            ("hit_rate", result.hit_rates),
            ("mrr", result.mrr),
            ("precision", result.precisions),
            ("recall", result.recalls),
        ]
        for metric_name, source in metric_map:
            for k in result.top_ks:
                writer.writerow([result.dataset_name, metric_name, k, source.get(k, 0.0)])
        return buf.getvalue()

    def to_details_csv(self, result: EvalRagResult) -> str:
        """逐 query 明细 CSV：列为 query/hit@k，并附 expected/retrieved。"""
        buf = io.StringIO()
        writer = csv.writer(buf)
        header = ["query"] + [f"hit@{k}" for k in result.top_ks] + ["expected", "retrieved"]
        writer.writerow(header)
        for detail in result.details:
            row = [detail.get("query", "")]
            for k in result.top_ks:
                row.append("1" if detail.get(f"hit@{k}") else "0")
            expected = detail.get("expected", [])
            retrieved = detail.get("retrieved", [])
            row.append(";".join(expected))
            row.append(";".join(retrieved))
            writer.writerow(row)
        return buf.getvalue()

    def to_csv(self, result: EvalRagResult) -> str:
        """单文件 CSV：先指标含义注释，再汇总区块，再明细区块，用 section 行分隔。"""
        parts = [
            *metric_legend_lines(),
            "# summary",
            self.to_summary_csv(result).rstrip("\n"),
            "",
            "# details",
            self.to_details_csv(result).rstrip("\n"),
        ]
        return "\n".join(parts) + "\n"

    def write_csv(
        self,
        result: EvalRagResult,
        path: str | Path,
        split: bool = False,
    ) -> list[Path]:
        """把 RAG 评估结果写成 CSV。返回写出的文件列表（含配套指标含义说明）。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

        # 配套指标含义说明文件（与 CSV 同目录、同前缀）
        legend_path = out.with_suffix(".metrics_legend.md")
        legend_path.write_text(metric_legend_markdown(), encoding="utf-8")

        if split:
            # 拆分为汇总 + 明细两个文件
            summary_path = out.with_suffix(".summary.csv")
            details_path = out.with_suffix(".details.csv")
            summary_path.write_text(self.to_summary_csv(result), encoding="utf-8-sig", newline="")
            details_path.write_text(self.to_details_csv(result), encoding="utf-8-sig", newline="")
            return [summary_path, details_path, legend_path]
        # 合并为单个 CSV（先指标含义注释，再汇总区块，再明细区块）
        out.write_text(self.to_csv(result), encoding="utf-8-sig", newline="")
        return [out, legend_path]
