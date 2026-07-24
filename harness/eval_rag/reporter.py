from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from harness.eval_rag.dataset import EvalRagResult


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
        """单文件 CSV：先汇总区块，再明细区块，用 section 行分隔。"""
        parts = [
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
        """
        把 RAG 评估结果写成 CSV。

        Args:
            result: 评估结果。
            path: 输出路径。
                - split=False：写单个 CSV（汇总+明细分块）。
                - split=True：path 视为前缀，写出 <path>.summary.csv 与 <path>.details.csv。
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if split:
            summary_path = out.with_suffix(".summary.csv")
            details_path = out.with_suffix(".details.csv")
            summary_path.write_text(self.to_summary_csv(result), encoding="utf-8-sig", newline="")
            details_path.write_text(self.to_details_csv(result), encoding="utf-8-sig", newline="")
            return [summary_path, details_path]
        out.write_text(self.to_csv(result), encoding="utf-8-sig", newline="")
        return [out]
