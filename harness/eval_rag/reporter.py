from __future__ import annotations

import json

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
            return f"| {name} | " + " | ".join(fmt(source[k]) for k in result.top_ks) + " |"

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
