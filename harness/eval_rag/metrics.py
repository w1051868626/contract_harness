from __future__ import annotations

from typing import Any

from harness.eval_rag.dataset import EvalRagResult


class RagMetricsCalculator:
    def compute(
        self,
        items: list[dict[str, Any]],
        top_ks: list[int],
        dataset_name: str = "default",
    ) -> EvalRagResult:
        n = len(items)
        max_k = max(top_ks)
        hit_counts = {k: 0 for k in top_ks}
        mrr_sums = {k: 0.0 for k in top_ks}
        precision_sums = {k: 0.0 for k in top_ks}
        recall_sums = {k: 0.0 for k in top_ks}
        details: list[dict[str, Any]] = []

        for item in items:
            expected = set(item["expected"])
            retrieved = item["retrieved"][:max_k]
            detail = {
                "expected": list(expected),
                "retrieved": retrieved,
                "query": item.get("query", ""),
            }

            for k in top_ks:
                top_k = retrieved[:k]
                hits = [c for c in top_k if c in expected]
                hit = len(hits) > 0
                if hit:
                    hit_counts[k] += 1
                detail[f"hit@{k}"] = hit

                if top_k:
                    precision_sums[k] += len(hits) / len(top_k)

                for rank, c in enumerate(top_k, 1):
                    if c in expected:
                        mrr_sums[k] += 1.0 / rank
                        break

                if expected:
                    recall_sums[k] += len(hits) / len(expected)

            details.append(detail)

        return EvalRagResult(
            dataset_name=dataset_name,
            top_ks=top_ks,
            hit_rates={k: hit_counts[k] / n if n else 0 for k in top_ks},
            mrr={k: mrr_sums[k] / n if n else 0 for k in top_ks},
            precisions={k: round(precision_sums[k] / n, 4) if n else 0 for k in top_ks},
            recalls={k: round(recall_sums[k] / n, 4) if n else 0 for k in top_ks},
            details=details,
        )
