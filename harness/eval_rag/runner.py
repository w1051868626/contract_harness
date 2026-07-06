from __future__ import annotations

from typing import Any

from harness.eval_rag.dataset import EvalRagItem, EvalRagResult
from harness.eval_rag.metrics import RagMetricsCalculator
from harness.utils.log import logger


class RagEvalRunner:
    def __init__(self):
        self._calculator = RagMetricsCalculator()

    def run(
        self,
        kb: Any,
        items: list[EvalRagItem],
        top_ks: list[int] | None = None,
        dataset_name: str = "default",
        expansion_threshold: float = 0.6,
    ) -> EvalRagResult:
        if top_ks is None:
            top_ks = [1, 3, 5]
        max_k = max(top_ks)
        logger.info("Running RAG eval on {} items with top_k={}", len(items), top_ks)

        raw_items: list[dict[str, Any]] = []
        for item in items:
            results = kb.query(item.query, top_k=max_k, expansion_threshold=expansion_threshold)
            retrieved = [r.id for r in results]
            raw_items.append(
                {
                    "expected": item.expected_chunk_ids,
                    "retrieved": retrieved,
                    "query": item.query,
                }
            )

        result = self._calculator.compute(raw_items, top_ks, dataset_name=dataset_name)
        logger.info("RAG eval complete: hit_rate@1={}", result.hit_rates.get(1))
        return result
