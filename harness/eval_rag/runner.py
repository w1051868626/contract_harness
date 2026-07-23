from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.eval_rag.dataset import EvalRagItem, EvalRagResult
from harness.eval_rag.metrics import RagMetricsCalculator
from harness.rag.knowledge_base import KnowledgeBase
from harness.utils.log import logger


class RagEvalRunner:
    def __init__(self):
        self._calculator = RagMetricsCalculator()

    def run(
        self,
        kb: KnowledgeBase,
        items: list[EvalRagItem],
        top_ks: list[int] | None = None,
        dataset_name: str = "default",
        expansion_threshold: float = 0.6,
        checkpoint_path: str | None = None,
    ) -> EvalRagResult:
        if top_ks is None:
            top_ks = [1, 3, 5]
        max_k = max(top_ks)
        logger.info("Running RAG eval on {} items with top_k={}", len(items), top_ks)

        # 断点恢复：加载 checkpoint 中已完成的 query→retrieved 映射
        raw_items: list[dict[str, Any]] = []
        completed: dict[str, list[str]] = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    completed[rec["query"]] = rec["retrieved"]
            if completed:
                logger.info("断点恢复：检测到 {} 个已完成 query", len(completed))

        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
        try:
            for item in items:
                if item.query in completed:
                    retrieved = completed[item.query]
                else:
                    results = kb.query(
                        item.query, top_k=max_k, expansion_threshold=expansion_threshold
                    )
                    retrieved = [r.id for r in results]
                    # 增量写盘，中断后可续跑
                    if checkpoint_file:
                        checkpoint_file.write(
                            json.dumps(
                                {"query": item.query, "retrieved": retrieved},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        checkpoint_file.flush()
                raw_items.append(
                    {
                        "expected": item.expected_chunk_ids,
                        "retrieved": retrieved,
                        "query": item.query,
                    }
                )
        finally:
            if checkpoint_file:
                checkpoint_file.close()

        result = self._calculator.compute(raw_items, top_ks, dataset_name=dataset_name)
        logger.info("RAG eval complete: hit_rate@1={}", result.hit_rates.get(1))
        return result
