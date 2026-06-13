from __future__ import annotations

"""评测评分引擎，驱动 Agent 执行评测并聚合指标。"""

from datetime import datetime, timezone
from typing import Any

from harness.agent.contract_agent import ContractAgent
from harness.core.types import EvalResult
from harness.eval.dataset import EvalDataset
from harness.eval.metrics import MetricsCalculator


class EvalScorer:
    """评测评分器，驱动 Agent 对数据集逐项审查并计算指标。"""
    def __init__(
        self,
        agent: ContractAgent | None = None,
        metrics_calc: MetricsCalculator | None = None,
    ):
        self._agent = agent or ContractAgent()
        self._metrics = metrics_calc or MetricsCalculator()

    def run(self, dataset: EvalDataset) -> list[EvalResult]:
        """对数据集逐项运行审查，返回各评测结果。"""
        results: list[EvalResult] = []
        for item in dataset.items:
            report, _ = self._agent.review(item.document)

            expected = {
                "clauses": item.expected_clauses,
                "risks": item.expected_risks,
                "compliance": item.expected_compliance,
                "overall_risk": item.expected_risk_level.value,
            }

            metrics = self._metrics.calculate(report, expected)

            results.append(
                EvalResult(
                    dataset_name=item.document.title,
                    agent_version="0.1.0",
                    metrics=metrics,
                    detailed_results=[
                        {
                            "document_id": item.document.id,
                            "metrics": {m.name: m.value for m in metrics},
                        }
                    ],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        return results

    def score(self, dataset: EvalDataset) -> dict[str, Any]:
        """运行评测并返回聚合后的评分结果。"""
        results = self.run(dataset)
        aggregated = self._metrics.aggregate(results)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": len(dataset.items),
            "aggregated_metrics": aggregated,
            "per_item_results": [
                {
                    "document_id": r.dataset_name,
                    "metrics": {m.name: m.value for m in r.metrics},
                }
                for r in results
            ],
        }
