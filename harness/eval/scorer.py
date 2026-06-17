"""评测评分引擎，驱动 Agent 执行评测并聚合指标。"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from typing import Any

from harness.agent.contract_agent import ContractAgent
from harness.agent.memory import MemoryStore
from harness.core.types import EvalResult, ExpectedMetrics
from harness.eval.dataset import EvalDataset
from harness.eval.metrics import MetricsCalculator
from harness.utils.log import logger


class EvalScorer:
    """评测评分器，驱动 Agent 对数据集逐项审查并计算指标。"""

    def __init__(
        self,
        agent: ContractAgent | None = None,
        metrics_calc: MetricsCalculator | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self._agent = agent or ContractAgent()
        self._metrics = metrics_calc or MetricsCalculator()
        self._memory = memory_store

    def run(self, dataset: EvalDataset, learn: bool = False) -> list[EvalResult]:
        """对数据集逐项运行审查，返回各评测结果。

        Args:
            dataset: 评测数据集。
            learn: 若为 True, 将评测期望结果作为修正信号存入记忆（自演进）。
        """
        logger.info("Starting scoring run on {} items (learn={})", len(dataset.items), learn)
        results: list[EvalResult] = []
        for item in dataset.items:
            report, _ = self._agent.review(item.document)

            expected: ExpectedMetrics = {
                "clauses": item.expected_clauses,
                "risks": item.expected_risks,
                "compliance": item.expected_compliance,
                "overall_risk": item.expected_risk_level.value,
            }

            metrics = self._metrics.calculate(report, expected)

            # 自演进：将评测期望作为修正信号存入记忆
            if learn and self._memory and self._memory.enabled:
                self._feed_corrections(report, item)

            results.append(
                EvalResult(
                    dataset_name=item.document.title,
                    agent_version=_pkg_version("contract-harness"),
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

        logger.info("Scoring completed: {} results", len(results))
        return results

    def _feed_corrections(self, report: Any, item: Any) -> None:
        """根据评测期望结果，将修正信号注入记忆。"""
        if not self._memory:
            return
        expected_risks = item.expected_risks or []
        expected_compliance = item.expected_compliance or []
        for i, clause in enumerate(report.clauses):
            if i < len(expected_risks):
                er = expected_risks[i]
                actual_risk = report.risks[i].risk_level.value if i < len(report.risks) else ""
                expected_risk = er.get("risk_level", "")
                if actual_risk != expected_risk:
                    self._memory.correct(
                        clause_type=clause.clause_type,
                        clause_content=clause.content,
                        field="risk_level",
                        correct_value=expected_risk,
                    )
            if i < len(expected_compliance):
                ec = expected_compliance[i]
                for check in ec:
                    regulation = check.get("regulation", "")
                    expected_status = check.get("status", True)
                    actual = next(
                        (c for c in report.compliance_checks if c.regulation == regulation),
                        None,
                    )
                    if actual and actual.status != expected_status:
                        self._memory.correct(
                            clause_type=clause.clause_type,
                            clause_content=clause.content,
                            field=f"compliance:{regulation}",
                            correct_value="合规" if expected_status else "不合规",
                        )

    def score(self, dataset: EvalDataset) -> dict[str, Any]:
        """运行评测并返回聚合后的评分结果。"""
        logger.debug("Scoring dataset with {} items", len(dataset.items))
        results = self.run(dataset)
        aggregated = self._metrics.aggregate(results)

        result = {
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
        logger.debug(
            "Score result: total_items={}, metrics={}",
            result["total_items"],
            result["aggregated_metrics"],
        )
        return result
