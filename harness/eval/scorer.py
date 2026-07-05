"""评测评分引擎，驱动 Agent 执行评测并聚合指标。"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import Any

from harness.agent.contract_agent import ContractAgent
from harness.agent.memory import MemoryStore
from harness.core.types import EvalResult, ExpectedMetrics
from harness.eval.dataset import EvalDataset
from harness.eval.metrics import MetricsCalculator
from harness.utils.io import now_iso
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
                    timestamp=now_iso(),
                )
            )

        logger.info("Scoring completed: {} results", len(results))
        return results

    def _feed_corrections(self, report: Any, item: Any) -> None:
        """根据评测期望结果，将修正信号注入记忆。

        匹配策略：expected_risks / expected_compliance 优先用 ``clause_index``
        对齐到 ``report.clauses``；未提供 index 时回退到按 ``clause_type``
        模糊匹配。避免用 ``enumerate(report.clauses)`` 下标强行对齐导致
        Agent 提取顺序/数量与期望不一致时修正信号错位注入。
        """
        if not self._memory:
            return
        expected_risks = item.expected_risks or []
        expected_compliance = item.expected_compliance or []
        clauses = report.clauses or []

        # clause_type -> [该 type 下所有 clause index]
        # 多个同 type 条款时轮转消费，避免全部对齐到第一个导致修正信号错位。
        # 注意：risks 和 compliance 各自维护独立的消费游标，否则 risks 循环
        # pop 掉的 index 会让 compliance 循环对齐失败（同 type 多条款场景）。
        type_to_indices: dict[str, list[int]] = {}
        for idx, clause in enumerate(clauses):
            type_to_indices.setdefault(clause.clause_type, []).append(idx)
        # 每个 field 维护独立的"已消费计数"，按 type 轮转取 index
        consumed: dict[str, dict[str, int]] = {"risk": {}, "compliance": {}}

        def _resolve_clause_index(er: dict, field: str) -> int | None:
            """从 expected 项解析对应的 clause index，找不到返回 None。

            优先用显式 ``clause_index``；缺失时按 ``clause_type``/``type``
            从同 type 队列里按 ``field`` 独立游标轮转取 index，保证多个同
            type 条款不会全部对齐到同一个，且 risks/compliance 互不抢占。
            """
            ci = er.get("clause_index")
            if isinstance(ci, int) and 0 <= ci < len(clauses):
                return ci
            key = er.get("clause_type") or er.get("type")
            if not key:
                return None
            indices = type_to_indices.get(key)
            if not indices:
                return None
            pos = consumed[field].get(key, 0)
            if pos >= len(indices):
                return None
            consumed[field][key] = pos + 1
            return indices[pos]

        for er in expected_risks:
            idx = _resolve_clause_index(er, "risk")
            if idx is None:
                continue
            clause = clauses[idx]
            actual_risk = report.risks[idx].risk_level.value if idx < len(report.risks) else ""
            expected_risk = er.get("risk_level", "")
            if expected_risk and actual_risk != expected_risk:
                self._memory.correct(
                    clause_type=clause.clause_type,
                    clause_content=clause.content,
                    field="risk_level",
                    correct_value=expected_risk,
                )

        # expected_compliance 为扁平 list[dict]，每条 check 独立对齐到条款
        # （与 EvalItem.expected_compliance 类型注解及 metrics._compliance_accuracy
        # 消费方式一致；同条款多条 check 各自用 clause_index 显式对齐或按
        # clause_type 轮转，避免嵌套分组遍历导致同条款第二条 check 错位
        # 挂到下一个同 type 条款）。
        for check in expected_compliance:
            if not isinstance(check, dict):
                continue
            idx = _resolve_clause_index(check, "compliance")
            if idx is None:
                continue
            clause = clauses[idx]
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
            "timestamp": now_iso(),
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
