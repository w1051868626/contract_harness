"""评测指标计算器。"""

from __future__ import annotations

from harness.core.types import EvalMetric, EvalResult, ExpectedMetrics, ReviewReport
from harness.utils.log import logger


class MetricsCalculator:
    """指标计算器，计算条款覆盖率、风险准确率等。"""

    def calculate(
        self,
        report: ReviewReport,
        expected: ExpectedMetrics,
    ) -> list[EvalMetric]:
        """计算单份审查报告的各项指标。"""
        logger.info("Calculating metrics for document_id={}", report.document_id)
        metrics = [
            self._clause_coverage(report, expected),
            self._risk_accuracy(report, expected),
            self._compliance_accuracy(report, expected),
            self._risk_level_accuracy(report, expected),
        ]
        logger.debug("Metrics: {}", {m.name: m.value for m in metrics})
        return metrics

    def aggregate(self, results: list[EvalResult]) -> dict[str, float]:
        """聚合多个评测结果为平均指标。"""
        logger.debug("Aggregating {} eval results", len(results))
        if not results:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for m in result.metrics:
                all_metrics.setdefault(m.name, []).append(m.value)

        aggregated = {name: sum(values) / len(values) for name, values in all_metrics.items()}
        logger.debug("Aggregated metrics: {}", aggregated)
        return aggregated

    def _clause_coverage(self, report: ReviewReport, expected: ExpectedMetrics) -> EvalMetric:
        """计算条款类型覆盖比例。"""
        expected_types = {c.get("clause_type") for c in expected.get("clauses", [])}
        actual_types = {c.clause_type for c in report.clauses}

        if not expected_types:
            return EvalMetric(name="clause_coverage", value=1.0)

        intersection = expected_types & actual_types
        value = len(intersection) / len(expected_types)
        return EvalMetric(name="clause_coverage", value=round(value, 4))

    def _risk_accuracy(self, report: ReviewReport, expected: ExpectedMetrics) -> EvalMetric:
        """计算风险识别匹配准确率。"""
        expected_risks = expected.get("risks", [])
        if not expected_risks:
            return EvalMetric(name="risk_accuracy", value=1.0)

        matches = 0
        for exp in expected_risks:
            for actual in report.risks:
                if (
                    exp.get("clause_type", "") == actual.clause.clause_type
                    and exp.get("risk_level", "") == actual.risk_level.value
                ):
                    matches += 1
                    break

        value = matches / len(expected_risks)
        return EvalMetric(name="risk_accuracy", value=round(value, 4))

    def _compliance_accuracy(self, report: ReviewReport, expected: ExpectedMetrics) -> EvalMetric:
        """计算合规检查匹配准确率。"""
        expected_checks = expected.get("compliance", [])
        if not expected_checks:
            return EvalMetric(name="compliance_accuracy", value=1.0)

        matches = 0
        for exp in expected_checks:
            for actual in report.compliance_checks:
                if (
                    exp.get("regulation", "") == actual.regulation
                    and exp.get("status") == actual.status
                ):
                    matches += 1
                    break

        value = matches / len(expected_checks)
        return EvalMetric(name="compliance_accuracy", value=round(value, 4))

    def _risk_level_accuracy(self, report: ReviewReport, expected: ExpectedMetrics) -> EvalMetric:
        """计算整体风险等级判断准确率。"""
        expected_level = expected.get("overall_risk", "info")
        actual_level = report.overall_risk.value
        value = 1.0 if expected_level == actual_level else 0.0
        return EvalMetric(name="risk_level_accuracy", value=value)
