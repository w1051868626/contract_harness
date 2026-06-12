from __future__ import annotations

from typing import Any

from harness.core.types import EvalMetric, EvalResult, ReviewReport


class MetricsCalculator:
    def calculate(
        self,
        report: ReviewReport,
        expected: dict[str, Any],
    ) -> list[EvalMetric]:
        return [
            self._clause_coverage(report, expected),
            self._risk_accuracy(report, expected),
            self._compliance_accuracy(report, expected),
            self._risk_level_accuracy(report, expected),
        ]

    def aggregate(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for m in result.metrics:
                all_metrics.setdefault(m.name, []).append(m.value)

        return {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }

    def _clause_coverage(
        self, report: ReviewReport, expected: dict
    ) -> EvalMetric:
        expected_types = {c.get("clause_type") for c in expected.get("clauses", [])}
        actual_types = {c.clause_type for c in report.clauses}

        if not expected_types:
            return EvalMetric(name="clause_coverage", value=1.0)

        intersection = expected_types & actual_types
        value = len(intersection) / len(expected_types)
        return EvalMetric(name="clause_coverage", value=round(value, 4))

    def _risk_accuracy(
        self, report: ReviewReport, expected: dict
    ) -> EvalMetric:
        expected_risks = expected.get("risks", [])
        if not expected_risks:
            return EvalMetric(name="risk_accuracy", value=1.0)

        matches = 0
        for exp in expected_risks:
            for actual in report.risks:
                if (exp.get("clause_type", "") == actual.clause.clause_type
                        and exp.get("risk_level", "") == actual.risk_level.value):
                    matches += 1
                    break

        value = matches / len(expected_risks)
        return EvalMetric(name="risk_accuracy", value=round(value, 4))

    def _compliance_accuracy(
        self, report: ReviewReport, expected: dict
    ) -> EvalMetric:
        expected_checks = expected.get("compliance", [])
        if not expected_checks:
            return EvalMetric(name="compliance_accuracy", value=1.0)

        matches = 0
        for exp in expected_checks:
            for actual in report.compliance_checks:
                if (exp.get("regulation", "") == actual.regulation
                        and exp.get("status") == actual.status):
                    matches += 1
                    break

        value = matches / len(expected_checks)
        return EvalMetric(name="compliance_accuracy", value=round(value, 4))

    def _risk_level_accuracy(
        self, report: ReviewReport, expected: dict
    ) -> EvalMetric:
        expected_level = expected.get("overall_risk", "info")
        actual_level = report.overall_risk.value
        value = 1.0 if expected_level == actual_level else 0.0
        return EvalMetric(name="risk_level_accuracy", value=value)
