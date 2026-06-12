from harness.core.types import (
    Clause,
    ReviewReport,
    RiskLevel,
)
from harness.eval.metrics import MetricsCalculator


class TestEval:
    def test_metrics_calculator_clause_coverage(self):
        report = ReviewReport(
            document_id="1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=[
                Clause(clause_type="保密", content="..."),
                Clause(clause_type="违约责任", content="..."),
            ],
        )
        expected = {
            "clauses": [
                {"clause_type": "保密"},
                {"clause_type": "违约责任"},
                {"clause_type": "管辖"},
            ],
        }
        calc = MetricsCalculator()
        metrics = calc._clause_coverage(report, expected)
        import pytest

        assert pytest.approx(metrics.value, 0.001) == 2 / 3

    def test_metrics_calculator_risk_level_accuracy(self):
        report = ReviewReport(
            document_id="1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            overall_risk=RiskLevel.HIGH,
        )
        expected = {"overall_risk": "high"}
        calc = MetricsCalculator()
        metrics = calc._risk_level_accuracy(report, expected)
        assert metrics.value == 1.0

        expected_wrong = {"overall_risk": "low"}
        metrics = calc._risk_level_accuracy(report, expected_wrong)
        assert metrics.value == 0.0
