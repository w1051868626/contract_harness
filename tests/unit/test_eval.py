"""评测系统指标计算单元测试。"""

import pytest

from harness.core.types import (
    Clause,
    ReviewReport,
    RiskLevel,
)
from harness.eval.metrics import MetricsCalculator


class TestEval:
    """评测指标计算器测试。"""

    def test_metrics_calculator_clause_coverage(self):
        """条款覆盖率应正确计算实际与预期条款的比例。"""
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
        assert pytest.approx(metrics.value, 0.001) == 2 / 3

    def test_metrics_calculator_risk_level_accuracy(self):
        """风险等级准确率应正确匹配预期值。"""
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
