"""评测系统指标计算与报告生成单元测试。"""

import pytest

from harness.core.types import (
    Clause,
    ReviewReport,
    RiskLevel,
)
from harness.eval.metrics import MetricsCalculator
from harness.eval.reporters import EvalReporter


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


class TestEvalReporter:
    """评测报告生成器测试。"""

    SAMPLE_DATA = {
        "timestamp": "2026-01-01T00:00:00",
        "total_items": 2,
        "aggregated_metrics": {"clause_coverage": 0.85, "risk_accuracy": 0.75},
        "per_item_results": [
            {"document_id": "doc1", "metrics": {"clause_coverage": 0.9, "risk_accuracy": 0.8}},
            {"document_id": "doc2", "metrics": {"clause_coverage": 0.8, "risk_accuracy": 0.7}},
        ],
    }

    def test_report_json(self, tmp_path):
        """生成 JSON 报告应写入文件且内容正确。"""
        reporter = EvalReporter(tmp_path)
        path = reporter.report_json(self.SAMPLE_DATA, "test_report")
        assert path.exists()
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["total_items"] == 2
        assert data["aggregated_metrics"]["clause_coverage"] == 0.85

    def test_report_markdown(self, tmp_path):
        """生成 Markdown 报告应包含指标和结果。"""
        reporter = EvalReporter(tmp_path)
        path = reporter.report_markdown(self.SAMPLE_DATA, "test_report")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "评测报告" in content
        assert "条款覆盖率" in content
        assert "85.00" in content
        assert "doc1" in content

    def test_report_html(self, tmp_path):
        """生成 HTML 报告应包含表格结构。"""
        reporter = EvalReporter(tmp_path)
        path = reporter.report_html(self.SAMPLE_DATA, "test_report")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html" in content
        assert "<table>" in content
        assert "条款覆盖率" in content
