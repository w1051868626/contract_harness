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


class _MockMemoryStore:
    """捕获 ``correct`` 调用的 mock，用于验证 _feed_corrections 对齐正确性。"""

    enabled = True

    def __init__(self):
        self.corrections: list[dict] = []

    def correct(self, clause_type, clause_content, field, correct_value):
        self.corrections.append(
            {
                "clause_type": clause_type,
                "clause_content": clause_content,
                "field": field,
                "correct_value": correct_value,
            }
        )


class TestFeedCorrectionsAlignment:
    """``EvalScorer._feed_corrections`` 条款对齐回归测试（R4）。"""

    def _make_scorer(self):
        from harness.eval.scorer import EvalScorer

        memory = _MockMemoryStore()
        scorer = EvalScorer(memory_store=memory)
        return scorer, memory

    def test_same_clause_type_multiple_clauses_rotates_index(self):
        """多个同 type 条款应轮转消费 type_to_indices 队列，
        而非全部对齐到第一个（R4 回归）。
        """
        scorer, memory = self._make_scorer()
        from harness.core.types import (
            Clause,
            ReviewReport,
            RiskAssessment,
        )

        clauses = [
            Clause(clause_type="保密", content="保密A"),
            Clause(clause_type="保密", content="保密B"),
        ]
        report = ReviewReport(
            document_id="d",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=[
                RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
                RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
            ],
            compliance_checks=[],
        )
        item = type(
            "Item",
            (),
            {
                "expected_risks": [
                    {"clause_type": "保密", "risk_level": "high"},
                    {"clause_type": "保密", "risk_level": "critical"},
                ],
                "expected_compliance": [],
            },
        )()
        scorer._feed_corrections(report, item)
        # 两条修正应分别落到保密A 和 保密B，而非都落到 A
        assert len(memory.corrections) == 2
        assert memory.corrections[0]["clause_content"] == "保密A"
        assert memory.corrections[0]["correct_value"] == "high"
        assert memory.corrections[1]["clause_content"] == "保密B"
        assert memory.corrections[1]["correct_value"] == "critical"

    def test_explicit_clause_index_takes_priority(self):
        """expected 提供 clause_index 时优先用 index，忽略 type 回退。"""
        scorer, memory = self._make_scorer()
        from harness.core.types import (
            Clause,
            ReviewReport,
            RiskAssessment,
        )

        clauses = [
            Clause(clause_type="保密", content="保密A"),
            Clause(clause_type="保密", content="保密B"),
        ]
        report = ReviewReport(
            document_id="d",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=[
                RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
                RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
            ],
            compliance_checks=[],
        )
        # clause_index=1 应精准对齐到保密B，即便 clause_type 也是"保密"
        item = type(
            "Item",
            (),
            {
                "expected_risks": [
                    {"clause_index": 1, "clause_type": "保密", "risk_level": "high"},
                ],
                "expected_compliance": [],
            },
        )()
        scorer._feed_corrections(report, item)
        assert len(memory.corrections) == 1
        assert memory.corrections[0]["clause_content"] == "保密B"

    def test_no_match_skips_correction(self):
        """expected 的 clause_type 在 report 中不存在时跳过，不抛错。"""
        scorer, memory = self._make_scorer()
        from harness.core.types import Clause, ReviewReport

        report = ReviewReport(
            document_id="d",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=[Clause(clause_type="保密", content="x")],
            risks=[],
            compliance_checks=[],
        )
        item = type(
            "Item",
            (),
            {
                "expected_risks": [{"clause_type": "不存在", "risk_level": "high"}],
                "expected_compliance": [],
            },
        )()
        scorer._feed_corrections(report, item)
        assert memory.corrections == []

    def test_risks_and_compliance_share_type_queue_independently(self):
        """同 type 多条款场景下，expected_risks 消费的 index 不应影响
        expected_compliance 的对齐（R1 回归）。

        旧实现 type_to_indices 队列被 risks/compliance 共用 pop，risks 循环
        消费完后 compliance 循环因队列已空而对齐失败返回 None，修正信号丢失。
        新实现每个 field 维护独立游标，互不抢占。
        """
        scorer, memory = self._make_scorer()
        from harness.core.types import (
            Clause,
            ComplianceCheck,
            ReviewReport,
            RiskAssessment,
        )

        clauses = [
            Clause(clause_type="保密", content="保密A"),
            Clause(clause_type="保密", content="保密B"),
        ]
        report = ReviewReport(
            document_id="d",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=[
                RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
                RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
            ],
            compliance_checks=[
                ComplianceCheck(regulation="数据安全法", status=True, detail=""),
            ],
        )
        # risks 消费掉两个保密 index；compliance 也应能对齐到保密A/B
        item = type(
            "Item",
            (),
            {
                "expected_risks": [
                    {"clause_type": "保密", "risk_level": "high"},
                    {"clause_type": "保密", "risk_level": "critical"},
                ],
                "expected_compliance": [
                    {"clause_type": "保密", "regulation": "数据安全法", "status": False},
                ],
            },
        )()
        scorer._feed_corrections(report, item)
        # 2 条 risk 修正 + 1 条 compliance 修正，共 3 条
        assert len(memory.corrections) == 3
        risk_corrections = [c for c in memory.corrections if c["field"] == "risk_level"]
        comp_corrections = [c for c in memory.corrections if c["field"].startswith("compliance:")]
        assert len(risk_corrections) == 2
        assert len(comp_corrections) == 1
        # compliance 应对齐到保密A（独立游标从 0 开始）
        assert comp_corrections[0]["clause_content"] == "保密A"
        assert comp_corrections[0]["correct_value"] == "不合规"

    def test_same_clause_multiple_compliance_checks_align_correctly(self):
        """同条款多条合规 check 应都对齐到该条款，而非嵌套遍历导致第二条
        check 错位挂到下一个同 type 条款（R5 回归）。

        旧实现按 ``list[list[dict]]`` 嵌套遍历 ``expected_compliance``，
        与 ``EvalItem.expected_compliance: list[dict]`` 类型注解及
        ``metrics._compliance_accuracy`` 的扁平消费方式不一致；同条款
        多条 check 时第二条会轮转到下一个同 type 条款，修正信号错位。
        """
        scorer, memory = self._make_scorer()
        from harness.core.types import (
            Clause,
            ComplianceCheck,
            ReviewReport,
            RiskAssessment,
        )

        clauses = [
            Clause(clause_type="保密", content="保密A"),
            Clause(clause_type="保密", content="保密B"),
        ]
        report = ReviewReport(
            document_id="d",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=[
                RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
                RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
            ],
            compliance_checks=[
                ComplianceCheck(regulation="数据安全法", status=True, detail=""),
                ComplianceCheck(regulation="个人信息法", status=True, detail=""),
            ],
        )
        # 同条款（保密A，clause_index=0）两条 check 都应对齐到保密A，
        # 而非嵌套遍历下第二条轮转挂到保密B
        item = type(
            "Item",
            (),
            {
                "expected_risks": [],
                "expected_compliance": [
                    {"clause_index": 0, "regulation": "数据安全法", "status": False},
                    {"clause_index": 0, "regulation": "个人信息法", "status": False},
                ],
            },
        )()
        scorer._feed_corrections(report, item)
        comp_corrections = [c for c in memory.corrections if c["field"].startswith("compliance:")]
        assert len(comp_corrections) == 2
        # 两条都应挂到保密A，而非一条挂 A 一条挂 B
        assert all(c["clause_content"] == "保密A" for c in comp_corrections)
        regulations = sorted(c["field"].split(":", 1)[1] for c in comp_corrections)
        assert regulations == ["个人信息法", "数据安全法"]
