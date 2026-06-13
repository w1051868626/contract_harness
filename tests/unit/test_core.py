"""核心类型模型单元测试。"""

from harness.core.types import (
    Clause,
    ContractDocument,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
)


class TestCoreTypes:
    """核心类型（RiskLevel、ContractDocument、ReviewReport 等）测试。"""

    def test_risk_level_enum(self):
        """RiskLevel 枚举值应正确映射。"""
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"

    def test_contract_document(self):
        """ContractDocument 应正确存储文档信息。"""
        doc = ContractDocument(id="test", title="测试合同", content="合同内容")
        assert doc.id == "test"
        assert doc.title == "测试合同"
        assert doc.content == "合同内容"

    def test_review_report_defaults(self):
        """ReviewReport 的列表字段默认应为空列表。"""
        report = ReviewReport(document_id="1", document_title="t", reviewed_at="now", summary="s")
        assert report.clauses == []
        assert report.risks == []
        assert report.compliance_checks == []
        assert report.overall_risk == RiskLevel.INFO

    def test_risk_assessment(self):
        """RiskAssessment 应正确关联 Clause 与风险等级。"""
        clause = Clause(clause_type="保密", content="保密内容")
        risk = RiskAssessment(clause=clause, risk_level=RiskLevel.HIGH, reason="风险原因")
        assert risk.clause.clause_type == "保密"
        assert risk.risk_level == RiskLevel.HIGH
