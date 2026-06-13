"""Agent 组件单元测试：条款提取、风险分析、合规检查与完整审查流程。"""

from __future__ import annotations

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMResponse
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import Clause, ContractDocument, RiskLevel
from tests.conftest import MockLLMClient

CLAUSE_JSON = """[
    {"type": "保密", "content": "双方应对本合同内容保密", "risk": "low"},
    {"type": "违约责任", "content": "违约方应赔偿守约方全部损失", "risk": "medium"}
]"""

RISK_JSON = '{"risk_level": "medium", "reason": "赔偿范围不明确", "suggestion": "建议明确赔偿上限"}'

COMPLIANCE_JSON = '{"status": true, "detail": "符合相关法律规定"}'


class TestClauseExtractor:
    """条款提取器测试。"""

    def test_extract_returns_clauses(self):
        """应正确从 LLM 响应中解析出 Clause 列表。"""
        llm = MockLLMClient([LLMResponse(content=CLAUSE_JSON, model="mock")])
        doc = ContractDocument(id="doc1", title="测试合同", content="本合同...")
        extractor = ClauseExtractor(llm)
        clauses = extractor.extract(doc)
        assert len(clauses) == 2
        assert clauses[0].clause_type == "保密"
        assert clauses[1].clause_type == "违约责任"
        assert clauses[0].risk == RiskLevel.LOW
        assert clauses[1].risk == RiskLevel.MEDIUM

    def test_extract_empty_on_bad_response(self):
        """LLM 返回非 JSON 时应返回空列表。"""
        llm = MockLLMClient([LLMResponse(content="not json", model="mock")])
        doc = ContractDocument(id="doc1", title="测试合同", content="本合同...")
        extractor = ClauseExtractor(llm)
        clauses = extractor.extract(doc)
        assert clauses == []


class TestRiskAnalyzer:
    """风险分析器测试。"""

    def test_analyze_returns_assessment(self):
        """应正确解析风险分析 JSON 并返回 RiskAssessment。"""
        llm = MockLLMClient([LLMResponse(content=RISK_JSON, model="mock")])
        clause = Clause(clause_type="违约责任", content="违约方应赔偿...")
        analyzer = RiskAnalyzer(llm)
        result = analyzer.analyze(clause)
        assert result.risk_level == RiskLevel.MEDIUM
        assert "赔偿" in result.reason

    def test_analyze_fallback_on_bad_response(self):
        """LLM 返回非 JSON 时应回退到 INFO 等级。"""
        llm = MockLLMClient([LLMResponse(content="not json", model="mock")])
        clause = Clause(clause_type="违约", content="...")
        analyzer = RiskAnalyzer(llm)
        result = analyzer.analyze(clause)
        assert result.risk_level == RiskLevel.INFO


class TestComplianceChecker:
    """合规检查器测试。"""

    def test_check_returns_checks(self):
        """应对所有预设法条返回合规检查结果。"""
        llm = MockLLMClient([LLMResponse(content=COMPLIANCE_JSON, model="mock") for _ in range(5)])
        clause = Clause(clause_type="保密", content="双方应保密...")
        checker = ComplianceChecker(llm)
        results = checker.check(clause)
        assert len(results) == 5
        assert all(r.status for r in results)

    def test_all_regulations_checked(self):
        """应覆盖所有预定义的法条类型。"""
        llm = MockLLMClient([LLMResponse(content=COMPLIANCE_JSON, model="mock") for _ in range(5)])
        clause = Clause(clause_type="保密", content="...")
        checker = ComplianceChecker(llm)
        results = checker.check(clause)
        regulations = {r.regulation for r in results}
        assert "中华人民共和国民法典（合同编）" in regulations
        assert "中华人民共和国劳动合同法" in regulations
        assert "中华人民共和国数据安全法" in regulations


class TestContractAgent:
    """完整合同审查 Agent 端到端测试。"""

    def test_review_returns_report_and_session(self, mock_llm):
        """审查应返回完整的报告与会话记录。"""
        doc = ContractDocument(id="doc1", title="测试合同", content="本合同...")
        agent = ContractAgent(mock_llm)
        report, session = agent.review(doc)
        assert report.document_id == "doc1"
        assert len(report.clauses) == 2
        assert len(report.risks) == 2
        assert len(report.compliance_checks) == 10
        assert report.overall_risk == RiskLevel.MEDIUM
        assert "审查完成" in report.summary
        assert len(session.steps) == 4
        assert session.report is not None

    def test_review_records_session(self, mock_llm):
        """审查过程应正确记录每个步骤的工具调用。"""
        doc = ContractDocument(id="doc2", title="NDA", content="保密内容...")
        agent = ContractAgent(mock_llm)
        _, session = agent.review(doc)
        assert session.session_id
        assert session.document.id == "doc2"
        assert len(session.steps[0].tool_calls) == 1
        assert len(session.steps[1].tool_calls) == 2
