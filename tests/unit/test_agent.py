"""Agent 组件单元测试：条款提取、风险分析、合规检查与完整审查流程。"""

from __future__ import annotations

import json
import tempfile

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMResponse
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import Clause, ContractDocument, RiskLevel
from harness.replay.storage import ReplayStorage
from tests.conftest import MockLLMClient

CLAUSE_JSON = """[
    {"type": "保密", "content": "双方应对本合同内容保密", "risk": "low"},
    {"type": "违约责任", "content": "违约方应赔偿守约方全部损失", "risk": "medium"}
]"""

RISK_JSON = '{"risk_level": "medium", "reason": "赔偿范围不明确", "suggestion": "建议明确赔偿上限"}'

COMPLIANCE_JSON = '{"status": true, "detail": "符合相关法律规定"}'

BATCH_COMPLIANCE_JSON = json.dumps(
    [
        {
            "clause_index": 0,
            "regulation": "中华人民共和国民法典（合同编）",
            "status": True,
            "detail": "符合",
        },
        {
            "clause_index": 0,
            "regulation": "中华人民共和国劳动合同法",
            "status": True,
            "detail": "符合",
        },
        {
            "clause_index": 0,
            "regulation": "中华人民共和国数据安全法",
            "status": True,
            "detail": "符合",
        },
        {
            "clause_index": 0,
            "regulation": "中华人民共和国个人信息保护法",
            "status": True,
            "detail": "符合",
        },
        {
            "clause_index": 0,
            "regulation": "中华人民共和国反垄断法",
            "status": True,
            "detail": "符合",
        },
    ],
    ensure_ascii=False,
)


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

    def test_batch_analyze_duplicate_index_keeps_first(self):
        """同 index 多条 item 应保留首个，而非 dict comprehension 静默覆盖。"""
        import json as _json

        # 两条 item 都 index=0，应保留第一个（medium），而非被 high 覆盖
        raw = _json.dumps(
            [
                {"index": 0, "risk_level": "medium", "reason": "首个"},
                {"index": 0, "risk_level": "high", "reason": "重复"},
            ],
            ensure_ascii=False,
        )
        llm = MockLLMClient([LLMResponse(content=raw, model="mock")])
        analyzer = RiskAnalyzer(llm)
        clauses = [Clause(clause_type="保密", content="内容")]
        results = analyzer.batch_analyze(clauses)
        assert len(results) == 1
        assert results[0].risk_level == RiskLevel.MEDIUM
        assert results[0].reason == "首个"


class TestComplianceChecker:
    """合规检查器测试。"""

    def test_check_returns_checks(self):
        """批量模式下应一次 LLM 调用返回 5 个法规的检查结果。"""
        llm = MockLLMClient([LLMResponse(content=BATCH_COMPLIANCE_JSON, model="mock")])
        clause = Clause(clause_type="保密", content="双方应保密...")
        checker = ComplianceChecker(llm)
        results = checker.check(clause)
        assert len(results) == 5
        assert all(r.status for r in results)

    def test_all_regulations_checked(self):
        """应覆盖所有预定义的法条类型。"""
        llm = MockLLMClient([LLMResponse(content=BATCH_COMPLIANCE_JSON, model="mock")])
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
        assert report.overall_risk == RiskLevel.HIGH
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
        assert len(session.steps[1].tool_calls) == 1

    def test_converse_nonexistent_session(self):
        """对不存在的会话追问应返回错误信息。"""
        # 用 MockLLMClient 显式隔离，避免默认 LLMClient 误打真实 API
        agent = ContractAgent(MockLLMClient([]))
        answer = agent.converse("nonexistent", "这个合同有什么风险？")
        assert "未找到会话" in answer

    def test_converse_returns_answer(self):
        """对已有会话追问应返回回答。"""
        # 使用独立 mock：4 次审查 + 1 次追问 = 5 个响应
        conv_llm = MockLLMClient(
            [
                LLMResponse(content=CLAUSE_JSON, model="mock"),
                LLMResponse(
                    content='[{"index": 0, "risk_level": "low", "reason": "标准", "suggestion": ""},{"index": 1, "risk_level": "medium", "reason": "模糊", "suggestion": ""}]',
                    model="mock",
                ),
                LLMResponse(content=BATCH_COMPLIANCE_JSON, model="mock"),
                LLMResponse(content="审查完成，无异议", model="mock"),
                LLMResponse(content="该保密条款风险为低，属于标准表述", model="mock"),
            ]
        )
        doc = ContractDocument(id="conv_test", title="测试", content="本合同保密条款...")
        agent = ContractAgent(conv_llm)
        report, session = agent.review(doc)
        assert session.session_id
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ReplayStorage(tmpdir)
            storage.save(
                session.session_id,
                {
                    "session_id": session.session_id,
                    "document": {
                        "id": session.document.id,
                        "title": session.document.title,
                        "content": session.document.content,
                    },
                    "started_at": session.started_at,
                    "finished_at": session.finished_at or "",
                    "steps": [],
                    "report": {
                        "summary": report.summary,
                        "overall_risk": report.overall_risk.value,
                        "clauses": [
                            {"clause_type": c.clause_type, "content": c.content}
                            for c in report.clauses
                        ],
                        "risks": [
                            {"risk_level": r.risk_level.value, "reason": r.reason}
                            for r in report.risks
                        ],
                        "compliance_checks": [
                            {"regulation": c.regulation, "status": c.status}
                            for c in report.compliance_checks
                        ],
                    },
                    "metadata": {},
                },
            )
            answer = agent.converse(session.session_id, "保密条款风险高吗？", replay_dir=tmpdir)
            assert answer, "应返回回答"
