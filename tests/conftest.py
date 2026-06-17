"""测试全局配置，包含 MockLLMClient 与通用 fixture。"""

from __future__ import annotations

from typing import Any

import pytest

from harness.agent.llm import LLMResponse
from harness.core.types import (
    Clause,
    ComplianceCheck,
    ContractDocument,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
)


class MockLLMClient:
    """模拟 LLM 客户端，返回预设响应列表。"""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, **kwargs):
        """记录调用并返回下一个预设响应，不足则返回空响应。"""
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return LLMResponse(content="", model="mock")


@pytest.fixture
def sample_document() -> ContractDocument:
    """返回示例合同文档。"""
    return ContractDocument(
        id="test-doc",
        title="测试合同",
        content="双方应保守商业秘密。违约方应赔偿损失。",
    )


@pytest.fixture
def sample_report(sample_document: ContractDocument) -> ReviewReport:
    """返回示例审查报告。"""
    clause = Clause(clause_type="保密", content="双方应保守商业秘密", risk=RiskLevel.LOW)
    risk = RiskAssessment(
        clause=clause,
        risk_level=RiskLevel.MEDIUM,
        reason="保密范围不明确",
        suggestion="建议明确保密范围",
    )
    compliance = ComplianceCheck(
        regulation="中华人民共和国民法典（合同编）",
        status=True,
        detail="符合相关法律规定",
    )
    return ReviewReport(
        document_id=sample_document.id,
        document_title=sample_document.title,
        reviewed_at="2026-01-01T00:00:00",
        summary="审查完成",
        clauses=[clause],
        risks=[risk],
        compliance_checks=[compliance],
        overall_risk=RiskLevel.MEDIUM,
    )


@pytest.fixture
def react_action_response() -> str:
    """返回模拟的 ReAct action JSON 响应。"""
    return '{"thought": "先提取条款", "action": "clause_extractor", "action_input": {}}'


@pytest.fixture
def react_final_response() -> str:
    """返回模拟的 ReAct final_answer JSON 响应。"""
    return '{"thought": "审查完成", "final_answer": true}'


@pytest.fixture
def clause_extract_response() -> str:
    """返回模拟的条款提取 JSON 响应。"""
    return """[
        {"type": "保密", "content": "双方应对本合同内容保密", "risk": "low"},
        {"type": "违约责任", "content": "违约方应赔偿守约方全部损失", "risk": "medium"}
    ]"""


@pytest.fixture
def risk_analysis_response() -> str:
    """返回模拟的单条风险分析 JSON 响应。"""
    return '{"risk_level": "medium", "reason": "赔偿范围不明确", "suggestion": "建议明确赔偿上限"}'


@pytest.fixture
def batch_risk_response() -> str:
    """返回模拟的批量风险分析 JSON 响应（覆盖多个条款）。"""
    return """[
        {"index": 0, "risk_level": "medium", "reason": "赔偿范围不明确", "suggestion": "建议明确赔偿上限"},
        {"index": 1, "risk_level": "high", "reason": "违约金过高", "suggestion": "建议调整违约金比例"}
    ]"""


@pytest.fixture
def compliance_response() -> str:
    """返回模拟的单条合规检查 JSON 响应。"""
    return '{"status": true, "detail": "符合相关法律规定"}'


@pytest.fixture
def batch_compliance_response() -> str:
    """返回模拟的批量合规检查 JSON 响应（2 个条款 × 5 部法规）。"""
    return """[
        {"clause_index": 0, "regulation": "中华人民共和国民法典（合同编）", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 0, "regulation": "中华人民共和国劳动合同法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 0, "regulation": "中华人民共和国数据安全法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 0, "regulation": "中华人民共和国个人信息保护法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 0, "regulation": "中华人民共和国反垄断法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 1, "regulation": "中华人民共和国民法典（合同编）", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 1, "regulation": "中华人民共和国劳动合同法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 1, "regulation": "中华人民共和国数据安全法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 1, "regulation": "中华人民共和国个人信息保护法", "status": true, "detail": "符合相关法律规定"},
        {"clause_index": 1, "regulation": "中华人民共和国反垄断法", "status": true, "detail": "符合相关法律规定"}
    ]"""


@pytest.fixture
def summary_response() -> str:
    """返回模拟的摘要 JSON 响应。"""
    return "审查完成，共发现2个条款，1个中风险项，0个不合规项。"


@pytest.fixture
def mock_llm(
    clause_extract_response: str,
    batch_risk_response: str,
    batch_compliance_response: str,
    summary_response: str,
) -> MockLLMClient:
    """批量处理模式：1 次条款提取 + 1 次风险批量 + 1 次合规批量 + 1 次摘要 = 4 个响应。"""
    responses: list[LLMResponse] = [
        LLMResponse(content=clause_extract_response, model="mock"),
        LLMResponse(content=batch_risk_response, model="mock"),
        LLMResponse(content=batch_compliance_response, model="mock"),
        LLMResponse(content=summary_response, model="mock"),
    ]
    return MockLLMClient(responses)
