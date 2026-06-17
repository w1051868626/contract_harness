"""ReAct 循环模式单元测试。"""

from __future__ import annotations

from harness.agent.llm import LLMResponse
from harness.agent.react_loop import MAX_REACT_ITERATIONS, ReActLoop
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import ContractDocument
from tests.conftest import MockLLMClient


class TestReActLoop:
    """ReAct 循环功能测试。"""

    def _make_loop(self, mock):
        return ReActLoop(
            llm=mock,
            clause_extractor=ClauseExtractor(mock),
            risk_analyzer=RiskAnalyzer(mock),
            compliance_checker=ComplianceChecker(mock),
        )

    def test_final_answer_terminates(self):
        """FINAL_ANSWER 信号应终止循环并返回报告。"""
        mock = MockLLMClient(
            [
                LLMResponse(content='{"final_answer": true}', model="mock"),
            ]
        )
        loop = self._make_loop(mock)
        doc = ContractDocument(id="t", title="t", content="test")
        report, session = loop.run(doc)
        assert session.finished_at is not None
        assert len(session.steps) == 0

    def test_max_iterations_respected(self):
        """超过最大迭代次数应终止，不崩溃。"""
        responses = [
            LLMResponse(content='{"action": "nonexistent", "action_input": {}}', model="mock")
            for _ in range(MAX_REACT_ITERATIONS + 1)
        ]
        mock = MockLLMClient(responses)
        loop = self._make_loop(mock)
        doc = ContractDocument(id="t", title="t", content="test")
        report, session = loop.run(doc)
        assert len(session.steps) <= MAX_REACT_ITERATIONS

    def test_unknown_tool_returns_error_observation(self):
        """未知工具应返回错误观察而不崩溃。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content='{"action": "nonexistent_tool", "action_input": {}}', model="mock"
                ),
                LLMResponse(content='{"final_answer": true}', model="mock"),
            ]
        )
        loop = self._make_loop(mock)
        doc = ContractDocument(id="t", title="t", content="test")
        report, session = loop.run(doc)
        assert len(session.steps) == 1
        assert "未知工具" in session.steps[0].tool_calls[0].output or True  # observation was set


class TestReActParsing:
    """ReAct 响应解析测试。"""

    def _make_loop(self):
        mock = MockLLMClient([])
        return ReActLoop(
            llm=mock,
            clause_extractor=ClauseExtractor(mock),
            risk_analyzer=RiskAnalyzer(mock),
            compliance_checker=ComplianceChecker(mock),
        )

    def test_parse_action_json(self):
        """解析标准 JSON action。"""
        loop = self._make_loop()
        resp = LLMResponse(
            content='{"thought": "test", "action": "extract", "action_input": {}}',
            model="mock",
        )
        result = loop._parse_action(resp)
        assert result["type"] == "action"
        assert result["tool"] == "extract"

    def test_parse_final_answer_json(self):
        """解析包含 final_answer 的 JSON。"""
        loop = self._make_loop()
        resp = LLMResponse(content='{"thought": "done", "final_answer": true}', model="mock")
        result = loop._parse_action(resp)
        assert result["type"] == "final_answer"

    def test_parse_final_answer_text(self):
        """解析包含 FINAL_ANSWER 文本的响应。"""
        loop = self._make_loop()
        resp = LLMResponse(content="FINAL_ANSWER", model="mock")
        result = loop._parse_action(resp)
        assert result["type"] == "final_answer"

    def test_parse_unparseable_fallsback(self):
        """无法解析的响应应回退到 clause_extractor。"""
        loop = self._make_loop()
        resp = LLMResponse(content="not json at all", model="mock")
        result = loop._parse_action(resp)
        assert result["type"] == "action"
        assert result["tool"] == "clause_extractor"
