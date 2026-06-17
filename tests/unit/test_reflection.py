"""Reflection 模式单元测试。"""

from __future__ import annotations

from harness.agent.llm import LLMResponse
from harness.agent.reflection import reflect_on_report
from harness.core.types import RiskLevel
from tests.conftest import MockLLMClient


class TestReflection:
    """自审修正功能测试。"""

    def test_reflection_fixes_issues(self, sample_report):
        """发现 issues 时应更新摘要和风险等级。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content="""{
                    "issues": [{"type": "完整性", "severity": "high", "description": "缺少违约责任分析", "fix": "补充分析"}],
                    "revised_summary": "修正后的摘要",
                    "revised_overall_risk": "high"
                }""",
                    model="mock",
                ),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.summary == "修正后的摘要"
        assert revised.overall_risk == RiskLevel.HIGH

    def test_reflection_no_issues(self, sample_report):
        """无 issues 时报告应保持不变。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content='{"issues": [], "revised_summary": "", "revised_overall_risk": ""}',
                    model="mock",
                ),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.summary == sample_report.summary
        assert revised.overall_risk == sample_report.overall_risk

    def test_reflection_parse_error(self, sample_report):
        """JSON 解析失败时应返回原报告。"""
        mock = MockLLMClient(
            [
                LLMResponse(content="not valid json", model="mock"),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.summary == sample_report.summary
        assert revised.overall_risk == sample_report.overall_risk

    def test_reflection_partial_update(self, sample_report):
        """只有 summary 更新时 risk 应保持不变。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content='{"issues": [], "revised_summary": "新摘要", "revised_overall_risk": ""}',
                    model="mock",
                ),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.summary == "新摘要"
        assert revised.overall_risk == sample_report.overall_risk

    def test_reflection_only_risk_update(self, sample_report):
        """只有 risk 更新时 summary 应保持不变。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content='{"issues": [], "revised_summary": "", "revised_overall_risk": "low"}',
                    model="mock",
                ),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.summary == sample_report.summary
        assert revised.overall_risk == RiskLevel.LOW

    def test_reflection_invalid_risk(self, sample_report):
        """无效风险等级时应保持原值。"""
        mock = MockLLMClient(
            [
                LLMResponse(
                    content='{"issues": [], "revised_summary": "", "revised_overall_risk": "invalid_risk"}',
                    model="mock",
                ),
            ]
        )
        revised = reflect_on_report(mock, sample_report)
        assert revised.overall_risk == sample_report.overall_risk
