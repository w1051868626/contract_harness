"""P0 修复回归测试：批量解析围栏、修正信号匹配、Worker 结构化输出、LLM 异常处理。

覆盖四个 P0 修复，确保：
- P0-1: risk_analyzer/compliance_checker 的 _parse_batch_response 正确处理 ```json 围栏
- P0-2: EvalScorer._feed_corrections 按 clause_index/clause_type 匹配而非下标对齐
- P0-3: WorkerAgent.execute 填充 structured 字段，交叉验证场景不解析
- P0-4: LLMClient 密钥缺失抛 AgentError、显式 mock 模式可用、真实 API 错误被捕获
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from harness.agent.llm import LLMClient, LLMResponse
from harness.agent.multi_agent.worker import WorkerAgent
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.exceptions import AgentError
from harness.core.types import (
    Clause,
    ContractDocument,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
)
from harness.eval.dataset import EvalItem
from harness.eval.scorer import EvalScorer
from tests.conftest import MockLLMClient

# ============================================================
# P0-1: 批量解析 ```json 围栏
# ============================================================


class TestBatchParseJsonFence:
    """P0-1: _parse_batch_response 应正确解析 ```json 围栏包裹的数组。"""

    BATCH_RISK_FENCED = """```json
[
    {"index": 0, "risk_level": "medium", "reason": "模糊", "suggestion": "明确"},
    {"index": 1, "risk_level": "high", "reason": "过高", "suggestion": "调整"}
]
```"""

    BATCH_COMPLIANCE_FENCED = """```json
[
    {"clause_index": 0, "regulation": "中华人民共和国民法典（合同编）", "status": true, "detail": "符合"},
    {"clause_index": 0, "regulation": "中华人民共和国劳动合同法", "status": false, "detail": "不符"}
]
```"""

    def test_risk_analyzer_parses_fenced_json_array(self):
        """risk_analyzer 应解析 ```json 围栏数组，而非回退到逐条 analyze。"""
        llm = MockLLMClient([LLMResponse(content=self.BATCH_RISK_FENCED, model="mock")])
        analyzer = RiskAnalyzer(llm)
        clauses = [
            Clause(clause_type="保密", content="保密内容"),
            Clause(clause_type="违约责任", content="违约内容"),
        ]
        results = analyzer.batch_analyze(clauses)
        assert len(results) == 2
        assert results[0].risk_level == RiskLevel.MEDIUM
        assert results[1].risk_level == RiskLevel.HIGH
        # 关键：只应调用 1 次 LLM（批量），不应回退到逐条 analyze 触发额外调用
        assert llm.call_count == 1

    def test_compliance_checker_parses_fenced_json_array(self):
        """compliance_checker 应解析 ```json 围栏数组。"""
        llm = MockLLMClient([LLMResponse(content=self.BATCH_COMPLIANCE_FENCED, model="mock")])
        checker = ComplianceChecker(llm)
        clauses = [Clause(clause_type="保密", content="保密内容")]
        results = checker.batch_check(clauses)
        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0].status is True
        assert results[0][1].status is False

    def test_risk_analyzer_handles_single_object(self):
        """LLM 返回单个 JSON 对象（非数组）时也应正确解析。"""
        single_obj = '{"index": 0, "risk_level": "low", "reason": "标准", "suggestion": ""}'
        llm = MockLLMClient([LLMResponse(content=single_obj, model="mock")])
        analyzer = RiskAnalyzer(llm)
        clauses = [Clause(clause_type="保密", content="内容")]
        results = analyzer.batch_analyze(clauses)
        assert len(results) == 1
        assert results[0].risk_level == RiskLevel.LOW


# ============================================================
# P0-2: EvalScorer._feed_corrections 按 clause_index/clause_type 匹配
# ============================================================


class TestFeedCorrectionsMatching:
    """P0-2: 修正信号应按 clause_index/clause_type 匹配，而非下标对齐。"""

    def test_correction_matches_by_clause_index(self):
        """expected_risks 用 clause_index 引用时，应正确定位 clause。"""
        corrected: list[tuple[str, str, str]] = []

        class StubMemory:
            enabled = True

            def correct(self, clause_type, clause_content, field, correct_value):
                corrected.append((clause_type, field, correct_value))

        clauses = [
            Clause(clause_type="保密", content="保密内容"),
            Clause(clause_type="违约责任", content="违约内容"),
        ]
        risks = [
            RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
            RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
        ]
        report = ReviewReport(
            document_id="d1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=risks,
            overall_risk=RiskLevel.LOW,
        )
        item = EvalItem(
            document=ContractDocument(id="d1", title="t", content="c"),
            expected_clauses=[],
            expected_risks=[
                {"clause_index": 1, "risk_level": "high", "reason": "", "suggestion": ""},
            ],
            expected_compliance=[],
            expected_risk_level=RiskLevel.HIGH,
        )

        scorer = EvalScorer(memory_store=StubMemory())
        scorer._feed_corrections(report, item)

        # 应只修正 clause_index=1（违约责任）的 risk_level，而非下标 0
        assert len(corrected) == 1
        assert corrected[0] == ("违约责任", "risk_level", "high")

    def test_correction_fallback_by_clause_type(self):
        """expected_risks 无 clause_index 时回退到 clause_type 匹配。"""
        corrected: list[tuple[str, str, str]] = []

        class StubMemory:
            enabled = True

            def correct(self, clause_type, clause_content, field, correct_value):
                corrected.append((clause_type, field, correct_value))

        clauses = [
            Clause(clause_type="保密", content="A"),
            Clause(clause_type="违约责任", content="B"),
        ]
        risks = [
            RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason=""),
            RiskAssessment(clause=clauses[1], risk_level=RiskLevel.LOW, reason=""),
        ]
        report = ReviewReport(
            document_id="d1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=risks,
            overall_risk=RiskLevel.LOW,
        )
        # 无 clause_index，靠 clause_type 定位
        item = EvalItem(
            document=ContractDocument(id="d1", title="t", content="c"),
            expected_clauses=[],
            expected_risks=[{"clause_type": "违约责任", "risk_level": "critical"}],
            expected_compliance=[],
            expected_risk_level=RiskLevel.CRITICAL,
        )

        scorer = EvalScorer(memory_store=StubMemory())
        scorer._feed_corrections(report, item)

        assert len(corrected) == 1
        assert corrected[0][0] == "违约责任"
        assert corrected[0][2] == "critical"

    def test_no_correction_when_mismatch(self):
        """expected 与实际 clause_type 都不匹配时不应注入任何修正。"""
        corrected: list = []

        class StubMemory:
            enabled = True

            def correct(self, *args, **kwargs):
                corrected.append(1)

        clauses = [Clause(clause_type="保密", content="A")]
        risks = [RiskAssessment(clause=clauses[0], risk_level=RiskLevel.LOW, reason="")]
        report = ReviewReport(
            document_id="d1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=risks,
            overall_risk=RiskLevel.LOW,
        )
        # clause_index 越界 + clause_type 不匹配
        item = EvalItem(
            document=ContractDocument(id="d1", title="t", content="c"),
            expected_clauses=[],
            expected_risks=[{"clause_index": 5, "clause_type": "不存在", "risk_level": "high"}],
            expected_compliance=[],
            expected_risk_level=RiskLevel.HIGH,
        )

        scorer = EvalScorer(memory_store=StubMemory())
        scorer._feed_corrections(report, item)

        assert corrected == []

    def test_no_correction_when_actual_matches_expected(self):
        """actual_risk == expected_risk 时不应注入修正。"""
        corrected: list = []

        class StubMemory:
            enabled = True

            def correct(self, *args, **kwargs):
                corrected.append(1)

        clauses = [Clause(clause_type="保密", content="A")]
        risks = [RiskAssessment(clause=clauses[0], risk_level=RiskLevel.HIGH, reason="")]
        report = ReviewReport(
            document_id="d1",
            document_title="t",
            reviewed_at="now",
            summary="s",
            clauses=clauses,
            risks=risks,
            overall_risk=RiskLevel.HIGH,
        )
        item = EvalItem(
            document=ContractDocument(id="d1", title="t", content="c"),
            expected_clauses=[],
            expected_risks=[{"clause_index": 0, "risk_level": "high"}],
            expected_compliance=[],
            expected_risk_level=RiskLevel.HIGH,
        )

        scorer = EvalScorer(memory_store=StubMemory())
        scorer._feed_corrections(report, item)

        assert corrected == []


# ============================================================
# P0-3: WorkerAgent.execute 填充 structured 字段
# ============================================================


class TestWorkerStructuredOutput:
    """P0-3: Worker execute 应填充 structured，交叉验证场景不解析。"""

    def test_worker_fills_structured_on_first_execute(self):
        """首次执行时 structured 应被解析填充。"""
        structured_json = json.dumps(
            [{"type": "保密", "content": "保密内容"}],
            ensure_ascii=False,
        )
        llm = MockLLMClient([LLMResponse(content=structured_json, model="mock")])
        worker = WorkerAgent(role="ClauseExpert", llm=llm)
        output = worker.execute("合同内容")
        assert output.structured is not None
        assert isinstance(output.structured, list)
        assert output.structured[0]["type"] == "保密"

    def test_worker_structured_none_on_parse_failure(self):
        """解析失败时 structured 应为 None。"""
        llm = MockLLMClient([LLMResponse(content="not json at all", model="mock")])
        worker = WorkerAgent(role="RiskExpert", llm=llm)
        output = worker.execute("合同内容")
        assert output.structured is None

    def test_worker_no_structured_on_cross_validation(self):
        """交叉验证场景（peer_results 非空）不应解析 structured。"""
        llm = MockLLMClient([LLMResponse(content='{"valid": "json"}', model="mock")])
        worker = WorkerAgent(role="ComplianceExpert", llm=llm)
        output = worker.execute("审阅内容", peer_results={"RiskExpert": "高风险"})
        assert output.structured is None
        assert output.content == '{"valid": "json"}'


# ============================================================
# P0-4: LLMClient 异常处理
# ============================================================


class TestLLMClientErrorHandling:
    """P0-4: LLMClient 密钥缺失抛 AgentError、显式 mock 可用、API 错误被捕获。"""

    def test_missing_api_key_raises_agent_error(self):
        """无 mock 模式 + 无密钥 → 抛 AgentError，不再静默回退。"""
        # 临时清空环境变量，确保不受 .env 影响
        import os

        from harness.core.config import LLMConfig

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "LLM_API_KEY": ""}, clear=False):
            client = LLMClient(LLMConfig(api_key="", api_base="https://example.com/v1"))
            with pytest.raises(AgentError, match="API 密钥"):
                client.chat([{"role": "user", "content": "hi"}])

    def test_explicit_mock_mode_returns_mock_response(self):
        """显式 mock=True 时无密钥也应返回模拟响应。"""
        client = LLMClient(mock=True)
        resp = client.chat([{"role": "user", "content": "合同条款提取：保密义务"}])
        assert resp.model == "mock"
        assert resp.content  # 非空

    def test_api_error_wrapped_as_agent_error(self):
        """openai.APIError 应被包装为 AgentError 向上抛出。"""
        from openai import APIError

        from harness.core.config import LLMConfig

        client = LLMClient(LLMConfig(api_key="sk-fake", api_base="https://example.com/v1"))
        # 模拟 client 属性返回一个会抛 APIError 的 mock
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = APIError(
            message="internal error",
            request=MagicMock(),
            body=None,
        )
        client._client = mock_openai
        with pytest.raises(AgentError, match="LLM API 调用失败"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_rate_limit_wrapped_as_agent_error(self):
        """openai.RateLimitError 应被包装为 AgentError。

        用 max_retries=1 关闭重试，避免测试真 sleep；此时应直接抛原始
        包装错误（message 含"限流"），而非"重试 N 次后仍失败"。
        """
        from openai import RateLimitError

        from harness.core.config import LLMConfig

        client = LLMClient(LLMConfig(api_key="sk-fake", api_base="https://example.com/v1"))
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = RateLimitError(
            message="rate limited",
            response=MagicMock(),
            body=None,
        )
        client._client = mock_openai
        with pytest.raises(AgentError, match="限流"):
            client.chat([{"role": "user", "content": "hi"}], max_retries=1)

    def test_retry_succeeds_after_transient_error(self):
        """网络层瞬时错误应自动重试，最终成功时返回正常响应。"""
        from openai import APIConnectionError

        from harness.core.config import LLMConfig

        client = LLMClient(LLMConfig(api_key="sk-fake", api_base="https://example.com/v1"))
        mock_openai = MagicMock()
        # 前 2 次抛网络错误，第 3 次成功
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "ok"
        mock_resp.model = "mock"
        mock_resp.usage = None
        mock_openai.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            mock_resp,
        ]
        client._client = mock_openai
        # 用 max_retries=3 + 0 退避避免真 sleep
        import harness.agent.llm as llm_mod

        orig = llm_mod._RETRY_BASE_DELAY
        llm_mod._RETRY_BASE_DELAY = 0.0
        try:
            resp = client.chat([{"role": "user", "content": "hi"}], max_retries=3)
        finally:
            llm_mod._RETRY_BASE_DELAY = orig
        assert resp.content == "ok"
        assert mock_openai.chat.completions.create.call_count == 3

    def test_retry_exhausted_raises_aggregated_error(self):
        """重试耗尽时应抛聚合错误（message 含"重试 N 次后仍失败"）。"""
        from openai import APIConnectionError

        from harness.core.config import LLMConfig

        client = LLMClient(LLMConfig(api_key="sk-fake", api_base="https://example.com/v1"))
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        client._client = mock_openai
        import harness.agent.llm as llm_mod

        orig = llm_mod._RETRY_BASE_DELAY
        llm_mod._RETRY_BASE_DELAY = 0.0
        try:
            with pytest.raises(AgentError, match="重试 2 次后仍失败"):
                client.chat([{"role": "user", "content": "hi"}], max_retries=2)
        finally:
            llm_mod._RETRY_BASE_DELAY = orig
        assert mock_openai.chat.completions.create.call_count == 2
