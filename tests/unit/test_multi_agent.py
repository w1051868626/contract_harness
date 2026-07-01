"""Multi-Agent 数据模型测试。"""

from __future__ import annotations

from harness.core.types import (
    AgentMode,
    Disagreement,
    WorkerOutput,
    WorkerTask,
)


class TestDataModels:
    """Multi-Agent 新增数据模型测试。"""

    def test_agent_mode_has_multi_agent(self):
        assert AgentMode.MULTI_AGENT == "multi_agent"

    def test_worker_task_creation(self):
        task = WorkerTask(worker_role="ClauseExpert", prompt="提取条款", input_data={})
        assert task.worker_role == "ClauseExpert"
        assert task.prompt == "提取条款"

    def test_worker_output_creation(self):
        output = WorkerOutput(
            worker_role="RiskExpert", content="高风险", structured={"level": "high"}
        )
        assert output.worker_role == "RiskExpert"
        assert output.structured["level"] == "high"

    def test_disagreement_creation(self):
        d = Disagreement(
            item_id="clause-0",
            field="risk_level",
            value_a="high",
            value_b="low",
            worker_a="RiskExpert",
            worker_b="ComplianceExpert",
        )
        assert d.field == "risk_level"
        assert d.worker_a == "RiskExpert"


class TestWorkerAgent:
    """WorkerAgent 单元测试。"""

    def test_worker_has_role_prompt(self):
        from harness.agent.multi_agent.worker import WORKER_PROMPTS

        assert "ClauseExpert" in WORKER_PROMPTS
        assert "RiskExpert" in WORKER_PROMPTS
        assert "ComplianceExpert" in WORKER_PROMPTS

    def test_worker_execute_returns_output(self):
        from harness.agent.llm import LLMResponse
        from harness.agent.multi_agent.worker import WorkerAgent
        from tests.conftest import MockLLMClient

        llm = MockLLMClient([LLMResponse(content='{"clauses": []}', model="mock")])
        worker = WorkerAgent(role="ClauseExpert", llm=llm)
        output = worker.execute("测试合同内容")
        assert output.worker_role == "ClauseExpert"
        assert output.content

    def test_worker_calls_llm_with_correct_messages(self):
        from harness.agent.llm import LLMResponse
        from harness.agent.multi_agent.worker import WorkerAgent
        from tests.conftest import MockLLMClient

        llm = MockLLMClient([LLMResponse(content="ok", model="mock")])
        worker = WorkerAgent(role="RiskExpert", llm=llm)
        worker.execute("审核内容")
        assert len(llm.calls) == 1
        messages = llm.calls[0]["messages"]
        assert any(m["role"] == "system" for m in messages)
        assert any("审核" in m.get("content", "") for m in messages if m["role"] == "user")

    def test_worker_with_peer_results(self):
        from harness.agent.llm import LLMResponse
        from harness.agent.multi_agent.worker import WorkerAgent
        from tests.conftest import MockLLMClient

        llm = MockLLMClient([LLMResponse(content="交叉验证完成", model="mock")])
        worker = WorkerAgent(role="ComplianceExpert", llm=llm)
        peer = {"RiskExpert": "高风险发现"}
        output = worker.execute("审核内容", peer_results=peer)
        assert output.content
        all_text = " ".join(m.get("content", "") for m in llm.calls[0]["messages"])
        assert "高风险发现" in all_text
