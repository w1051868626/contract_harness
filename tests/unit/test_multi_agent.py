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
