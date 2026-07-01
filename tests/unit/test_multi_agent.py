"""Multi-Agent 数据模型测试。"""

from __future__ import annotations

from harness.agent.llm import LLMResponse
from harness.core.types import (
    AgentMode,
    Disagreement,
    WorkerOutput,
    WorkerTask,
)
from tests.conftest import MockLLMClient


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


class TestCrossValidator:
    """CrossValidator 分歧仲裁测试。"""

    def test_same_level_no_disagreement(self):
        from harness.agent.multi_agent.validator import CrossValidator

        validator = CrossValidator()
        disc: list[Disagreement] = []
        result = validator.arbitrate(disc)
        assert len(result) == 0

    def test_one_level_diff_uses_stricter(self):
        from harness.agent.multi_agent.validator import CrossValidator

        d = Disagreement(
            item_id="clause-0",
            field="risk_level",
            value_a="medium",
            value_b="low",
            worker_a="A",
            worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d])
        assert result[0]["resolved"] == "medium"  # 更严格

    def test_two_level_diff_triggers_llm(self):
        from harness.agent.multi_agent.validator import CrossValidator

        d = Disagreement(
            item_id="clause-0",
            field="risk_level",
            value_a="critical",
            value_b="low",
            worker_a="A",
            worker_b="B",
        )
        llm = MockLLMClient(
            [
                LLMResponse(
                    content='{"resolution": "critical", "explanation": "条款涉及核心义务"}',
                    model="mock",
                )
            ]
        )
        validator = CrossValidator(llm=llm)
        result = validator.arbitrate([d])
        assert result[0]["resolved"] == "critical"

    def test_compliance_status_disagreement(self):
        from harness.agent.multi_agent.validator import CrossValidator

        d = Disagreement(
            item_id="clause-0",
            field="compliance_status",
            value_a=True,
            value_b=False,
            worker_a="A",
            worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d])
        assert result[0]["needs_human_review"] is True

    def test_llm_fallback_keeps_majority(self):
        from harness.agent.multi_agent.validator import CrossValidator

        d1 = Disagreement(
            item_id="c0",
            field="risk_level",
            value_a="medium",
            value_b="high",
            worker_a="A",
            worker_b="B",
        )
        d2 = Disagreement(
            item_id="c1",
            field="clause_type",
            value_a="保密",
            value_b="违约责任",
            worker_a="A",
            worker_b="B",
        )
        validator = CrossValidator()
        result = validator.arbitrate([d1, d2])
        # clause_type 分歧 → 都保留
        assert "c1" in [r["item_id"] for r in result]


SAMPLE_CLAUSES = [
    {"type": "保密", "content": "双方应保守商业秘密", "risk": "low"},
    {"type": "违约责任", "content": "违约方应赔偿损失", "risk": "medium"},
]


class TestSupervisorAgent:
    """SupervisorAgent 单元测试。"""

    def test_validate_consensus_no_disagreement(self):
        from harness.agent.multi_agent.supervisor import SupervisorAgent

        outputs = {
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low"},
                    {"clause_type": "违约责任", "risk_level": "medium"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[
                    {"clause_index": 0, "status": True},
                    {"clause_index": 1, "status": True},
                ],
            ),
        }
        supervisor = SupervisorAgent()
        disagreements = supervisor.validate_consensus(outputs)
        assert len(disagreements) == 0

    def test_validate_consensus_finds_disagreement(self):
        from harness.agent.multi_agent.supervisor import SupervisorAgent

        outputs = {
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[
                    {"clause_index": 0, "status": True, "risk_level_note": "high"},
                ],
            ),
        }
        supervisor = SupervisorAgent()
        disagreements = supervisor.validate_consensus(outputs)
        assert len(disagreements) > 0

    def test_synthesize_report_returns_report(self, sample_document):
        from harness.agent.multi_agent.supervisor import SupervisorAgent
        from harness.core.types import ReviewReport

        outputs = {
            "ClauseExpert": WorkerOutput(
                worker_role="ClauseExpert",
                content="条款提取完成",
                structured=SAMPLE_CLAUSES,
            ),
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_type": "保密", "risk_level": "low", "reason": "标准条款"},
                    {"clause_type": "违约责任", "risk_level": "medium", "reason": "赔偿范围模糊"},
                ],
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert",
                content="合规检查完成",
                structured=[],
            ),
        }
        supervisor = SupervisorAgent()
        report = supervisor.synthesize_report(sample_document, outputs, [])
        assert isinstance(report, ReviewReport)
        assert report.document_id == sample_document.id
