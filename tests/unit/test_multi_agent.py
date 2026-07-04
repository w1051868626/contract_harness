"""Multi-Agent 数据模型测试。"""

from __future__ import annotations

import json

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMResponse
from harness.agent.multi_agent.coordinator import MultiAgentCoordinator
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

        # 两个 Worker 用同一 clause_index 对齐到同一条款，风险等级不同 → 分歧
        outputs = {
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert",
                content="风险分析完成",
                structured=[
                    {"clause_index": 0, "clause_type": "保密", "risk_level": "low"},
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

    def test_validate_consensus_no_false_disagreement_on_misaligned_index(self):
        """RiskExpert 与 ComplianceExpert 输出顺序/数量不一致时，
        不应按下标强行对齐产出假分歧（R5 回归）。

        旧实现用 range(max(len)) 下标对齐：RiskExpert[0]=type:保密/low
        与 ComplianceExpert[0]=idx:0/high 会被错位比较判为分歧。
        新实现按 key 分组，type:保密 与 idx:0 是不同 key 不比较 → 无分歧。
        """
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
        assert disagreements == []

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


class TestMultiAgentCoordinator:
    """MultiAgentCoordinator 集成测试。"""

    def test_run_returns_report_and_session(self, sample_document):
        """全流程应返回报告和会话。"""
        llm = MockLLMClient(
            [
                LLMResponse(
                    content=json.dumps(
                        [
                            {"type": "保密", "content": "双方应保守商业秘密"},
                            {"type": "违约责任", "content": "违约方应赔偿损失"},
                        ],
                        ensure_ascii=False,
                    ),
                    model="mock",
                ),
                LLMResponse(
                    content=json.dumps(
                        [
                            {
                                "clause_type": "保密",
                                "risk_level": "low",
                                "reason": "标准",
                                "suggestion": "",
                            },
                            {
                                "clause_type": "违约责任",
                                "risk_level": "medium",
                                "reason": "模糊",
                                "suggestion": "明确",
                            },
                        ],
                        ensure_ascii=False,
                    ),
                    model="mock",
                ),
                LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
                # 3 次交叉验证 LLM 调用
                LLMResponse(content="无分歧", model="mock"),
                LLMResponse(content="无分歧", model="mock"),
                LLMResponse(content="无分歧", model="mock"),
            ]
        )
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report.document_id == sample_document.id
        assert session.session_id
        assert len(session.steps) >= 3

    def test_run_with_worker_failure_falls_back(self, sample_document):
        """单个 Worker 失败时不影响整体流程。"""
        llm = MockLLMClient(
            [
                LLMResponse(
                    content=json.dumps(
                        [
                            {"type": "保密", "content": "保密内容"},
                        ],
                        ensure_ascii=False,
                    ),
                    model="mock",
                ),
                # RiskExpert 返回空（模拟失败）
                LLMResponse(content="{}", model="mock"),
                LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
                # 交叉验证（跳过失败的 RiskExpert）
                LLMResponse(content="ok", model="mock"),
            ]
        )
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report.document_id == sample_document.id
        assert report.summary

    def test_all_workers_fail_fallsback_to_pipeline(self, sample_document):
        """全部 Worker 失败时降级到 Pipeline。"""
        llm = MockLLMClient(
            [
                LLMResponse(content="{}", model="mock"),
            ]
        )
        coordinator = MultiAgentCoordinator(llm=llm)
        report, session = coordinator.run(sample_document)
        assert report is not None


class TestContractAgentMultiAgent:
    """ContractAgent Multi-Agent 模式集成测试。"""

    def test_multi_agent_mode_dispatch(self, sample_document):
        """MULTI_AGENT 模式应正确分发到 MultiAgentCoordinator。"""
        llm = MockLLMClient(
            [
                LLMResponse(
                    content=json.dumps(
                        [
                            {"type": "保密", "content": "双方应保守商业秘密"},
                            {"type": "违约责任", "content": "违约方应赔偿损失"},
                        ],
                        ensure_ascii=False,
                    ),
                    model="mock",
                ),
                LLMResponse(
                    content=json.dumps(
                        [
                            {
                                "clause_type": "保密",
                                "risk_level": "low",
                                "reason": "标准",
                                "suggestion": "",
                            },
                            {
                                "clause_type": "违约责任",
                                "risk_level": "medium",
                                "reason": "模糊",
                                "suggestion": "明确",
                            },
                        ],
                        ensure_ascii=False,
                    ),
                    model="mock",
                ),
                LLMResponse(content=json.dumps([], ensure_ascii=False), model="mock"),
                LLMResponse(content="无分歧", model="mock"),
                LLMResponse(content="无分歧", model="mock"),
                LLMResponse(content="无分歧", model="mock"),
            ]
        )
        agent = ContractAgent(llm=llm, mode=AgentMode.MULTI_AGENT)
        report, session = agent.review(sample_document)
        assert report.document_id == sample_document.id
        assert session.session_id


class TestSynthesizeReportAlignment:
    """``synthesize_report`` 条款对齐回归测试（R3）。"""

    def test_risk_aligned_by_clause_type_not_index(self, sample_document):
        """RiskExpert.structured 与 ClauseExpert 数量/顺序不一致时，
        应按 clause_type 匹配而非下标强行对齐，避免 risk 挂错条款。
        """
        from harness.agent.multi_agent.supervisor import SupervisorAgent

        # ClauseExpert 提取 3 个条款，RiskExpert 只评估 2 个且顺序打乱
        clauses = [
            {"type": "保密", "content": "保密条款"},
            {"type": "违约责任", "content": "违约条款"},
            {"type": "管辖", "content": "管辖条款"},
        ]
        risk_struct = [
            # 故意把"违约责任"放在第 0 位，若按下标对齐会错误挂到"保密"
            {"clause_type": "违约责任", "risk_level": "high", "reason": "赔偿过高"},
            {"clause_type": "保密", "risk_level": "low", "reason": "标准条款"},
        ]
        outputs = {
            "ClauseExpert": WorkerOutput(
                worker_role="ClauseExpert", content="", structured=clauses
            ),
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert", content="", structured=risk_struct
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert", content="", structured=[]
            ),
        }
        report = SupervisorAgent().synthesize_report(sample_document, outputs, [])
        # 按 clause_type 找到对应 risk
        risk_by_type = {r.clause.clause_type: r for r in report.risks}
        assert "违约责任" in risk_by_type
        assert risk_by_type["违约责任"].risk_level.value == "high"
        assert risk_by_type["违约责任"].clause.content == "违约条款"
        assert "保密" in risk_by_type
        assert risk_by_type["保密"].risk_level.value == "low"
        assert risk_by_type["保密"].clause.content == "保密条款"

    def test_risk_aligned_by_explicit_clause_index(self, sample_document):
        """RiskExpert.structured 提供 clause_index 时优先用 index 对齐。"""
        from harness.agent.multi_agent.supervisor import SupervisorAgent

        clauses = [
            {"type": "保密", "content": "保密A"},
            {"type": "保密", "content": "保密B"},
        ]
        # 两个同 type 条款，必须靠 clause_index 区分
        risk_struct = [
            {"clause_index": 1, "clause_type": "保密", "risk_level": "high", "reason": "B 风险高"},
            {"clause_index": 0, "clause_type": "保密", "risk_level": "low", "reason": "A 标准"},
        ]
        outputs = {
            "ClauseExpert": WorkerOutput(
                worker_role="ClauseExpert", content="", structured=clauses
            ),
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert", content="", structured=risk_struct
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert", content="", structured=[]
            ),
        }
        report = SupervisorAgent().synthesize_report(sample_document, outputs, [])
        assert len(report.risks) == 2
        assert report.risks[0].clause.content == "保密B"
        assert report.risks[0].risk_level.value == "high"
        assert report.risks[1].clause.content == "保密A"
        assert report.risks[1].risk_level.value == "low"

    def test_risk_same_type_multiple_clauses_rotates(self, sample_document):
        """同 type 多个条款、RiskExpert 未提供 clause_index 时应轮转消费，
        而非全部对齐到第一个（R4 回归，对齐 scorer 行为）。
        """
        from harness.agent.multi_agent.supervisor import SupervisorAgent

        clauses = [
            {"type": "保密", "content": "保密A"},
            {"type": "保密", "content": "保密B"},
        ]
        # 两条 risk 都用 clause_type="保密" 回退，应分别落到 A 和 B
        risk_struct = [
            {"clause_type": "保密", "risk_level": "high", "reason": "A 风险高"},
            {"clause_type": "保密", "risk_level": "critical", "reason": "B 风险极高"},
        ]
        outputs = {
            "ClauseExpert": WorkerOutput(
                worker_role="ClauseExpert", content="", structured=clauses
            ),
            "RiskExpert": WorkerOutput(
                worker_role="RiskExpert", content="", structured=risk_struct
            ),
            "ComplianceExpert": WorkerOutput(
                worker_role="ComplianceExpert", content="", structured=[]
            ),
        }
        report = SupervisorAgent().synthesize_report(sample_document, outputs, [])
        assert len(report.risks) == 2
        assert report.risks[0].clause.content == "保密A"
        assert report.risks[0].risk_level.value == "high"
        assert report.risks[1].clause.content == "保密B"
        assert report.risks[1].risk_level.value == "critical"
