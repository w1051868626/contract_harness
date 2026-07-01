"""Multi-Agent 协同审查编排器。"""

from __future__ import annotations

from datetime import datetime, timezone

from harness.agent.llm import LLMClient
from harness.agent.memory import MemoryStore
from harness.agent.multi_agent.supervisor import SupervisorAgent
from harness.agent.multi_agent.validator import CrossValidator
from harness.agent.multi_agent.worker import WorkerAgent
from harness.core.types import (
    AgentSession,
    AgentStep,
    ContractDocument,
    ReviewReport,
    ToolCall,
    WorkerOutput,
)
from harness.utils.io import make_id
from harness.utils.log import logger


class MultiAgentCoordinator:
    """编排 Supervisor + Worker + CrossValidator 多 Agent 审查流程。"""

    def __init__(
        self,
        llm: LLMClient,
        memory_store: MemoryStore | None = None,
    ):
        self._llm = llm
        self._memory = memory_store
        self._supervisor = SupervisorAgent()
        self._validator = CrossValidator(llm=self._llm)
        self._workers: dict[str, WorkerAgent] = {
            role: WorkerAgent(role=role, llm=self._llm)
            for role in ("ClauseExpert", "RiskExpert", "ComplianceExpert")
        }

    def run(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """执行多 Agent 协同审查。"""
        session = AgentSession(
            session_id=make_id(),
            document=document,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Starting multi-agent review for document_id={}", document.id)

        # Phase 1: 分配任务
        tasks = self._supervisor.assign_tasks(document)

        # Phase 2-3: 串行执行（ClauseExpert → RiskExpert → ComplianceExpert）
        outputs: dict[str, WorkerOutput] = {}
        phase_order = ["ClauseExpert", "RiskExpert", "ComplianceExpert"]

        for phase_idx, role in enumerate(phase_order):
            step = AgentStep(
                step_index=phase_idx,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            step.agent_message = f"正在执行 {role}..."
            tc = ToolCall(
                tool_name=role,
                input={"document_id": document.id},
                started_at=datetime.now(timezone.utc).isoformat(),
            )

            try:
                if role == "ClauseExpert":
                    content = tasks.get(role, document.content)
                else:
                    clause_out = outputs.get("ClauseExpert")
                    if clause_out and clause_out.structured:
                        content = f"合同条款：\n{clause_out.content}"
                    else:
                        content = document.content

                worker_out = self._workers[role].execute(content)
                outputs[role] = worker_out
                tc.output = f"{role} 执行完成"
                logger.info("{} completed successfully", role)
            except Exception as e:
                logger.warning("{} failed: {}", role, e)
                tc.output = str(e)
            finally:
                tc.finished_at = datetime.now(timezone.utc).isoformat()
                step.tool_calls.append(tc)
                session.steps.append(step)

        if not any(out for out in outputs.values()):
            logger.warning("All workers failed, synthesizing partial report")
            return self._build_partial_report(document, session)

        # Phase 4: 交叉验证
        step = AgentStep(
            step_index=len(session.steps),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        step.agent_message = "正在交叉验证..."
        tc = ToolCall(
            tool_name="cross_validation",
            input={},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        cross_validation_outputs: dict[str, str] = {}
        for role, worker in self._workers.items():
            if role not in outputs:
                continue
            try:
                peer = {k: v.content for k, v in outputs.items() if k != role}
                if peer:
                    cv_out = worker.execute("确认审查结果", peer_results=peer)
                    cross_validation_outputs[role] = cv_out.content[:500]
            except Exception as e:
                logger.warning("Cross-validation for {} failed: {}", role, e)
        tc.output = "交叉验证完成"
        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step.tool_calls.append(tc)
        session.steps.append(step)
        session.metadata["cross_validation"] = cross_validation_outputs

        # Phase 5: 分歧检测
        disagreements = self._supervisor.validate_consensus({k: v for k, v in outputs.items() if v})

        # Phase 6: 仲裁
        arbitration = self._validator.arbitrate(disagreements)

        # Phase 7: 合成报告
        report = self._supervisor.synthesize_report(document, outputs, arbitration)
        report.reviewed_at = datetime.now(timezone.utc).isoformat()
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()

        if disagreements:
            logger.info("Multi-agent review found {} disagreements", len(disagreements))

        logger.info("Multi-agent review completed for document_id={}", document.id)
        return report, session

    def _build_partial_report(
        self, document: ContractDocument, session: AgentSession
    ) -> tuple[ReviewReport, AgentSession]:
        """全部 Worker 失败时生成简化报告。"""
        report = ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            summary="多 Agent 审查全部失败，请重试或切换为 Pipeline 模式",
            clauses=[],
            risks=[],
            compliance_checks=[],
            overall_risk=None,  # type: ignore[arg-type]
        )
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()
        return report, session
