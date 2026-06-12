from __future__ import annotations

import uuid
from datetime import datetime, timezone

from harness.agent.llm import LLMClient
from harness.agent.prompts import REVIEW_SUMMARY_PROMPT, SYSTEM_PROMPT
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import (
    AgentSession,
    AgentStep,
    ContractDocument,
    ReviewReport,
    RiskLevel,
    ToolCall,
)


class ContractAgent:
    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()
        self._clause_extractor = ClauseExtractor(self._llm)
        self._risk_analyzer = RiskAnalyzer(self._llm)
        self._compliance_checker = ComplianceChecker(self._llm)

    def review(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        session = AgentSession(
            session_id=uuid.uuid4().hex[:12],
            document=document,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: 条款提取
        step1 = AgentStep(step_index=1, timestamp=datetime.now(timezone.utc).isoformat())
        step1.agent_message = "正在提取合同条款..."
        tc = ToolCall(
            tool_name="clause_extractor",
            input={"document_id": document.id},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        clauses = self._clause_extractor.extract(document)
        tc.output = [c.__dict__ for c in clauses]
        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step1.tool_calls.append(tc)
        session.steps.append(step1)

        # Step 2: 风险分析
        step2 = AgentStep(step_index=2, timestamp=datetime.now(timezone.utc).isoformat())
        step2.agent_message = "正在进行风险分析..."
        risks = []
        for clause in clauses:
            tc = ToolCall(
                tool_name="risk_analyzer",
                input={"clause_type": clause.clause_type},
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            risk = self._risk_analyzer.analyze(clause)
            tc.output = risk.__dict__
            tc.finished_at = datetime.now(timezone.utc).isoformat()
            step2.tool_calls.append(tc)
            risks.append(risk)
        session.steps.append(step2)

        # Step 3: 合规检查
        step3 = AgentStep(step_index=3, timestamp=datetime.now(timezone.utc).isoformat())
        step3.agent_message = "正在进行合规检查..."
        all_compliance = []
        for clause in clauses:
            tc = ToolCall(
                tool_name="compliance_checker",
                input={"clause_type": clause.clause_type},
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            checks = self._compliance_checker.check(clause)
            tc.output = [c.__dict__ for c in checks]
            tc.finished_at = datetime.now(timezone.utc).isoformat()
            step3.tool_calls.append(tc)
            all_compliance.extend(checks)
        session.steps.append(step3)

        # Step 4: 生成报告摘要
        step4 = AgentStep(step_index=4, timestamp=datetime.now(timezone.utc).isoformat())
        step4.agent_message = "正在生成审查报告..."
        overall_risk = self._compute_overall_risk(risks)
        summary = self._generate_summary(clauses, risks, all_compliance)
        session.steps.append(step4)

        report = ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            clauses=clauses,
            risks=risks,
            compliance_checks=all_compliance,
            overall_risk=overall_risk,
        )
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()

        return report, session

    def _compute_overall_risk(self, risks: list) -> RiskLevel:
        if not risks:
            return RiskLevel.INFO
        levels = [r.risk_level for r in risks]
        if RiskLevel.CRITICAL in levels:
            return RiskLevel.CRITICAL
        if RiskLevel.HIGH in levels:
            return RiskLevel.HIGH
        if RiskLevel.MEDIUM in levels:
            return RiskLevel.MEDIUM
        if RiskLevel.LOW in levels:
            return RiskLevel.LOW
        return RiskLevel.INFO

    def _generate_summary(self, clauses, risks, compliance) -> str:
        clauses_summary = f"共发现 {len(clauses)} 个条款"
        high_risks = [r for r in risks if r.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
        risks_summary = f"高风险项: {len(high_risks)} 个"
        non_compliant = [c for c in compliance if not c.status]
        compliance_summary = f"不合规项: {len(non_compliant)} 个"

        prompt = REVIEW_SUMMARY_PROMPT.format(
            clauses_summary=clauses_summary,
            risks_summary=risks_summary,
            compliance_summary=compliance_summary,
        )
        resp = self._llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return resp.content
