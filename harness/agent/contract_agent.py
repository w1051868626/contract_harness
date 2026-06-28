"""合同审查 Agent 主流程编排，协调各工具完成条款提取、风险分析、合规检查及报告生成。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from harness.agent.llm import LLMClient
from harness.agent.memory import MemoryStore
from harness.agent.prompts import REVIEW_SUMMARY_PROMPT, SYSTEM_PROMPT
from harness.agent.react_loop import ReActLoop
from harness.agent.reflection import reflect_on_report
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.knowledge_retriever import KnowledgeRetriever
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import (
    AgentMode,
    AgentSession,
    AgentStep,
    Clause,
    ComplianceCheck,
    ContractDocument,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
    ToolCall,
)
from harness.replay.storage import ReplayStorage
from harness.utils.log import logger

if TYPE_CHECKING:
    from harness.rag.knowledge_base import KnowledgeBase

CONVERSE_PROMPT = """你是一位资深法律合同审查专家。以下是之前对一份合同的审查报告：

## 审查摘要
{summary}

## 整体风险
{overall_risk}

## 条款清单
{clauses_section}

## 风险分析
{risks_section}

## 合规检查
{compliance_section}

用户现在有一个后续问题，请结合以上审查结果回答：

{query}"""


class ContractAgent:
    """合同审查 Agent，串联 LLM 与各分析工具执行完整审查流程。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        knowledge_base: KnowledgeBase | None = None,
        memory_store: MemoryStore | None = None,
        mode: AgentMode = AgentMode.PIPELINE,
    ):
        """初始化 Agent，注入 LLM 客户端与知识库实例。"""
        self._llm = llm or LLMClient()
        self._kb = knowledge_base
        self._memory = memory_store
        self._mode = mode
        self._clause_extractor = ClauseExtractor(self._llm)
        self._risk_analyzer = RiskAnalyzer(self._llm)
        self._compliance_checker = ComplianceChecker(self._llm)
        self._knowledge_retriever = KnowledgeRetriever(knowledge_base)

    def review(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """执行合同审查全流程，返回审查报告和会话记录。"""
        if self._mode == AgentMode.REACT:
            return self._review_react(document)
        if self._mode == AgentMode.REFLECTION:
            return self._review_reflection(document)
        return self._review_pipeline(document)

    def _review_react(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """ReAct 模式：LLM 自主决策调用工具。"""
        loop = ReActLoop(
            llm=self._llm,
            clause_extractor=self._clause_extractor,
            risk_analyzer=self._risk_analyzer,
            compliance_checker=self._compliance_checker,
            knowledge_retriever=self._knowledge_retriever,
            memory_store=self._memory,
        )
        return loop.run(document)

    def _review_reflection(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """Reflection 模式：管道审查 + 自审修正。"""
        report, session = self._review_pipeline(document)
        revised_report = reflect_on_report(self._llm, report)
        session.report = revised_report
        return revised_report, session

    def _review_pipeline(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """管道模式：固定步骤串联执行（默认行为）。"""
        session = AgentSession(
            session_id=uuid.uuid4().hex[:12],
            document=document,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Starting pipeline review for document_id={}", document.id)

        # Step 0: 知识库检索（可选）
        kb_context = ""
        if self._knowledge_retriever.available:
            step0 = AgentStep(step_index=0, timestamp=datetime.now(timezone.utc).isoformat())
            step0.agent_message = "正在检索知识库..."
            tc = ToolCall(
                tool_name="knowledge_retriever",
                input={"query": document.content[:500]},
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            kb_context = self._knowledge_retriever.retrieve(document.content)
            tc.output = kb_context
            tc.finished_at = datetime.now(timezone.utc).isoformat()
            step0.tool_calls.append(tc)
            session.steps.append(step0)

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
        logger.info("Extracted {} clauses from document", len(clauses))

        # 从记忆中检索相似案例作为参考上下文
        mem_risk_ctx = ""
        mem_compliance_ctx = ""
        if self._memory and self._memory.enabled:
            memories: list = []
            seen: set[str] = set()
            for c in clauses:
                for m in self._memory.recall(c.content, top_k=2):
                    if m.clause_content not in seen:
                        seen.add(m.clause_content)
                        memories.append(m)
            if memories:
                mem_risk_ctx = self._memory.format_memory_context(memories)
                mem_compliance_ctx = mem_risk_ctx

        # Step 2: 批量风险分析（单次 LLM 调用）
        step2 = AgentStep(step_index=2, timestamp=datetime.now(timezone.utc).isoformat())
        step2.agent_message = "正在进行风险分析..."
        tc = ToolCall(
            tool_name="risk_analyzer",
            input={"clause_count": len(clauses)},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        risks = self._risk_analyzer.batch_analyze(clauses, memory_context=mem_risk_ctx)
        tc.output = [r.__dict__ for r in risks]
        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step2.tool_calls.append(tc)
        session.steps.append(step2)
        logger.info("Analyzed {} clauses for risk", len(risks))

        # Step 3: 批量合规检查（单次 LLM 调用）
        step3 = AgentStep(step_index=3, timestamp=datetime.now(timezone.utc).isoformat())
        step3.agent_message = "正在进行合规检查..."
        tc = ToolCall(
            tool_name="compliance_checker",
            input={"clause_count": len(clauses)},
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        compliance_results = self._compliance_checker.batch_check(
            clauses, memory_context=mem_compliance_ctx
        )
        all_compliance: list[ComplianceCheck] = []
        for checks in compliance_results:
            all_compliance.extend(checks)
        tc.output = [c.__dict__ for c in all_compliance]
        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step3.tool_calls.append(tc)
        session.steps.append(step3)
        logger.info("Performed {} compliance checks", len(all_compliance))

        # Step 4: 生成报告摘要
        step4 = AgentStep(step_index=4, timestamp=datetime.now(timezone.utc).isoformat())
        step4.agent_message = "正在生成审查报告..."
        overall_risk = self._compute_overall_risk(risks)
        summary = self._generate_summary(clauses, risks, all_compliance, kb_context)
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

        # 将本次审查结果存入持久化记忆
        if self._memory and self._memory.enabled:
            self._memory.remember_session(
                clauses=clauses,
                risks=risks,
                compliance=compliance_results,
                session_id=session.session_id,
            )

        logger.info("Pipeline review completed for document_id={}", document.id)

        return report, session

    def converse(self, session_id: str, query: str, replay_dir: str | None = None) -> str:
        """根据会话 ID 继续对话，回答对上次审查的追问。

        Args:
            session_id: 已有审查会话的 ID。
            query: 用户的后续问题。
            replay_dir: 回放存储目录（测试用），默认使用配置路径。

        Returns:
            Agent 的回答文本。
        """
        storage = ReplayStorage(replay_dir) if replay_dir else ReplayStorage()
        data = storage.load(session_id)
        if not data:
            return f"未找到会话: {session_id}"

        report = data.get("report")
        if not report:
            return "该会话没有审查报告，无法继续对话"

        # 重建报告上下文
        summary = report.get("summary", "无摘要")
        overall_risk = report.get("overall_risk", "info")
        clauses = report.get("clauses", [])
        risks = report.get("risks", [])
        compliance = report.get("compliance_checks", [])

        clauses_section = (
            "\n".join(
                f"- [{c.get('clause_type', '?')}] {c.get('content', '')[:200]}" for c in clauses
            )
            or "无"
        )

        risks_section = (
            "\n".join(f"- [{r.get('risk_level', '?')}] {r.get('reason', '')[:200]}" for r in risks)
            or "无"
        )

        compliance_section = (
            "\n".join(
                f"- {c.get('regulation', '?')}: {'✅合规' if c.get('status') else '❌不合规'}"
                for c in compliance[:10]
            )
            or "无"
        )

        # 读取历史对话记录
        conversation: list[dict[str, str]] = data.get("metadata", {}).get("conversation", [])  # type: ignore[arg-type]

        prompt = CONVERSE_PROMPT.format(
            summary=summary,
            overall_risk=overall_risk,
            clauses_section=clauses_section,
            risks_section=risks_section,
            compliance_section=compliance_section,
            query=query,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "你是一位资深法律合同审查专家。"},
        ]
        # 注入历史对话作为上下文
        for turn in conversation:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        resp = self._llm.chat(messages)
        answer = resp.content

        # 更新对话记录
        conversation.append({"role": "user", "content": query})
        conversation.append({"role": "assistant", "content": answer})
        data.setdefault("metadata", {})["conversation"] = conversation
        storage.save(session_id, data)  # type: ignore[arg-type]

        logger.info("Converse updated: session_id={}, turns={}", session_id, len(conversation) // 2)
        return answer

    def _compute_overall_risk(self, risks: list[RiskAssessment]) -> RiskLevel:
        """根据所有风险项计算综合风险等级。"""
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

    def _generate_summary(
        self,
        clauses: list[Clause],
        risks: list[RiskAssessment],
        compliance: list[ComplianceCheck],
        kb_context: str = "",
    ) -> str:
        """调用 LLM 生成审查报告摘要文本。"""
        clauses_summary = f"共发现 {len(clauses)} 个条款"
        high_risks = [r for r in risks if r.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
        risks_summary = f"高风险项: {len(high_risks)} 个"
        non_compliant = [c for c in compliance if not c.status]
        compliance_summary = f"不合规项: {len(non_compliant)} 个"

        kb_section = f"\n\n## 知识库参考\n{kb_context}" if kb_context else ""
        summary_prompt = REVIEW_SUMMARY_PROMPT.format(
            clauses_summary=clauses_summary,
            risks_summary=risks_summary,
            compliance_summary=compliance_summary,
        )
        user_content = summary_prompt + kb_section
        resp = self._llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        return resp.content
