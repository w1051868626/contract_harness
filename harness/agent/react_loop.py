"""ReAct 循环模式：LLM 自主决策调用工具，支持推理轨迹记录。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from harness.agent.llm import LLMClient, LLMResponse
from harness.agent.memory import MemoryStore
from harness.agent.prompts import SYSTEM_PROMPT
from harness.agent.tools.clause_extractor import ClauseExtractor
from harness.agent.tools.compliance_checker import ComplianceChecker
from harness.agent.tools.knowledge_retriever import KnowledgeRetriever
from harness.agent.tools.risk_analyzer import RiskAnalyzer
from harness.core.types import (
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
from harness.utils.log import logger

REACT_SYSTEM_PROMPT = """你是一位资深法律合同审查专家。你有以下工具可用：

1. **knowledge_retriever**: 检索法律知识库，输入 {{"query": "检索关键词"}}
2. **clause_extractor**: 从合同中提取所有关键条款，无需输入
3. **risk_analyzer**: 分析已提取条款的风险等级，无需输入
4. **compliance_checker**: 检查条款合规性，无需输入

请一步步推理，每次调用一个工具。输出 JSON 格式：
{{"thought": "你的推理过程", "action": "工具名", "action_input": {{...}} }}

完成后输出 FINAL_ANSWER。"""

MAX_REACT_ITERATIONS = 15


class ReActLoop:
    """ReAct 循环：LLM 自主决策调用工具，逐步完成合同审查。"""

    def __init__(
        self,
        llm: LLMClient,
        clause_extractor: ClauseExtractor,
        risk_analyzer: RiskAnalyzer,
        compliance_checker: ComplianceChecker,
        knowledge_retriever: KnowledgeRetriever | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self._llm = llm
        self._clause_extractor = clause_extractor
        self._risk_analyzer = risk_analyzer
        self._compliance_checker = compliance_checker
        self._knowledge_retriever = knowledge_retriever
        self._memory = memory_store

    def run(self, document: ContractDocument) -> tuple[ReviewReport, AgentSession]:
        """执行 ReAct 循环审查，返回报告和会话。"""
        session = AgentSession(
            session_id=uuid.uuid4().hex[:12],
            document=document,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Starting ReAct review for document_id={}", document.id)

        state: dict[str, Any] = {
            "document": document,
            "clauses": [],
            "risks": [],
            "compliance": [],
            "kb_context": "",
        }
        history: list[dict[str, str]] = []

        for iteration in range(MAX_REACT_ITERATIONS):
            messages = self._build_messages(document, state, history)
            resp = self._llm.chat(messages)
            action = self._parse_action(resp)

            if action["type"] == "final_answer":
                logger.info("ReAct loop finished after {} iterations", iteration + 1)
                break

            step, observation = self._execute_tool(action, state, session)
            session.steps.append(step)
            history.append({"role": "user", "content": str(observation)})

            if iteration == MAX_REACT_ITERATIONS - 1:
                logger.warning("ReAct loop reached max iterations ({})", MAX_REACT_ITERATIONS)

        report = self._build_report(document, state)
        session.report = report
        session.finished_at = datetime.now(timezone.utc).isoformat()

        if self._memory and self._memory.enabled:
            self._memory.remember_session(
                clauses=state["clauses"],
                risks=state["risks"],
                compliance=state["compliance"],
                session_id=session.session_id,
            )

        logger.info("ReAct review completed for document_id={}", document.id)
        return report, session

    def _build_messages(
        self, document: ContractDocument, state: dict[str, Any], history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """构建 ReAct 对话消息。"""
        kb_available = self._knowledge_retriever and self._knowledge_retriever.available
        kb_status = "可用" if kb_available else "不可用"
        clause_status = f"已提取 {len(state['clauses'])} 个条款"
        risk_status = f"已完成 {len(state['risks'])} 项风险评估"
        compliance_status = f"已完成 {len(state['compliance'])} 项合规检查"

        context = (
            f"合同标题: {document.title}\n"
            f"合同内容（前500字）: {document.content[:500]}...\n\n"
            "当前状态:\n"
            f"- 知识库: {kb_status}\n"
            f"- {clause_status}\n"
            f"- {risk_status}\n"
            f"- {compliance_status}\n"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"请审查以下合同:\n\n{context}"},
        ]
        for h in history:
            messages.append(h)
        return messages

    def _parse_action(self, resp: LLMResponse) -> dict[str, Any]:
        """解析 LLM 响应为 action dict。"""
        content = resp.content.strip()
        content = content.removeprefix("```json").removesuffix("```").strip()

        if "FINAL_ANSWER" in content or '"final_answer"' in content:
            try:
                parsed = json.loads(content) if content.startswith("{") else {}
                if parsed.get("final_answer"):
                    return {"type": "final_answer"}
            except (json.JSONDecodeError, ValueError):
                pass
            return {"type": "final_answer"}

        try:
            parsed = json.loads(content)
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            if action_name:
                return {"type": "action", "tool": action_name, "input": action_input}
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("Could not parse ReAct action from: {}", content[:200])
        return {"type": "action", "tool": "clause_extractor", "input": {}}

    def _execute_tool(
        self, action: dict[str, Any], state: dict[str, Any], session: AgentSession
    ) -> tuple[AgentStep, str]:
        """执行工具调用并更新状态。"""
        tool = action.get("tool", "")
        step = AgentStep(
            step_index=len(session.steps),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        tc = ToolCall(
            tool_name=tool,
            input=action.get("input", {}),
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            if tool == "knowledge_retriever":
                query = action.get("input", {}).get("query", state["document"].content[:500])
                kr = self._knowledge_retriever
                result = kr.retrieve(query) if kr else ""
                state["kb_context"] = result
                observation = f"知识库检索结果: {result[:500]}"
            elif tool == "clause_extractor":
                clauses = self._clause_extractor.extract(state["document"])
                state["clauses"] = clauses
                observation = f"提取到 {len(clauses)} 个条款:\n"
                for i, c in enumerate(clauses):
                    observation += f"  [{i}] {c.clause_type}: {c.content[:200]}\n"
            elif tool == "risk_analyzer":
                if not state["clauses"]:
                    observation = "错误: 尚未提取条款，请先调用 clause_extractor"
                else:
                    mem_ctx = self._load_memory_context(state["clauses"]) if self._memory else ""
                    risks = self._risk_analyzer.batch_analyze(
                        state["clauses"], memory_context=mem_ctx
                    )
                    state["risks"] = risks
                    observation = "风险分析完成:\n"
                    for r in risks:
                        obs_line = (
                            f"  [{r.clause.clause_type}] {r.risk_level.value}: {r.reason[:200]}\n"
                        )
                        observation += obs_line
            elif tool == "compliance_checker":
                if not state["clauses"]:
                    observation = "错误: 尚未提取条款，请先调用 clause_extractor"
                else:
                    mem_ctx = self._load_memory_context(state["clauses"]) if self._memory else ""
                    results = self._compliance_checker.batch_check(
                        state["clauses"], memory_context=mem_ctx
                    )
                    state["compliance"] = results
                    total = sum(len(r) for r in results)
                    observation = f"合规检查完成，共 {total} 项检查:\n"
                    for i, checks in enumerate(results):
                        for c in checks:
                            status = "✅" if c.status else "❌"
                            observation += f"  [{i}] {c.regulation}: {status} {c.detail[:100]}\n"
            else:
                known = "knowledge_retriever, clause_extractor, risk_analyzer, compliance_checker"
                observation = f"未知工具: {tool}，可用: {known}"

            tc.output = observation
        except Exception as e:
            observation = f"工具执行出错: {e}"
            tc.output = str(e)

        tc.finished_at = datetime.now(timezone.utc).isoformat()
        step.tool_calls.append(tc)
        step.agent_message = f"调用工具: {tool}"
        return step, observation

    def _load_memory_context(self, clauses: list[Clause]) -> str:
        """加载历史记忆作为上下文。"""
        if not self._memory:
            return ""
        memories = []
        seen: set[str] = set()
        for c in clauses:
            for m in self._memory.recall(c.content, top_k=2):
                if m.clause_content not in seen:
                    seen.add(m.clause_content)
                    memories.append(m)
        return self._memory.format_memory_context(memories) if memories else ""

    def _build_report(self, document: ContractDocument, state: dict[str, Any]) -> ReviewReport:
        """从状态构建 ReviewReport。"""
        all_risks = state.get("risks", [])
        all_compliance: list[ComplianceCheck] = []
        for checks in state.get("compliance", []):
            if isinstance(checks, list):
                all_compliance.extend(checks)

        overall_risk = self._compute_overall_risk(all_risks)

        clauses_summary = f"共发现 {len(state['clauses'])} 个条款"
        high_risks = [r for r in all_risks if r.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
        risks_summary = f"高风险项: {len(high_risks)} 个"
        non_compliant = [c for c in all_compliance if not c.status]
        compliance_summary = f"不合规项: {len(non_compliant)} 个"

        summary_prompt = (
            f"请根据以下审查结果，生成合同审查报告摘要：\n\n"
            f"条款提取结果：{clauses_summary}\n"
            f"风险评估结果：{risks_summary}\n"
            f"合规检查结果：{compliance_summary}\n\n"
            f"请生成：\n1. 整体风险评级\n2. 核心风险总结（3-5点）\n3. 总体修改建议"
        )
        kb_section = f"\n\n## 知识库参考\n{state['kb_context']}" if state.get("kb_context") else ""
        resp = self._llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": summary_prompt + kb_section},
            ]
        )
        summary = resp.content

        return ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            clauses=state.get("clauses", []),
            risks=all_risks,
            compliance_checks=all_compliance,
            overall_risk=overall_risk,
        )

    @staticmethod
    def _compute_overall_risk(risks: list[RiskAssessment]) -> RiskLevel:
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
