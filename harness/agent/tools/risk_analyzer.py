"""基于 LLM 的风险分析工具，评估合同条款的风险等级并提供建议。"""

from __future__ import annotations

from harness.agent.llm import LLMClient
from harness.agent.prompts import RISK_ANALYSIS_PROMPT
from harness.agent.tools.llm_utils import extract_json_object
from harness.core.types import Clause, RiskAssessment, RiskLevel
from harness.utils.log import logger

_BATCH_RISK_PROMPT = """请对以下合同条款批量进行风险分析，返回 JSON 数组，
每个元素格式为
{{"index": ..., "risk_level": "...", "reason": "...", "suggestion": "..."}}。

{clause_sections}"""


class RiskAnalyzer:
    """风险分析器，调用 LLM 评估条款风险等级及给出修改建议。"""

    def __init__(self, llm: LLMClient):
        """注入 LLM 客户端。"""
        self._llm = llm

    def analyze(self, clause: Clause) -> RiskAssessment:
        """对单一条款进行风险分析。"""
        logger.info("Analyzing risk for clause_type={}", clause.clause_type)
        prompt = RISK_ANALYSIS_PROMPT.format(
            clause_type=clause.clause_type,
            clause_content=clause.content,
        )
        resp = self._llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是一个合同风险分析专家。请严格按照 JSON 格式输出结果。",
                },
                {"role": "user", "content": prompt},
            ]
        )
        result = self._parse_response(resp.content, clause)
        logger.debug("Risk level for {}: {}", clause.clause_type, result.risk_level.value)
        return result

    def batch_analyze(self, clauses: list[Clause]) -> list[RiskAssessment]:
        """批量分析多个条款的风险，单次 LLM 调用完成。"""
        if not clauses:
            return []

        logger.info("Batch analyzing risk for {} clauses", len(clauses))
        sections = []
        for i, c in enumerate(clauses):
            sections.append(f"--- 条款 {i} ---\n类型: {c.clause_type}\n内容: {c.content}")
        prompt = _BATCH_RISK_PROMPT.format(clause_sections="\n\n".join(sections))

        resp = self._llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是一个合同风险分析专家。请严格按照 JSON 数组格式输出结果。",
                },
                {"role": "user", "content": prompt},
            ]
        )

        results = self._parse_batch_response(resp.content, clauses)
        logger.info("Batch risk analysis completed: {} results", len(results))
        return results

    def _parse_batch_response(self, content: str, clauses: list[Clause]) -> list[RiskAssessment]:
        """解析批量风险分析的 LLM 响应。"""
        import json as _json

        raw = content.strip()
        raw = _json.loads(f"[{raw}]" if not raw.startswith("[") else raw) if raw else []
        if not isinstance(raw, list):
            raw = []
        items = {item.get("index"): item for item in raw if isinstance(item, dict)}

        results: list[RiskAssessment] = []
        for i, clause in enumerate(clauses):
            item = items.get(i)
            if item:
                results.append(
                    RiskAssessment(
                        clause=clause,
                        risk_level=RiskLevel(item.get("risk_level", "info")),
                        reason=item.get("reason", ""),
                        suggestion=item.get("suggestion", ""),
                    )
                )
            else:
                results.append(self.analyze(clause))
        return results

    def _parse_response(self, content: str, clause: Clause) -> RiskAssessment:
        """解析 LLM 返回的 JSON 响应为 RiskAssessment 对象。"""
        data = extract_json_object(content)
        if data:
            return RiskAssessment(
                clause=clause,
                risk_level=RiskLevel(data.get("risk_level", "info")),
                reason=data.get("reason", ""),
                suggestion=data.get("suggestion", ""),
            )
        return RiskAssessment(
            clause=clause,
            risk_level=RiskLevel.INFO,
            reason="无法解析风险分析结果",
        )
