"""基于 LLM 的风险分析工具，评估合同条款的风险等级并提供建议。"""

from __future__ import annotations

import json
import re

from harness.agent.llm import LLMClient
from harness.agent.prompts import RISK_ANALYSIS_PROMPT
from harness.core.types import Clause, RiskAssessment, RiskLevel


class RiskAnalyzer:
    """风险分析器，调用 LLM 评估条款风险等级及给出修改建议。"""

    def __init__(self, llm: LLMClient):
        """注入 LLM 客户端。"""
        self._llm = llm

    def analyze(self, clause: Clause) -> RiskAssessment:
        """对单一条款进行风险分析。"""
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
        return self._parse_response(resp.content, clause)

    def _parse_response(self, content: str, clause: Clause) -> RiskAssessment:
        """解析 LLM 返回的 JSON 响应为 RiskAssessment 对象。"""
        json_str = re.search(r"\{.*?\}", content, re.DOTALL)
        if not json_str:
            return RiskAssessment(
                clause=clause,
                risk_level=RiskLevel.INFO,
                reason="无法解析风险分析结果",
            )

        try:
            data = json.loads(json_str.group())
            return RiskAssessment(
                clause=clause,
                risk_level=RiskLevel(data.get("risk_level", "info")),
                reason=data.get("reason", ""),
                suggestion=data.get("suggestion", ""),
            )
        except (json.JSONDecodeError, ValueError):
            return RiskAssessment(
                clause=clause,
                risk_level=RiskLevel.INFO,
                reason="解析失败",
            )
