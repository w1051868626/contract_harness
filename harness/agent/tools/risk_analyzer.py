from __future__ import annotations

from harness.agent.llm import LLMClient
from harness.agent.prompts import RISK_ANALYSIS_PROMPT
from harness.core.types import Clause, RiskAssessment, RiskLevel


class RiskAnalyzer:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def analyze(self, clause: Clause) -> RiskAssessment:
        prompt = RISK_ANALYSIS_PROMPT.format(
            clause_type=clause.clause_type,
            clause_content=clause.content,
        )
        resp = self._llm.chat([
            {"role": "system",
             "content": "你是一个合同风险分析专家。请严格按照 JSON 格式输出结果。"},
            {"role": "user", "content": prompt},
        ])
        return self._parse_response(resp.content, clause)

    def _parse_response(self, content: str, clause: Clause) -> RiskAssessment:
        import json
        import re

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
