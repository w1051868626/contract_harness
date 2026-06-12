from __future__ import annotations

from harness.agent.llm import LLMClient
from harness.agent.prompts import COMPLIANCE_CHECK_PROMPT
from harness.core.types import Clause, ComplianceCheck


class ComplianceChecker:
    REGULATIONS = [
        "中华人民共和国民法典（合同编）",
        "中华人民共和国劳动合同法",
        "中华人民共和国数据安全法",
        "中华人民共和国个人信息保护法",
        "中华人民共和国反垄断法",
    ]

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def check(self, clause: Clause) -> list[ComplianceCheck]:
        results: list[ComplianceCheck] = []
        for regulation in self.REGULATIONS:
            prompt = COMPLIANCE_CHECK_PROMPT.format(
                topic=regulation,
                clause_content=clause.content,
            )
            resp = self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": "你是一个法律合规专家。请严格按照 JSON 格式输出结果。",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            results.append(self._parse_response(resp.content, regulation))
        return results

    def _parse_response(self, content: str, regulation: str) -> ComplianceCheck:
        import json
        import re

        json_str = re.search(r"\{.*?\}", content, re.DOTALL)
        if not json_str:
            return ComplianceCheck(regulation=regulation, status=True, detail="无法解析")

        try:
            data = json.loads(json_str.group())
            return ComplianceCheck(
                regulation=regulation,
                status=data.get("status", True),
                detail=data.get("detail", ""),
            )
        except (json.JSONDecodeError, ValueError):
            return ComplianceCheck(regulation=regulation, status=True, detail="解析失败")
