from __future__ import annotations

from harness.agent.llm import LLMClient
from harness.agent.prompts import CLAUSE_EXTRACT_PROMPT
from harness.core.types import Clause, ContractDocument, RiskLevel


class ClauseExtractor:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def extract(self, document: ContractDocument) -> list[Clause]:
        prompt = CLAUSE_EXTRACT_PROMPT.format(contract_content=document.content)
        resp = self._llm.chat([
            {"role": "system",
             "content": "你是一个合同条款提取专家。请严格按照 JSON 格式输出结果。"},
            {"role": "user", "content": prompt},
        ])
        return self._parse_response(resp.content)

    def _parse_response(self, content: str) -> list[Clause]:
        import json
        import re

        json_str = re.search(r"\[.*?\]", content, re.DOTALL)
        if not json_str:
            return []

        try:
            data = json.loads(json_str.group())
            return [Clause(
                clause_type=item.get("type", "unknown"),
                content=item.get("content", ""),
                start_pos=item.get("start_pos"),
                end_pos=item.get("end_pos"),
                risk=RiskLevel(item.get("risk", "info")),
                comment=item.get("comment", ""),
            ) for item in data]
        except (json.JSONDecodeError, ValueError):
            return []
