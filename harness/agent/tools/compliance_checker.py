"""基于 LLM 的合规检查工具，对照多部法规逐条审查合同合规性。"""

from __future__ import annotations

from harness.agent.llm import LLMClient
from harness.agent.tools.llm_utils import extract_json_object
from harness.core.types import Clause, ComplianceCheck
from harness.utils.log import logger

_BATCH_COMPLIANCE_PROMPT = """请对照以下法律法规，批量检查合同条款是否合规，
返回 JSON 数组，每个元素格式为
{{"index": ..., "regulation": "...", "status": true/false, "detail": "..."}}。

合同条款内容：
{clause_content}

法律法规列表：
{regulations}"""


class ComplianceChecker:
    """合规检查器，逐条对照法规库对条款进行合规审查。"""

    REGULATIONS = [
        "中华人民共和国民法典（合同编）",
        "中华人民共和国劳动合同法",
        "中华人民共和国数据安全法",
        "中华人民共和国个人信息保护法",
        "中华人民共和国反垄断法",
    ]

    def __init__(self, llm: LLMClient):
        """注入 LLM 客户端。"""
        self._llm = llm

    def check(self, clause: Clause) -> list[ComplianceCheck]:
        """对单一条款执行多法规合规检查。"""
        return self.batch_check([clause])[0] if clause else []

    def batch_check(self, clauses: list[Clause]) -> list[list[ComplianceCheck]]:
        """批量对多个条款执行合规检查，每个条款一次 LLM 调用完成所有法规检查。"""
        if not clauses:
            return []

        all_results: list[list[ComplianceCheck]] = []
        for clause in clauses:
            logger.debug("Checking compliance for clause_type={}", clause.clause_type)
            prompt = _BATCH_COMPLIANCE_PROMPT.format(
                clause_content=clause.content,
                regulations="\n".join(f"{i + 1}. {r}" for i, r in enumerate(self.REGULATIONS)),
            )
            resp = self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": "你是一个法律合规专家。请严格按照 JSON 数组格式输出结果。",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            results = self._parse_batch_response(resp.content)
            all_results.append(results)
            logger.debug(
                "Completed {} compliance checks for clause_type={}",
                len(results),
                clause.clause_type,
            )
        return all_results

    def _parse_batch_response(self, content: str) -> list[ComplianceCheck]:
        """解析批量合规检查的 LLM 响应。"""
        import json as _json

        raw = content.strip()
        try:
            parsed = _json.loads(raw) if raw.startswith("[") else _json.loads(f"[{raw}]")
        except (_json.JSONDecodeError, ValueError):
            parsed = []
        if not isinstance(parsed, list):
            parsed = []

        results: list[ComplianceCheck] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            results.append(
                ComplianceCheck(
                    regulation=item.get("regulation", ""),
                    status=bool(item.get("status", True)),
                    detail=item.get("detail", ""),
                )
            )

        if not results:
            for reg in self.REGULATIONS:
                results.append(ComplianceCheck(regulation=reg, status=True, detail="无法解析"))
        return results

    def _parse_response(self, content: str, regulation: str) -> ComplianceCheck:
        """解析 LLM 返回的 JSON 响应为 ComplianceCheck 对象。"""
        data = extract_json_object(content)
        if data:
            return ComplianceCheck(
                regulation=regulation,
                status=data.get("status", True),
                detail=data.get("detail", ""),
            )
        return ComplianceCheck(regulation=regulation, status=True, detail="无法解析")
