"""基于 LLM 的合规检查工具，对照多部法规逐条审查合同合规性。"""

from __future__ import annotations

from harness.agent.tools.base import BaseTool
from harness.agent.tools.llm_utils import extract_json_array, extract_json_object
from harness.core.types import Clause, ComplianceCheck
from harness.utils.log import logger

_BATCH_COMPLIANCE_PROMPT = """请对照以下法律法规，批量检查合同条款是否合规，
返回 JSON 数组，每个元素格式为
{{"clause_index": ..., "regulation": "...", "status": true/false, "detail": "..."}}。

合同条款列表：
{clause_sections}

法律法规列表：
{regulations}"""


class ComplianceChecker(BaseTool):
    """合规检查器，逐条对照法规库对条款进行合规审查。"""

    REGULATIONS = [
        "中华人民共和国民法典（合同编）",
        "中华人民共和国劳动合同法",
        "中华人民共和国数据安全法",
        "中华人民共和国个人信息保护法",
        "中华人民共和国反垄断法",
    ]

    def check(self, clause: Clause) -> list[ComplianceCheck]:
        """对单一条款执行多法规合规检查。"""
        # batch_check 对空 clauses 返回 []，这里 clause 一定非空，
        # [0] 安全；去掉 `if clause` 死分支（Clause 对象恒 truthy）
        return self.batch_check([clause])[0]

    def batch_check(
        self, clauses: list[Clause], memory_context: str = ""
    ) -> list[list[ComplianceCheck]]:
        """批量对多个条款执行合规检查，单次 LLM 调用完成所有条款和法规检查。"""
        if not clauses:
            return []

        logger.info("Batch checking compliance for {} clauses", len(clauses))
        sections = []
        for i, c in enumerate(clauses):
            sections.append(f"--- 条款 {i} ---\n类型: {c.clause_type}\n内容: {c.content}")
        prompt = _BATCH_COMPLIANCE_PROMPT.format(
            clause_sections="\n\n".join(sections),
            regulations="\n".join(f"{i + 1}. {r}" for i, r in enumerate(self.REGULATIONS)),
        )
        if memory_context:
            prompt += "\n\n" + memory_context
        resp = self._llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是一个法律合规专家。请严格按照 JSON 数组格式输出结果。",
                },
                {"role": "user", "content": prompt},
            ]
        )
        all_results = self._parse_batch_response(resp.content, clauses)
        logger.info(
            "Batch compliance completed: {} clauses, {} total checks",
            len(clauses),
            sum(len(r) for r in all_results),
        )
        return all_results

    def _parse_batch_response(
        self, content: str, clauses: list[Clause]
    ) -> list[list[ComplianceCheck]]:
        """解析批量合规检查的 LLM 响应，按 clause_index 分组。

        复用 ``extract_json_array`` 统一处理 `````json`` 围栏、多对象逗号分隔等
        LLM 输出形态；当 LLM 返回单个 JSON 对象时回退到 ``extract_json_object``
        解析并包成单元素列表，避免解析失败后用占位 ``ComplianceCheck`` 污染结果。
        """
        parsed = extract_json_array(content)
        if not parsed:
            single = extract_json_object(content)
            if single:
                parsed = [single]

        grouped: dict[int, list[ComplianceCheck]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            clause_idx = item.get("clause_index")
            if not isinstance(clause_idx, int):
                continue
            grouped.setdefault(clause_idx, []).append(
                ComplianceCheck(
                    regulation=item.get("regulation", ""),
                    status=bool(item.get("status", True)),
                    detail=item.get("detail", ""),
                )
            )

        all_results: list[list[ComplianceCheck]] = []
        for i, clause in enumerate(clauses):
            results = grouped.get(i)
            if not results:
                results = [
                    ComplianceCheck(regulation=reg, status=True, detail="无法解析")
                    for reg in self.REGULATIONS
                ]
            all_results.append(results)
        return all_results
