"""Supervisor Agent：任务分配 + 分歧检测 + 报告合成。"""

from __future__ import annotations

import json
from typing import Any

from harness.core.types import (
    Clause,
    ComplianceCheck,
    ContractDocument,
    Disagreement,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
    WorkerOutput,
)
from harness.utils.agent import build_risk_summary, compute_overall_risk
from harness.utils.log import logger


class SupervisorAgent:
    """协调 Agent，负责任务分配、分歧检测和报告合成。"""

    def assign_tasks(self, document: ContractDocument) -> dict[str, str]:
        """根据合同内容决定任务列表。
        当前实现：始终分配 ClauseExpert → RiskExpert → ComplianceExpert。
        返回 {worker_role: prompt} 字典。
        """
        return {
            "ClauseExpert": (
                f"请提取以下合同中的所有关键条款，输出 JSON 数组：\n\n{document.content}"
            ),
            "RiskExpert": "",
            "ComplianceExpert": "",
        }

    def validate_consensus(self, outputs: dict[str, WorkerOutput]) -> list[Disagreement]:
        """比对各 Worker 输出，找出分歧项。"""
        disagreements: list[Disagreement] = []

        clause_results: list[dict[str, Any]] = []
        for role, out in outputs.items():
            if out.structured and isinstance(out.structured, list):
                clause_results.append({"role": role, "data": out.structured})

        if len(clause_results) < 2:
            return disagreements

        # 按条款位置对比风险等级
        for i in range(max(len(r["data"]) for r in clause_results)):
            values: dict[str, Any] = {}
            for r in clause_results:
                if i < len(r["data"]):
                    item = r["data"][i]
                    if isinstance(item, dict):
                        rl = item.get("risk_level") or item.get("risk_level_note")
                        if rl:
                            values[r["role"]] = rl

            if len(values) >= 2:
                unique = set(values.values())
                if len(unique) > 1:
                    items = list(values.items())
                    for j in range(len(items)):
                        for k in range(j + 1, len(items)):
                            if items[j][1] != items[k][1]:
                                disagreements.append(
                                    Disagreement(
                                        item_id=f"clause-{i}",
                                        field="risk_level",
                                        value_a=items[j][1],
                                        value_b=items[k][1],
                                        worker_a=items[j][0],
                                        worker_b=items[k][0],
                                    )
                                )

        logger.info("Supervisor found {} disagreements", len(disagreements))
        return disagreements

    def synthesize_report(
        self,
        document: ContractDocument,
        outputs: dict[str, WorkerOutput],
        arbitration_results: list[dict[str, Any]],
    ) -> ReviewReport:
        """汇总 Worker 输出为 ReviewReport。"""
        clauses: list[Clause] = []
        risks: list[RiskAssessment] = []
        compliance: list[ComplianceCheck] = []

        clause_out = outputs.get("ClauseExpert")
        if clause_out and clause_out.structured:
            for c in clause_out.structured:
                if isinstance(c, dict):
                    clauses.append(
                        Clause(
                            clause_type=c.get("type", "未知"),
                            content=c.get("content", ""),
                        )
                    )

        risk_out = outputs.get("RiskExpert")
        if risk_out and risk_out.structured:
            for i, r in enumerate(risk_out.structured):
                if isinstance(r, dict):
                    clause = (
                        clauses[i] if i < len(clauses) else Clause(clause_type="未知", content="")
                    )
                    risks.append(
                        RiskAssessment(
                            clause=clause,
                            risk_level=RiskLevel(r.get("risk_level", "info")),
                            reason=r.get("reason", ""),
                            suggestion=r.get("suggestion", ""),
                        )
                    )

        compliance_out = outputs.get("ComplianceExpert")
        if compliance_out and compliance_out.structured:
            for c in compliance_out.structured:
                if isinstance(c, dict):
                    compliance.append(
                        ComplianceCheck(
                            regulation=c.get("regulation", ""),
                            status=c.get("status", False),
                            detail=c.get("detail", ""),
                        )
                    )

        report_raw = ""
        if arbitration_results:
            report_raw = json.dumps({"arbitration": arbitration_results}, ensure_ascii=False)

        overall_risk = compute_overall_risk(risks)
        summary = build_risk_summary(clauses, risks, compliance)
        return ReviewReport(
            document_id=document.id,
            document_title=document.title,
            reviewed_at="",
            summary=summary,
            clauses=clauses,
            risks=risks,
            compliance_checks=compliance,
            overall_risk=overall_risk,
            raw_output=report_raw,
        )
