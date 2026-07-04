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
        """比对各 Worker 输出，找出分歧项。

        对齐策略与 ``synthesize_report`` 一致：优先用 ``clause_index`` 显式
        对齐，缺失时按 ``clause_type`` 匹配。避免下标强行对齐导致
        RiskExpert / ComplianceExpert 输出顺序或数量不一致时拿错位的
        风险等级比较，产出假分歧或漏掉真分歧。
        """
        disagreements: list[Disagreement] = []

        # 收集每个 role 的 {clause_key -> risk_level}，clause_key 优先用
        # clause_index，回退到 clause_type；同一 key 多次出现取最后一次。
        role_risks: dict[str, dict[str, str]] = {}
        for role, out in outputs.items():
            if not (out.structured and isinstance(out.structured, list)):
                continue
            key_to_level: dict[str, str] = {}
            for item in out.structured:
                if not isinstance(item, dict):
                    continue
                rl = item.get("risk_level") or item.get("risk_level_note")
                if not rl:
                    continue
                ci = item.get("clause_index")
                if isinstance(ci, int):
                    key = f"idx:{ci}"
                else:
                    key = f"type:{item.get('clause_type') or item.get('type') or ''}"
                key_to_level[key] = rl
            if key_to_level:
                role_risks[role] = key_to_level

        if len(role_risks) < 2:
            return disagreements

        # 取所有 key 的并集，逐 key 比对各 role 的 risk_level
        all_keys: set[str] = set()
        for levels in role_risks.values():
            all_keys.update(levels.keys())

        for key in all_keys:
            values: dict[str, str] = {}
            for role, levels in role_risks.items():
                if key in levels:
                    values[role] = levels[key]
            if len(values) < 2:
                continue
            unique = set(values.values())
            if len(unique) <= 1:
                continue
            items = list(values.items())
            for j in range(len(items)):
                for k in range(j + 1, len(items)):
                    if items[j][1] != items[k][1]:
                        disagreements.append(
                            Disagreement(
                                item_id=key,
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
            # 优先用 clause_index 显式对齐；缺 index 时按 clause_type 匹配
            # clauses，避免下标强行对齐导致 Agent 提取顺序/数量与
            # ClauseExpert 不一致时 risk 挂错条款。
            clauses_by_type: dict[str, Clause] = {}
            for c in clauses:
                clauses_by_type.setdefault(c.clause_type, c)
            for r in risk_out.structured:
                if not isinstance(r, dict):
                    continue
                ci = r.get("clause_index")
                if isinstance(ci, int) and 0 <= ci < len(clauses):
                    clause = clauses[ci]
                else:
                    key = r.get("clause_type") or r.get("type")
                    clause = clauses_by_type.get(key or "", Clause(clause_type="未知", content=""))
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
