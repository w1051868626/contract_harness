"""两份审查报告的差异对比工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.core.types import (
    AgentSession,
    Clause,
    ComplianceCheck,
    DiffResult,
    RiskAssessment,
)
from harness.replay.player import SessionPlayer
from harness.utils.io import write_text
from harness.utils.log import logger


class OutputComparator:
    """比较两个会话的输出差异，支持条款、风险、合规三个维度。"""

    def __init__(self, player: SessionPlayer | None = None):
        """注入 SessionPlayer，用于按 ID 加载会话。"""
        self._player = player or SessionPlayer()

    def compare(self, session_a: AgentSession, session_b: AgentSession) -> DiffResult:
        """对比两个会话的审查结果，返回差异字典。"""
        logger.info("Comparing sessions {} and {}", session_a.session_id, session_b.session_id)
        diff: DiffResult = {
            "summary_changed": False,
            "risk_level_changed": False,
            "clause_diffs": [],
            "risk_diffs": [],
            "compliance_diffs": [],
        }

        report_a = session_a.report
        report_b = session_b.report
        if not report_a or not report_b:
            return diff

        if report_a.summary != report_b.summary:
            diff["summary_changed"] = True
        if report_a.overall_risk != report_b.overall_risk:
            diff["risk_level_changed"] = True

        diff["clause_diffs"] = self._compare_clauses(report_a.clauses, report_b.clauses)
        diff["risk_diffs"] = self._compare_risks(report_a.risks, report_b.risks)
        diff["compliance_diffs"] = self._compare_compliance(
            report_a.compliance_checks, report_b.compliance_checks
        )

        logger.info(
            "Comparison complete: {} clause diffs, {} risk diffs, {} compliance diffs",
            len(diff["clause_diffs"]),
            len(diff["risk_diffs"]),
            len(diff["compliance_diffs"]),
        )
        return diff

    def compare_by_session_id(self, session_id_a: str, session_id_b: str) -> DiffResult:
        """通过会话 ID 加载并对比。"""
        logger.debug("Comparing sessions by ID: {} vs {}", session_id_a, session_id_b)
        session_a = self._player.load(session_id_a)
        session_b = self._player.load(session_id_b)
        if not session_a or not session_b:
            raise ValueError("无法加载会话")
        return self.compare(session_a, session_b)

    def _compare_clauses(
        self, clauses_a: list[Clause], clauses_b: list[Clause]
    ) -> list[dict[str, Any]]:
        """对比条款列表，返回增删改差异。"""
        diffs = []
        types_a = {c.clause_type: c for c in clauses_a}
        types_b = {c.clause_type: c for c in clauses_b}

        for ctype in types_a.keys() | types_b.keys():
            if ctype not in types_a:
                diffs.append({"type": ctype, "change": "removed"})
            elif ctype not in types_b:
                diffs.append({"type": ctype, "change": "added"})
            elif types_a[ctype].content != types_b[ctype].content:
                diffs.append({"type": ctype, "change": "modified"})

        return diffs

    def _compare_risks(
        self, risks_a: list[RiskAssessment], risks_b: list[RiskAssessment]
    ) -> list[dict[str, Any]]:
        """对比风险评估列表，返回等级变化与增删差异。"""
        risk_map_a = {r.clause.clause_type: r for r in risks_a}
        risk_map_b = {r.clause.clause_type: r for r in risks_b}
        all_keys = set(risk_map_a) | set(risk_map_b)
        diffs = []
        for key in sorted(all_keys):
            ra = risk_map_a.get(key)
            rb = risk_map_b.get(key)
            if ra is None and rb is not None:
                diffs.append(
                    {"type": "added", "clause_type": key, "risk_level": rb.risk_level.value}
                )
            elif rb is None and ra is not None:
                diffs.append(
                    {"type": "removed", "clause_type": key, "risk_level": ra.risk_level.value}
                )
            elif ra is not None and rb is not None and ra.risk_level != rb.risk_level:
                diffs.append(
                    {
                        "type": "changed",
                        "clause_type": key,
                        "from": ra.risk_level.value,
                        "to": rb.risk_level.value,
                    }
                )
        return diffs

    def _compare_compliance(
        self, comp_a: list[ComplianceCheck], comp_b: list[ComplianceCheck]
    ) -> list[dict[str, Any]]:
        """对比合规检查列表，返回状态变化与增删差异。"""
        diffs = []
        regs_a = {c.regulation: c for c in comp_a}
        regs_b = {c.regulation: c for c in comp_b}

        for reg in regs_a.keys() | regs_b.keys():
            if reg not in regs_a:
                diffs.append({"regulation": reg, "change": "removed"})
            elif reg not in regs_b:
                diffs.append({"regulation": reg, "change": "added"})
            elif regs_a[reg].status != regs_b[reg].status:
                diffs.append(
                    {
                        "regulation": reg,
                        "change": "status_changed",
                        "from": regs_a[reg].status,
                        "to": regs_b[reg].status,
                    }
                )

        return diffs

    def generate_diff_report(
        self, diff: dict[str, Any], output_path: str | Path | None = None
    ) -> str:
        """生成可读的 Markdown 差异报告，可选写入文件。"""
        lines = ["# 回归对比报告\n"]

        if diff.get("risk_level_changed"):
            lines.append("## ⚠️ 整体风险等级变化\n")
        if diff.get("summary_changed"):
            lines.append("## 📝 摘要内容变化\n")

        if diff.get("clause_diffs"):
            lines.append("## 条款变化")
            for d in diff["clause_diffs"]:
                lines.append(f"- {d['type']}: {d['change']}")
            lines.append("")

        if diff.get("risk_diffs"):
            lines.append("## 风险评估变化")
            for d in diff["risk_diffs"]:
                if d["type"] == "changed":
                    lines.append(f"- {d['clause_type']}: {d['from']} → {d['to']}")
                else:
                    lines.append(f"- {d['clause_type']}: {d['type']}")
            lines.append("")

        if diff.get("compliance_diffs"):
            lines.append("## 合规检查变化")
            for d in diff["compliance_diffs"]:
                lines.append(f"- {d['regulation']}: {d['change']}")
            lines.append("")

        content = "\n".join(lines)
        if output_path:
            write_text(Path(output_path), content)
        return content
