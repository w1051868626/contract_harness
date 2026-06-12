from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.core.types import AgentSession
from harness.replay.player import SessionPlayer
from harness.utils.io import write_text


class OutputComparator:
    def __init__(self, player: SessionPlayer | None = None):
        self._player = player or SessionPlayer()

    def compare(self, session_a: AgentSession, session_b: AgentSession) -> dict[str, Any]:
        diff = {
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

        return diff

    def compare_by_session_id(self, session_id_a: str, session_id_b: str) -> dict[str, Any]:
        session_a = self._player.load(session_id_a)
        session_b = self._player.load(session_id_b)
        if not session_a or not session_b:
            raise ValueError("无法加载会话")
        return self.compare(session_a, session_b)

    def _compare_clauses(self, clauses_a: list, clauses_b: list) -> list[dict]:
        diffs = []
        types_a = {c.clause_type: c for c in clauses_a}
        types_b = {c.clause_type: c for c in clauses_b}

        for ctype in set(list(types_a.keys()) + list(types_b.keys())):
            if ctype not in types_a:
                diffs.append({"type": ctype, "change": "removed"})
            elif ctype not in types_b:
                diffs.append({"type": ctype, "change": "added"})
            elif types_a[ctype].content != types_b[ctype].content:
                diffs.append({"type": ctype, "change": "modified"})

        return diffs

    def _compare_risks(self, risks_a: list, risks_b: list) -> list[dict]:
        diffs = []
        min_len = min(len(risks_a), len(risks_b))
        for i in range(min_len):
            if risks_a[i].risk_level != risks_b[i].risk_level:
                diffs.append(
                    {
                        "index": i,
                        "from": risks_a[i].risk_level.value,
                        "to": risks_b[i].risk_level.value,
                    }
                )

        if len(risks_a) > len(risks_b):
            for i in range(len(risks_b), len(risks_a)):
                diffs.append({"index": i, "change": "removed"})
        elif len(risks_b) > len(risks_a):
            for i in range(len(risks_a), len(risks_b)):
                diffs.append({"index": i, "change": "added"})

        return diffs

    def _compare_compliance(self, comp_a: list, comp_b: list) -> list[dict]:
        diffs = []
        regs_a = {c.regulation: c for c in comp_a}
        regs_b = {c.regulation: c for c in comp_b}

        for reg in set(list(regs_a.keys()) + list(regs_b.keys())):
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
                if "from" in d:
                    lines.append(f"- 第{d['index']}项: {d['from']} → {d['to']}")
                else:
                    lines.append(f"- 第{d['index']}项: {d['change']}")
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
