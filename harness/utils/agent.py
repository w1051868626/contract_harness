from harness.core.types import Clause, ComplianceCheck, RiskAssessment, RiskLevel

_RISK_PRIORITY = [
    RiskLevel.CRITICAL,
    RiskLevel.HIGH,
    RiskLevel.MEDIUM,
    RiskLevel.LOW,
]


def compute_overall_risk(risks: list[RiskAssessment]) -> RiskLevel:
    if not risks:
        return RiskLevel.INFO
    levels = {r.risk_level for r in risks}
    for level in _RISK_PRIORITY:
        if level in levels:
            return level
    return RiskLevel.INFO


def build_risk_summary(
    clauses: list[Clause], risks: list[RiskAssessment], compliance: list[ComplianceCheck]
) -> str:
    parts = [f"共发现 {len(clauses)} 个条款"]
    high = [r for r in risks if r.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
    parts.append(f"高风险项: {len(high)} 个")
    bad = [c for c in compliance if not c.status]
    parts.append(f"不合规项: {len(bad)} 个")
    return "；".join(parts)
