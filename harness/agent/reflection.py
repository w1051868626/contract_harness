"""Reflection 模式：管道审查后追加自审修正，提升报告质量。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from harness.agent.llm import LLMClient
from harness.core.types import (
    ReviewReport,
    RiskLevel,
)
from harness.utils.log import logger

REFLECTION_PROMPT = """你是一位法律合同审查质量审核专家。请对以下审查报告进行复核，
检查是否存在以下问题：

1. **完整性**：是否有遗漏的重要条款或风险点？
2. **一致性**：风险评级是否与具体分析一致？合规判断是否合理？
3. **准确性**：法律依据是否准确？分析是否存在错误？
4. **建议质量**：修改建议是否具体可行？

## 当前审查报告

### 整体风险: {overall_risk}

### 条款清单
{clauses_section}

### 风险分析
{risks_section}

### 合规检查
{compliance_section}

### 摘要
{summary}

请输出 JSON 格式：
{{
  "issues": [
    {{"type": "完整性/一致性/准确性/建议质量", "severity": "high/medium/low",
      "description": "问题描述", "fix": "修复建议"}}
  ],
  "revised_summary": "修正后的摘要（如有必要，否则填空字符串）",
  "revised_overall_risk": "修正后的整体风险等级（如无变化则和原来一致）"
}}
"""


def reflect_on_report(
    llm: LLMClient,
    report: ReviewReport,
) -> ReviewReport:
    """对审查报告执行自审修正，返回（可能）改进后的报告。"""
    logger.info("Running reflection on report document_id={}", report.document_id)

    clauses_section = (
        "\n".join(f"- [{c.clause_type}] {c.content[:200]}" for c in report.clauses) or "无"
    )
    risks_section = (
        "\n".join(
            f"- [{r.clause.clause_type}] {r.risk_level.value}: {r.reason[:200]}"
            for r in report.risks
        )
        or "无"
    )
    compliance_section = (
        "\n".join(
            f"- {c.regulation}: {'✅合规' if c.status else '❌不合规'} {c.detail[:100]}"
            for c in report.compliance_checks
        )
        or "无"
    )

    prompt = REFLECTION_PROMPT.format(
        overall_risk=report.overall_risk.value,
        clauses_section=clauses_section,
        risks_section=risks_section,
        compliance_section=compliance_section,
        summary=report.summary,
    )

    resp = llm.chat(
        [
            {"role": "system", "content": "你是一位严谨的法律合同审查质量审核专家。"},
            {"role": "user", "content": prompt},
        ]
    )

    try:
        raw = resp.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Reflection parse failed: {}", e)
        return report

    issues = data.get("issues", [])
    new_summary = data.get("revised_summary", "")
    new_risk = data.get("revised_overall_risk", "")

    if not issues and not new_summary and not new_risk:
        logger.info("Reflection: no issues found, report unchanged")
        return report

    revised_summary = new_summary or report.summary
    valid_risks = {r.value for r in RiskLevel}
    if new_risk and new_risk in valid_risks:
        revised_risk = RiskLevel(new_risk)
    else:
        revised_risk = report.overall_risk

    logger.info(
        "Reflection found {} issues, summary_changed={}, risk_changed={}",
        len(issues),
        new_summary != "",
        new_risk != "",
    )

    return ReviewReport(
        document_id=report.document_id,
        document_title=report.document_title,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        summary=revised_summary,
        clauses=report.clauses,
        risks=report.risks,
        compliance_checks=report.compliance_checks,
        overall_risk=revised_risk,
        raw_output=json.dumps(
            {"reflection_issues": issues, "original_summary": report.summary},
            ensure_ascii=False,
        ),
    )
