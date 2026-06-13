"""回归比较器单元测试。"""

from harness.core.types import (
    AgentSession,
    ContractDocument,
    ReviewReport,
    RiskLevel,
)
from harness.regression.comparator import OutputComparator


def _make_session(report: ReviewReport) -> AgentSession:
    return AgentSession(
        session_id="test",
        document=ContractDocument(id="t", title="t", content=""),
        started_at="now",
        report=report,
    )


class TestRegression:
    """OutputComparator 差异对比功能测试。"""

    def test_comparator_no_changes(self):
        """两个相同报告应无差异。"""

        def _report(risk=RiskLevel.LOW):
            return ReviewReport(
                document_id="1",
                document_title="t",
                reviewed_at="now",
                summary="s",
                overall_risk=risk,
            )

        r1 = _report(RiskLevel.LOW)
        r2 = _report(RiskLevel.LOW)
        s1 = _make_session(r1)
        s2 = _make_session(r2)

        comp = OutputComparator()
        result = comp.compare(s1, s2)
        assert result["risk_level_changed"] is False
        assert result["summary_changed"] is False

    def test_comparator_risk_change(self):
        """风险等级不同时应被检测到。"""

        def _report(risk=RiskLevel.LOW):
            return ReviewReport(
                document_id="1",
                document_title="t",
                reviewed_at="now",
                summary="s",
                overall_risk=risk,
            )

        r1 = _report(RiskLevel.LOW)
        r2 = _report(RiskLevel.HIGH)
        s1 = _make_session(r1)
        s2 = _make_session(r2)

        comp = OutputComparator()
        result = comp.compare(s1, s2)
        assert result["risk_level_changed"] is True
