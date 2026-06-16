"""回归比较器与套件单元测试。"""

from unittest.mock import MagicMock

from harness.core.types import (
    AgentSession,
    ContractDocument,
    EvalResult,
    ReviewReport,
    RiskLevel,
)
from harness.eval.dataset import EvalDataset
from harness.regression.comparator import OutputComparator
from harness.regression.suite import RegressionSuite


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


class TestRegressionSuite:
    """回归测试套件功能测试。"""

    def test_save_baseline_creates_file(self, tmp_path):
        """保存基线应在指定目录创建文件。"""
        dataset = EvalDataset()
        scorer = MagicMock()
        scorer.run.return_value = [
            EvalResult(
                dataset_name="test",
                agent_version="0.1.0",
                metrics=[],
            )
        ]
        scorer.run.return_value[0].metrics = MagicMock()
        scorer.run.return_value[0].metrics.__iter__ = lambda s: iter([])
        scorer.run.return_value[0].metrics.__len__ = lambda s: 0

        suite = RegressionSuite(baseline_dir=tmp_path, scorer=scorer)
        path = suite.save_baseline(dataset, version="v1")
        assert path.exists()
        import json as _json

        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "v1"
        assert "metrics" in data
        assert "timestamp" in data

    def test_load_baseline_none_when_missing(self):
        """无基线文件时应返回 None。"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            suite = RegressionSuite(baseline_dir=Path(tmpdir))
            assert suite._load_baseline() is None

    def test_save_and_load_baseline(self, tmp_path):
        """保存后应能正确加载。"""
        suite = RegressionSuite(baseline_dir=tmp_path)
        suite._save_baseline({"accuracy": 0.95}, version="v2")
        loaded = suite._load_baseline()
        assert loaded is not None
        assert loaded["version"] == "v2"
        assert loaded["metrics"]["accuracy"] == 0.95
