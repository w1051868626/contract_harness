"""回归测试套件，用于对比基线指标与当前评测结果。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.core.types import EvalResult, RegressionResult
from harness.eval.dataset import EvalDataset
from harness.eval.scorer import EvalScorer
from harness.utils.io import read_json, write_json
from harness.utils.log import logger

_REGRESSION_THRESHOLD = 0.05


class RegressionSuite:
    """回归测试套件，管理基线保存、加载与指标对比。"""

    def __init__(
        self,
        baseline_dir: str | Path | None = None,
        scorer: EvalScorer | None = None,
    ):
        """初始化回归套件，指定基线目录与评分器。"""
        default_dir = Path.cwd() / ".regression_baseline"
        self._baseline_dir = Path(baseline_dir) if baseline_dir else default_dir
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        self._scorer = scorer or EvalScorer()

    def _aggregate_metrics(self, results: list[EvalResult]) -> dict[str, float]:
        """汇总多条评测结果的指标，计算平均值。"""
        metrics: dict[str, float] = {}
        for r in results:
            for m in r.metrics:
                metrics[m.name] = metrics.get(m.name, 0.0) + m.value
        if results:
            for k in metrics:
                metrics[k] /= len(results)
        return metrics

    def run(self, dataset: EvalDataset, version: str = "") -> RegressionResult:
        """运行回归测试：评分、对比基线、返回结果。"""
        logger.info("Running regression test on {} items", len(dataset.items))
        results = self._scorer.run(dataset)
        current_metrics = self._aggregate_metrics(results)

        baseline = self._load_baseline()
        result = RegressionResult(
            baseline_version=baseline.get("version", "unknown") if baseline else "none",
            current_version=version or f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        )

        if baseline:
            baseline_metrics = baseline.get("metrics", {})
            for name, current_val in current_metrics.items():
                baseline_val = baseline_metrics.get(name, current_val)
                diff = current_val - baseline_val
                result.metrics_diff[name] = round(diff, 4)

                if diff < -_REGRESSION_THRESHOLD:
                    result.regressions.append(
                        f"{name}: {baseline_val:.2%} → {current_val:.2%} (下降 {abs(diff):.2%})"
                    )
                    result.passed = False
                elif diff > _REGRESSION_THRESHOLD:
                    result.improvements.append(
                        f"{name}: {baseline_val:.2%} → {current_val:.2%} (提升 {diff:.2%})"
                    )

        self._save_baseline(current_metrics, result.current_version)
        logger.info(
            "Regression test completed: passed={}, regressions={}, improvements={}",
            result.passed,
            len(result.regressions),
            len(result.improvements),
        )
        return result

    def save_baseline(self, dataset: EvalDataset, version: str) -> Path:
        """手动保存当前评测结果作为基线。"""
        logger.debug("Saving baseline version={}", version)
        results = self._scorer.run(dataset)
        metrics = self._aggregate_metrics(results)

        result_path = self._save_baseline(metrics, version)
        logger.debug("Baseline saved to {}", result_path)
        return result_path

    def _load_baseline(self) -> dict[str, Any] | None:
        """从磁盘加载基线 JSON。"""
        filepath = self._baseline_dir / "baseline.json"
        if not filepath.exists():
            return None
        return read_json(filepath)

    def _save_baseline(self, metrics: dict[str, float], version: str) -> Path:
        """将指标与版本写入基线文件。"""
        data = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        filepath = self._baseline_dir / "baseline.json"
        write_json(filepath, data)
        return filepath
