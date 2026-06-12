from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.core.types import RegressionResult
from harness.eval.dataset import EvalDataset
from harness.eval.scorer import EvalScorer
from harness.utils.io import read_json, write_json


class RegressionSuite:
    def __init__(
        self,
        baseline_dir: str | Path | None = None,
        scorer: EvalScorer | None = None,
    ):
        default_dir = Path.cwd() / ".regression_baseline"
        self._baseline_dir = Path(baseline_dir) if baseline_dir else default_dir
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        self._scorer = scorer or EvalScorer()

    def run(self, dataset: EvalDataset, version: str = "") -> RegressionResult:
        results = self._scorer.run(dataset)
        current_metrics: dict[str, float] = {}
        for r in results:
            for m in r.metrics:
                current_metrics[m.name] = current_metrics.get(m.name, 0.0) + m.value
        if results:
            for k in current_metrics:
                current_metrics[k] /= len(results)

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

                if diff < -0.05:
                    result.regressions.append(
                        f"{name}: {baseline_val:.2%} → {current_val:.2%} (下降 {abs(diff):.2%})"
                    )
                    result.passed = False
                elif diff > 0.05:
                    result.improvements.append(
                        f"{name}: {baseline_val:.2%} → {current_val:.2%} (提升 {diff:.2%})"
                    )

        self._save_baseline(current_metrics, result.current_version)
        return result

    def save_baseline(self, dataset: EvalDataset, version: str) -> Path:
        results = self._scorer.run(dataset)
        metrics: dict[str, float] = {}
        for r in results:
            for m in r.metrics:
                metrics[m.name] = metrics.get(m.name, 0.0) + m.value
        if results:
            for k in metrics:
                metrics[k] /= len(results)

        return self._save_baseline(metrics, version)

    def _load_baseline(self) -> dict[str, Any] | None:
        filepath = self._baseline_dir / "baseline.json"
        if not filepath.exists():
            return None
        return read_json(filepath)

    def _save_baseline(self, metrics: dict[str, float], version: str) -> Path:
        data = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        filepath = self._baseline_dir / "baseline.json"
        write_json(filepath, data)
        return filepath
