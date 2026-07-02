from __future__ import annotations

import tempfile

from harness.eval_rag.dataset import EvalRagItem, EvalRagResult, load_jsonl, save_jsonl
from harness.eval_rag.metrics import RagMetricsCalculator


class TestDataModels:
    def test_eval_rag_item_creation(self):
        item = EvalRagItem(query="违约金上限是多少？", expected_chunk_ids=["c1", "c2"])
        assert item.query == "违约金上限是多少？"
        assert item.expected_chunk_ids == ["c1", "c2"]

    def test_eval_rag_result_creation(self):
        result = EvalRagResult(
            dataset_name="test",
            top_ks=[1, 3, 5],
            hit_rates={1: 0.8, 3: 0.95},
            mrr={1: 0.75, 3: 0.85},
            precisions={1: 0.8, 3: 0.4},
            recalls={5: 0.9},
            details=[],
        )
        assert result.hit_rates[1] == 0.8

    def test_jsonl_roundtrip(self):
        items = [
            EvalRagItem(query="q1", expected_chunk_ids=["c1"]),
            EvalRagItem(query="q2", expected_chunk_ids=["c2"], metadata={"source": "doc1"}),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            save_jsonl(f.name, items)
            loaded = load_jsonl(f.name)
        assert len(loaded) == 2
        assert loaded[0].query == "q1"
        assert loaded[1].metadata["source"] == "doc1"


class TestRagMetrics:
    def test_hit_rate_perfect(self):
        """全部命中 top-1。"""
        items = [
            {"expected": ["c1"], "retrieved": ["c1", "c2", "c3"]},
            {"expected": ["c2"], "retrieved": ["c2", "c3", "c1"]},
        ]
        calc = RagMetricsCalculator()
        result = calc.compute(items, top_ks=[1, 3])
        assert result.hit_rates[1] == 1.0
        assert result.hit_rates[3] == 1.0

    def test_hit_rate_partial(self):
        items = [
            {"expected": ["c1"], "retrieved": ["c1", "c2"]},
            {"expected": ["c99"], "retrieved": ["c1", "c2"]},
            {"expected": ["c3"], "retrieved": ["c1", "c2", "c3"]},
        ]
        calc = RagMetricsCalculator()
        result = calc.compute(items, top_ks=[1, 3])
        assert result.hit_rates[1] == 1.0 / 3
        assert result.hit_rates[3] == 2.0 / 3

    def test_mrr(self):
        items = [
            {"expected": ["c2"], "retrieved": ["c1", "c2", "c3"]},
            {"expected": ["c1"], "retrieved": ["c1", "c2", "c3"]},
        ]
        calc = RagMetricsCalculator()
        result = calc.compute(items, top_ks=[3])
        assert result.mrr[3] == 0.75

    def test_precision_at_k(self):
        items = [
            {"expected": ["c1", "c2"], "retrieved": ["c1", "c3", "c4"]},
        ]
        calc = RagMetricsCalculator()
        result = calc.compute(items, top_ks=[3])
        assert result.precisions[3] == round(1.0 / 3, 4)
