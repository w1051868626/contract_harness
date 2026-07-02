from __future__ import annotations

import tempfile

from harness.agent.llm import LLMResponse
from harness.eval_rag.dataset import EvalRagItem, EvalRagResult, load_jsonl, save_jsonl
from harness.eval_rag.generator import RagDatasetGenerator
from harness.eval_rag.metrics import RagMetricsCalculator
from tests.conftest import MockLLMClient


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


class TestRagDatasetGenerator:
    def test_generate_returns_items(self):
        """从 mock KB chunk 生成 eval 数据集。"""
        llm = MockLLMClient(
            [
                LLMResponse(content="违约金的上限是多少？", model="mock"),
                LLMResponse(content="保密义务的期限是多久？", model="mock"),
            ]
        )

        class MockChunk:
            id = "c1"
            content = "违约金不得超过实际损失的30%"

        class MockChunk2:
            id = "c2"
            content = "保密义务期限为合同终止后三年"

        class MockKB:
            def list_chunks(self):
                return [MockChunk(), MockChunk2()]

        generator = RagDatasetGenerator()
        items = generator.generate(MockKB(), llm, queries_per_chunk=1)
        assert len(items) == 2
        assert items[0].query == "违约金的上限是多少？"
        assert items[0].expected_chunk_ids == ["c1"]
        assert items[1].expected_chunk_ids == ["c2"]
