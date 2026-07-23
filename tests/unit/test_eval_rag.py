from __future__ import annotations

import json
import os
import tempfile

from click.testing import CliRunner
from openai import APIError

from harness.agent.llm import LLMResponse
from harness.cli.main import cli
from harness.eval_rag.dataset import EvalRagItem, EvalRagResult, load_jsonl, save_jsonl
from harness.eval_rag.generator import RagDatasetGenerator
from harness.eval_rag.metrics import RagMetricsCalculator
from harness.eval_rag.reporter import RagEvalReporter
from harness.eval_rag.runner import RagEvalRunner
from harness.rag.vector_store import Chunk
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


class TestRagEvalRunner:
    def test_run_returns_result(self):
        items = [
            EvalRagItem(query="q1", expected_chunk_ids=["c1"]),
            EvalRagItem(query="q2", expected_chunk_ids=["c2"]),
        ]

        class MockKB:
            def query(self, query, top_k=5, **kwargs):
                if query == "q1":
                    return [
                        Chunk(id="c1", document_id="d1", content="test", chunk_index=0, score=0.9)
                    ]
                return [Chunk(id="c2", document_id="d1", content="test", chunk_index=0, score=0.8)]

        runner = RagEvalRunner()
        result = runner.run(MockKB(), items, top_ks=[1, 3])
        assert isinstance(result, EvalRagResult)
        assert result.hit_rates[1] == 1.0

    def test_run_checkpoint_resume(self):
        """断点续跑：第二次 run 跳过 checkpoint 中已完成的 query。"""

        class CountingKB:
            """记录 query 调用次数，模拟「已跑过的不再跑」。"""

            def __init__(self):
                self.queries: list[str] = []

            def query(self, query, top_k=5, **kwargs):
                self.queries.append(query)
                return [Chunk(id="c1", document_id="d1", content="t", chunk_index=0, score=0.9)]

        items = [
            EvalRagItem(query="q1", expected_chunk_ids=["c1"]),
            EvalRagItem(query="q2", expected_chunk_ids=["c1"]),
            EvalRagItem(query="q3", expected_chunk_ids=["c1"]),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            ckpt = f.name
        try:
            # 第一轮：跑完 3 条，checkpoint 写 3 行
            kb1 = CountingKB()
            runner = RagEvalRunner()
            runner.run(kb1, items, top_ks=[1], checkpoint_path=ckpt)
            assert len(kb1.queries) == 3
            with open(ckpt, encoding="utf-8") as f:
                assert sum(1 for _ in f) == 3

            # 第二轮：同样 3 条 items + 同一 checkpoint，应全部命中缓存、不再调 kb.query
            kb2 = CountingKB()
            runner2 = RagEvalRunner()
            result = runner2.run(kb2, items, top_ks=[1], checkpoint_path=ckpt)
            assert kb2.queries == []  # 全部从 checkpoint 恢复
            assert result.hit_rates[1] == 1.0  # 结果与第一轮一致
        finally:
            os.unlink(ckpt)

    def test_run_checkpoint_partial_resume(self):
        """断点续跑：checkpoint 只含部分 query 时，跳过已完成的、补跑剩余的。"""

        class CountingKB:
            def __init__(self):
                self.queries: list[str] = []

            def query(self, query, top_k=5, **kwargs):
                self.queries.append(query)
                return [Chunk(id="c_new", document_id="d1", content="t", chunk_index=0, score=0.9)]

        items = [
            EvalRagItem(query="done_q", expected_chunk_ids=["c1"]),
            EvalRagItem(query="new_q", expected_chunk_ids=["c_new"]),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"query": "done_q", "retrieved": ["c1"]}, ensure_ascii=False) + "\n")
            ckpt = f.name
        try:
            kb = CountingKB()
            runner = RagEvalRunner()
            result = runner.run(kb, items, top_ks=[1], checkpoint_path=ckpt)
            # done_q 命中缓存，只 new_q 实际调用 kb.query
            assert kb.queries == ["new_q"]
            assert result.hit_rates[1] == 1.0
        finally:
            os.unlink(ckpt)


class TestRagEvalReporter:
    def test_to_markdown(self):
        result = EvalRagResult(
            dataset_name="test",
            top_ks=[1, 3, 5],
            hit_rates={1: 0.85, 3: 0.95, 5: 1.0},
            mrr={1: 0.85, 3: 0.88, 5: 0.88},
            precisions={1: 0.85, 3: 0.35, 5: 0.22},
            recalls={1: 0.50, 3: 0.70, 5: 0.85},
            details=[],
        )
        reporter = RagEvalReporter()
        md = reporter.to_markdown(result)
        assert "85.00%" in md
        assert "test" in md

    def test_to_json(self):
        result = EvalRagResult(
            dataset_name="test",
            top_ks=[1],
            hit_rates={1: 1.0},
            mrr={1: 1.0},
            precisions={1: 1.0},
            recalls={1: 1.0},
            details=[],
        )
        reporter = RagEvalReporter()
        js = reporter.to_json(result)
        assert '"hit_rates"' in js


class TestRagDatasetGenerator:
    def test_generate_returns_items(self):
        """从 mock KB chunk 生成 eval 数据集。"""
        llm = MockLLMClient(
            [
                LLMResponse(content='{"questions": ["违约金的上限是多少？"]}', model="mock"),
                LLMResponse(content='{"questions": ["保密义务的期限是多久？"]}', model="mock"),
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

    def test_generate_skips_chunk_on_llm_error(self):
        """LLM 调用失败时跳过该 chunk，继续处理后续。"""

        class FailingLLM:
            def __init__(self):
                self.call_count = 0

            def chat(self, messages, tools=None, **kwargs):
                self.call_count += 1
                if self.call_count <= 3:
                    raise APIError("模拟 API 错误", request=None, body=None)
                return LLMResponse(
                    content='{"questions": ["保密义务的期限是多久？"]}',
                    model="mock",
                )

        class MockChunk1:
            id = "c1"
            content = "违约金不得超过实际损失的30%"

        class MockChunk2:
            id = "c2"
            content = "保密义务期限为合同终止后三年"

        class MockKB:
            def list_chunks(self):
                return [MockChunk1(), MockChunk2()]

        generator = RagDatasetGenerator()
        items = generator.generate(MockKB(), FailingLLM(), queries_per_chunk=1)
        assert len(items) == 1
        assert items[0].expected_chunk_ids == ["c2"]

    def test_generate_resume_from_output(self):
        """断点恢复：第二次 generate 跳过已有输出中已处理的 chunk。"""

        class MockChunk1:
            id = "c1"
            content = "违约金不得超过实际损失的30%"

        class MockChunk2:
            id = "c2"
            content = "保密义务期限为合同终止后三年"

        class MockKB:
            def list_chunks(self):
                return [MockChunk1(), MockChunk2()]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            # 第一轮：只处理 chunk 1
            llm1 = MockLLMClient(
                [
                    LLMResponse(content='{"questions": ["违约金的上限是多少？"]}', model="mock"),
                ]
            )
            generator = RagDatasetGenerator()
            items1 = generator.generate(
                MockKB(), llm1, queries_per_chunk=1, output_path=output_path
            )
            assert len(items1) == 1
            assert items1[0].expected_chunk_ids == ["c1"]

            # 第二轮：重新指定同一 output_path，应跳过 c1 只处理 c2
            llm2 = MockLLMClient(
                [
                    LLMResponse(content='{"questions": ["保密义务的期限是多久？"]}', model="mock"),
                ]
            )
            items2 = generator.generate(
                MockKB(), llm2, queries_per_chunk=1, output_path=output_path
            )
            assert len(items2) == 1
            assert items2[0].expected_chunk_ids == ["c2"]

            # 验证输出文件包含两条
            loaded = load_jsonl(output_path)
            assert len(loaded) == 2
        finally:
            os.unlink(output_path)


class TestRagEvalCLI:
    def test_kb_eval_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "eval", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output or "运行" in result.output

    def test_kb_eval_generate_no_kb(self, tmp_path, monkeypatch):
        """无 KB 时 generate 应报 Missing 错误，而非静默成功或读残留数据。

        用临时空数据目录隔离，避免读到全局 ``.harness/`` 残留 KB/JSONL
        导致 ``UnicodeDecodeError``（Windows 预存 bug）。
        """
        monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "eval", "generate"])
        assert result.exit_code != 0
        assert result.exception is not None
        assert "Missing" in str(result.exception)
