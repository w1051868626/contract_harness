# RAG 检索质量评估实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 RAG 知识库新增检索质量评估系统，支持 LLM 自动生成数据集 + 标准检索指标计算 + CLI 命令。

**Architecture:** 新建 `harness/eval_rag/` 模块（与 `harness/eval/` 平级），包含 dataset（数据模型+JSONL）、metrics（HitRate/MRR/Precision/Recall）、generator（LLM 从 chunk 生成 query）、runner（评估流水线）、reporter（报告输出）。挂载到 `harness kb eval` CLI 子命令组。

**Tech Stack:** Python 3.11+, pytest, MockLLMClient, KnowledgeBase.query()

## Global Constraints

- 所有文件使用 `from __future__ import annotations`
- 遵循项目现有的 dataclass 模式
- 测试使用 `MockLLMClient` 来自 `tests.conftest`
- ruff check + format 通过后提交

---

### Task 1: 数据模型 + 数据集 I/O

**Files:**
- Create: `harness/eval_rag/__init__.py`
- Create: `harness/eval_rag/dataset.py`

**Interfaces:**
- Produces: `EvalRagItem(query, expected_chunk_ids, expected_texts, metadata)`, `EvalRagResult(dataset_name, top_ks, hit_rates, mrr, precisions, recalls, details)`
- Produces: `save_jsonl(path, items)`, `load_jsonl(path) → list[EvalRagItem]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_rag.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness.eval_rag.dataset import EvalRagItem, EvalRagResult, load_jsonl, save_jsonl


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestDataModels -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement dataset.py**

```python
# harness/eval_rag/__init__.py
"""RAG 检索质量评估模块。"""


# harness/eval_rag/dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalRagItem:
    query: str
    expected_chunk_ids: list[str]
    expected_texts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRagResult:
    dataset_name: str
    top_ks: list[int]
    hit_rates: dict[int, float]
    mrr: dict[int, float]
    precisions: dict[int, float]
    recalls: dict[int, float]
    details: list[dict[str, Any]] = field(default_factory=list)


def save_jsonl(path: str, items: list[EvalRagItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({
                "query": item.query,
                "expected_chunk_ids": item.expected_chunk_ids,
                "expected_texts": item.expected_texts,
                "metadata": item.metadata,
            }, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[EvalRagItem]:
    items: list[EvalRagItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            items.append(EvalRagItem(
                query=data["query"],
                expected_chunk_ids=data["expected_chunk_ids"],
                expected_texts=data.get("expected_texts", []),
                metadata=data.get("metadata", {}),
            ))
    return items
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestDataModels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/eval_rag/ tests/unit/test_eval_rag.py
git commit -m "feat: add RAG eval data models and JSONL I/O"
```

---

### Task 2: 检索指标计算

**Files:**
- Create: `harness/eval_rag/metrics.py`

**Interfaces:**
- Consumes: `list[EvalRagItem]`, `KnowledgeBase.query()`
- Produces: `RagMetricsCalculator.compute(items, results, top_ks) → EvalRagResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_rag.py (append)
from harness.eval_rag.metrics import RagMetricsCalculator


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
        # MRR = (1/2 + 1/1) / 2 = (0.5 + 1.0) / 2 = 0.75
        assert result.mrr[3] == 0.75

    def test_precision_at_k(self):
        items = [
            {"expected": ["c1", "c2"], "retrieved": ["c1", "c3", "c4"]},
        ]
        calc = RagMetricsCalculator()
        result = calc.compute(items, top_ks=[3])
        # Precisio@3 = 1/3
        assert result.precisions[3] == round(1.0 / 3, 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagMetrics -v`
Expected: FAIL (metrics module not found)

- [ ] **Step 3: Implement metrics.py**

```python
# harness/eval_rag/metrics.py
from __future__ import annotations

from typing import Any

from harness.eval_rag.dataset import EvalRagResult


class RagMetricsCalculator:
    def compute(
        self,
        items: list[dict[str, Any]],
        top_ks: list[int],
        dataset_name: str = "default",
    ) -> EvalRagResult:
        n = len(items)
        max_k = max(top_ks)
        hit_counts = {k: 0 for k in top_ks}
        mrr_sums = {k: 0.0 for k in top_ks}
        precision_sums = {k: 0.0 for k in top_ks}
        recall_sums = {k: 0.0 for k in top_ks}
        details: list[dict[str, Any]] = []

        for item in items:
            expected = set(item["expected"])
            retrieved = item["retrieved"][:max_k]
            detail = {"expected": list(expected), "retrieved": retrieved}

            for k in top_ks:
                top_k = retrieved[:k]
                hits = [c for c in top_k if c in expected]
                hit = len(hits) > 0
                if hit:
                    hit_counts[k] += 1
                detail[f"hit@{k}"] = hit

                # Precision@K
                if top_k:
                    precision_sums[k] += len(hits) / len(top_k)

                # MRR@K
                for rank, c in enumerate(top_k, 1):
                    if c in expected:
                        mrr_sums[k] += 1.0 / rank
                        break

                # Recall@K
                if expected:
                    recall_sums[k] += len(hits) / len(expected)

            details.append(detail)

        return EvalRagResult(
            dataset_name=dataset_name,
            top_ks=top_ks,
            hit_rates={k: hit_counts[k] / n if n else 0 for k in top_ks},
            mrr={k: mrr_sums[k] / n if n else 0 for k in top_ks},
            precisions={k: round(precision_sums[k] / n, 4) if n else 0 for k in top_ks},
            recalls={k: round(recall_sums[k] / n, 4) if n else 0 for k in top_ks},
            details=details,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagMetrics -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/eval_rag/metrics.py tests/unit/test_eval_rag.py
git commit -m "feat: add RAG retrieval metrics (HitRate/MRR/Precision/Recall)"
```

---

### Task 3: LLM 数据集生成器

**Files:**
- Create: `harness/eval_rag/generator.py`

**Interfaces:**
- Consumes: `KnowledgeBase`, `LLMClient`
- Produces: `RagDatasetGenerator.generate(kb, llm, queries_per_chunk) → list[EvalRagItem]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_rag.py (append)
from harness.agent.llm import LLMResponse
from harness.eval_rag.generator import RagDatasetGenerator
from tests.conftest import MockLLMClient


class TestRagDatasetGenerator:
    def test_generate_returns_items(self):
        """从 mock KB chunk 生成 eval 数据集。"""
        # Mock: 2 chunks, each generates 1 query
        llm = MockLLMClient([
            LLMResponse(content="违约金的上限是多少？", model="mock"),
            LLMResponse(content="保密义务的期限是多久？", model="mock"),
        ])
        # Mock KB
        class MockChunk:
            id = "c1"
            text = "违约金不得超过实际损失的30%"
        class MockChunk2:
            id = "c2"
            text = "保密义务期限为合同终止后三年"

        class MockKB:
            def list_chunks(self):
                return [MockChunk(), MockChunk2()]

        generator = RagDatasetGenerator()
        items = generator.generate(MockKB(), llm, queries_per_chunk=1)
        assert len(items) == 2
        assert items[0].query == "违约金的上限是多少？"
        assert items[0].expected_chunk_ids == ["c1"]
        assert items[1].expected_chunk_ids == ["c2"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagDatasetGenerator -v`
Expected: FAIL (generator module not found)

- [ ] **Step 3: Implement generator.py**

```python
# harness/eval_rag/generator.py
from __future__ import annotations

from typing import Any

from harness.agent.llm import LLMClient
from harness.eval_rag.dataset import EvalRagItem
from harness.utils.log import logger

GENERATOR_PROMPT = """你是一位法律知识库的测试数据生成专家。
请根据以下法律文本，生成 {count} 个用户可能会问的自然语言问题。
每个问题应该：
1. 是真实用户会使用的问法（口语化、自然）
2. 答案可以从该文本中找到
3. 各不相同，覆盖不同角度

文本：
{text}

请直接输出问题，每行一个，不要编号。"""


class RagDatasetGenerator:
    def generate(
        self,
        kb: Any,
        llm: LLMClient,
        queries_per_chunk: int = 2,
    ) -> list[EvalRagItem]:
        items: list[EvalRagItem] = []
        chunks = kb.list_chunks()
        logger.info("Generating eval dataset from {} chunks", len(chunks))

        for chunk in chunks:
            if not chunk.text.strip():
                continue
            prompt = GENERATOR_PROMPT.format(count=queries_per_chunk, text=chunk.text[:1000])
            resp = llm.chat([
                {"role": "system", "content": "你是一个测试数据生成助手。"},
                {"role": "user", "content": prompt},
            ])
            queries = [q.strip().removeprefix(f"{i+1}.").strip()
                       for i, q in enumerate(resp.content.strip().split("\n"))
                       if q.strip()]
            for q in queries[:queries_per_chunk]:
                items.append(EvalRagItem(
                    query=q,
                    expected_chunk_ids=[chunk.id],
                    expected_texts=[chunk.text[:200]],
                    metadata={"source_chunk": chunk.id},
                ))

        logger.info("Generated {} eval items", len(items))
        return items
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagDatasetGenerator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/eval_rag/generator.py tests/unit/test_eval_rag.py
git commit -m "feat: add LLM-based RAG eval dataset generator"
```

---

### Task 4: 评估执行器 + 报告器

**Files:**
- Create: `harness/eval_rag/runner.py`
- Create: `harness/eval_rag/reporter.py`

**Interfaces:**
- Runner: `RagEvalRunner.run(kb, items, top_ks) → EvalRagResult`
- Reporter: `RagEvalReporter.to_markdown(result) → str`, `RagEvalReporter.to_json(result) → str`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_rag.py (append)
from harness.eval_rag.dataset import EvalRagItem, EvalRagResult
from harness.eval_rag.runner import RagEvalRunner
from harness.eval_rag.reporter import RagEvalReporter


class TestRagEvalRunner:
    def test_run_returns_result(self):
        items = [
            EvalRagItem(query="q1", expected_chunk_ids=["c1"]),
            EvalRagItem(query="q2", expected_chunk_ids=["c2"]),
        ]

        class MockKB:
            def query(self, query, top_k=5):
                if query == "q1":
                    return [type("", (), {"id": "c1", "score": 0.9})()]
                return [type("", (), {"id": "c2", "score": 0.8})()]

        runner = RagEvalRunner()
        result = runner.run(MockKB(), items, top_ks=[1, 3])
        assert isinstance(result, EvalRagResult)
        assert result.hit_rates[1] == 1.0


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
            dataset_name="test", top_ks=[1], hit_rates={1: 1.0},
            mrr={1: 1.0}, precisions={1: 1.0}, recalls={1: 1.0}, details=[],
        )
        reporter = RagEvalReporter()
        js = reporter.to_json(result)
        assert '"hit_rates"' in js
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagEvalRunner -v`
Expected: FAIL

- [ ] **Step 3: Implement runner.py**

```python
# harness/eval_rag/runner.py
from __future__ import annotations

from typing import Any

from harness.eval_rag.dataset import EvalRagItem, EvalRagResult
from harness.eval_rag.metrics import RagMetricsCalculator
from harness.utils.log import logger


class RagEvalRunner:
    def __init__(self):
        self._calculator = RagMetricsCalculator()

    def run(
        self,
        kb: Any,
        items: list[EvalRagItem],
        top_ks: list[int] | None = None,
        dataset_name: str = "default",
    ) -> EvalRagResult:
        if top_ks is None:
            top_ks = [1, 3, 5]
        max_k = max(top_ks)
        logger.info("Running RAG eval on {} items with top_k={}", len(items), top_ks)

        raw_items: list[dict[str, Any]] = []
        for item in items:
            results = kb.query(item.query, top_k=max_k)
            retrieved = [r.id for r in results]
            raw_items.append({
                "expected": item.expected_chunk_ids,
                "retrieved": retrieved,
                "query": item.query,
            })

        result = self._calculator.compute(raw_items, top_ks, dataset_name=dataset_name)
        logger.info("RAG eval complete: hit_rate@1={}", result.hit_rates.get(1))
        return result
```

- [ ] **Step 4: Implement reporter.py**

```python
# harness/eval_rag/reporter.py
from __future__ import annotations

import json

from harness.eval_rag.dataset import EvalRagResult


class RagEvalReporter:
    def to_markdown(self, result: EvalRagResult) -> str:
        lines = [f"# RAG 评估报告: {result.dataset_name}", ""]
        lines.append("| 指标 | " + " | ".join(f"@{k}" for k in result.top_ks) + " |")
        lines.append("|---" * (len(result.top_ks) + 1) + "|")

        def fmt(val: float) -> str:
            return f"{val:.2%}"

        lines.append("| Hit Rate | " + " | ".join(fmt(result.hit_rates[k]) for k in result.top_ks) + " |")
        lines.append("| MRR | " + " | ".join(fmt(result.mrr[k]) for k in result.top_ks) + " |")
        lines.append("| Precision | " + " | ".join(fmt(result.precisions[k]) for k in result.top_ks) + " |")
        lines.append("| Recall | " + " | ".join(fmt(result.recalls[k]) for k in result.top_ks) + " |")
        lines.append("")
        lines.append(f"共 {len(result.details)} 条评估项")
        return "\n".join(lines)

    def to_json(self, result: EvalRagResult) -> str:
        return json.dumps({
            "dataset_name": result.dataset_name,
            "top_ks": result.top_ks,
            "hit_rates": result.hit_rates,
            "mrr": result.mrr,
            "precisions": result.precisions,
            "recalls": result.recalls,
            "item_count": len(result.details),
        }, ensure_ascii=False, indent=2)
```

- [ ] **Step 5: Run to verify it passes**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagEvalRunner -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add harness/eval_rag/runner.py harness/eval_rag/reporter.py tests/unit/test_eval_rag.py
git commit -m "feat: add RAG eval runner and reporter"
```

---

### Task 5: CLI 集成

**Files:**
- Modify: `harness/cli/main.py` — 新增 `kb eval` 子命令组

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_rag.py (append)
from click.testing import CliRunner
from harness.cli.main import cli


class TestRagEvalCLI:
    def test_kb_eval_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "eval", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output or "运行" in result.output

    def test_kb_eval_generate_no_kb(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["kb", "eval", "generate"])
        # No KB initialized, should error gracefully
        assert result.exit_code != 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagEvalCLI -v`
Expected: FAIL (CLI not wired)

- [ ] **Step 3: Add CLI subcommand group**

```python
# In harness/cli/main.py, after the kb group definition, add:
@kb.group()
def eval() -> None:
    """RAG 检索质量评估。"""


@eval.command()
@click.option("--queries-per-chunk", default=2, help="每个 chunk 生成的问题数")
@click.option("--output", default=None, help="输出数据集路径")
@click.pass_context
def generate(ctx: click.Context, queries_per_chunk: int, output: str | None) -> None:
    """从知识库 chunk 自动生成评估数据集。"""
    config: HarnessConfig = ctx.obj["config"]
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    llm = LLMClient(config.llm)
    from harness.eval_rag.generator import RagDatasetGenerator
    generator = RagDatasetGenerator()
    items = generator.generate(kb_instance, llm, queries_per_chunk=queries_per_chunk)
    path = output or str(Path(config.data_dir) / "rag_eval_dataset.jsonl")
    from harness.eval_rag.dataset import save_jsonl
    save_jsonl(path, items)
    logger.info("生成 {} 条评估数据 -> {}", len(items), path)


@eval.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--top-ks", default="1,3,5", help="评估的 K 值，逗号分隔")
@click.pass_context
def run(ctx: click.Context, dataset: str, top_ks: str) -> None:
    """执行 RAG 检索质量评估。"""
    config: HarnessConfig = ctx.obj["config"]
    config.ensure_dirs()
    kb_instance = KnowledgeBase.from_config(config)
    from harness.eval_rag.dataset import EvalRagItem, load_jsonl
    from harness.eval_rag.runner import RagEvalRunner
    from harness.eval_rag.reporter import RagEvalReporter
    items = load_jsonl(dataset)
    top_ks_list = [int(k.strip()) for k in top_ks.split(",")]
    runner = RagEvalRunner()
    result = runner.run(kb_instance, items, top_ks=top_ks_list, dataset_name=Path(dataset).stem)
    reporter = RagEvalReporter()
    print(reporter.to_markdown(result))
```

Add imports to the top of `harness/cli/main.py`:
```python
from pathlib import Path                   # likely already there
from harness.agent.llm import LLMClient     # add if not present
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n contract-harness pytest tests/unit/test_eval_rag.py::TestRagEvalCLI -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add harness/cli/main.py tests/unit/test_eval_rag.py
git commit -m "feat: add harness kb eval CLI commands"
```

---

### Task 6: 文档更新

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 README.md 知识库管理节**

在 `harness kb search` 行后追加：
```
harness kb eval generate            # 从 KB 生成 eval 数据集
harness kb eval import <file>       # 导入外部数据集
harness kb eval run <dataset>       # 执行 RAG 检索质量评估
```

- [ ] **Step 2: 更新关键命令节（AGENTS.md + CLAUDE.md）**

同上追加命令说明

- [ ] **Step 3: Commit**

```bash
git add README.md AGENTS.md CLAUDE.md
git commit -m "docs: add RAG eval CLI docs"
```
