# RAG 检索质量评估设计

## 背景

项目现有 `harness/eval/` 只评估 Agent 合同审查质量，缺少对 RAG 知识库检索质量的可量化评估。需新增 RAG 评估系统，支持检索质量和分块质量的量化指标。

## 架构

新增 `harness/eval_rag/` 模块，与现有 `harness/eval/` 平级。

```
harness/eval_rag/
├── dataset.py      # EvalRagItem 数据模型 + 数据集加载/导出
├── generator.py    # LLM 从 KB chunk 自动生成 (query, chunk_id) 对
├── metrics.py      # HitRate@K, MRR@K, Precision@K, Recall@K
├── runner.py       # 评估流水线编排
└── reporter.py     # JSON / Markdown / HTML 报告生成
```

挂载到 `harness kb eval` 子命令组下。

## 数据模型

```python
@dataclass
class EvalRagItem:
    query: str                          # 检索查询
    expected_chunk_ids: list[str]       # 期望命中的 chunk ID
    expected_texts: list[str] = ...     # 期望命中的文本（冗余，便于人工核验）
    metadata: dict[str, Any] = ...      # 来源文档等

@dataclass
class EvalRagResult:
    dataset_name: str
    top_ks: list[int]                   # [1, 3, 5]
    hit_rates: dict[int, float]         # {1: 0.85, 3: 0.95}
    mrr: dict[int, float]
    precisions: dict[int, float]
    recalls: dict[int, float]
    details: list[dict[str, Any]]       # 逐项明细
```

## 流程

### 1. 数据集生成（generator.py）

对 KB 中每个 chunk，用 LLM 生成 1-3 条 query，chunk ID 作为 ground truth。

```
chunks = kb.list_chunks()  # 或遍历文档
for chunk in chunks:
    query = llm.chat(f"根据以下法律文本生成一个用户可能问的问题：\n{chunk.text}")
    dataset.append(EvalRagItem(query=query, expected_chunk_ids=[chunk.id]))
```

### 2. 数据集导入/导出（dataset.py）

- `save_jsonl(path, items)` — 导出为 JSONL，便于人工校验和复用
- `load_jsonl(path)` — 导入外部数据集（或 LLM 生成后编辑过的版本）
- 开源格式兼容：支持 BEIR 格式（`{"query": ..., "corpus_id": ...}`）

### 3. 评估执行（runner.py）

```
for item in dataset:
    results = kb.query(item.query, top_k=max(top_ks))
    hit = any(r.id in item.expected_chunk_ids for r in results[:k])
    # 计算各指标
```

### 4. 报告（reporter.py）

支持三种输出格式：
- JSON（供程序消费）
- Markdown（终端展示 + PR 注释）
- HTML（可视化报表）

## CLI

```
harness kb eval generate [--top-ks 1,3,5]     # 从 KB 生成 eval 数据集
harness kb eval import <file>                  # 导入外部数据集
harness kb eval run [--dataset <path>]         # 执行评估
harness kb eval report [--format md]           # 生成报告
```

## 指标

| 指标 | 公式 | 含义 |
|---|---|---|
| Hit Rate@K | 至少一个命中 / 总数 | 是否有用 |
| MRR@K | 1/rank(第一个命中) 的均值 | 第一个答案多靠前 |
| Precision@K | 命中的 / K | top-K 中有多少是相关的 |
| Recall@K | 命中的 / 期望命中数 | 期望的召回全不全 |

## 测试策略

- `generator.py`: mock LLM 响应，验证 query 数量和格式
- `metrics.py`: 预设命中/未命中场景，验证各指标计算结果
- `runner.py`: mock KB.query，验证流水线完整性
- `dataset.py`: JSONL 导入导出往返测试
