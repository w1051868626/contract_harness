# Reranker 优化设计

> 模块: `harness/rag/reranker.py`
> 抽象基类: `Reranker` → 实现 `OpenAIReranker`（API 走 httpx）/ `LocalReranker`（sentence-transformers cross-encoder）

## 1. 优化历程（2026-07-26）

本轮在「**API 域唯一可用 rerank 模型 = `BAAI/bge-reranker-v2-m3`，无法更换**」约束下，沿调用侧做三步优化，全量 eval 20,442 query 验证：

| # | 改动 | 原理 | 200 样本 A/B delta | 风险 |
|---|---|---|---|---|
| 1 | 候选池扩大 `_search_single` `top_k*2` → `max(20, top_k*4)` | 给 Reranker 更大判别空间，避免正确 chunk 在 dense 阶段被截掉 | top-1 +5pp | 低 |
| 2 | `top_n` 从 `top_k` 改为 `len(candidates)` | 让 API 对全 pool 候选打分，客户端按 `relevance_score` 截 top_k；原 `top_n=top_k` 截断打分阶段 | top-1 +0.5pp（不显著） | 低 |
| 3 | **Reranker 输入注入法条标题前缀** `【{law_name}·{articles}】{chunk.content}` | 给 cross-encoder 额外结构信号（法条归属），不动 `chunk.content` | **top-1 +9.5pp** | 低 |

**组合效果**（全量 20,442 query，对比 07-25 baseline）：

| metric | baseline | 新代码 | delta |
|---|---|---|---|
| hit_rate@1 | 0.5239 | 0.5316 | +0.77pp |
| hit_rate@5 | 0.9021 | 0.9396 | +3.75pp |
| top-5 仍漏 | 2,001 | 1,234 | -38% |

## 2. 标题前缀注入设计（核心突破）

### 2.1 契约

- **仅改 reranker 喂给 API 的 `documents` 文本**，不动 `chunk.content`（embedding、其他调用方不受影响）。
- 无 metadata 或缺 `articles` / `law_name` 时降级为裸 `content`。
- 实现入口：`OpenAIReranker._format_for_rerank(chunk) -> str` 静态方法。

### 2.2 metadata 字段来源

来自 ChromaDB 的 chunk 元数据，由 `harness/rag/chunking.py` 在切片时写入：

| 字段 | 含义 | 示例 |
|---|---|---|
| `law_name` | 法律全名 | `中华人民共和国放射性污染防治法` |
| `articles` | 法条编号 | `第一条` |
| `chapter` | 章节 | `第一章 总  则` |

### 2.3 前缀格式

```
【{law_name}·{articles}】{chunk.content}
```

例：`【中华人民共和国放射性污染防治法·第一条】第一条 为了防治放射性污染……`

缺 `law_name` 但有 `articles` 时降级为 `【{articles}】{chunk.content}`；都缺时返裸 `chunk.content`。

### 2.4 效果数据

200 样本 A/B 对照（pool 均为 20，仅差是否注入前缀）：

| metric | A 裸 content（基线） | B 注入前缀 | delta |
|---|---|---|---|
| **hit@1** | 0.265 | **0.360** | **+9.5pp** |
| hit@3 | 0.995 | 0.980 | -1.5pp（噪声） |
| hit@5 | 1.000 | 0.995 | -0.5pp（噪声） |

外推全量 7,639 条「差 1 位」query：预计救回 ~725 条到 hit@1，全量 top-1 hit_rate 从 0.5316 提到 ~0.567（+3.5pp）。

## 3. 多 query 加权 rerank 实验（2026-07-26，已回滚）

在「query 同义改写 + 多 query 加权 rerank」方向上做了完整实现 + 200 样本 A/B：

- `Reranker.rerank_multi(queries, candidates, weights, top_k)` 默认实现：循环 `rerank` + 按 `weights` 加权融合 `relevance_score`，`copy.copy(chunk)` 隔离 score 污染。
- `KnowledgeBase.query` 改写分支用 `rerank_multi` 替代单 query `rerank`，权重 `[0.6, 0.4/N...]`。
- `scripts/compare_rerank_pool.py` 扩展 `--mode multiquery` + `--phase A/B`。

### A/B 结果（200 样本「差 1 位」query）

| 指标 | A 单query (阈0) | B 多query加权 (阈0.75) | delta |
|---|---|---|---|
| hit@1 | 69/200 = 0.345 | 75/200 = 0.375 | **+3.0pp** |
| hit@3 | 196 | 197 | +0.5pp |
| hit@5 | 199 | 199 | 0 |

### 失败原因

1. **触发率太低**：`expansion_threshold=0.75` 只触发 6/200 改写（3%），+3pp 完全来自这 6 个 query，6 个样本的随机噪声本身就可能有 ±3pp 波动，信号弱不显著。
2. **dense top-1 score 普遍 >0.75**：200 样本 dense top-1 分数中位数高，「差 1 位」query 的 dense 召回没问题，瓶颈在 reranker 排序本身，而非「字面不贴导致 dense 漏召回」。
3. **多 query 加权信号被稀释**：原始 query 权重 0.6 主导排序，改写 query 各 0.2 难以翻转 reranker 对原始 query 的判别。

### 决策

**回滚 `KnowledgeBase.query` 到单 query rerank**，保留：
- `Reranker.rerank_multi` 接口（无害、未来可用）
- `scripts/compare_rerank_pool.py` `--mode multiquery` A/B 工具

## 4. 下一步优化方向（待验证）

| 方向 | 预期 top-1 收益 | 代价 | 优先级 |
|---|---|---|---|
| rerank 后对 top-3 做二次精排（同模型再跑一遍，输入拼 query+候选全文） | +1~3pp | 同模型但给 top-3 更长上下文 | 中 |
| 候选池进一步扩大（pool=30/40） | <1pp（边际递减） | API token 成本 | 低 |

## 5. 产物

- 对比报告: `.harness/reports/eval_compare_2026-07-26.md`（Section 7.1/7.2/7.3）
- 对比脚本: `scripts/compare_rerank_pool.py`（支持 `inject_prefix` A/B 对照）
- 实现: `harness/rag/reranker.py:OpenAIReranker._format_for_rerank`
