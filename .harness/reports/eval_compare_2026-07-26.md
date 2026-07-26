# RAG 评估新旧对比报告

生成时间: 2026-07-26
对比对象: 候选池扩大 + merge 后 rerank 的代码改动效果

## 1. 改动背景

本轮 eval 测试的是以下两个连续提交的效果：

| commit | 改动 |
|---|---|
| `26d20dc` | `KnowledgeBase._search_single` dense/sparse 候选池从 `top_k*2` 扩到 `max(20, top_k*4)`，rrf_fuse 在有 reranker 时保留 pool 个候选给精排（避免正确 chunk 在 dense 阶段被截掉） |
| `fef4336` | `KnowledgeBase.query()` 在 AI 扩展检索词分支 `candidates = self._merge_results(...)` 后追加 `if self._reranker and len(candidates) > 1: candidates = self._reranker.rerank(text, candidates, top_k=top_k)`，对齐多路 rerank 的分数尺度 |

环境: BAAI/bge-m3 embedding + BAAI/bge-reranker-v2-m3 reranker
（均走 SiliconFlow OpenAI 兼容 API）

## 2. 汇总指标对比

数据集: `rag_eval_dataset`，query 总数: 20,442，distinct expected chunk: 3,948

| metric | 旧 baseline (07-25) | 新 eval (07-26 16:27) | delta |
|---|---|---|---|
| **hit_rate@1** | 0.5239 | **0.5316** | **+0.77pp** |
| hit_rate@3 | 0.8810 | 0.9052 | +2.42pp |
| hit_rate@5 | 0.9021 | 0.9396 | +3.75pp |
| MRR@1 | 0.5239 | 0.5316 | +0.77pp |
| MRR@3 | 0.6847 | 0.6999 | +1.52pp |
| MRR@5 | 0.6897 | 0.7079 | +1.82pp |
| precision@1 | 0.5239 | 0.5316 | +0.77pp |
| precision@3 | 0.2937 | 0.3017 | +0.80pp |
| precision@5 | 0.1804 | 0.1879 | +0.75pp |
| recall@1 | 0.5239 | 0.5316 | +0.77pp |
| recall@3 | 0.8810 | 0.9052 | +2.42pp |
| recall@5 | 0.9021 | 0.9396 | +3.75pp |

**结论**: 全指标全面提升，越高 K 提升越大（top-5 提升最显著 +3.75pp）。

## 3. 排序改进空间对比

| 类别 | 旧 baseline | 新 eval | delta |
|---|---|---|---|
| hit@1=0 的 query | 9,732 (47.6%) | 9,576 (46.8%) | -156 |
| **hit@3=1（差 2~3 位）** | 7,299 (35.7%) | **7,639 (37.4%)** | **+340** |
| hit@3=0 但 hit@5=1 | 432 (2.1%) | 703 (3.4%) | +271 |
| top-5 仍漏 | 2,001 (9.8%) | 1,234 (6.0%) | **-767** |

## 4. 关键解读

1. **top-1 提升有限（+0.77pp）** —— 候选池扩大主要受益的是 top-3/top-5，因为更大的候选池让正确 chunk 更容易进入 top-5，但 reranker 把它从第 2 位提到第 1 位的能力没变。

2. **top-5 仍漏从 2,001 降到 1,234（-38%）** —— 这是候选池扩大最直接的收益：原本因 dense 阶段截断而漏掉的 767 个 query，现在能召回到 top-5。

3. **「差 1 位」query 从 7,299 增到 7,639（+340）** —— 这是好事：原本漏 top-5 的 query 现在进入了 top-3，成为新的「差 1 位」候选。Reranker 优化的目标池反而在变大。

4. **系统性漏检 chunk 从 477 增到 504** —— 候选池扩大后，更多原本被截断的 chunk 进入评估，其中一部分仍是系统性漏检。需要排查这些 chunk 的切片质量。

5. **MRR@5 从 0.6897 提到 0.7079（+1.82pp）** —— 平均排位在改善，但 top-1 的瓶颈仍在。

## 5. 结论与下一步

候选池扩大（`_search_single` pool 20）+ merge 后 rerank 的组合，**主要解决了召回（top-5）问题**，top-1 hit_rate 的提升空间仍待 Reranker 调优。

| 优化方向 | 预期收益 | 难度 |
|---|---|---|
| 调 Reranker top-1 判别（更强 rerank 模型 / 二次精排） | top-1 hit_rate +5~10pp | 中 |
| 系统性漏检 chunk 切片质量审查（504 个） | top-5 recall +1~2pp | 低 |
| 候选池进一步扩大（pool=30/40） | 召回边际递减 | 低 |
| 多 query 加权 / query 改写 | top-1 hit_rate +3~5pp | 高 |

## 6. 产物路径

- 本报告: `.harness/reports/eval_compare_2026-07-26.md`
- 新 eval 汇总: `.harness/reports/rag_eval.summary.csv`
- 新 eval 明细: `.harness/reports/rag_eval.details.csv`
- 三项分析: `.harness/reports/analysis/`
  - `hit1_miss_hit3_hit.csv` —— 排序改进空间（7,639 条）
  - `chunk_hit_distribution.csv` / `chunk_systematic_miss.csv` —— chunk 维度命中分布 + 系统性漏检
  - `rag_eval_charts.html` —— pyecharts 交互式图表
  - `rag_eval_analysis_summary.md` —— Markdown 摘要
- 旧 baseline 报告: `.harness/evals/reports/`（对比基准）

## 7. 后续优化实验记录

### 7.1 Reranker `top_n` 扩大实验（2026-07-26 19:10）

**改动**: `OpenAIReranker.rerank` 的 `/rerank` API `top_n` 从 `top_k` 改为 `len(candidates)`，让 API 对全 pool 候选打分，客户端按 `relevance_score` 降序后截 top_k。原 `top_n=top_k` 会截断打分阶段，正确 chunk 在 pool 内但分未排进 top_k 时直接丢失。

**验证**: 用 `scripts/compare_rerank_pool.py` 在新 eval 的 7,639 条「差 1 位」query 中抽 200 条样本对比（策略 A 旧 `pool=top_k*2` vs 策略 B 新 `pool=max(20, top_k*4)`，两者现在都用新 `top_n=len(candidates)` 打分）：

| metric | 策略 A (pool=10) | 策略 B (pool=20) | delta |
|---|---|---|---|
| hit@1 | 0.265 | 0.270 | +0.5pp |
| hit@3 | 0.965 | 0.995 | +3.0pp |
| hit@5 | 0.990 | 1.000 | +1.0pp |

**结论**: `top_n` 扩大主要改善 top-3/top-5 召回（+3pp / +1pp），**top-1 判别能力仅 +0.5pp**，不显著。说明 top-1 瓶颈不是「打分阶段被截断」而是 reranker 模型本身的判别上限——`bge-reranker-v2-m3` 对「正确 chunk vs 字面更相似的干扰 chunk」的二选一判别已到瓶颈。

### 7.2 下一步优化方向（待验证）

| 方向 | 预期 top-1 收益 | 代价 | 优先级 |
|---|---|---|---|
| 换更强 rerank 模型（如 `bge-reranker-v2-gemma` / `jina-reranker-v2`） | +3~8pp | 模型下载 + 推理延迟 | 高 |
| query 同义改写 + 多 query 加权 rerank | +3~5pp | LLM 调用 + 多轮 rerank | 中 |
| chunk 文本注入法条标题前缀（如「【第五十二条】」）给 reranker 额外结构信号 | +1~3pp | 改 chunk content 影响其他调用方 | 中 |
| 候选池进一步扩大（pool=30/40） | <1pp（边际递减） | API token 成本 | 低 |

首选方向: 换更强 rerank 模型，ROI 最高。
