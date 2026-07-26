# RAG 检索优化笔记

记录 contract-harness 项目 RAG 检索质量优化的完整历程，供后续迭代参考。

## 2026-07-26 Reranker 调用侧三步优化

**约束**: API 域唯一可用 rerank 模型 = `BAAI/bge-reranker-v2-m3`（SiliconFlow OpenAI 兼容端点），**无法更换模型**。

**baseline**: 07-25 全量 eval 20,442 query，hit_rate@1=0.5239、hit_rate@5=0.9021、top-5 仍漏 2,001 条。

### Step 1: 候选池扩大（commit `26d20dc`）

`KnowledgeBase._search_single` dense/sparse 候选池从 `top_k*2` 扩到 `max(20, top_k*4)`，rrf_fuse 在有 reranker 时保留 pool 个候选给精排。

- **原理**: 给 Reranker 更大判别空间，避免正确 chunk 在 dense 阶段就被截掉。
- **200 样本 A/B**: top-1 hit_rate +5pp（35.5% → 40.5%）。
- **风险**: 低（仅扩大检索范围，无新增 API 调用）。

### Step 2: `top_n` 扩大（commit `f83b1e2`）

`OpenAIReranker.rerank` 的 `/rerank` API `top_n` 从 `top_k` 改为 `len(candidates)`，让 API 对全 pool 候选打分，客户端按 `relevance_score` 降序后截 top_k。

- **原理**: 原 `top_n=top_k` 截断打分阶段，正确 chunk 在 pool 内但分未排进 top_k 时直接丢失。
- **200 样本 A/B**: top-1 +0.5pp（不显著），top-3 +3pp、top-5 +1pp。
- **结论**: 主要改善 top-3/top-5 召回，top-1 瓶颈不是「打分阶段被截断」而是 reranker 模型本身的判别上限。
- **风险**: 低。

### Step 3: Reranker 输入注入法条标题前缀（commit `ff68af4`，**核心突破**）

`OpenAIReranker.rerank` 喂给 `/rerank` API 的 `documents` 从裸 `chunk.content` 改为 `【{law_name}·{articles}】{chunk.content}`（注入法条标题前缀）。

- **原理**: 给 cross-encoder 额外结构信号——「【中华人民共和国放射性污染防治法·第一条】正文」比裸正文更易判别 chunk 的法条归属，尤其当 query 字面与正确 chunk 不贴但语义指向同一法条时。
- **契约**: **不动 `chunk.content`**（embedding、其他调用方不受影响）；无 metadata 或缺 `articles`/`law_name` 时降级为裸 content。实现入口 `OpenAIReranker._format_for_rerank(chunk)` 静态方法。
- **200 样本 A/B**（pool 均为 20，仅差是否注入前缀）:

  | metric | A 裸 content（基线） | B 注入前缀 | delta |
  |---|---|---|---|
  | **hit@1** | 0.265 | **0.360** | **+9.5pp** |
  | hit@3 | 0.995 | 0.980 | -1.5pp（噪声） |
  | hit@5 | 1.000 | 0.995 | -0.5pp（噪声） |

- **外推全量**: 7,639 条「差 1 位」query 预计救回 ~725 条到 hit@1，全量 top-1 hit_rate 从 0.5316 提到 ~0.567（+3.5pp）。
- **风险**: 低（仅多几字符 token，无新增 API 调用）。

### 组合效果（全量 20,442 query）

| metric | baseline (07-25) | 新代码 (07-26) | delta |
|---|---|---|---|
| hit_rate@1 | 0.5239 | 0.5316 | +0.77pp |
| hit_rate@3 | 0.8810 | 0.9052 | +2.42pp |
| hit_rate@5 | 0.9021 | 0.9396 | +3.75pp |
| top-5 仍漏 | 2,001 | 1,234 | -38% |

候选池扩大主要解决召回（top-5），标题前缀注入是 top-1 瓶颈的有效突破口。

## 下一步优化方向（待验证）

| 方向 | 预期 top-1 收益 | 代价 | 优先级 |
|---|---|---|---|
| query 同义改写 + 多 query 加权 rerank | +3~5pp | LLM 调用 + 多轮 rerank | 中 |
| rerank 后对 top-3 做二次精排（同模型再跑一遍，输入拼 query+候选全文） | +1~3pp | 同模型但给 top-3 更长上下文 | 中 |
| 候选池进一步扩大（pool=30/40） | <1pp（边际递减） | API token 成本 | 低 |

## 产物索引

| 产物 | 路径 |
|---|---|
| 对比报告 | `.harness/reports/eval_compare_2026-07-26.md` |
| 设计文档 | `specs/reranker.md` |
| 对比脚本 | `scripts/compare_rerank_pool.py`（支持 `inject_prefix` A/B 对照） |
| 实现 | `harness/rag/reranker.py:OpenAIReranker._format_for_rerank` |
| 本笔记 | `docs/optimization_notes.md` |
