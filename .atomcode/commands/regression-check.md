---
name: regression-check
description: 本地运行合同审查回归测试，对比上次基线报告，快速发现指标退化。封装 harness regression run + diff。
---

# /regression-check

本地一键跑回归，无需等 CI（CI 回归只在周一 + push main 时触发）。

## 前置检查

1. 确认在 conda 环境：`conda activate contract-harness`（或当前 shell 已激活）
2. 确认 `harness` 命令可用：`harness --version`，不可用则 `pip install -e .`
3. 确认 `OPENAI_API_KEY` 是否设置：
   - 已设置 → 真实 LLM 运行
   - 未设置 → 提示「将使用 Mock LLM 运行，仅验证流程不验证质量」

## 执行步骤

1. 生成版本号：`local-$(date +%Y%m%d-%H%M%S)`（本地运行用时间戳，避免与 CI 的 `ci-` 前缀冲突）

2. 运行回归：
   ```bash
   harness regression run examples/contracts/ --version "$VERSION"
   ```
   默认数据集 `examples/contracts/`；若用户指定 `examples/contracts_creval/` 则用那个。

3. 列出历史基线：
   ```bash
   harness regression list
   ```
   找到最近一个非 `local-` 前缀的基线（通常是上一次 CI 的 `ci-*` 或上一次本地运行）。

4. 对比：
   ```bash
   harness regression diff <上一个版本> "$VERSION"
   ```

5. 输出报告：

```
🔁 回归测试报告
================
本次版本：local-20260704-205600
对比基线：ci-20260704-1b96ea3
数据集：  examples/contracts/ (5 用例)
LLM 模式：Mock / 真实

指标变化：
  precision:  0.82 → 0.85  (+0.03) ✓
  recall:     0.78 → 0.78  ( 0.00) ✓
  f1:         0.80 → 0.81  (+0.01) ✓
  avg_latency: 1.2s → 1.3s (+0.1s) ⚠ 略增

退化项：
  ⚠ case_c 的 risk 检出数从 3 降到 2 — 建议检查 supervisor.py 最近改动

报告路径：~/.harness/reports/local-20260704-205600/
```

## 异常处理

- `harness regression run` 失败 → 显示完整 stderr，提示检查 `harness/agent/` 改动
- 无历史基线 → 跳过 diff，只输出本次结果并提示「首次运行，已建立基线」
- 数据集不存在 → 提示可用数据集：`examples/contracts/`、`examples/contracts_creval/`

## 规约参考

来自 `AGENTS.md`：
> harness regression run <dataset>
> harness regression diff <a> <b>
