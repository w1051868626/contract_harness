---
name: eval-run
description: 一键运行合同审查评测。跑指定数据集 → 生成评测报告 → 汇总关键指标。支持 /eval-run <dataset> 和 /eval-run（用默认数据集 examples/contracts_creval/）。
---

# /eval-run 命令

执行合同审查 Agent 的端到端评测流程。

## 用法

```
/eval-run                       # 用默认数据集 examples/contracts_creval/
/eval-run <dataset-path>        # 跑指定数据集目录
/eval-run <dataset-path> --report  # 跑完额外生成 HTML 报告
```

## 执行步骤

当用户调用此命令时，按以下顺序执行：

1. **解析参数**
   - 无参数 → `dataset = "examples/contracts_creval/"`
   - 有参数 → 用第一个参数作为 dataset 路径
   - `--report` 标志 → 跑完调用 `harness eval report` 生成报告

2. **前置检查**
   - 确认 dataset 路径存在（`list_directory`）
   - 确认数据集含 `dataset.json` 或 `items.jsonl`（否则提示用户先 `harness eval create`）

3. **运行评测**
   ```bash
   harness eval run <dataset>
   ```
   - 捕获完整 stdout/stderr
   - 如果失败（非零退出码），读错误输出诊断根因，不要盲目重试

4. **生成报告**（仅当 `--report` 或默认数据集时）
   ```bash
   harness eval report
   ```

5. **汇总结果**

   输出表格汇总关键指标：

   | 指标 | 值 |
   |------|-----|
   | 数据集 | `<dataset>` |
   | 用例数 | N |
   | 通过数 | N |
   | 失败数 | N |
   | 准确率 | N% |
   | 风险识别 F1 | N |
   | 合规检查准确率 | N% |
   | 平均耗时 | Ns |
   | 总 token 用量 | N |

6. **失败用例下钻**（仅当有失败用例时）
   - 列出失败用例 ID + 失败原因摘要
   - 提示用户可用 `/replay-check <session_id>` 复现失败用例

## 约束

- ✅ 评测必须在隔离环境跑（`LLM_MOCK=1` 或真实 API key），不要在评测中途切换 LLM 配置
- ✅ 失败时先诊断（读 stderr），不要盲目重试
- ✅ 大数据集（>50 用例）跑前提示用户预计耗时
- ❌ 不要在评测运行中并发修改 `harness/agent/` 代码（会导致评测结果不稳定）
- ❌ 不要自动 commit 评测报告（让用户决定是否固化基线）

## 关联命令

- `/replay-check` — 复现失败用例的回放
- `/sync-baseline` — 固化当前评测结果为回归基线
- `/regression-check` — 跑回归套件对比基线
