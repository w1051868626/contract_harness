---
name: sync-baseline
description: 跑全套 pytest 测试并把结果固化为新的回归基线（写入 .regression_baseline/baseline.json 并提交）。用于功能稳定后更新基线供后续回归对比。
---

# /sync-baseline

把当前测试全跑通的结果固化为 contract_harness 的新回归基线。

## 执行步骤

1. **前置检查**：
   - 确认在 `contract-harness` conda 环境（`conda activate contract-harness`）
   - 确认 `LLM_MOCK=1`（避免烧真实 API 额度，回归基线应可离线复现）

2. **跑全套测试**：
   ```bash
   pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_sync.log
   ```
   - 若有失败：**停止**，报告失败用例，不更新基线
   - 若全过：进入下一步

3. **生成 baseline.json**：
   ```python
   {
     "updated_at": "<ISO 8601>",
     "git_head": "<git rev-parse HEAD>",
     "total_tests": <N>,
     "passed": <N>,
     "failed": 0,
     "duration_secs": <float>,
     "pytest_version": "<pytest --version>",
     "llm_mock": true
   }
   ```
   写入 `.regression_baseline/baseline.json`

4. **提交基线**：
   ```bash
   git add .regression_baseline/baseline.json
   git commit -m "chore: sync regression baseline

   total: <N> tests, all passing
   head: <git HEAD>"
   ```
   （遵循 AGENTS.md 规则：先更 README/AGENTS/CLAUDE 再提交——本命令仅同步基线，无功能变更，README/AGENTS/CLAUDE 无需改动）

5. **输出汇总**：
   ```
   ✅ 回归基线已同步
   total: <N> tests
   duration: <secs>s
   git head: <sha>
   committed: <commit sha>
   ```

## 边界

- 只在 `LLM_MOCK=1` 下运行（基线必须可离线复现，不依赖外部 API）
- 不跑 `pytest --cov`（覆盖率不纳入基线）
- 不更新 `.regression_baseline/` 下其他文件（只更新 baseline.json）
- 若任一测试失败，**不**更新基线，只报告失败用例让用户先修
