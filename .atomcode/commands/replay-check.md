---
name: replay-check
description: 回放最近一次合同审查会话并对比关键指标，快速发现审查质量退化。封装 harness sessions → harness replay <id> → 与录制元数据对比。
---

# /replay-check

回放最近一次 session，对比审查耗时、风险条数、合规命中数等关键指标，发现退化。

## 执行步骤

### 1. 列出最近会话

```bash
harness sessions 2>&1 | head -20
```

从输出中提取最近一个 `session_id`（第一行非表头记录）。若没有 session，提示用户先跑一次 `harness review <file>` 生成会话，然后退出。

### 2. 回放该会话

```bash
harness replay <session_id>
```

捕获完整输出。重点提取：
- 审查耗时（`elapsed_ms` 或类似字段）
- 风险条数（`risks` 列表长度）
- 合规命中数（`compliance` 列表长度）
- clause 总数
- overall_risk 等级

### 3. 对比录制元数据

读取 `.harness/sessions/<session_id>/` 下的录制文件（`meta.json` 或类似），对比回放输出与原始录制：

| 指标 | 录制值 | 回放值 | 差异 |
|------|--------|--------|------|
| elapsed_ms | … | … | … |
| risk_count | … | … | … |
| compliance_count | … | … | … |
| clause_count | … | … | … |

### 4. 报告

- 若所有指标差异 < 5%，输出 `✅ 回放一致性良好`。
- 若风险/合规条数变化，标记 `⚠️ 审查结果退化`，并提示检查 `harness/agent/` 或 `harness/eval/scorer.py` 改动。
- 若回放直接报错，输出完整 stderr，提示检查 `harness/replay/` 模块。

## 边界情况

- **session 不存在**：`harness replay` 会抛 `FileNotFoundError`，捕获并提示 `会话 <id> 不存在，请用 harness sessions 查看可用会话`。
- **录制损坏**：JSON 解析失败时，提示 `录制文件损坏，建议重新审查生成新 session`。
- **无 session 可回放**：步骤 1 输出为空，直接退出并提示用户先跑 `harness review`。
