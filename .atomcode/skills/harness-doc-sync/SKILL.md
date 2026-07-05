---
name: harness-doc-sync
description: 校验本次代码变更是否已同步更新 README.md、AGENTS.md、CLAUDE.md，并提示缺失项。AGENTS.md 明确规定「每次变更后必须先更新 README.md、AGENTS.md、CLAUDE.md，再提交并推送」。
user_invocable: true
disable_model_invocation: false
---

# harness-doc-sync

校验 contract_harness 项目的「三文档同步」规约是否被遵守。

## 触发时机

- 用户在 commit 前手动调用 `/harness-doc-sync`
- AtomCode 在准备 `git commit` 前自动调用

## 执行步骤

1. 运行 `git diff --name-only HEAD` 列出本次变更的文件。
2. 把变更文件分为三类：
   - **代码**：`harness/**/*.py`、`tests/**/*.py`
   - **依赖**：`pyproject.toml`、`environment.yml`、`requirements.txt`
   - **文档**：`README.md`、`AGENTS.md`、`CLAUDE.md`、`specs/**/*.md`
3. 判断是否需要更新文档：
   - 新增/删除了 `harness/` 下的模块 → 必须更新 `README.md` 架构图 + `AGENTS.md` 架构章节
   - 修改了 `harness/agent/`、`harness/rag/`、`harness/eval/`、`harness/eval_rag/`、`harness/regression/`、`harness/replay/`、`harness/web/`、`harness/cli/` 任一子模块的公开行为 → 必须更新 `README.md` + `AGENTS.md`
   - 新增/变更依赖 → 必须更新 `pyproject.toml` + `environment.yml` + `requirements.txt`（三者一致）+ `AGENTS.md` 技术栈/环境变量 + `README.md` 安装说明
   - 新增/变更环境变量 → 必须更新 `AGENTS.md` 环境变量章节 + `.env.example`
   - 新增测试用例 → 必须更新 `AGENTS.md` 更新记录里的「累计 N 个测试用例」计数
4. 检查 `AGENTS.md` 末尾的「更新记录」是否有今天日期的条目（格式 `- YYYY-MM-DD: ...`）。若没有，提示用户补一条。
5. 输出报告：

```
文档同步检查
============
代码变更：N 个文件
  - harness/agent/xxx.py
  - ...
文档状态：
  ✓ README.md      已更新 / ⚠ 未更新（需补充：架构图、安装说明）
  ✓ AGENTS.md      已更新 / ⚠ 未更新（需补充：更新记录条目、累计测试数）
  ✓ CLAUDE.md      已更新 / ⚠ 未更新
依赖一致性：
  ✓ pyproject.toml / environment.yml / requirements.txt 三者一致
  ⚠ 检测到不一致：pyproject.toml 新增 chromadb>=0.5，environment.yml 未同步

行动建议：
  1. 在 README.md 的「架构」章节补充 ...
  2. 在 AGENTS.md 末尾追加 `- 2026-07-04: ...` 条目
  3. ...
```

## 重要约束

- **不要自动修改文档内容**——只报告缺失项与建议位置，让用户或后续编辑步骤补全。文档措辞需要人工把关。
- **不要阻止 commit**——这是提醒型 skill，不是 hook。报告完即可。
- 若所有文档都已同步，输出 `✓ 文档同步检查通过，可以提交。` 并结束。

## 参考规约

来自 `AGENTS.md`：
> - **⚠️ 每次变更后必须先更新 README.md、AGENTS.md、CLAUDE.md，再提交（commit）并推送（push）**
> - 新增依赖时同步更新 `pyproject.toml`、`environment.yml`、`requirements.txt`（三者必须保持一致）
> - 增加新功能时必须同步添加相应的测试用例
