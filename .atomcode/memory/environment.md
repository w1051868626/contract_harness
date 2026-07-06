# Python 环境与运行环境记忆

> 跨会话持久化的项目运行环境信息，避免每次重新摸索。

## 2026-07-06: 项目 Python 环境

### conda 环境
- **环境名称**：`contract-harness`
- **环境路径**：`D:\Programs\miniforge\envs\contract-harness`
- **Python 版本**：3.11.*（最低 3.11+）
- **定义文件**：项目根目录 `environment.yml`

### 激活方式
```bash
conda activate contract-harness
```

### 依赖安装
```bash
pip install -e ".[dev]"       # 开发依赖
pip install -e ".[local]"     # 本地模型依赖（sentence-transformers）
```

### 依赖清单（三处必须保持一致，AGENTS.md 规则）
- `environment.yml`（conda 环境定义）
- `pyproject.toml`（Python 打包元数据）
- `requirements.txt`（pip 安装列表）

### 行为准则（以后必须遵守）
1. **跑任何项目命令前**先 `conda activate contract-harness`，再执行 `harness ...` / `pytest` / `ruff` / `pyright`。
2. **新增依赖时**必须同步更新 `pyproject.toml`、`environment.yml`、`requirements.txt` 三处，并跑 `sync-deps` skill 校验一致性。
3. **每次代码变更后**先更新 `README.md`、`AGENTS.md`、`CLAUDE.md`，再 `commit` + `push`（AGENTS.md 明确规定）。

### 平台注意
- 项目在 Windows 上开发，`bash` 工具走 Git Bash（POSIX 路径 `/c/` 或正斜杠 `C:/`），不要用 `cmd.exe` 命令。
- ChromaDB 在 Windows 上有 sqlite3 文件锁定问题，已通过 `ChromaVectorStore.close()` 调 `client.close()` 修复（2026-07-05 决策，见 `decisions.md`）。
