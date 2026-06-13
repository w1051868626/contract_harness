# contract-harness 项目记忆

## 项目概述
可回放、可评测、可回归的**法律合同审查 Agent** 系统，基于自研 Loop 框架（Python 3.11+）。

## 目录结构
```
harness/
├── agent/        合同审查 Agent（LLM 编排 + 工具调用）
├── replay/       回放系统（录制 + 回放 + 存储管理）
├── eval/         评测系统（数据集 + 指标 + 评分流水线）
├── regression/   回归系统（测试套件 + 对比器）
├── rag/          知识库（Embedding + 向量存储 + 检索 + Reranker）
├── web/          FastAPI Web 界面（审查 + 会话）
├── cli/          命令行入口（click + rich）
├── core/         核心类型（pydantic）、配置、异常
└── utils/        工具函数
```

## 关键命令
```bash
# 审查合同
harness review <file>

# 回放
harness replay <session_id>
harness sessions

# 评测
harness eval run <dataset>
harness eval report

# 回归
harness regression run <dataset>
harness regression diff <a> <b>
```

## Conda 环境
```bash
conda activate contract-harness
```

环境定义在 `environment.yml`，位于项目根目录。

## 开发命令
```bash
conda activate contract-harness
pip install -e ".[dev]"       # 安装开发依赖
pip install -e ".[local]"     # 安装本地模型依赖（sentence-transformers）
pytest tests/ -v
ruff check harness/ tests/
ruff format --check harness/ tests/
pyright harness/
```

## 环境变量
- `OPENAI_API_KEY` — LLM API 密钥
- `EMBEDDING_API_KEY/BASE` — Embedding 独立密钥和地址
- `LLM_PROVIDER/PROXY` — LLM 供应商和代理
- `RERANK_PROVIDER/API_KEY/API_BASE/MODEL` — Reranker 配置
- `HTTP_PROXY` — 通用代理回退

## 技术栈
- Python 3.11+ / 自研 Agent Loop
- click + rich（CLI）
- openai（LLM 客户端）
- pydantic（数据模型）
- pytest（测试）
- jinja2（报告模板）
- fastapi + uvicorn（Web 界面）
- pypdf + python-docx（文档解析）
- sentence-transformers（本地 embedding / reranker）
- ruff + pyright（代码规范）

## 规则
- 每次更改后都必须提交（commit）并推送（push）到远程仓库
- 每次更新后同步更新 README.md
- 新增依赖时同步更新 `environment.yml` 和 `pyproject.toml`（二者必须保持一致）

## 目标
构建一套合同审查 Agent 的 Harness Engineering 体系，确保 Agent 可回放、可评测、可回归。
