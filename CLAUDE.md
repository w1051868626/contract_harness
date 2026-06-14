# contract-harness

可回放、可评测、可回归的法律合同审查 Agent 系统，基于自研 Agent Loop 框架（Python 3.11+）。

## 架构

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

## 关键文件

- `harness/rag/collector.py` — 法律条文采集器（SeedLawSource / NPCLawSource / PlaywrightNPCSource / LocalFileSource），`pw-npc` 源用 Playwright 建立浏览器 session 后提取 cookies，再用 httpx 调 NPC API
- `harness/rag/knowledge_base.py` — KnowledgeBase，`from_config()` 类方法创建实例
- `harness/rag/embedding.py` — EmbeddingProvider / OpenAIEmbeddingProvider（离线 hash 回退）
- `harness/rag/reranker.py` — Reranker ABC / OpenAIReranker / LocalReranker
- `harness/rag/seed_laws.py` — 7 部内置法律种子数据
- `harness/rag/vector_store.py` — SQLite 向量存储
- `harness/cli/main.py` — 所有 Click CLI 命令入口（408 行）
- `harness/agent/llm.py` — LLMClient（无 key 时自动回退 Mock 响应）
- `harness/core/config.py` — HarnessConfig / LLMConfig / EmbeddingConfig
- `debug.py` — PyCharm 调试入口

## 开发命令

```bash
conda activate contract-harness
pip install -e ".[dev]"        # 开发依赖
pip install -e ".[local]"      # 本地模型
pytest tests/ -v
ruff check harness/ tests/
ruff format --check harness/ tests/
pyright harness/
```

## 关键命令

```bash
harness collect --source seed              # 内置种子采集
harness collect --source npc --query 民法典  # httpx 采集（有 cookie 时可用）
harness collect --source pw-npc --query 民法典  # Playwright 浏览器采集
harness kb seed                            # 导入内置法律条文
harness kb search <query>                  # 检索知识库
harness review <file>                      # 审查合同
harness replay <session_id>               # 回放会话
harness eval run <dataset>                 # 评测
harness regression run <dataset>           # 回归测试
harness serve                              # 启动 Web 界面
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | LLM API 密钥 |
| `EMBEDDING_API_KEY/BASE` | Embedding 独立密钥和地址 |
| `LLM_PROVIDER` / `LLM_PROXY` | LLM 供应商和代理 |
| `RERANK_PROVIDER/API_KEY/API_BASE/MODEL` | Reranker 配置 |
| `HTTP_PROXY` | 通用代理回退 |
| `HARNESS_DATA_DIR` | 数据根目录（默认项目下 `.harness/`） |

## 约定

- 每次更改后 commit + push
- 新增依赖同步更新 `environment.yml` 和 `pyproject.toml`
- 新增功能必须添加测试用例
- docstring 必须在 `from __future__ import annotations` 之前（ruff E402）
- LLMClient 空 key 延迟到 `client` 属性报错（非 `__init__`）
- 数据目录默认 `project_root/.harness/`，可被 `HARNESS_DATA_DIR` 覆盖
- CI type-check 安装 `.[dev,local]` 以引入 playwright

## 已知问题

- 国家法律法规数据库（flk.npc.gov.cn）有反爬保护
- `NPCLawSource`（`--source npc`）直接 httpx 请求返回 HTML，需 `--source pw-npc`
- `PlaywrightNPCSource` 某些 Playwright 版本 `page.goto()` 有 Union 序列化缺陷，使用 `page.evaluate("window.location.href = ...")` + `context.cookies()` + httpx 替代
- 安装 playwright 后需要运行 `playwright install chromium`
