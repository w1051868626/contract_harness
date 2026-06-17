# contract-harness 项目记忆

## 项目概述

可回放、可评测、可回归的**法律合同审查 Agent** 系统，基于自研 Agent Loop 框架（Python 3.11+）。

## 架构

```
harness/
├── agent/        合同审查 Agent（LLM 编排 + 工具调用 + 记忆）
│   └── memory.py         持久化记忆 + 自演进（ChromaDB）
├── replay/       回放系统（录制 + 回放 + 存储管理）
├── eval/         评测系统（数据集 + 指标 + 评分流水线）
├── regression/   回归系统（测试套件 + 对比器）
├── rag/          知识库（Embedding + 向量存储 + 检索 + Reranker）
├── web/          FastAPI Web 界面（审查 + 会话 + 追问）
├── cli/          命令行入口（click + rich）
├── core/         核心类型（pydantic）、配置、异常
└── utils/        工具函数（含 load_dotenv、文件读写等）
```

## 开发命令

```bash
conda activate contract-harness
pip install -e ".[dev]"       # 安装开发依赖
pip install -e ".[local]"     # 安装本地模型依赖（sentence-transformers）
git pull origin main           # 拉取最新代码
pytest tests/ -v
ruff check harness/ tests/
ruff format --check harness/ tests/
pyright harness/
```

## 关键命令

```bash
# 审查合同
harness review <file>

# 继续对话（session_id 见审查输出的「会话 ID」）
harness converse <session_id> <追问内容>

# 回放
harness replay <session_id>
harness sessions

# 评测
harness eval run <dataset>
harness eval report

# 回归
harness regression run <dataset>
harness regression diff <a> <b>

# 知识库
harness kb seed                    # 导入内置法律条文
harness kb import-file <file>      # 导入单个文件（支持 txt/md/json/pdf/docx/zip）；--docling 启用结构解析
harness kb import-dir <dir>        # 批量导入；--docling 启用结构解析
harness kb list                    # 列出文档
harness kb search <query>          # 检索

## Conda 环境

```bash
conda activate contract-harness
```

环境定义在 `environment.yml`，位于项目根目录。

## 技术栈

- Python 3.11+ / 自研 Agent Loop
- click + rich（CLI）
- openai（LLM 客户端）
- pydantic（数据模型）
- pytest（测试）
- jinja2（报告模板）
- fastapi + uvicorn（Web 界面）
- pypdf + python-docx（文档解析）
- docling（可选，高质量结构化文档解析）
- sentence-transformers（本地 embedding / reranker）
- chromadb（向量数据库）
- python-dotenv（.env 加载）
- ruff + pyright（代码规范）

## 环境变量

- `OPENAI_API_KEY` — LLM API 密钥
- `EMBEDDING_API_KEY/BASE/MODEL` — Embedding 独立密钥、地址和模型
- `LLM_PROVIDER/PROXY` — LLM 供应商和代理
- `RERANK_PROVIDER/API_KEY/API_BASE/MODEL` — Reranker 配置
- `CHUNK_API_KEY/BASE/MODEL` — AI 分块 LLM 独立配置
- `VECTOR_STORE_BACKEND` — 向量存储后端（已废弃，仅支持 chroma）
- `HTTP_PROXY` — 通用代理回退
- `HARNESS_DATA_DIR` — 数据根目录

支持 `.env` 文件（项目根目录自动加载）。

## 规则

- 每次更改后都必须提交（commit）并推送（push）到远程仓库
- 每次更新后同步更新 README.md、AGENTS.md、CLAUDE.md。
- 新增依赖时同步更新 `environment.yml` 和 `pyproject.toml`（二者必须保持一致）
- 增加新功能时必须同步添加相应的测试用例

## 目标

构建一套合同审查 Agent 的 Harness Engineering 体系，确保 Agent 可回放、可评测、可回归。

## 更新记录

- 2026-06-14: 新增 `load_dotenv()` 工具函数，CLI/Web 入口自动加载 `.env`；新增 `python-dotenv` 依赖；同步更新 pyproject.toml/environment.yml/README.md/AGENTS.md/CLAUDE.md。
- 2026-06-15: 移除 SQLite 向量存储后端，统一使用 Chroma 向量数据库；添加 ChromaVectorStore 集成测试（4 个用例）；更新 create_vector_store 及 KnowledgeBase.from_config 默认后端为 chroma。
- 2026-06-16: 全面优化——性能（风险/合规批量 LLM 调用、正则预编译、hash_embed 确定性）、安全（Web 上传大小限制、assert 替换运行时检查）、错误处理（修复静默吞异常、补全 reranker 异常处理）、代码质量（新增 llm_utils.py 消除重复、补全类型注解、清理未使用依赖 pyyaml/tabulate、Config 参数校验）；同步更新 pyproject.toml/tests/conftest.py/tests/unit/test_agent.py。
- 2026-06-17: 大规模优化——合规检查 `batch_check` 合并为单次 LLM 调用、risk_analyzer 移除单条款特殊分支、依赖清单清理（移除 pyyaml/tabulate）、新增测试覆盖（CLI/Web/Reporter/Suite/Storage 共 28 个新用例，总计 77 个）、修复 5 处静默吞异常（新增 ChunkingError/EmbeddingError 异常类）、Env 回退逻辑去重、TypedDict 注解替代 dict[str, Any]、AI 分块 `CHUNK_MAX_CHARS` 配置化支持超长文本分段、Web 日志 verbose 统一跟随 config、VectorStore.add_chunk 标记废弃、Agent 版本号从 importlib.metadata 动态获取。
- 2026-06-17: RAG 查询扩展——`KnowledgeBase.query()` 新增 `expansion_threshold` 参数，检索分数低时自动用 LLM 生成同义查询重试+合并去重；新增 6 个测试用例。
- 2026-06-17: 持久化记忆 + 自演进——新增 `MemoryStore`（ChromaDB 存储条款级分析结果），支持 `remember_session`/`recall`/`correct`/`format_memory_context`；`EvalScorer.run(learn=True)` 自动回写修正信号实现自演进；新增 9 个测试用例。
- 2026-06-17: 继续对话——`ContractAgent.converse(session_id, query)` 加载历史会话重建上下文并回答追问；CLI `harness converse` + Web `POST /sessions/{id}/converse`；对话记录持久化到 session metadata；新增 2 个测试用例。
- 2026-06-17: ReAct + Reflection 模式——新增 `AgentMode` 枚举（pipeline/react/reflection）；`react_loop.py` 实现 LLM 自主决策工具调用的 ReAct 循环；`reflection.py` 实现管道审查后追加自审修正；`ContractAgent.review()` 按 mode 自动 dispatch；`HarnessConfig` 新增 `agent_mode` 配置字段；新增 13 个测试用例，累计 107 个。
- 2026-06-17: Docling 文档解析器——新增 `DoclingParser` 封装（可选依赖），PDF/DOCX/PPTX/图片 → 结构化 Markdown；`HarnessConfig.use_docling` 配置字段；`KnowledgeBase._parse_file()` 支持 docling 优先解析，不替换原有 pypdf/python-docx 功能；新增 9 个测试用例，累计 116 个。
- 2026-06-17: CLI `kb import-file`/`import-dir` 新增 `--docling` 标志，控制台即可启用 Docling 解析。
