# contract-harness

可回放、可评测、可回归的法律合同审查 Agent 系统，基于自研 Agent Loop 框架（Python 3.11+）。

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

## 关键文件

- `harness/rag/knowledge_base.py` — KnowledgeBase，`from_config()` 类方法创建实例，`_parse_file()` 支持 txt/md/json/pdf/docx/zip
- `harness/rag/embedding.py` — EmbeddingProvider / OpenAIEmbeddingProvider（离线 hash 回退）
- `harness/rag/reranker.py` — Reranker ABC / OpenAIReranker / LocalReranker
- `harness/rag/seed_laws.py` — 7 部内置法律种子数据
- `harness/rag/vector_store.py` — Chroma 向量存储（ANN 近似搜索）
- `harness/cli/main.py` — 所有 Click CLI 命令入口（408 行）
- `harness/agent/llm.py` — LLMClient（无 key 时自动回退 Mock 响应）
- `harness/agent/memory.py` — MemoryStore（ChromaDB 持久化记忆 + 自演进）
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
harness kb seed                            # 导入内置法律条文
harness kb import-file <file>              # 导入单个文件（支持 txt/md/json/pdf/docx/zip）；--docling 启用结构解析
harness kb search <query>                  # 检索知识库
harness review <file>                      # 审查合同
harness converse <session_id> <query>      # 继续对话
harness replay <session_id>               # 回放会话
harness eval run <dataset>                 # 评测
harness regression run <dataset>           # 回归测试
harness serve                              # 启动 Web 界面
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | LLM API 密钥 |
| `EMBEDDING_API_KEY/BASE/MODEL` | Embedding 独立密钥、地址和模型 |
| `LLM_PROVIDER` / `LLM_PROXY` | LLM 供应商和代理 |
| `RERANK_PROVIDER/API_KEY/API_BASE/MODEL` | Reranker 配置 |
| `HTTP_PROXY` | 通用代理回退 |
| `HARNESS_DATA_DIR` | 数据根目录（默认项目下 `.harness/`） |
| `CHUNK_API_KEY` | AI 分块 API 密钥 |
| `CHUNK_API_BASE` | AI 分块 API 地址 |
| `CHUNK_MODEL` | AI 分块模型 |
| `VECTOR_STORE_BACKEND` | 向量存储后端（已废弃，仅支持 chroma） |

### .env 文件

项目启动时（CLI / Web）自动调用 `harness.utils.io.load_dotenv()`，从项目目录向上搜索 `.env` 并加载。支持 `CHUNK_API_KEY`、`CHUNK_API_BASE` 等环境变量。

## 约定

- **⚠️ 每次变更后必须先更新 README.md、AGENTS.md、CLAUDE.md，再 commit + push**
- 新增依赖同步更新 `environment.yml` 和 `pyproject.toml`
- 新增功能必须添加测试用例
- docstring 必须在 `from __future__ import annotations` 之前（ruff E402）
- LLMClient 空 key 延迟到 `client` 属性报错（非 `__init__`）
- 数据目录默认 `project_root/.harness/`，可被 `HARNESS_DATA_DIR` 覆盖
- CI type-check 安装 `.[dev]`

## 更新记录

- 2026-06-14: 新增 `load_dotenv()` 工具函数（`harness/utils/io.py`），CLI 入口和 Web 入口自动加载 `.env`；新增依赖 `python-dotenv`；同步更新文档。
- 2026-06-15: 移除 SQLite 向量存储后端，统一使用 Chroma 向量数据库；添加 ChromaVectorStore 集成测试（4 个用例）；更新 create_vector_store 及 KnowledgeBase.from_config 默认后端为 chroma。
- 2026-06-16: 全面优化——性能（风险/合规批量 LLM 调用、正则预编译、hash_embed 确定性）、安全（Web 上传大小限制、assert 替换运行时检查）、错误处理（修复静默吞异常、补全 reranker 异常处理）、代码质量（新增 llm_utils.py 消除重复、补全类型注解、清理未使用依赖 pyyaml/tabulate、Config 参数校验）；同步更新 pyproject.toml/tests/conftest.py/tests/unit/test_agent.py。
- 2026-06-17: 大规模优化——合规检查 `batch_check` 合并为单次 LLM 调用、risk_analyzer 移除单条款特殊分支、依赖清单清理（移除 pyyaml/tabulate）、新增测试覆盖（CLI/Web/Reporter/Suite/Storage 共 28 个新用例，总计 77 个）、修复 5 处静默吞异常（新增 ChunkingError/EmbeddingError 异常类）、Env 回退逻辑去重、TypedDict 注解替代 dict[str, Any]、AI 分块 `CHUNK_MAX_CHARS` 配置化支持超长文本分段、Web 日志 verbose 统一跟随 config、VectorStore.add_chunk 标记废弃、Agent 版本号从 importlib.metadata 动态获取。
- 2026-06-17: RAG 查询扩展 + 持久化记忆 + 继续对话。新增 `expansion_threshold` 参数、MemoryStore、`converse()` 方法；CLI/Web 全面打通；累计 94 个测试用例。
- 2026-06-17: ReAct + Reflection 模式。`AgentMode` 枚举、`react_loop.py`、`reflection.py`；新增 13 个测试，累计 107 个。
- 2026-06-17: Docling 文档解析器。`DoclingParser`（可选依赖），`HarnessConfig.use_docling`；新增 9 个测试，累计 116 个。
- 2026-06-18: `_chunk_markdown` 仅响应 `#` 标题。`heading_pat` 缩减为仅 `#{1,6}`；无 `#` 的结构（`第一章`、`一、`、`1.` 等）回退到 `_chunk_legal_text` / `_chunk_text`；同步更新 specs/chunking.md 及 5 个测试用例。
