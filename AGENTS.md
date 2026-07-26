# contract-harness 项目记忆

## 项目概述

可回放、可评测、可回归的**法律合同审查 Agent** 系统，基于自研 Agent Loop 框架（Python 3.11+）。

## 架构

```
harness/
├── agent/        合同审查 Agent（LLM 编排 + 工具调用 + 记忆）
│   ├── multi_agent/     多 Agent 协同（Worker/Supervisor/CrossValidator）
│   └── memory.py         持久化记忆 + 自演进（ChromaDB）
├── replay/       回放系统（录制 + 回放 + 存储管理）
├── eval/         评测系统（数据集 + 指标 + 评分流水线）
├── eval_rag/     RAG 检索质量评估（数据模型 + 指标 + 生成 + 执行 + 报告）
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

## 查看 GitHub Actions

```bash
# 查看最新 5 次运行状态
curl -s https://api.github.com/repos/w1051868626/contract_harness/actions/runs?per_page=5 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for run in data.get('workflow_runs', []):
    print(f\"{run['name']}: {run['conclusion']} ({run['status']})\")
    print(f\"  commit: {run['head_commit']['message'][:60]}\")
    print(f\"  url: {run['html_url']}\")
"

# 查看某次运行中失败的 job
curl -s https://api.github.com/repos/w1051868626/contract_harness/actions/runs/<RUN_ID>/jobs | python3 -c "
import json, sys
data = json.load(sys.stdin)
for job in data.get('jobs', []):
    print(f\"Job: {job['name']} - {job['conclusion']}\")
    for step in job.get('steps', []):
        if step['conclusion'] == 'failure':
            print(f\"  Failed step: {step['name']}\")
"
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
harness kb import-file <file>      # 导入单个文件（支持 txt/md/json/pdf/docx/zip）；--docling 启用结构解析；--work-dir 指定临时目录
harness kb import-dir <dir>        # 批量导入；--docling 启用结构解析；--work-dir 指定临时目录
harness kb list-docs               # 列出文档
harness kb search <query>          # 检索
harness kb eval generate           # 从 KB 自动生成评估数据集
harness kb eval run <dataset>      # 执行 RAG 检索质量评估

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
- rank-bm25（稀疏检索，混合检索用）
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
- `EMBEDDING_MAX_RPM` — Embedding API 每分钟最大请求数（0=不限）
- `EMBEDDING_MAX_TPM` — Embedding API 每分钟最大 Token 数（0=不限）

支持 `.env` 文件（项目根目录自动加载）。

## 规则

- **⚠️ 每次变更后必须先更新 README.md、AGENTS.md、CLAUDE.md，再提交（commit）并推送（push）**
- 新增依赖时同步更新 `pyproject.toml`、`environment.yml`、`requirements.txt`（三者必须保持一致）
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
- 2026-06-18: 法律文本切片重构——参考 legal_rag 实现"编→章→节→条→款→项"递归层级分割（RecursiveCharacterTextSplitter 风格），替代原有单层 split+merge 方案；新增 `_hierarchical_split`、`_split_keep_separator`、`_extract_law_metadata`、`_extract_case_metadata`、`_inject_contextual_header` 五个辅助方法；`_chunk_markdown` 新增 section 变化 flush、章节变化 flush 后 continue 修复（防止跨章节合并）；新增 4 个扩展标题模式测试用例；累计 129 个测试用例（后续 2026-06-19 增至 139）；同步更新 specs/chunking.md。
- 2026-06-18: 对齐依赖清单——`pyproject.toml`、`environment.yml`、`requirements.txt` 三者一致；environment.yml 分组注释核心/开发/可选依赖；新增 requirements.txt；AGENTS.md 规则更新。
- 2026-06-18: `_chunk_markdown` 的 `heading_pat` 缩减为仅 `#{1,6}`——只有包含 Markdown `#` 标题的文档才走 Markdown 分块，无 `#` 的 `第一章`、`一、`、`1.` 等结构回退到 `_chunk_legal_text` / `_chunk_text`；同步更新 specs/chunking.md 及 5 个测试用例。
- 2026-06-18: 提取 `MetaKey`/`DocType` 枚举 + 新建 `harness/rag/constants.py`——所有全局常量和枚举集中管理；`_chunk_markdown` 内部函数 `_detect_meta` 提升为类级静态方法 `_detect_md_heading_meta`；缩减代码约 40 行。
- 2026-06-19: Embedding 截断 + openai 库替换——`add_text` 入口清洗全角空格 `\u3000`；`EMBED_MAX_CHARS=1024` 句子边界截断；`OpenAIEmbeddingProvider` 改用 `openai` 库替代裸 httpx；`_chunk_legal_text` 补上 `overlap` 支持；章/节元数据提取从首行改为扫描 part 全部行，修复跨章节合并时的元数据丢失；累计 139 个测试用例。
- 2026-06-22: 新增评测数据集 `examples/contracts_creval/`——基于开源 [Contract-Reviewer-Agent-Eval](https://github.com/evan66547/Contract-Reviewer-Agent-Eval) 的 25 个中文民法典测试用例（MIT），覆盖违约责任、越权担保、数据合规、竞业限制等高风险条款场景；`harness eval run examples/contracts_creval/` 即可使用。
- 2026-06-29: 混合检索（稠密 + BM25 稀疏 + RRF 融合）——新增 `SparseRetriever`（`harness/rag/sparse.py`）、`rrf_fuse` RRF 融合函数；`KnowledgeBase._search_single()` 支持双路检索；`EmbeddingConfig` 新增 `enable_hybrid_search` / `rrf_k` 配置；新增 `rank-bm25` 依赖；新增 7 个测试用例，累计 146 个。
- 2026-07-01: Multi-Agent 协同审查——新增 `AgentMode.MULTI_AGENT` 模式；`WorkerAgent`（ClauseExpert/RiskExpert/ComplianceExpert 三个专业子 Agent，独立 LLM + system prompt）；`SupervisorAgent`（任务分配+分歧检测+报告合成）；`CrossValidator`（规则优先+LLM 兜底仲裁）；`MultiAgentCoordinator`（7 阶段全流程编排）；新增 20 个测试用例，累计 166 个。
- 2026-07-01: ZIP 解压改用 `TemporaryDirectory` 保留原始文件名；`extract_zip_texts`/`add_zip` 新增 `--work-dir` 参数指定临时目录。
- 2026-07-01: Embedding 速率限制——`OpenAIEmbeddingProvider` 集成滑动窗口速率限制器；`EmbeddingConfig` 新增 `max_rpm`/`max_tpm` 配置；支持 `EMBEDDING_MAX_RPM`/`EMBEDDING_MAX_TPM` 环境变量。
- 2026-07-02: 新增 RAG 检索质量评估模块——`harness/eval_rag/`（数据模型 + JSONL I/O + 指标 + LLM 生成器 + 执行器 + 报告器 + CLI）；同步更新文档；新增 13 个测试用例，累计 179 个。
- 2026-07-04: P0 架构审查修复——`extract_json_array`/`extract_json_object` 改用括号配平算法替代非贪婪正则，修复 JSON 字符串内含 `]`/`}` 时解析被截断导致静默回退逐条 LLM 调用（成本/延迟暴增）；`LLMClient` 显式 `mock=True` 模式取代密钥缺失静默回退，`client` 属性返回 `OpenAI | None` 消除 `# type: ignore`，并区分 `APIConnectionError`/`RateLimitError`/`APIError` 抛 `AgentError`；`SupervisorAgent.synthesize_report` 风险对齐从下标改为 `clause_index` 优先 + `clause_type` 回退，修复 RiskExpert 输出顺序/数量与 ClauseExpert 不一致时 risk 挂错条款；`EvalScorer._feed_corrections` 同 type 多条款轮转消费 `type_to_indices` 队列而非全部对齐第一个，避免修正信号错位污染记忆库；新增 `tests/unit/test_llm_utils.py`（9 用例）+ 多 Agent 对齐回归（2 用例）+ scorer 对齐回归（3 用例），累计 193 个。
- 2026-07-04: P0 对齐 bug 修复——`EvalScorer._feed_corrections` 的 `type_to_indices` 队列原被 `expected_risks`/`expected_compliance` 共用 `pop(0)`，risks 循环消费完后 compliance 循环因队列已空而对齐失败、修正信号丢失；改为按 field（`risk`/`compliance`）维护独立消费游标，互不抢占；`SupervisorAgent.validate_consensus` 从 `range(max(len))` 下标强行对齐改为 `clause_index` 优先 + `clause_type` 回退、按 key 分组逐 key 比对，与 `synthesize_report` 对齐逻辑一致，修复 RiskExpert/ComplianceExpert 输出顺序或数量不一致时产出假分歧/漏掉真分歧；新增 scorer 队列独立回归（1 用例）+ consensus 错位回归（1 用例）+ 改写共识分歧用例，累计 195 个。
- 2026-07-04: 代码坏味道修复——`LLMClient.client` 属性 mock 模式短路返回 None，避免每次访问重跑密钥检查；`LLMClient.chat` 新增 `max_retries` 参数（默认 3），对网络/限流瞬时错误指数退避重试，鉴权/请求格式等非瞬时错误不重试直接抛 `AgentError`，`max_retries=1` 时关闭重试抛原始包装错误；可重试异常（`APIConnectionError`/`APITimeoutError`/`RateLimitError`）必须在 `APIError` 之前捕获避免被兜底走"不重试"路径；`SupervisorAgent.synthesize_report` 的 `clauses_by_type` 从字典改为按 type 队列轮转消费，修复同 type 多条款全部对齐到第一个；删除 `llm_utils.py` 中已无引用的 `_JSON_ARRAY_RE`/`_JSON_OBJECT_RE` 死代码；新增重试成功/重试耗尽/supervisor 轮转回归（3 用例），改写限流测试用 `max_retries=1` 避免真 sleep，累计 198 个。
- 2026-07-05: CI 回归测试修复——`LLMClient` 密钥缺失从静默回退改为抛 `AgentError` 后，CI 无 `OPENAI_API_KEY` secret 时回归 workflow 直接失败；新增 `LLM_MOCK` 环境变量（`1`/`true`/`yes`）自动启用 mock 模式，回归 workflow 在无 secret 时 `export LLM_MOCK=1` 跑通；修复 `ruff format --check`（4 文件格式化）和 `pyright`（`llm.py` `self.client` 可空访问，提取局部变量 + `assert is not None` 缩窄类型）；新增 `LLM_MOCK` 启用/默认禁用回归（2 用例），累计 200 个。
- 2026-07-05: 测试隔离修复——`tests/conftest.py` 新增 autouse fixture `_isolate_llm_env`，全局强制 `LLM_MOCK=1` + 清空所有 provider key 环境变量（`OPENAI_API_KEY`/`DEEPSEEK_API_KEY`/`LLM_PROVIDER`/`LLM_PROXY` 等），防止测试误读 `.env` 真实 API 密钥烧钱；`TestLLMClientErrorHandling` 类级 fixture 覆盖关闭全局 mock，便于测真实错误路径（密钥缺失/API 异常/重试），所有 `LLMClient` 构造显式传 `mock=False`；`test_converse_nonexistent_session` 改用 `MockLLMClient` 隔离；清理 `test_p0_fixes.py` 冗余 `patch.dict`/`import os`。本地验证：92 用例 1.26s 通过无真实 API 调用，即使 `DEEPSEEK_API_KEY=sk-fake` 环境变量下仍隔离生效。
- 2026-07-05: `test_kb_eval_generate_no_kb` Windows 失败修复——`RagDatasetGenerator.generate` 原对空 KB 静默成功（exit 0），测试却断言 `exit_code != 0` + `"Missing" in exception`，且测试未隔离 `.harness/` 全局状态，读到残留 19MB 损坏 JSONL（中途混入非法字节）抛 `UnicodeDecodeError`；修复：`generate` 在 `kb.list_chunks()` 为空时抛 `EvalError("Missing knowledge base chunks: ...")` 提示用户先 `kb seed`，测试用 `monkeypatch.setenv("HARNESS_DATA_DIR", tmp_path)` 隔离到临时空目录确保前置条件成立。
- 2026-07-05: ChromaDB Windows 文件锁定修复——`ChromaVectorStore.close()` 原为空实现（`pass`），ChromaDB `PersistentClient` 内部 sqlite3 连接未释放，Windows 上 `TemporaryDirectory` 清理时 `chroma.sqlite3` 被占用抛 `PermissionError [WinError 32]`（17 个 test_rag + 3 个 test_memory 失败）；修复：`close()` 调用 ChromaDB 1.5+ 的 `client.close()`（用 `hasattr` 防御 + `pyright: ignore` 绕过存根缺失）释放 sqlite3 连接；`MemoryStore` 新增 `close()` 转发；`test_rag.py` 4 个 `test_query_expansion_*` + `test_memory.py` 3 个测试补 `store.close()`。本地验证：227 用例全过（从 17+3 失败降至 0），lint + pyright 全清。
- 2026-07-05: 代码坏味道修复（第二轮）——`ContractAgent._review_pipeline` 的 step4 重复构造（`_step_report_generation` 内部构造的 step4 没用，外层又重新构造），改为方法返回 step 外层直接用；`ComplianceChecker.check` 的 `if clause else []` 死分支（`Clause` 对象恒 truthy）+ 空 clauses 时 `[0]` IndexError 风险，去掉死分支；`RiskAnalyzer._parse_batch_response` 的 `items = {item.get("index"): item ...}` dict comprehension 同 index 静默覆盖丢失，改为显式循环保留首个；新增 `test_batch_analyze_duplicate_index_keeps_first` 回归（1 用例），累计 228 个。
- 2026-07-05: 代码坏味道修复（第三轮）——`ReActLoop._parse_action` 的 final_answer 分支冗余逻辑（`if parsed.get("final_answer")` 判断后无论 True/False 都 return，判断毫无意义）简化为直接 return；`react_loop.py` 和 `reflection.py` 两处手写 `removeprefix("```json").removesuffix("```")` 重复代码，统一复用 `llm_utils.strip_json_fences`；`reflection.py` 的 `summary_changed={new_summary != ""}` / `risk_changed={new_risk != ""}` 判断不准（空字符串算"未改"但与原值相同也应算未改），改为 `revised_summary != report.summary` / `revised_risk != report.overall_risk` 真正判断是否改了；`web/app.py:review_submit` 只捕获 `OSError/ValueError/RuntimeError`，`AgentError`（`HarnessError` 子类）未被捕获导致审查失败 500，加入 `HarnessError` 捕获显示错误页。
- 2026-07-05: 代码坏味道修复（第四轮）——抽取 `now_iso()` 工具函数消除 30+ 处 `datetime.now(timezone.utc).isoformat()` 重复；`web/app.py:_format_session` `delta.total_seconds()` 重复 5 次改为 `secs` 单次计算；`session_converse` 异常捕获合并（去除重复 `ValueError/RuntimeError` 分支，补 `HarnessError`/`FileNotFoundError`/`json.JSONDecodeError`），`player.load` 只调一次；`eval/scorer.py` 冗余 `report.risks[idx]` 检查（RiskAssessment 恒 truthy）简化；`agent/contract_agent.py:_recall_memories` 返回同值双变量简化为单值；`agent/react_loop.py:_execute_tool` 循环内 `observation +=` 拼接改为 `list.append` + `"\n".join`。
- 2026-07-05: 高优先级审查修复（3 项）——`EvalScorer._feed_corrections` 的 `expected_compliance` 原按嵌套 `list[list[dict]]` 遍历，与 `EvalItem.expected_compliance: list[dict]` 类型注解及 `metrics._compliance_accuracy` 扁平消费方式不一致，同条款多条合规 check 时第二条会轮转挂到下一个同 type 条款导致修正信号错位；改为扁平 `list[dict]` 遍历，每条 check 独立对齐；`MemoryStore.recall` 硬编码 `score > 0.3` 阈值改为 `__init__(recall_min_score=0.3)` 可配置参数（不同 embedding 模型分数尺度差异大）；`ContractAgent._recall_memories` 的 `memories: list` 补全为 `list[MemoryEntry]` 并补 `MemoryEntry` 导入；修正 `test_eval.py` 嵌套 compliance 用例为扁平；新增 `test_same_clause_multiple_compliance_checks_align_correctly` 回归（1 用例），累计 229 个。
- 2026-07-05: 中优先级审查修复（4 项）——`MemoryStore` 三处 `except (ValueError, RuntimeError)` 改为 `except Exception`，ChromaDB/Embedding 真实异常类型多样（sqlite3.Error/openai.APIError/ImportError 等）显式列举易漏导致主审查流程被打断，记忆是辅助功能任何异常都应降级；`rag/parsing.py` 模块级 `_docling_parser` 单例加注释说明「仅启动时 `enable_docling()` 写一次，运行期只读，FastAPI 单进程多线程下并发读无需加锁」；`web/app.py` 模块导入全局副作用加注释说明单例契约与测试隔离方式（`HARNESS_DATA_DIR` 指向临时目录）；`react_loop._build_report` 的 `isinstance(checks, list)` 冗余防御改为 `assert` 表达「`state["compliance"]` 恒为嵌套列表」契约，避免异常状态污染被静默跳过。
- 2026-07-05: 低优先级审查修复（4 项）——`cli/main.py` `kb_eval_run` 的 `print` 改为 `logger.info` 与项目输出方式一致并受 `--verbose` 控制；`llm.py` 的 `assert client is not None` 改为显式 `if client is None: raise AgentError(...)`，避免 `python -O` 剥离后 `client` 为 None 时抛无意义 `AttributeError`；`LLMConfig`/`EmbeddingConfig` 加注释明确「空 `api_key` 语义 = 未设置，触发环境变量回退」契约；`MemoryStore.__init__` 新增 `embedding_config: EmbeddingConfig | None` 参数，调用方可传主 RAG 的 `EmbeddingConfig` 对齐模型，避免记忆检索用错模型查询导致分数尺度不一致，缺省时仍用空密钥 openai provider 保持向后兼容。
- 2026-07-05: `eval_rag/generator.py` `_call_llm_with_retry` 漏捕获 `AgentError` 修复——`LLMClient.chat` 现会抛 `AgentError`（密钥缺失/鉴权失败/重试耗尽等非瞬时错误），原 `except (APIError, APITimeoutError, RateLimitError, httpx.HTTPError)` 未覆盖导致向上抛打断整个生成流程；新增 `except AgentError` 分支不重试直接降级返回 None，与「失败时返回 None」契约一致。
- 2026-07-05: `eval_rag/generator.py` `RagDatasetGenerator.generate` 的 `kb: Any` 参数补全类型注解——改为 `kb: KnowledgeBase` 并导入 `harness.rag.knowledge_base.KnowledgeBase`，清理未使用的 `from typing import Any`。
- 2026-07-05: `eval_rag/generator.py` `generate` 内补 4 行中文注释（断点恢复/组 prompt/每条 query 对应一个 EvalRagItem/增量写入），无功能改动。
- 2026-07-05: `eval_rag/generator.py` LLM token 用量日志——`_call_llm_with_retry` 返回值改为 `(queries, usage)` 元组携带单次调用 usage（prompt/completion/total），`generate` 累计本轮成功调用 token 并在末尾汇总日志输出，便于成本核算；无测试用例变化，累计 229 个。
- 2026-07-05: `agent/llm.py` `chat` 内 LLM token 用量日志统一升级——原 `logger.debug` 仅输出 total_tokens，改为 `logger.info` 输出 prompt/completion/total 三项 + 模型名，所有调用方（12 处）自动受益无需逐处改；usage 字段缺失时降级 `logger.debug`。
- 2026-07-06: `eval_rag/generator.py` `_call_llm_with_retry` 内 token 日志去重——`LLMClient.chat` 已统一输出每次用量，generator 内重复打的「chunk LLM 用量…」删掉，仅保留 `generate` 末尾本轮累计汇总（chat 内不知「本轮」语义）。
- 2026-07-06: `eval_rag/generator.py` `generate` 循环内加进度百分比日志——每个 chunk 处理完输出「进度 {done}/{total} ({pct}%) — chunk {id} 产出 {n} 条」，跳过/空 chunk 也计入分母保证百分比稳定到 100%，无需进度条。
- 2026-07-06: `eval_rag/generator.py` 删除本轮 token 累计汇总日志——`LLMClient.chat` 内已统一输出每次用量，累计汇总多余删掉；`_call_llm_with_retry` 签名回滚为只返 `list[str] | None`（不再带 usage 元组），循环里累计变量一并清掉。
- 2026-07-06: `eval_rag/generator.py` 重构——统计值修正（进度起点 & 最终汇总改用 `processed_in_sample = processed & current_ids` 交集）；新增 `seed` 参数 + `random.Random(seed).sample()` 实现可复现采样；将 LLM 调用 & Item 创建抽成 `_process_chunk()` 方法缩短 `generate()` 主体；`chunk: object` 改为 `chunk: Chunk` 类型注解；`harness/cli/main.py` 新增 `--seed` CLI 选项。
- 2026-07-06: YAML 自动执行修复——`run.arguments` 中 `click.Argument`（如 `dataset`）不再强制转 `--key=value`，改为根据目标命令参数类型自动区分 Option（`--key=value`）与位置参数（直接追加值）；`RagEvalRunner.run()` 新增 `expansion_threshold` 参数透传到 `kb.query()`；CLI `kb eval run` 新增 `--expansion-threshold` 选项（设 0 禁用 AI 查询扩展）。
- 2026-07-06: Reranker 速率限制——抽取 `RateLimiter` 到 `harness/rag/rate_limit.py`（从 `embedding.py` 移出，公开化）；`OpenAIReranker` 集成滑动窗口限速（RPM + TPM），`EmbeddingConfig` 新增 `rerank_max_rpm`/`rerank_max_tpm` 字段 + `RERANK_MAX_RPM`/`RERANK_MAX_TPM` 环境变量；`create_reranker`/`KnowledgeBase.from_config` 透传限速配置；新增 `_estimate_rerank_tokens` 估算 rerank 请求 token 数。
- 2026-07-06: Embedding/Reranker 重试机制——抽取 `retry_with_backoff` 到 `harness/rag/retry.py`（指数退避，非瞬时错误立即抛）；`OpenAIEmbeddingProvider.embed_batch` 对网络层错误（`APIConnectionError`/`APITimeoutError`/`RateLimitError`/`httpx.RequestError`/`OSError`）重试，鉴权/请求格式错误不重试直接抛 `EmbeddingError`；`OpenAIReranker.rerank` 对网络错误 + 5xx 重试，4xx 直接降级返回原始排序；`OpenAIEmbeddingProvider`/`OpenAIReranker`/`create_embedding_provider`/`create_reranker` 新增 `max_retries` 参数（默认 3）。
- 2026-07-07: AtomCode 自动化配置扩充——`.atomcode/skills/` 新增 4 个 skill：`pytest-coverage`/`ruff-recursive-fix`（在线 registry 装）+ `eval-dataset-reviewer`/`rag-retrieval-reviewer`（本地创建 subagent，编辑 eval/eval_rag/rag 模块时自动介入审格式契约与检索质量）；`.atomcode/commands/sync-baseline.md` 新增 `/sync-baseline` 命令（跑全套 pytest 固化回归基线并提交）；`.atomcode/settings.json` PostToolUse 追加 pyright 类型检查（与原 ruff format/check 串行，均在 `>/dev/null` 静默不打扰）。
- 2026-07-07: GitHub MCP 集成 + 密钥安全修复——`.mcp.json` 新增 `github` MCP server（`@modelcontextprotocol/server-github`，2025-04 已 deprecated 但本机无 Docker 暂用旧版，待装 Docker 升级官方 `ghcr.io/github/github-mcp-server`）；`.mcp.json` 加进 `.gitignore` + `git rm --cached` 脱离 git 跟踪（含 GitHub PAT 密钥禁止入库，team 各自本地维护）；PAT 改读 `.env` 的 `GITHUB_PAT` 环境变量（`.mcp.json` 用 `${GITHUB_PAT}` 占位符），`.env.example` 同步加示例引导；历史审计确认 PAT 从未进过 git commit（`git log -- .mcp.json` 仅一次提交且那次内容只有 context7 无 github）。
- 2026-07-09: 消除裸 `Any` 类型声明——18 处 `Any` 替换为具体类型（`np.ndarray`/`dict[str, Any] | list[Chunk]`/`ReviewReport`/`EvalItem`/`TypeVar` 等），第三方无 stub 库用 `TYPE_CHECKING` + `pyright: ignore`；ruff + pyright 全清。涉及 9 个文件。
- 2026-07-23: RAG eval 断点续跑——`RagEvalRunner.run()` 新增 `checkpoint_path` 参数，每跑完一条 query 增量写 JSONL（query→retrieved），下次 run 同一 checkpoint 时跳过已完成的 query 只补跑剩余；CLI `kb eval run` 新增 `--checkpoint` 选项；新增 `test_run_checkpoint_resume`/`test_run_checkpoint_partial_resume` 2 个回归用例，累计 231 个。
- 2026-07-25: RAG 评估报告支持 CSV 输出——`RagEvalReporter` 新增 `to_summary_csv`/`to_details_csv`/`to_csv`/`write_csv` 四个方法，CSV 用 `csv.writer` 正确转义、`utf-8-sig` 编码 Excel 双击不乱码；CLI `kb eval run` 新增 `--csv`/`--csv-prefix` 选项，YAML `run.arguments` 中 `csv: true` 同样生效；新增 5 个测试用例，累计 239 个。
- 2026-07-25: RAG eval 报告三项分析脚本——新增 `scripts/analyze_rag_eval.py`，基于已产出的 `summary.csv`/`details.csv` 做三项分析：①`hit1_miss_hit3_hit.csv`（hit@1=0 但 hit@3=1 的 query，排序改进空间）②`chunk_hit_distribution.csv`/`chunk_systematic_miss.csv`（按 expected chunk 维度的命中率 + 系统性漏检 hit1_rate≤0.3 且 query≥3）③`rag_eval_charts.html`（pyecharts 交互式图表，原生缩放/悬停 tooltip/dataZoom）+ `rag_eval_analysis_summary.md`（Markdown 摘要）；同步更新 README.md/AGENTS.md/CLAUDE.md；新增 `pyecharts` 依赖到 contract-harness 环境。
- 2026-07-26: KB.query 扩展分支 merge 后最终重排——`KnowledgeBase.query()` 在 AI 扩展检索词分支 `candidates = self._merge_results(...)` 后追加 `if self._reranker and len(candidates) > 1: candidates = self._reranker.rerank(text, candidates, top_k=top_k)`；原 merge 只取 max 分数但 candidates 来自多次 `_search_single` 各自 rerank 的分数尺度不一致，追加最终重排对齐排序；`_search_single` 内 rerank 不动（粗排重排分工）；新增 2 个回归用例 `test_query_expansion_rerank_after_merge`/`test_query_expansion_no_final_rerank_without_reranker`，累计 241 个。
- 2026-07-26: _search_single 候选池扩大 + 对比 eval 脚本——`KnowledgeBase._search_single` dense/sparse 候选池从 `top_k*2` 扩到 `max(20, top_k*4)`，rrf_fuse 在有 reranker 时保留 pool 个候选给精排（避免正确 chunk 在 dense 阶段被截掉）；新增 `scripts/compare_rerank_pool.py` 对比脚本，200 条「差 1 位」query 样本上 top-1 hit_rate 提升 +5pp（35.5% → 40.5%）。
- 2026-07-26: 评测/对比脚本通用 tqdm 进度条——`RagEvalRunner.run` 主循环用 `tqdm(items, desc="RAG eval", unit="query")` 包装，显示 N/total、ETA、query/s；`scripts/compare_rerank_pool.py` 的 `run_eval` 改用 tqdm + `set_postfix(h1, h3, h5)` 实时显示命中指标（替代手写 `% 20 == 0` 打印）；新增 `tqdm>=4.65` 到 pyproject.toml/environment.yml/requirements.txt 三份依赖清单。
- 2026-07-26: 三项分析合并入 CLI——新增 `harness/eval_rag/analyzer.py` 模块（把 `scripts/analyze_rag_eval.py` 的 4 个 task 函数 `task_rank_improvement`/`task_chunk_distribution`/`task_charts`/`task_markdown_summary` 重构为可复用模块，顶层 `run_analysis(summary_csv, details_csv, out_dir)` 入口）；`harness kb eval run` 新增 `--analyze` 标志，eval 跑完后自动派生三项分析（排序改进空间 / chunk 维度分布 / pyecharts 图表 + Markdown 摘要）到 `{eval_dir}/reports/analysis/`；`scripts/analyze_rag_eval.py` 保留作为独立脚本入口。
