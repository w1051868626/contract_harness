# 项目踩坑点

> contract_harness 项目特有的坑，跨会话共享。

## 环境与配置

### .env 文件
- 项目根目录 `.env` 被 `harness.utils.io.load_dotenv()` 自动加载（CLI/Web 入口）
- `.env` 里有真实 `DEEPSEEK_API_KEY`，`LLMConfig.__post_init__` 会读到
- **测试不加载 .env**（conftest 已隔离），但手动跑 `harness` 命令会加载
- `.env` 不入库（`.gitignore` 有 `*.env`）

### Provider 配置链
`LLMConfig._load_defaults_from_env`：
1. `provider` 默认 `openai`，可被 `LLM_PROVIDER` 覆盖
2. `api_key` 从 `{PROVIDER}_API_KEY` 读，回退 `OPENAI_API_KEY`
3. `api_base` 从 `{PROVIDER}_API_BASE` 读，回退 `https://api.openai.com/v1`

所以 `LLM_PROVIDER=deepseek` 时读 `DEEPSEEK_API_KEY`/`DEEPSEEK_API_BASE`。

### 多密钥配置
LLM/Embedding/Chunk/Expansion/Reranker 各有独立密钥配置：
- `OPENAI_API_KEY` / `LLM_*`
- `EMBEDDING_API_KEY` / `EMBEDDING_*`
- `CHUNK_API_KEY` / `CHUNK_*`
- `EXPANSION_API_KEY` / `EXPANSION_*`
- `RERANK_API_KEY` / `RERANK_*`

## 测试相关

### .gitignore 的 `*test*` 规则
`.gitignore` 有 `*test*` 会匹配 `tests/unit/test_*.py` 文件名片段。`git add tests/` 批量加会报"被忽略"。两种解法：
1. `git add -f tests/unit/test_xxx.py` 强制加（已用）
2. 把 `*test*` 改成 `/*test*` 仅匹配根目录（工作区有此改动，但未提交）

### Windows 平台问题（已修复）
- ~~`tests/unit/test_rag.py` 22 个 ChromaDB 测试在 Windows 上失败~~ → **2026-07-05 已修复**：`ChromaVectorStore.close()` 原为空实现，ChromaDB `PersistentClient` 内部 sqlite3 连接未释放，`TemporaryDirectory` 清理时 `chroma.sqlite3` 被占用抛 `PermissionError [WinError 32]`。修复：`close()` 调用 ChromaDB 1.5+ 的 `client.close()`（`hasattr` 防御 + `pyright: ignore`）；`MemoryStore` 新增 `close()` 转发；相关测试补 `store.close()`。227 用例全过。
- ~~`tests/unit/test_eval_rag.py::TestRagEvalCLI::test_kb_eval_generate_no_kb` 在 Windows 上失败~~ → **2026-07-05 已修复**：`RagDatasetGenerator.generate` 对空 KB 改为抛 `EvalError("Missing ...")`，测试用 `monkeypatch.setenv("HARNESS_DATA_DIR", tmp_path)` 隔离到临时空目录。

### CI 与本地差异
- CI（Ubuntu）跑 `pytest` + `ruff check` + `ruff format --check` + `pyright`
- 本地容易只跑 `ruff check`（语法）漏掉 `ruff format --check`（格式）和 `pyright`（类型）
- **提交前必须本地跑全**：`ruff format --check harness/ tests/` + `ruff check` + `pyright harness/`

## 多 Agent 架构

### Worker 输出对齐
- `ClauseExpert` / `RiskExpert` / `ComplianceExpert` 三个 Worker 独立调 LLM
- 输出顺序和数量**不保证一致**，不能用下标强行对齐
- 必须用 `clause_index` 优先 + `clause_type` 回退 + 同 type 队列轮转

### CrossValidator
- 规则优先（risk_level 取更严、compliance 取更严）
- LLM 兜底仲裁（分歧 ≥2 级时触发）

## 开发流程（AGENTS.md 规则）

### 三文档同步
每次变更后**必须先更新 README.md、AGENTS.md、CLAUDE.md，再提交推送**。AGENTS.md 末尾有「更新记录」段，格式 `- YYYY-MM-DD: 描述...累计 N 个测试用例`。

### 三依赖同步
新增依赖时同步更新 `pyproject.toml`、`environment.yml`、`requirements.txt`（三者必须一致）。

### 提交规范
- commit message 用 HEREDOC 保留空行
- 末尾加 trailer（HEREDOC 保留空行）：
  ```
  Co-Authored-By: AtomCode (GLM-5.2) <noreply@atomgit.com>
  ```
- `git commit --amend` 和 `git revert` 不加 trailer

## 常用命令

### 本地测试（安全，已隔离）
```bash
conda run -n contract-harness pytest tests/unit/ -q
```

### 本地复现 CI（安全）
```bash
LLM_MOCK=1 harness regression run examples/contracts/ --version "local-test"
```

### 查看 GitHub Actions
```bash
# 最近运行
curl -s "https://api.github.com/repos/w1051868626/contract_harness/actions/runs?per_page=5" | python -c "import json,sys;[print(r['name'],r['conclusion'],r['head_sha'][:7]) for r in json.load(sys.stdin)['workflow_runs']]"

# 失败 job 详情
curl -s "https://api.github.com/repos/w1051868626/contract_harness/actions/runs/<RUN_ID>/jobs" | python -c "import json,sys;[print(j['name'],[s['name'] for s in j['steps'] if s['conclusion']=='failure']) for j in json.load(sys.stdin)['jobs'] if j['conclusion']=='failure']"
```
