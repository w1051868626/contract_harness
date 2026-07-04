# 工程决策与教训记忆

> 跨会话持久化的关键决策和踩过的坑，避免重复犯错。

## 2026-07-05: 测试隔离 — 防止误烧 API 余额

### 教训（重要）
本地 `.env` 里有 `DEEPSEEK_API_KEY` 真实密钥时，`LLMConfig.__post_init__` 会从 `os.environ` 读到它。**任何用默认 `LLMClient()` / `ContractAgent()` 构造的测试都会打真实 API**， silently 烧钱（表现为 DeepSeek 返回 402 Insufficient Balance）。

### 已落地的隔离机制
`tests/conftest.py` 的 autouse fixture `_isolate_llm_env`：
- 全局强制 `LLM_MOCK=1` → 默认 `LLMClient()` 走 mock
- 清空所有 provider key 环境变量（`OPENAI_API_KEY`/`DEEPSEEK_API_KEY`/`LLM_PROVIDER`/`LLM_PROXY`/`HTTP_PROXY`）

需要测真实错误路径的测试（如 `TestLLMClientErrorHandling`）用类级 fixture 覆盖关闭 mock，并显式 `mock=False` + `MagicMock` 注入 `_client`。

### 行为准则（以后必须遵守）
1. **跑任何会触发 LLM 调用的命令前**，确认 `LLM_MOCK=1` 已设置，或用 `MockLLMClient`。
2. **新增测试**用 `MockLLMClient`，不要用默认 `LLMClient()`/`ContractAgent()`。
3. **本地复现 CI 失败**时，用 `LLM_MOCK=1 harness regression run ...`，不要直接跑真实命令。
4. `pytest` 全套跑现在安全（conftest 已隔离），单测/集成测都不会打真实 API。

---

## 2026-07-05: LLM 密钥缺失策略 — 抛错而非静默回退

### 决策
`LLMClient` 密钥缺失时**抛 `AgentError`**，不再静默回退 mock。理由：静默回退会产出假审查报告，比直接失败更危险。

### 配套机制
- 显式 `mock=True` 参数：测试/本地无 API 场景显式启用
- `LLM_MOCK` 环境变量（`1`/`true`/`yes`）：CI 无 secret 时自动启用 mock
- CI 回归 workflow 在无 `OPENAI_API_KEY` secret 时 `export LLM_MOCK=1`

### 注意
改这个策略时**必须同步改 CI workflow**，否则无 secret 的 CI 会直接挂（已踩过：回归测试连续 3 次失败）。

---

## 2026-07-04: 多 Agent 对齐策略 — clause_index 优先 + clause_type 回退

### 决策
`SupervisorAgent` 和 `EvalScorer` 里，Worker 输出与 ClauseExpert 条款对齐时：
1. 优先用显式 `clause_index`
2. 缺 index 时按 `clause_type` 回退
3. **同 type 多条款用队列轮转消费**，不要用字典 `setdefault`（会全对齐到第一个）

### 踩过的坑
- `validate_consensus` 原用 `range(max(len))` 下标强行对齐 → RiskExpert/ComplianceExpert 输出顺序不一致时产出假分歧
- `_feed_corrections` 的 `type_to_indices` 队列被 risks/compliance 共用 `pop(0)` → risks 消费完后 compliance 对齐失败
- `synthesize_report` 的 `clauses_by_type` 字典 `setdefault` → 同 type 多条款全挂到第一个

**修复统一用"按 field 独立游标轮转"或"按 key 分组逐 key 比对"。**

---

## 2026-07-04: JSON 提取 — 括号配平替代非贪婪正则

### 决策
`extract_json_array`/`extract_json_object` 用括号配平算法，不用非贪婪正则 `\[.*?\]`。

### 原因
非贪婪正则会被 JSON 字符串内的 `]`/`}` 截断（如 `{"reason":"a]b"}`），导致解析失败静默回退到逐条 LLM 调用，成本/延迟暴增。

### 实现
`_balance_bracket` 按字符串感知配平（跳过 JSON 字符串内的括号），先剥围栏整段解析，再括号配平回退。

---

## 2026-07-05: ChromaDB 资源释放 — close() 必须真正释放 sqlite3

### 决策
`ChromaVectorStore.close()` 调用 ChromaDB 1.5+ 的 `client.close()` 释放底层 sqlite3 连接，不能是空实现。

### 原因
ChromaDB `PersistentClient` 内部持有 sqlite3 连接，Windows 上若不显式释放，`TemporaryDirectory` 清理时 `chroma.sqlite3` 被占用抛 `PermissionError [WinError 32]`（20 个测试失败）。

### 实现要点
- ChromaDB 类型存根未声明 `close()`，用 `hasattr(client, "close")` 防御 + `# pyright: ignore[reportAttributeAccessIssue]` 绕过
- **不要把 `self._client` 置 None**：pyright 会推断其为 `ClientAPI | None`，导致所有方法访问报 `reportOptionalMemberAccess`（9 个错误）。close 后再访问属于调用方误用，不在类型层面防御
- `MemoryStore` 也需加 `close()` 转发到底层 store

### 行为准则
新增任何持有 ChromaDB 的类（如未来的 Reranker 缓存等），都要在 `close()` 里转发 `store.close()`，并在测试的 `with TemporaryDirectory` 块末尾调用。

---

## 2026-07-04: LLM 调用重试策略

### 决策
`LLMClient.chat` 加 `max_retries=3` 参数：
- **可重试**：`APIConnectionError`/`APITimeoutError`/`RateLimitError`/`httpx.HTTPError`（瞬时错误，指数退避）
- **不重试**：`APIError` 兜底（鉴权/请求格式等非瞬时错误）
- `max_retries=1` 时关闭重试，抛原始包装错误（保留"限流"/"网络错误"字样）

### 注意
可重试异常**必须在 `APIError` 之前捕获**，因为 `APIConnectionError`/`RateLimitError` 都是 `APIError` 子类，否则会被兜底走"不重试"路径。pyright 还要求 `self.client` 用局部变量 + `assert is not None` 缩窄类型（`client` 属性返回 `OpenAI | None`）。

---

## 2026-07-05: JSON 围栏剥离统一用 strip_json_fences

### 决策
所有剥除 ```` ```json ```` 围栏的地方统一用 `harness.agent.tools.llm_utils.strip_json_fences`，不要手写 `removeprefix("```json").removesuffix("```")`。

### 原因
`react_loop.py` 和 `reflection.py` 原各手写一份，逻辑重复且易漏处理（如 ` ```json ` 带空格、多行围栏）。`strip_json_fences` 用正则 `^```(?:json)?\s*|\s*```$` 处理更全。

---

## 2026-07-05: Web 异常捕获必须包含 HarnessError

### 决策
`web/app.py` 的 `review_submit` 等 endpoint 捕获异常时必须包含 `HarnessError`（`AgentError`/`EvalError`/`ReplayError` 等的基类）。

### 原因
原来只捕获 `OSError/ValueError/RuntimeError`，`AgentError`（LLM 密钥缺失、API 调用失败等）是 `HarnessError` 子类未被捕获，导致审查失败时 FastAPI 返回 500 而非显示用户友好的错误页。

### 行为准则
新增任何调用 Agent/LLM 的 Web endpoint，异常捕获列表都要含 `HarnessError`。
