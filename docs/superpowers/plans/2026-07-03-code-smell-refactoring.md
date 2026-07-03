# 代码坏味道重构计划

> **For agentic workers:** Use subagent-driven-development to implement task-by-task.

**Goal:** 消除 4 项高优先级代码坏味道：knowledge_base 拆分、_compute_overall_risk 三重重复、_raw_to_chunks 重复、CLI 重复模式

**Architecture:** 提取 chunking/parsing 独立模块、合并公共函数到 core/utils.py、 CLI 提取辅助函数消除重复

---

## 全局约束

- 保持现有 API 兼容（必要时加兼容导入）
- 每次变更后运行 `ruff format` + `pytest tests/ -v`
- 拆分的模块在 `harness/rag/__init__.py` 中重导出

---

### Task 1: 提取 `parsing.py`（文件解析独立模块）

- **Create:** `harness/rag/parsing.py`
- **Modify:** `harness/rag/knowledge_base.py` (删除对应静态方法 + 更新引用)  
- **Modify:** `harness/cli/main.py` (更新 import)
- **Test:** `tests/unit/test_rag.py` (更新 import)

从 `KnowledgeBase` 搬出以下模块级函数：
- `parse_txt(path: Path) -> str`
- `parse_json(path: Path) -> str`
- `parse_pdf(path: Path) -> str`
- `parse_docx(path: Path) -> str`
- `parse_file(path: Path) -> str` (含 Docling 逻辑)
- `extract_zip_texts(path: Path, work_dir: str | None = None) -> list[tuple[str, str]]`
- `enable_docling() -> None` (模块级状态)
- `_PARSERS` 字典

知识库侧：`KnowledgeBase.parse_file` 改为委托 `from harness.rag.parsing import parse_file as _parse_file`；`add_file`/`extract_zip_texts` 也相应委托。

CLI 侧：`main.py:58-59` 改为 `from harness.rag.parsing import enable_docling, parse_file`

---

### Task 2: 提取 `chunking.py`（分块独立模块）

- **Create:** `harness/rag/chunking.py`
- **Modify:** `harness/rag/knowledge_base.py`  
- **Modify:** `harness/cli/main.py` (如果需要)
- **Test:** `tests/unit/test_rag.py`

从 `KnowledgeBase` 搬出以下模块级函数：
- `_chinese_to_int(text: str) -> int` (去掉 _ 前缀，公开)
- `chunk_with_ai(text, doc_id, chunk_llm, chunk_model) -> list[Chunk]` (非 `self`，参数传入 llm)
- `chunk_markdown(text, doc_id, chunk_size, overlap) -> list[Chunk] | None`
- `chunk_law_text(text, doc_id, chunk_size, overlap=0) -> list[Chunk] | None`
- `chunk_legal_text(text, doc_id, chunk_size, overlap) -> list[Chunk] | None`
- `chunk_text(text, doc_id, chunk_size, overlap) -> list[Chunk]`
- `_detect_md_heading_meta`, `_split_keep_separator`, `_hierarchical_split`, `_split_recursive`, `_md_article_range`, `_flush_law_article`, `_split_segments`, `_align_chunk_end`, `_align_overlap_start`, `_split_long`, `_carry_overlap`, `_make_chunk`, `_LawContext`, `_MdAccumulator`
- `_extract_law_metadata`, `_extract_case_metadata`, `_inject_contextual_header`

`KnowledgeBase._resolve_chunks` 改为调用模块级函数（传参而非 self 访问）。

---

### Task 3: 提取 `_compute_overall_risk` 和报告摘要统计到 `core/utils.py`

- **Modify:** `harness/core/utils.py` (新建或使用 `harness/utils/`)
- **Modify:** `harness/agent/contract_agent.py` (删除重复方法)
- **Modify:** `harness/agent/react_loop.py` (删除重复 staticmethod)  
- **Modify:** `harness/agent/multi_agent/supervisor.py` (删除重复 staticmethod)

提取到 `harness/utils/io.py` 或新建 `harness/utils/agent.py`：
- `compute_overall_risk(risks: list[RiskAssessment]) -> RiskLevel`

`contract_agent.py` 和 `supervisor.py` 中的 `_build_summary` 可考虑后续提取。

---

### Task 4: `vector_store.py` 提取 `_raw_to_chunks()`

- **Modify:** `harness/rag/vector_store.py`

`search()` (行 147-165) 和 `get_all_chunks()` (行 210-226) 中重复的 Chunk 重建逻辑提取为 `_raw_to_chunks(raw: dict) -> list[Chunk]`。

---

### Task 5: CLI `main.py` 重复模式消除

- **Modify:** `harness/cli/main.py`

提取辅助函数：
- `_get_config(ctx) -> HarnessConfig` 
- `_get_kb(ctx) -> KnowledgeBase`
- `_get_replay_storage(ctx) -> ReplayStorage`

消除 `config: HarnessConfig = ctx.obj["config"]` 30+ 次重复和 `config.ensure_dirs(); kb_instance = KnowledgeBase.from_config(config)` 7 次重复。
