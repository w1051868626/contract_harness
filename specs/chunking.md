# 结构化分块模块规约

## 1. 概述

知识库导入文档时，需要将文本切分为语义完整的 Chunk，以便后续向量检索。  
系统提供四种分块策略，按优先级依次尝试：

```
AI 分块 → Markdown/结构化分块 → 法律条文分块 → 通用文本分块
```

## 2. 分块策略

### 2.1 AI 分块 (`_chunk_with_ai`)

- **触发条件**: `use_ai=True` 且 `chunk_llm` 已配置
- **输入**: 原始文本全文
- **输出**: LLM 返回的 JSON 格式分块列表
- **回退**: 解析失败时静默降级到下一策略

### 2.2 Markdown/结构化分块 (`_chunk_markdown`)

- **触发条件**: 文本包含 `#` Markdown 标题或中文法律章节标题（`第X章/节/编/条`）
- **输入**: 原始文本 + doc_id + chunk_size + overlap
- **分割边界**: `# ` / `## ` / `### ` 或 `第X章` / `第X节` / `第X编` / `第X条`
- **合并规则**: 相邻小段合并，总长不超过 chunk_size
- **超长处理**: 超过 chunk_size 的单段按段落（空行）切分
- **输出**: `list[Chunk] | None`（无匹配标题时返回 None）
- **元数据**: chunk 需要携带 `chapter` / `section` / `articles` 信息（TODO）

### 2.3 法律条文分块 (`_chunk_legal_text`)

- **触发条件**: 文本包含 `第X条` 模式
- **分割边界**: `第X条` / `第X章` / `第X节` / `第X编`
- **元数据**: 每个 chunk 记录 `chapter`、`section`、`articles` 范围
- **超长单条**: 按 `（X）` 款/项切分

### 2.4 通用文本分块 (`_chunk_text`)

- **分割边界**: 空行（段落）→ 滑动窗口
- **超长段落**: 按句子（`。！？;`）切分
- **回退**: 上述均无法处理时的最终策略

## 3. Chunk 数据结构

```python
@dataclass
class Chunk:
    id: str           # uuid4 hex[:12]
    document_id: str  # 所属文档 ID
    content: str      # 文本内容
    chunk_index: int  # 在文档内的序号
    embedding: list[float] | None = None
    metadata: dict = {}   # 见 §3.1
    score: float = 0.0
```

### 3.1 Metadata 契约

| Key | 类型 | 来源 | 说明 |
|---|---|---|---|
| `chapter` | `str` | 结构化 / 法律分块 | 当前章节标题，如 `"第一章 总则"` |
| `section` | `str` | 同上 | 当前节标题，如 `"第一节 一般规定"` |
| `articles` | `str` | 同上 | 包含的条款范围，如 `"第一条—第五条"` |

metadata 为空 dict 时不产生额外字段。

## 4. 回退链

```
_chunk_with_ai → 失败?
  ↓
_chunk_markdown → 返回 None?（无标题结构）
  ↓
_chunk_legal_text → 返回 None?（无「第X条」）
  ↓
_chunk_text（最终保底）
```

## 5. 待实现

- [x] Markdown 标题分割
- [x] 中文法律章节标题分割
- [ ] chunk metadata 追踪（chapter / section / articles）
