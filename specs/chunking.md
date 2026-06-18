# 结构化分块模块规约

## 1. 概述

知识库导入文档时，需要将文本切分为语义完整的 Chunk，以便后续向量检索。  
系统提供四种分块策略，按优先级依次尝试：

```
AI 分块 → Markdown/结构化分块 → 法律条文层级分块 → 通用文本分块
```

## 2. 分块策略

### 2.1 AI 分块 (`_chunk_with_ai`)

- **触发条件**: `use_ai=True` 且 `chunk_llm` 已配置
- **输入**: 原始文本全文
- **输出**: LLM 返回的 JSON 格式分块列表
- **回退**: 解析失败时静默降级到下一策略

### 2.2 Markdown/结构化分块 (`_chunk_markdown`)

- **触发条件**: 文本包含 `#` Markdown 标题（`#` / `##` / `###`）
- **输入**: 原始文本 + doc_id + chunk_size + overlap
- **分割边界**: 正则 `heading_pat` 定义的标题行
- **合并规则**: 相邻小段合并，总长不超过 chunk_size；chapter/section 变化触发强制刷新
- **超长处理**: 超过 chunk_size 的单段按段落（空行）切分
- **输出**: `list[Chunk] | None`（无匹配标题时返回 None）
- **元数据**: chunk 携带 `chapter` / `section` / `articles` 信息

### 2.3 法律条文层级分块 (`_chunk_legal_text`)

- **触发条件**: 文本包含 `第X条` 模式
- **算法**: 模拟 RecursiveCharacterTextSplitter 的**递归层级分割**，优先级从高到低：
  1. `第X编`
  2. `第X章`
  3. `第X节`
  4. `第X条`（核心分割点）
  5. `一、` / `1.`（款）
  6. `（一）` / `(1)`（项）
  7. `\n`（换行，最终回退）
- **分割规则**: 分隔符保留在后续片段开头（`keep_separator=True`）
- **递归**: 片段超 chunk_size 时尝试下一级分隔符
- **不合并**: 每个分割片段独立成 Chunk（片段已 ≤ chunk_size）
- **元数据**: 每个 chunk 记录 `chapter`、`section`、`articles` 范围
- **参考**: [Teeeeen/legal_rag](https://github.com/Teeeeen/legal_rag)

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
| `law_name` | `str` | 元数据提取 | 法律名称，如 `"中华人民共和国合同法"` |
| `effective_date` | `str` | 元数据提取 | 生效日期 |
| `guiding_number` | `str` | 元数据提取 | 指导案例编号，如 `"指导案例1号"` |
| `case_number` | `str` | 元数据提取 | 案号 |
| `keywords` | `str` | 元数据提取 | 关键词（分号分隔） |
| `doc_type` | `str` | 元数据提取 | `"law"` 或 `"case"` |

metadata 为空 dict 时不产生额外字段。

## 4. 回退链

```
_chunk_with_ai → 失败?
  ↓
_chunk_markdown → 返回 None?（无 # 标题）
  ↓
_chunk_legal_text → 返回 None?（无「第X条」）
  ↓
_chunk_text（最终保底）
```

## 5. 辅助方法

### 5.1 `_split_keep_separator(text, pattern) → list[str]`

按正则分割并保留分隔符在后续片段开头。用 `re.split` 的 capturing group 捕获分隔符，然后合并分隔符与后续内容。

### 5.2 `_hierarchical_split(text, separators, chunk_size) → list[str]`

递归层级分割核心。尝试当前级分隔符，超长片段用下级分隔符递归。

### 5.3 `_extract_law_metadata(text, filename) → dict`

从法律文本前 500/2000 字符中提取 `law_name`（按优先级匹配：全称→书名号→首行）和 `effective_date`。

### 5.4 `_extract_case_metadata(text, filename) → dict`

从指导案例文本中提取 `guiding_number`、`case_number`、`keywords`。

### 5.5 `_inject_contextual_header(content, metadata, doc_meta) → str`

在 chunk 内容前插入 `[法律名称: X | 章节: Y | 条号: 第Z条]` 格式标头，提升 Embedding 质量。

## 6. 待实现

- [x] Markdown 标题分割
- [x] 中文法律章节标题分割
- [x] chunk metadata 追踪（chapter / section / articles）
- [x] 法律元数据提取（law_name / effective_date / case metadata）
- [x] 上下文标头注入
- [ ] AI 分块支持 contextual header
- [ ] 通用分块支持 overlap 参数
