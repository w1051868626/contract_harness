"""RAG 模块全局常量与枚举。"""

from __future__ import annotations

import re
from enum import Enum

# ===== 分块参数 =====

CHUNK_MAX_CHARS = 8000
EMBED_MAX_CHARS = 1024

# ===== 提示词模板 =====

CHUNK_PROMPT = """你是一个文档分块专家。请将以下文档按逻辑结构拆分成有意义的片段。
每个片段应该是一个完整的主题、章节或逻辑段落，不要切割句子。
直接输出 JSON 数组，每个元素的格式为 {{"content": "..."}}。

文档：
{text}"""

QUERY_EXPANSION_PROMPT = (
    "你是一个法律合同检索专家。用户的原始查询可能措辞不准确，导致语义检索效果不佳。\n"
    "请根据原始查询，生成 {num_variants} 个同义但措辞不同的搜索查询，"
    "用于从法律知识库中检索相关条款。\n"
    "要求：\n"
    "- 保持法律含义不变\n"
    "- 使用专业法律术语\n"
    "- 每个查询独立成行，不要编号\n"
    "- 直接输出查询文本，每行一个，不要额外解释\n"
    "\n"
    "原始查询：{query}"
)

# ===== 预编译正则 =====

_MD_HEADING_RE = re.compile(r"^(?:#{1,6})(?:\s|$|(?=[^\s]))", re.MULTILINE)
_MD_SPLIT_RE = re.compile(r"(?=^(?:#{1,6})(?:\s|$|(?=[^\s])))", re.MULTILINE)

_ART_PAT_RE = re.compile(r"第([一二三四五六七八九十百千零\d]+)条")
_ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千零\d]+条)\s*")
_BOUNDARY_RE = re.compile(r"(?=\n第[一二三四五六七八九十百千零\d]+[条章节分编])")

# ===== 元数据键名 =====


class MetaKey(str, Enum):
    """Chunk metadata 键名常量。"""

    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLES = "articles"
    DOC_TYPE = "doc_type"
    SOURCE_FILE = "source_file"
    LAW_NAME = "law_name"
    EFFECTIVE_DATE = "effective_date"
    CASE_NUMBER = "case_number"
    CASE_TITLE = "case_title"
    GUIDING_NUMBER = "guiding_number"
    KEYWORDS = "keywords"


# ===== 文档类型 =====


class DocType(str, Enum):
    """文档类型常量。"""

    LAW = "law"
    CASE = "case"
