"""RAG 模块全局常量与枚举。"""

from __future__ import annotations

import re
from enum import Enum

# ===== 分块参数 =====

CHUNK_MAX_CHARS = 8000
EMBED_MAX_CHARS = 1024 * 8

# ===== 提示词模板 =====

CHUNK_SYSTEM_PROMPT = "你是文档分块专家，严格按 JSON 格式输出。"
EXPANSION_SYSTEM_PROMPT = "你是法律合同检索专家，输出每行一个搜索查询。"

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

# ===== 法律编号字符类（减少 knowlege_base.py 中重复） =====

_CN_DIGIT = r"[一二三四五六七八九十百千零\d]"
_CN_CHAPTER = r"[一二三四五六七八九十百千]"  # 纯中文数字（章节专用，不含零/阿拉伯数字）
_CN_ARTICLE = r"[一二三四五六七八九十百千零〇\d]"  # 条号专用（含零和阿拉伯数字）

# ===== 中文数字映射 & 法律元数据正则 =====

_CN_NUM: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_TITLE_RE = re.compile(r"^中华人民共和国.*")
_DATE_RE = re.compile(r"（(\d{4})年(\d+)月(\d+)日.*?通过")

_CHAPTER_LAW_RE = re.compile(r"^(第" + _CN_CHAPTER + r"+章.*)")
_SECTION_LAW_RE = re.compile(r"^(第" + _CN_CHAPTER + r"+节.*)")
_ARTICLE_LAW_RE = re.compile(r"^(第(" + _CN_ARTICLE + r"+)条)")

_CHAPTER_DIVISION_RE = re.compile(r"第" + _CN_DIGIT + r"+[章编]")
_SECTION_HEAD_RE = re.compile(r"第" + _CN_DIGIT + r"+节")
_NUM_LIST_RE = re.compile(_CN_DIGIT + r"+[、．]")
_PAREN_NUM_RE = re.compile(r"[（(]" + _CN_DIGIT + r"+[）)]")
_ART_SEARCH_RE = re.compile(r"第" + _CN_DIGIT + r"+条")

_DECIMAL_RE = re.compile(r"\d+\.(?!\d)")
_DECIMAL_RANGE_RE = re.compile(r"\d+\.\d+")


# ===== 元数据键名 =====


class MetaKey(str, Enum):
    """Chunk metadata 键名常量。"""

    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLES = "articles"
    ARTICLE_NO = "article_no"
    CHUNK_TOTAL = "chunk_total"
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
