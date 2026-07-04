"""RAG 模块全局配置常量与枚举。"""

from __future__ import annotations

from enum import Enum

# ===== 分块参数 =====

CHUNK_MAX_CHARS = 8000
EMBED_MAX_CHARS = 1024 * 8

# ===== Embedding / Reranker 默认值 =====

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_RERANK_MODEL = "rerank-v1"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ===== 混合检索参数 =====

RRF_K = 60

# ===== 跨模块共享的正则 =====

import re  # noqa: E402

ART_PAT_RE = re.compile(r"第([一二三四五六七八九十百千零\d]+)条")
ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百千零\d]+条)\s*")
BOUNDARY_RE = re.compile(r"(?=\n第[一二三四五六七八九十百千零\d]+[条章节分编])")


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
