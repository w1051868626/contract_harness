"""分块模块入口，聚合重导出各子模块的公开函数。"""

from harness.rag.chunking_law import (
    chunk_law_text,
    chunk_legal_text,
    chunk_markdown,
)
from harness.rag.chunking_text import (
    CHUNK_PROMPT,
    CHUNK_SYSTEM_PROMPT,
    chunk_text,
    chunk_with_ai,
)

__all__ = [
    "CHUNK_PROMPT",
    "CHUNK_SYSTEM_PROMPT",
    "chunk_law_text",
    "chunk_legal_text",
    "chunk_markdown",
    "chunk_text",
    "chunk_with_ai",
]
