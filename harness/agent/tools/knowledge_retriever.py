"""知识库检索工具，从向量知识库中检索合同审查相关的参考信息。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.rag.knowledge_base import KnowledgeBase


class KnowledgeRetriever:
    """知识库检索器，支持查询可用性判断和相似内容检索。"""

    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        """注入知识库实例。"""
        self._kb = knowledge_base

    @property
    def available(self) -> bool:
        """判断知识库是否可用。"""
        return self._kb is not None

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """根据查询文本检索知识库，返回格式化参考信息。"""
        if not self._kb:
            return ""

        chunks = self._kb.query(query, top_k=top_k)
        if not chunks:
            return ""

        lines = ["## 知识库参考信息\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"### 参考 {i} (相似度: {chunk.score:.2%})\n")
            lines.append(chunk.content)
            lines.append("")
        return "\n".join(lines)
