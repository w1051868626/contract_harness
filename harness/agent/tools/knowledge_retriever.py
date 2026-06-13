from __future__ import annotations

from harness.rag.knowledge_base import KnowledgeBase


class KnowledgeRetriever:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self._kb = knowledge_base

    @property
    def available(self) -> bool:
        return self._kb is not None

    def retrieve(self, query: str, top_k: int = 3) -> str:
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
