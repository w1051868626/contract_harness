"""BM25 稀疏检索模块，配合稠密向量检索实现混合检索。"""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from harness.rag.vector_store import Chunk
from harness.utils.log import logger

_RRF_K = 60


def rrf_fuse(
    dense: list[Chunk],
    sparse: list[Chunk],
    top_k: int,
    k: int = _RRF_K,
) -> list[Chunk]:
    """Reciprocal Rank Fusion 融合两路检索结果。

    Args:
        dense: 稠密检索结果（按相关性降序）。
        sparse: 稀疏检索结果（按相关性降序）。
        top_k: 返回数量。
        k: RRF 常数，默认 60。

    Returns:
        融合后的结果列表（按 RRF 分降序）。
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, c in enumerate(dense):
        chunk_map[c.id] = c
        scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k + rank + 1)

    for rank, c in enumerate(sparse):
        chunk_map[c.id] = c
        scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, s in ranked[:top_k]:
        chunk = chunk_map[cid]
        chunk.score = s
        result.append(chunk)
    return result


class SparseRetriever:
    """BM25 稀疏检索器，维护语料库并支持关键词检索。"""

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._id_map: list[str] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """向语料库添加 Chunk。"""
        self._docs = [c.content for c in chunks]
        self._id_map = [c.id for c in chunks]
        tokenized = [self._tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 索引已构建: docs={}", len(self._docs))

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """执行 BM25 检索。

        Args:
            query: 查询文本。
            top_k: 返回结果数量。

        Returns:
            按 BM25 分降序的 Chunk 列表。
        """
        if self._bm25 is None or not self._docs:
            logger.debug("BM25 索引为空，返回空结果")
            return []
        tokenized = self._tokenize(query)
        scores: Any = self._bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[Chunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = Chunk(
                id=self._id_map[idx],
                document_id="",
                content=self._docs[idx],
                chunk_index=idx,
                score=float(scores[idx]),
            )
            results.append(chunk)
        logger.debug("BM25 检索: query={}, results={}", query[:30], len(results))
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """BM25 分词：中文单字 + 英文单词。"""
        import re

        tokens: list[str] = []
        for match in re.finditer(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text):
            tokens.append(match.group(0).lower())
        return tokens
