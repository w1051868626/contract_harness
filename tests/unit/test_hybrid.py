"""混合检索（稠密 + BM25 稀疏 + RRF 融合）测试。"""

from __future__ import annotations

from harness.rag.sparse import SparseRetriever, rrf_fuse
from harness.rag.vector_store import Chunk


class TestRrfFuse:
    """RRF 融合功能测试。"""

    def test_rrf_fuse_basic(self):
        dense = [
            Chunk(id="a", document_id="d1", content="条款A", chunk_index=0, score=0.9),
            Chunk(id="b", document_id="d1", content="条款B", chunk_index=1, score=0.8),
        ]
        sparse = [
            Chunk(id="c", document_id="d2", content="条款C", chunk_index=0, score=0.7),
            Chunk(id="a", document_id="d1", content="条款A", chunk_index=0, score=0.6),
        ]
        fused = rrf_fuse(dense, sparse, top_k=3, k=60)
        assert len(fused) <= 3
        ids = [c.id for c in fused]
        assert "a" in ids
        # "a" 在两路中都出现，RRF 分应最高
        assert fused[0].id == "a"

    def test_rrf_fuse_empty_dense(self):
        sparse = [Chunk(id="x", document_id="d", content="X", chunk_index=0, score=0.9)]
        fused = rrf_fuse([], sparse, top_k=5, k=60)
        assert len(fused) == 1
        assert fused[0].id == "x"

    def test_rrf_fuse_empty_sparse(self):
        dense = [Chunk(id="x", document_id="d", content="X", chunk_index=0, score=0.9)]
        fused = rrf_fuse(dense, [], top_k=5, k=60)
        assert len(fused) == 1
        assert fused[0].id == "x"

    def test_rrf_fuse_both_empty(self):
        fused = rrf_fuse([], [], top_k=5, k=60)
        assert fused == []


class TestSparseRetriever:
    """BM25 稀疏检索功能测试。"""

    def test_build_and_search(self):
        chunks = [
            Chunk(id="1", document_id="d", content="违约责任包括继续履行和赔偿损失", chunk_index=0),
            Chunk(
                id="2",
                document_id="d",
                content="当事人一方不履行合同义务应当承担违约责任",
                chunk_index=1,
            ),
            Chunk(id="3", document_id="d", content="知识产权归属及保密条款", chunk_index=2),
        ]
        retriever = SparseRetriever()
        retriever.add_chunks(chunks)
        results = retriever.search("违约", top_k=2)
        assert len(results) == 2
        assert results[0].id in ("1", "2")
        assert results[0].score > 0

    def test_search_empty_index(self):
        retriever = SparseRetriever()
        assert retriever.search("违约", top_k=5) == []

    def test_search_no_match(self):
        chunks = [
            Chunk(id="1", document_id="d", content="知识产权归属", chunk_index=0),
        ]
        retriever = SparseRetriever()
        retriever.add_chunks(chunks)
        results = retriever.search("xyzzy", top_k=5)
        # BM25 无匹配返回空列表
        assert results == []
