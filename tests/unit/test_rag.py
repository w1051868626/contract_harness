from __future__ import annotations

"""RAG 系统单元测试：向量存储、知识库、嵌入提供者与文件解析。"""

import tempfile
from pathlib import Path

from harness.rag.embedding import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.vector_store import Chunk, Document, VectorStore


class _MockEmbeddingProvider(EmbeddingProvider):
    """模拟嵌入提供者，基于字符哈希生成伪向量。"""

    def embed(self, text: str) -> list[float]:
        return [hash(c) % 100 / 100.0 for c in text[:4]] or [0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TestVectorStore:
    """向量存储的增删查操作测试。"""

    def test_add_and_list_documents(self):
        """添加文档后应能列出。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            store.add_document(Document(id="d1", title="Test Doc", source="test.txt"))
            docs = store.list_documents()
            assert len(docs) == 1
            assert docs[0].id == "d1"
            assert docs[0].title == "Test Doc"
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_add_chunk_and_search(self):
        """添加 Chunk 后应能按向量相似度搜索到。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            store.add_document(Document(id="d1", title="法律条款"))
            emb1 = [1.0, 0.0, 0.0]
            emb2 = [0.0, 1.0, 0.0]
            store.add_chunks(
                [
                    Chunk(
                        id="c1", document_id="d1", content="保密条款", embedding=emb1, chunk_index=0
                    ),
                    Chunk(
                        id="c2", document_id="d1", content="违约责任", embedding=emb2, chunk_index=1
                    ),
                ]
            )
            results = store.search([0.9, 0.1, 0.0], top_k=2)
            assert len(results) == 2
            assert results[0].id == "c1"
            assert results[0].score > results[1].score
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_delete_document(self):
        """删除文档后列表应清空。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            store.add_document(Document(id="d1", title="Test"))
            store.delete_document("d1")
            assert store.list_documents() == []
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestKnowledgeBase:
    """知识库的添加、查询、分块与文件解析测试。"""

    def test_add_text_and_query(self):
        """添加文本后应能通过向量搜索查询到。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            kb.add_text("保密法规", "双方应对合同内容严格保密，未经对方书面同意不得向第三方披露")
            chunks = kb.query("保密义务", top_k=3)
            assert len(chunks) >= 1
            assert "保密" in chunks[0].content
            kb.delete_document(chunks[0].document_id)
            assert kb.list_documents() == []
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_add_file(self):
        """添加文件应自动分块并入库。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            kb = KnowledgeBase(store, _MockEmbeddingProvider())
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as src:
                src.write("test content for knowledge base")
                src_path = src.name
            try:
                doc_id = kb.add_file(src_path)
                docs = kb.list_documents()
                assert len(docs) == 1
                assert docs[0].id == doc_id
                assert docs[0].source == src_path
            finally:
                Path(src_path).unlink(missing_ok=True)
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_chunk_text(self):
        """长文本应按 chunk_size 正确分块。"""
        text = "A" * 1000
        chunks = KnowledgeBase._chunk_text(text, "doc1", chunk_size=300, overlap=30)
        assert len(chunks) >= 3
        assert chunks[0].document_id == "doc1"

    def test_chunk_text_small(self):
        """短文本应只返回一个 Chunk。"""
        chunks = KnowledgeBase._chunk_text("small text", "doc1", chunk_size=300, overlap=30)
        assert len(chunks) == 1


class TestEmbeddingProvider:
    """嵌入提供者工厂与 Mock 实现测试。"""

    def test_create_openai_provider(self):
        """create_embedding_provider 应返回 OpenAIEmbeddingProvider 实例。"""
        provider = create_embedding_provider("openai", api_key="sk-test")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model == "text-embedding-3-small"

    def test_mock_embed(self):
        """Mock 嵌入提供者应生成非空向量。"""
        emb = _MockEmbeddingProvider()
        result = emb.embed("hello")
        assert len(result) > 0
        batch = emb.embed_batch(["a", "b"])
        assert len(batch) == 2


class TestAIChunking:
    def test_ai_chunking_fallback_on_no_llm(self):
        """无 llm 时自动回退到规则分块"""
        emb = _MockEmbeddingProvider()
        store = VectorStore(":memory:")
        kb = KnowledgeBase(store, emb, llm=None)
        chunks = kb._resolve_chunks("hello world", "doc1", 512, 64, use_ai=True)
        # 无 llm → use_ai=True 但 llm is None → 走规则分块
        assert len(chunks) >= 1

    def test_ai_chunking_parse_response(self, mock_llm):
        """验证 AI chunking 能正确解析 LLM 返回的 JSON"""
        emb = _MockEmbeddingProvider()
        store = VectorStore(":memory:")
        kb = KnowledgeBase(store, emb, llm=mock_llm)

        kb.add_text(
            "测试合同",
            "本合同由甲乙双方签订。\n\n第一条 保密义务。\n\n第二条 违约责任。\n\n第三条 管辖。",
            use_ai_chunking=True,
        )
        chunks = store.search([0.1, 0.2, 0.3], top_k=5)
        assert any("保密" in c.content for c in chunks)

    def test_ai_chunking_malformed_json(self, mock_llm):
        """AI 返回非 JSON 时优雅回退到规则分块"""
        mock_llm.responses = []
        emb = _MockEmbeddingProvider()
        store = VectorStore(":memory:")
        kb = KnowledgeBase(store, emb, llm=mock_llm)
        result = kb.add_text("test", "AAA BBB CCC", use_ai_chunking=True)
        assert result


class TestFileParsing:
    """文件解析器（txt / json / pdf）测试。"""

    def test_parse_txt(self):
        """应正确读取 .txt 文件内容。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("hello world")
            path = f.name
        try:
            content = KnowledgeBase._parse_file(Path(path))
            assert content == "hello world"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_json_list(self):
        """应正确解析 JSON 数组文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('[{"a": 1}, {"b": 2}]')
            path = f.name
        try:
            content = KnowledgeBase._parse_file(Path(path))
            assert "a" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_json_dict(self):
        """应正确解析 JSON 对象文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"title": "test", "content": "hello"}')
            path = f.name
        try:
            content = KnowledgeBase._parse_file(Path(path))
            assert "hello" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_pdf_fallback(self):
        """当 pypdf 未安装时，退回到按字节读取"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 garbage content")
            path = f.name
        try:
            content = KnowledgeBase._parse_file(Path(path))
            assert content is not None
        finally:
            Path(path).unlink(missing_ok=True)
