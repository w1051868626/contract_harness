from __future__ import annotations

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
    def embed(self, text: str) -> list[float]:
        return [hash(c) % 100 / 100.0 for c in text[:4]] or [0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TestVectorStore:
    def test_add_and_list_documents(self):
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
    def test_add_text_and_query(self):
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
        text = "A" * 1000
        chunks = KnowledgeBase._chunk_text(text, "doc1", chunk_size=300, overlap=30)
        assert len(chunks) >= 3
        assert chunks[0].document_id == "doc1"

    def test_chunk_text_small(self):
        chunks = KnowledgeBase._chunk_text("small text", "doc1", chunk_size=300, overlap=30)
        assert len(chunks) == 1


class TestEmbeddingProvider:
    def test_create_openai_provider(self):
        provider = create_embedding_provider("openai", api_key="sk-test")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model == "text-embedding-3-small"

    def test_mock_embed(self):
        emb = _MockEmbeddingProvider()
        result = emb.embed("hello")
        assert len(result) > 0
        batch = emb.embed_batch(["a", "b"])
        assert len(batch) == 2


class TestFileParsing:
    def test_parse_txt(self):
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
