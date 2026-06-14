"""RAG 系统单元测试：向量存储、知识库、嵌入提供者与文件解析。"""

from __future__ import annotations

import re
import tempfile
import unittest.mock
from pathlib import Path

from harness.rag.embedding import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.reranker import LocalReranker, OpenAIReranker, Reranker, create_reranker
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

    def test_chunk_legal_non_legal(self):
        """非法律文本应返回 None。"""
        result = KnowledgeBase._chunk_legal_text("hello world", "doc1", 200, 30)
        assert result is None

    def test_chunk_legal_preserves_article_boundary(self):
        """法律条文分块不应切割「条」。"""
        text = """第三章 合同的效力

第五百零二条 依法成立的合同，自成立时生效。

第五百零三条 无权代理人以被代理人的名义订立合同，被代理人已经开始履行合同义务或者接受相对人履行的，视为对合同的追认。

第五百零四条 法人的法定代表人超越权限订立的合同，除相对人知道或者应当知道其超越权限外，该代表行为有效。

第五百零五条 当事人超越经营范围订立的合同的效力，应当依照本法有关规定确定。"""
        chunks = KnowledgeBase._chunk_legal_text(text, "doc1", 200, 30)
        assert chunks is not None
        for c in chunks:
            assert re.match(
                r"(第[一二三四五六七八九十百千零\d]+[章节分编]|第[一二三四五六七八九十百千零\d]+条)",
                c.content.strip(),
            ), f"Chunk starts mid-article: {c.content[:40]}"

    def test_chunk_legal_split_long_article(self):
        """超长法律条文应按「（X）」款/项分割。"""
        text = """第十三条 符合下列情形之一的，个人信息处理者方可处理个人信息：
（一）取得个人的同意；
（二）为订立、履行个人作为一方当事人的合同所必需；
（三）为履行法定职责或者法定义务所必需；
（四）为应对突发公共卫生事件，或者紧急情况下为保护自然人的生命健康和财产安全所必需；
（五）为公共利益实施新闻报道、舆论监督等行为，在合理的范围内处理个人信息；
（六）依照本法规定在合理的范围内处理个人自行公开或者其他已经合法公开的个人信息；
（七）法律、行政法规规定的其他情形。"""
        chunks = KnowledgeBase._chunk_legal_text(text, "doc1", 150, 30)
        assert chunks is not None
        assert len(chunks) > 1, "Long article should be split into multiple chunks"

    def test_chunk_legal_preserves_header(self):
        """章节标题应作为新 chunk 起点。"""
        text = """第一章 总则

第一条 为了保护合同当事人的合法权益，维护社会经济秩序，制定本法。

第二章 合同的订立

第二条 本法所称合同是平等主体的自然人、法人、其他组织之间设立、变更、终止民事权利义务关系的协议。"""
        chunks = KnowledgeBase._chunk_legal_text(text, "doc1", 200, 30)
        assert chunks is not None
        assert any("第一章" in c.content for c in chunks)
        assert any("第二章" in c.content for c in chunks)
        assert any("第一条" in c.content for c in chunks)
        assert any("第二条" in c.content for c in chunks)


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
            with unittest.mock.patch.dict("sys.modules", {"pypdf": None}):
                content = KnowledgeBase._parse_file(Path(path))
                assert content is not None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_zip(self):
        """zip 应解压并以内部文件名作为标题分别导入。"""
        import zipfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        db_path = ""
        try:
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("合同A.txt", "第一条内容")
                zf.writestr("合同B.txt", "第二条内容")

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
                db_path = f2.name
            from harness.rag.vector_store import VectorStore
            store = VectorStore(db_path)
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            result = kb.add_file(zip_path)
            docs = store.list_documents()
            assert len(docs) == 2
            titles = {d.title for d in docs}
            assert "合同A" in titles
            assert "合同B" in titles
            assert result  # first doc_id is returned
            store.close()
        finally:
            Path(zip_path).unlink(missing_ok=True)
            if db_path:
                Path(db_path).unlink(missing_ok=True)

    def test_parse_zip_skip_unsupported(self):
        """zip 中的不支持格式应被跳过。"""
        import zipfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        db_path = ""
        try:
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("doc.txt", "文本内容")
                zf.writestr("image.png", b"fake_png")

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
                db_path = f2.name
            from harness.rag.vector_store import VectorStore
            store = VectorStore(db_path)
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            kb.add_file(zip_path)
            docs = store.list_documents()
            assert len(docs) == 1
            assert docs[0].title == "doc"
            store.close()
        finally:
            Path(zip_path).unlink(missing_ok=True)
            if db_path:
                Path(db_path).unlink(missing_ok=True)


class _MockReranker(Reranker):
    """模拟重排序器，反转候选顺序。"""

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:
        candidates.reverse()
        return candidates[:top_k]


class TestReranker:
    """重排序器单元测试。"""

    def test_create_openai_reranker(self):
        """create_reranker 应返回 OpenAIReranker 实例。"""
        r = create_reranker("openai", api_key="sk-test")
        assert isinstance(r, OpenAIReranker)
        assert r.model == "rerank-v1"

    def test_create_reranker_none(self):
        """provider 为空时应返回 None。"""
        assert create_reranker("") is None

    def test_create_reranker_invalid(self):
        """不支持的 provider 应抛出 ValueError。"""
        try:
            create_reranker("unknown")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_local_reranker_init(self):
        """LocalReranker 应正确保存模型名称。"""
        r = LocalReranker("test-model")
        assert r.model_name == "test-model"

    def test_reranker_empty_candidates(self):
        """空候选列表应返回空列表。"""
        r = _MockReranker()
        assert r.rerank("query", []) == []

    def test_reranker_reorders(self):
        """重排序器应调整候选顺序。"""
        r = _MockReranker()
        chunks = [
            Chunk(id="c1", document_id="d1", content="a", chunk_index=0),
            Chunk(id="c2", document_id="d1", content="b", chunk_index=1),
        ]
        result = r.rerank("query", chunks, top_k=2)
        assert result[0].id == "c2"
        assert result[1].id == "c1"

    def test_query_with_reranker(self):
        """使用 reranker 时 query 应通过 reranker 返回结果。"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = VectorStore(db_path)
            store.add_document(Document(id="d1", title="条款"))
            store.add_chunks(
                [
                    Chunk(
                        id="c1",
                        document_id="d1",
                        content="保密",
                        embedding=[1.0, 0.0],
                        chunk_index=0,
                    ),
                    Chunk(
                        id="c2",
                        document_id="d1",
                        content="违约",
                        embedding=[0.0, 1.0],
                        chunk_index=1,
                    ),
                ]
            )
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb, reranker=_MockReranker())
            results = kb.query("保密", top_k=2)
            # _MockReranker 反转了顺序，结果应与原始搜索顺序不同
            raw = store.search(emb.embed("保密"), top_k=4)
            assert len(results) == 2
            assert results[0].id != raw[0].id
            store.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


