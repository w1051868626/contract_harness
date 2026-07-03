"""RAG 系统单元测试：向量存储、知识库、嵌入提供者与文件解析。"""

from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

from harness.agent.llm import LLMResponse
from harness.rag.chunking import (
    chunk_legal_text,
    chunk_markdown,
    chunk_text,
)
from harness.rag.embedding import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    _truncate_at_boundary,
    create_embedding_provider,
)
from harness.rag.knowledge_base import KnowledgeBase
from harness.rag.parsing import parse_file as _parse_file
from harness.rag.reranker import LocalReranker, OpenAIReranker, Reranker, create_reranker
from harness.rag.vector_store import ChromaVectorStore, Chunk, Document
from tests.conftest import MockLLMClient


class _MockEmbeddingProvider(EmbeddingProvider):
    """模拟嵌入提供者，基于有序索引生成确定性伪向量。"""

    def embed(self, text: str) -> list[float]:
        return [float(ord(c) % 100) / 100.0 for c in text[:4]] or [0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TestChromaVectorStore:
    """Chroma 向量存储的增删查操作测试。"""

    def test_add_and_list_documents(self):
        """添加文档后应能列出。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            store.add_document(Document(id="d1", title="Test Doc", source="test.txt"))
            docs = store.list_documents()
            assert len(docs) == 1
            assert docs[0].id == "d1"
            assert docs[0].title == "Test Doc"
            store.close()

    def test_add_chunk_and_search(self):
        """添加 Chunk 后应能按向量相似度搜索到。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
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

    def test_delete_document(self):
        """删除文档后列表应清空。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            store.add_document(Document(id="d1", title="Test"))
            store.add_chunks(
                [Chunk(id="c1", document_id="d1", content="test", embedding=[1.0, 0.0, 0.0])]
            )
            store.delete_document("d1")
            assert store.list_documents() == []
            store.close()

    def test_empty_search(self):
        """空库搜索应返回空列表。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            results = store.search([1.0, 0.0, 0.0], top_k=5)
            assert results == []
            store.close()


class TestKnowledgeBase:
    """知识库的添加、查询、分块与文件解析测试。"""

    def test_add_text_and_query(self):
        """添加文本后应能通过向量搜索查询到。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            kb.add_text("保密法规", "双方应对合同内容严格保密，未经对方书面同意不得向第三方披露")
            chunks = kb.query("保密义务", top_k=3)
            assert len(chunks) >= 1
            assert "保密" in chunks[0].content
            kb.delete_document(chunks[0].document_id)
            assert kb.list_documents() == []
            store.close()

    def test_add_file(self):
        """添加文件应自动分块并入库。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
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

    def test_chunk_text(self):
        """长文本应按 chunk_size 正确分块。"""
        text = "A" * 1000
        chunks = chunk_text(text, "doc1", chunk_size=300, overlap=30)
        assert len(chunks) >= 3
        assert chunks[0].document_id == "doc1"

    def test_chunk_text_small(self):
        """短文本应只返回一个 Chunk。"""
        chunks = chunk_text("small text", "doc1", chunk_size=300, overlap=30)
        assert len(chunks) == 1

    def test_chunk_legal_non_legal(self):
        """非法律文本应返回 None。"""
        result = chunk_legal_text("hello world", "doc1", 200, 30)
        assert result is None

    def test_chunk_legal_preserves_article_boundary(self):
        """法律条文分块不应切割「条」。"""
        text = """第三章 合同的效力

第五百零二条 依法成立的合同，自成立时生效。

第五百零三条 无权代理人以被代理人的名义订立合同，
被代理人已经开始履行合同义务或者接受相对人履行的，视为对合同的追认。

第五百零四条 法人的法定代表人超越权限订立的合同，
除相对人知道或者应当知道其超越权限外，该代表行为有效。

第五百零五条 当事人超越经营范围订立的合同的效力，应当依照本法有关规定确定。"""
        chunks = chunk_legal_text(text, "doc1", 200, 30)
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
        chunks = chunk_legal_text(text, "doc1", 150, 30)
        assert chunks is not None
        assert len(chunks) > 1, "Long article should be split into multiple chunks"

    def test_chunk_legal_preserves_header(self):
        """章节标题应作为新 chunk 起点。"""
        text = """第一章 总则

第一条 为了保护合同当事人的合法权益，维护社会经济秩序，制定本法。

第二章 合同的订立

第二条 本法所称合同是平等主体的自然人、法人、其他组织之间设立、变更、终止民事权利义务关系的协议。"""
        chunks = chunk_legal_text(text, "doc1", 200, 30)
        assert chunks is not None
        assert any("第一章" in c.content for c in chunks)
        assert any("第二章" in c.content for c in chunks)
        assert any("第一条" in c.content for c in chunks)
        assert any("第二条" in c.content for c in chunks)

    def test_chunk_legal_metadata(self):
        """法律分块应在 metadata 中记录章/条层级。"""
        text = """第一章 总则

第一条 为了保护合同当事人的合法权益，制定本法。

第二章 合同的订立

第二条 合同是平等主体之间设立、变更、终止民事权利义务关系的协议。

第三条 当事人订立合同，应当具有相应的民事权利能力和民事行为能力。"""
        chunks = chunk_legal_text(text, "doc1", 300, 30)
        assert chunks is not None
        assert chunks[0].metadata.get("chapter") == "第一章 总则"
        has_articles = any("articles" in c.metadata for c in chunks)
        assert has_articles, "应包含 articles 范围信息"

    def test_query_expansion_called_on_low_score(self):
        """分数低于阈值时调用 LLM 扩展检索词。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            mock_llm = MockLLMClient(
                [
                    LLMResponse(content="保密条款\n保密义务\n信息披露", model="mock"),
                ]
            )
            store.add_document(Document(id="d_low", title="示例"))
            # 插入零向量 chunk，与任何查询的余弦相似度均为 0 → score ≈ 0
            store.add_chunks(
                [
                    Chunk(
                        id="c_low",
                        document_id="d_low",
                        content="劳动法相关条款内容",
                        embedding=[0.0, 0.0, 0.0, 0.0],
                        chunk_index=0,
                    ),
                ]
            )
            kb = KnowledgeBase(store, emb, expansion_llm=mock_llm)
            chunks = kb.query("保密义务", top_k=3, expansion_threshold=0.5)
            assert len(chunks) >= 1
            assert mock_llm.call_count > 0

    def test_query_expansion_skipped_on_high_score(self):
        """分数高于阈值时不进行扩展。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            mock_llm = MockLLMClient(
                [
                    LLMResponse(content="保密条款\n保密义务\n信息披露", model="mock"),
                ]
            )
            kb = KnowledgeBase(store, emb, expansion_llm=mock_llm)
            kb.add_text("测试", "双方应对合同内容严格保密")
            chunks = kb.query("双方应对合同内容严格保密", top_k=3, expansion_threshold=0.6)
            assert len(chunks) >= 1
            assert mock_llm.call_count == 0

    def test_query_expansion_disabled_with_zero_threshold(self):
        """threshold=0 时不进行扩展。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            mock_llm = MockLLMClient(
                [
                    LLMResponse(content="保密条款\n保密义务\n信息披露", model="mock"),
                ]
            )
            kb = KnowledgeBase(store, emb, expansion_llm=mock_llm)
            kb.add_text("保密法规", "双方应对合同内容严格保密")
            chunks = kb.query("保密义务", top_k=3, expansion_threshold=0.0)
            assert len(chunks) >= 1
            assert mock_llm.call_count == 0

    def test_query_expansion_no_llm(self):
        """无 LLM 时不进行扩展。"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            kb.add_text("保密法规", "双方应对合同内容严格保密")
            chunks = kb.query("保密义务", top_k=3, expansion_threshold=0.9)
            assert len(chunks) >= 1

    def test_merge_results_dedup(self):
        """_merge_results 应对相同 id 去重并保留最高分。"""
        c1 = Chunk(id="a", document_id="d1", content="第一段", chunk_index=0, score=0.5)
        c2 = Chunk(id="b", document_id="d1", content="第二段", chunk_index=1, score=0.3)
        c3 = Chunk(id="a", document_id="d1", content="第一段", chunk_index=0, score=0.7)
        merged = KnowledgeBase._merge_results([c1, c2], [c3], top_k=5)
        assert len(merged) == 2
        ids = {c.id for c in merged}
        assert ids == {"a", "b"}
        for c in merged:
            if c.id == "a":
                assert c.score == 0.7

    def test_merge_results_top_k(self):
        """_merge_results 应截断到 top_k。"""
        c1 = Chunk(id="a", document_id="d1", content="第一段", chunk_index=0, score=0.9)
        c2 = Chunk(id="b", document_id="d1", content="第二段", chunk_index=1, score=0.8)
        c3 = Chunk(id="c", document_id="d1", content="第三段", chunk_index=2, score=0.7)
        merged = KnowledgeBase._merge_results([c1, c2], [c3], top_k=2)
        assert len(merged) == 2
        assert merged[0].id == "a"
        assert merged[1].id == "b"


class TestChunkMarkdown:
    """Markdown 结构化分块测试。"""

    def test_non_markdown_returns_none(self):
        """不含 Markdown 标题时返回 None。"""
        text = "这是一段纯文本\n没有标题\n只有段落"
        result = chunk_markdown(text, "doc1", 200, 30)
        assert result is None

    def test_markdown_single_chunk(self):
        """短 Markdown 文本应合并为单个 Chunk。"""
        text = "# 第一章\n\n第一条 依法成立的合同，自成立时生效。"
        chunks = chunk_markdown(text, "doc1", 500, 30)
        assert chunks is not None
        assert len(chunks) == 1
        assert "第一章" in chunks[0].content

    def test_markdown_split_by_heading(self):
        """标题应作为 chunk 分割边界。"""
        text = (
            "# 第一章 总则\n\n"
            "第一条 为了保护合同当事人的合法权益，制定本法。\n\n"
            "# 第二章 合同的订立\n\n"
            "第二条 本法所称合同是平等主体的自然人之间设立民事权利义务关系的协议。"
        )
        chunks = chunk_markdown(text, "doc1", 40, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        assert any("第一章" in c.content for c in chunks)
        assert any("第二章" in c.content for c in chunks)

    def test_markdown_heading_stays_with_content(self):
        """标题应与其后内容在同个 Chunk 中。"""
        text = "# 保密条款\n\n双方应对合同内容严格保密。"
        chunks = chunk_markdown(text, "doc1", 500, 30)
        assert chunks is not None
        assert len(chunks) == 1
        assert chunks[0].content.startswith("# 保密条款")

    def test_markdown_metadata_chapter(self):
        """Markdown 标题应设置 chunk metadata.chapter。"""
        text = "# 第一章 总则\n\n第一条 合同自成立时生效。\n\n" + "# 第二章 合同的订立\n\n" * 3
        chunks = chunk_markdown(text, "doc1", 40, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        ch1 = [c for c in chunks if "第一章" in c.content]
        ch2 = [c for c in chunks if "第二章" in c.content]
        assert ch1[0].metadata.get("chapter") == "第一章 总则"
        assert ch2[0].metadata.get("chapter") == "第二章 合同的订立"

    def test_markdown_metadata_section(self):
        """子标题应设置 chunk metadata.section，同时保留 chapter。"""
        text = "# 第一章 总则\n\n## 第一节 一般规定\n\n第一条 内容。\n\n# 第二章 合同的订立\n\n## 第一节 订立方式\n\n第二条 内容。"
        chunks = chunk_markdown(text, "doc1", 200, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        s1 = [c for c in chunks if "第一节 一般规定" in c.content]
        s2 = [c for c in chunks if "第一节 订立方式" in c.content]
        assert s1[0].metadata.get("chapter") == "第一章 总则"
        assert s1[0].metadata.get("section") == "第一节 一般规定"
        assert s2[0].metadata.get("chapter") == "第二章 合同的订立"
        assert s2[0].metadata.get("section") == "第一节 订立方式"

    def test_markdown_metadata_articles(self):
        """条款范围应记入 metadata.articles。"""
        text = "# 第一章\n\n第一条 内容一。\n\n第二条 内容二。\n\n第三条 内容三。"
        chunks = chunk_markdown(text, "doc1", 100, 10)
        assert chunks is not None
        assert any(
            c.metadata.get("articles") and "第一条" in c.metadata["articles"] for c in chunks
        )

    def test_chinese_numbered_heading(self):
        """# 下的中文数字编号（一、）应作为章节边界。"""
        text = "# 一、总则\n\n第一条 内容。\n\n# 二、合同的订立\n\n第二条 内容。"
        chunks = chunk_markdown(text, "doc1", 40, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        c1 = [c for c in chunks if "总则" in c.content]
        c2 = [c for c in chunks if "合同的订立" in c.content]
        assert c1[0].metadata.get("chapter") == "一、总则"
        assert c2[0].metadata.get("chapter") == "二、合同的订立"

    def test_arabic_numbered_heading(self):
        """# 下的阿拉伯数字编号（1. / 2.）应作为章节边界。"""
        text = "# 1. General\n\nArt 1 Content.\n\n# 2. Contract\n\nArt 2 Content."
        chunks = chunk_markdown(text, "doc1", 40, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        c1 = [c for c in chunks if "General" in c.content]
        c2 = [c for c in chunks if "Contract" in c.content]
        assert c1[0].metadata.get("chapter") == "1. General"
        assert c2[0].metadata.get("chapter") == "2. Contract"

    def test_subsection_numbered_heading(self):
        """# 下的子编号（1.1 / 2.1）应作为 section 边界。"""
        text = "# 1. Chapter\n\n## 1.1 Section A\n\nContent A.\n\n## 1.2 Section B\n\nContent B."
        chunks = chunk_markdown(text, "doc1", 200, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        s1 = [c for c in chunks if "Section A" in c.content]
        s2 = [c for c in chunks if "Section B" in c.content]
        assert s1[0].metadata.get("section") == "1.1 Section A"
        assert s2[0].metadata.get("section") == "1.2 Section B"

    def test_parenthesized_heading(self):
        """# 下的括号编号（（一））应作为 section 边界。"""
        text = "# 第一章 总则\n\n## （一）保密义务\n\n保密内容。\n\n## （二）违约责任\n\n违约后果。"
        chunks = chunk_markdown(text, "doc1", 200, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        s1 = [c for c in chunks if "保密义务" in c.content]
        s2 = [c for c in chunks if "违约责任" in c.content]
        assert s1[0].metadata.get("section") == "（一）保密义务"
        assert s2[0].metadata.get("section") == "（二）违约责任"

    def test_legal_plain_metadata_chapter(self):
        """纯文本章节标题应设置 metadata.chapter。"""
        text = (
            "# 第一章 总则\n\n第一条 为了保护合同当事人的合法权益，制定本法。\n\n"
            "# 第二章 合同的订立\n\n第二条 依法成立的合同，自成立时生效。\n\n"
            "# 第三章 违约责任\n\n第三条 违约方应当承担责任。"
        )
        chunks = chunk_markdown(text, "doc1", 60, 10)
        assert chunks is not None
        assert len(chunks) >= 2
        ch1 = [c for c in chunks if "第一章" in c.content]
        ch2 = [c for c in chunks if "第二章" in c.content]
        assert ch1[0].metadata.get("chapter") == "第一章 总则"
        assert ch2[0].metadata.get("chapter") == "第二章 合同的订立"

    def test_markdown_in_resolve_chains(self):
        """_resolve_chunks 应为 Markdown 文本选择结构化分块。"""
        text = "# 第一章\n\n第一条 保密义务。\n\n# 第二章\n\n第二条 违约责任。"
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            from harness.rag.vector_store import ChromaVectorStore

            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            emb = _MockEmbeddingProvider()
            kb = KnowledgeBase(store, emb)
            chunks = kb._resolve_chunks(text, "doc1", 200, 20, use_ai=False)
            assert len(chunks) >= 1
            store.close()


class TestChunkLegalText:
    """法律条文结构化分块测试。"""

    def test_non_legal_returns_none(self):
        """不含第X条时返回 None。"""
        text = "这是一段普通文本\n没有法律条文结构"
        result = chunk_legal_text(text, "doc1", 200, 30)
        assert result is None

    def test_basic_legal_chunking(self):
        """法律条文应按条分割为多个 chunk（小 chunk_size 强制按条切）。"""
        text = "第一条 保护合同当事人。\n第二条 依法成立合同。\n第三条 违约承担责任。"
        chunks = chunk_legal_text(text, "doc1", 30, 0)
        assert chunks is not None
        assert len(chunks) >= 2
        assert all(c.chunk_index == i for i, c in enumerate(chunks))

    def test_legal_chapter_metadata(self):
        """章标题应设置 metadata.chapter。"""
        text = "第一章 总则\n第一条 保护合同当事人。\n第二条 依法成立合同。\n第二章 订立\n第三条 违约承担责任。"
        chunks = chunk_legal_text(text, "doc1", 30, 0)
        assert chunks is not None
        ch1 = [c for c in chunks if "第一条" in c.content]
        ch2 = [c for c in chunks if "第三条" in c.content]
        assert ch1[0].metadata.get("chapter") == "第一章 总则"
        assert ch2[0].metadata.get("chapter") == "第二章 订立"

    def test_legal_overlap(self):
        """chunk 间 overlap 应携带上一片尾部内容。"""
        long_texts = "\n".join(f"第{i}条 " + "内容" * 10 + "。" for i in range(1, 21))
        chunks = chunk_legal_text(long_texts, "doc1", 80, 30)
        assert chunks is not None
        assert len(chunks) >= 2
        tail = chunks[0].content[-30:]
        assert any(tail in c.content for c in chunks[1:])

    def test_legal_single_chunk(self):
        """短法律文本应合并为单个 chunk。"""
        text = "第一条 依法成立的合同，自成立时生效。"
        chunks = chunk_legal_text(text, "doc1", 500, 30)
        assert chunks is not None
        assert len(chunks) == 1

    def test_legal_article_range_metadata(self):
        """多条 chunk 应标注文章范围。"""
        text = "第一条 内容一。\n第二条 内容二。\n第三条 内容三。\n第四条 内容四。"
        chunks = chunk_legal_text(text, "doc1", 20, 0)
        assert chunks is not None
        assert any("articles" in c.metadata and "条" in c.metadata["articles"] for c in chunks)


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

    def test_truncate_short_text_unchanged(self):
        """短文本不应截断。"""
        assert _truncate_at_boundary("你好世界", 1024) == "你好世界"

    def test_truncate_at_sentence_boundary(self):
        """应在句子边界截断（边界在 max_chars 后半段内）。"""
        long = "正常句子。" + "A" * 25 + "中间句。结束。" + "B" * 100
        result = _truncate_at_boundary(long, 50)
        assert result.endswith("。")

    def test_truncate_fallback_hard_cut(self):
        """无句子边界时回退到硬切。"""
        long = "x" * 2000
        result = _truncate_at_boundary(long, 100)
        assert len(result) == 100

    def test_truncate_newline_boundary(self):
        """换行符作为截断边界。"""
        long = "第一行\n" + "x" * 80 + "\n" + "y" * 200
        result = _truncate_at_boundary(long, 100)
        assert "\n" in result


class TestAIChunking:
    def test_ai_chunking_fallback_on_no_llm(self):
        """无 llm 时自动回退到规则分块"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            emb = _MockEmbeddingProvider()
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            kb = KnowledgeBase(store, emb, llm=None)
            chunks = kb._resolve_chunks("hello world", "doc1", 512, 64, use_ai=True)
            assert len(chunks) >= 1
            store.close()

    def test_ai_chunking_parse_response(self, mock_llm):
        """验证 AI chunking 能正确解析 LLM 返回的 JSON"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            emb = _MockEmbeddingProvider()
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            kb = KnowledgeBase(store, emb, llm=mock_llm, chunk_llm=mock_llm)
            kb.add_text(
                "测试合同",
                "本合同由甲乙双方签订。\n\n第一条 保密义务。\n\n第二条 违约责任。\n\n第三条 管辖。",
                use_ai_chunking=True,
            )
            chunks = store.search([0.1, 0.2, 0.3, 0.0], top_k=5)
            assert any("保密" in c.content for c in chunks)
            store.close()

    def test_ai_chunking_malformed_json(self, mock_llm):
        """AI 返回非 JSON 时优雅回退到规则分块"""
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            mock_llm.responses = []
            emb = _MockEmbeddingProvider()
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
            kb = KnowledgeBase(store, emb, llm=mock_llm, chunk_llm=mock_llm)
            result = kb.add_text("test", "AAA BBB CCC", use_ai_chunking=True)
            assert result
            store.close()


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
            content = _parse_file(Path(path))
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
            content = _parse_file(Path(path))
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
            content = _parse_file(Path(path))
            assert "hello" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_pdf_fallback(self):
        """PDF 解析失败时退回到按文本读取"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"not a real pdf content")
            path = f.name
        try:
            content = _parse_file(Path(path))
            assert content is not None
            assert "not a real pdf content" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_zip(self):
        """zip 应解压并以内部文件名作为标题分别导入。"""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("合同A.txt", "第一条内容")
                zf.writestr("合同B.txt", "第二条内容")

            with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
                store = ChromaVectorStore(tmpdir, collection_name="test_coll")
                emb = _MockEmbeddingProvider()
                kb = KnowledgeBase(store, emb)
                result = kb.add_file(zip_path)
                docs = store.list_documents()
                assert len(docs) == 2
                titles = {d.title for d in docs}
                assert "合同A" in titles
                assert "合同B" in titles
                assert result
                store.close()
        finally:
            Path(zip_path).unlink(missing_ok=True)

    def test_parse_zip_skip_unsupported(self):
        """zip 中的不支持格式应被跳过。"""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = f.name
        try:
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("doc.txt", "文本内容")
                zf.writestr("image.png", b"fake_png")

            with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
                store = ChromaVectorStore(tmpdir, collection_name="test_coll")
                emb = _MockEmbeddingProvider()
                kb = KnowledgeBase(store, emb)
                kb.add_file(zip_path)
                docs = store.list_documents()
                assert len(docs) == 1
                assert docs[0].title == "doc"
                store.close()
        finally:
            Path(zip_path).unlink(missing_ok=True)


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
        with tempfile.TemporaryDirectory(prefix="chroma_test_") as tmpdir:
            store = ChromaVectorStore(tmpdir, collection_name="test_coll")
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
