"""知识库模块，支持文档管理、智能分块与语义检索。"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path
from typing import Any

from openai import APIError

from harness.agent.llm import LLMClient, LLMResponse
from harness.core.config import HarnessConfig, LLMConfig
from harness.core.exceptions import ChunkingError
from harness.rag.chunking import (
    chunk_law_text as _chunk_law_text,
)
from harness.rag.chunking import (
    chunk_legal_text as _chunk_legal_text,
)
from harness.rag.chunking import (
    chunk_markdown as _chunk_markdown,
)
from harness.rag.chunking import (
    chunk_text as _chunk_text,
)
from harness.rag.chunking import (
    chunk_with_ai as _chunk_with_ai,
)
from harness.rag.constants import RRF_K
from harness.rag.embedding import EmbeddingProvider, create_embedding_provider
from harness.rag.parsing import enable_docling
from harness.rag.parsing import extract_zip_texts as _extract_zip_texts
from harness.rag.parsing import parse_file as _parse_file
from harness.rag.reranker import Reranker, create_reranker
from harness.rag.sparse import SparseRetriever, rrf_fuse
from harness.rag.vector_store import Chunk, Document, VectorStore, create_vector_store
from harness.utils.io import normalize_text as _util_normalize
from harness.utils.log import logger

EXPANSION_SYSTEM_PROMPT = "你是法律合同检索专家，输出每行一个搜索查询。"

QUERY_EXPANSION_PROMPT = (
    "你是一个法律合同检索专家。用户的原始查询可能措辞不准确，导致语义检索效果不佳。\n"
    "请根据原始查询，生成 {num_variants} 个同义但措辞不同的搜索查询，"
    "用于从法律知识库中检索相关条款。\n"
    "要求：\n"
    "- 保持法律含义不变\n"
    "- 使用专业法律术语\n"
    "- 每个查询独立成行，不要编号\n"
    "- 直接输出查询文本，每行一个，不要额外解释\n"
    "\n"
    "原始查询：{query}"
)

_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_CHUNK_OVERLAP = 64


class KnowledgeBase:
    """知识库，管理文档的添加、分块、嵌入与检索。"""

    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        llm: LLMClient | None = None,
        reranker: Reranker | None = None,
        chunk_model: str = "gpt-4o-mini",
        chunk_llm: LLMClient | None = None,
        expansion_llm: LLMClient | None = None,
        sparse_retriever: SparseRetriever | None = None,
        rrf_k: int = RRF_K,
    ):
        """初始化知识库。"""
        self._store = store
        self._embedding = embedding
        self._llm = llm
        self._reranker = reranker
        self._chunk_model = chunk_model
        self._chunk_llm = chunk_llm
        self._expansion_llm = expansion_llm or llm
        self._sparse_retriever = sparse_retriever
        self._rrf_k = rrf_k

    @property
    def store(self) -> VectorStore:
        """返回底层向量存储。"""
        return self._store

    @classmethod
    def from_config(cls, config: HarnessConfig | None = None) -> KnowledgeBase:
        """从 HarnessConfig 创建知识库实例。"""
        cfg = config or HarnessConfig()
        if cfg.use_docling:
            enable_docling()
        store = create_vector_store(cfg.kb_dir)
        embedding = create_embedding_provider(
            provider=cfg.embedding.provider,
            api_key=cfg.embedding.api_key,
            api_base=cfg.embedding.api_base,
            model=cfg.embedding.model,
            proxy=cfg.embedding.proxy,
            max_rpm=cfg.embedding.max_rpm,
            max_tpm=cfg.embedding.max_tpm,
        )
        reranker = create_reranker(
            provider=cfg.embedding.rerank_provider,
            api_key=cfg.embedding.rerank_api_key,
            api_base=cfg.embedding.rerank_api_base,
            model=cfg.embedding.rerank_model,
        )
        llm = LLMClient(cfg.llm) if cfg.llm.api_key else None
        chunk_cfg = LLMConfig(
            api_key=cfg.llm.chunk_api_key,
            api_base=cfg.llm.chunk_api_base,
            model=cfg.llm.chunk_model,
            proxy=cfg.llm.proxy,
        )
        chunk_llm = LLMClient(chunk_cfg) if chunk_cfg.api_key and chunk_cfg.model else None
        expansion_cfg = LLMConfig(
            api_key=cfg.llm.expansion_api_key,
            api_base=cfg.llm.expansion_api_base,
            model=cfg.llm.expansion_model,
            proxy=cfg.llm.proxy,
        )
        expansion_llm = (
            LLMClient(expansion_cfg) if expansion_cfg.api_key and expansion_cfg.model else None
        )
        sparse = SparseRetriever() if cfg.embedding.enable_hybrid_search else None
        logger.debug(
            "KnowledgeBase.from_config: kb_dir={}, embedding={}, reranker={},"
            " chunk_llm={}, expansion_llm={}, hybrid={}",
            cfg.kb_dir,
            cfg.embedding.provider,
            cfg.embedding.rerank_provider or "none",
            chunk_cfg.model if chunk_llm else "none",
            expansion_cfg.model if expansion_llm else "none",
            "on" if sparse else "off",
        )
        return cls(
            store=store,
            embedding=embedding,
            reranker=reranker,
            llm=llm,
            chunk_llm=chunk_llm,
            chunk_model=cfg.llm.chunk_model,
            expansion_llm=expansion_llm,
            sparse_retriever=sparse,
            rrf_k=cfg.embedding.rrf_k,
        )

    @staticmethod
    def _new_id() -> str:
        """生成短 ID。"""
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """清洗文本中的非常规字符。"""
        return _util_normalize(text)

    def add_text(
        self,
        title: str,
        content: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        use_ai_chunking: bool = True,
    ) -> str:
        """将文本添加到知识库。"""
        content = KnowledgeBase._normalize_text(content)
        doc_id = KnowledgeBase._new_id()
        doc = Document(
            id=doc_id,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        self._store.add_document(doc)
        chunks = self._resolve_chunks(content, doc_id, chunk_size, chunk_overlap, use_ai_chunking)
        if chunks:
            texts = [c.content for c in chunks]
            embeddings = self._embedding.embed_batch(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self._store.add_chunks(chunks)
            if self._sparse_retriever is not None:
                self._sparse_retriever.add_chunks(chunks)
            logger.debug("添加文本: title={}, doc_id={}, chunks={}", title, doc_id, len(chunks))
        else:
            logger.warning("添加文本无分块: title={}, doc_id={}", title, doc_id)
        return doc_id

    def _resolve_chunks(
        self,
        content: str,
        doc_id: str,
        chunk_size: int,
        chunk_overlap: int,
        use_ai: bool,
    ) -> list[Chunk]:
        """根据配置选择 AI 分块、法律条文分块或传统分块。"""
        if use_ai and self._chunk_llm is not None:
            logger.debug("使用 AI 分块")
            try:
                return _chunk_with_ai(content, doc_id, self._chunk_llm, self._chunk_model)
            except (ValueError, json.JSONDecodeError, RuntimeError):
                logger.warning("AI 分块失败，回退到传统分块", exc_info=True)
        try:
            md = _chunk_markdown(content, doc_id, chunk_size, chunk_overlap)
            if md is not None:
                logger.debug("使用 Markdown 结构化分块: chunks={}", len(md))
                return md
            law = _chunk_law_text(content, doc_id, chunk_size)
            if law is not None:
                logger.debug("使用逐条法律分块: chunks={}", len(law))
                return law
            legal = _chunk_legal_text(content, doc_id, chunk_size, chunk_overlap)
            if legal is not None:
                logger.debug("使用法律条文分块: chunks={}", len(legal))
                return legal
            logger.debug("使用通用文本分块")
            return _chunk_text(content, doc_id, chunk_size, chunk_overlap)
        except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
            raise ChunkingError(f"所有分块策略均失败: {exc}") from exc

    def add_file(
        self,
        file_path: str | Path,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        use_ai_chunking: bool = True,
    ) -> str:
        """添加文件到知识库。"""
        path = Path(file_path)
        logger.info("添加文件: {}", path.name)
        if path.suffix.lower() == ".zip":
            doc_ids = self.add_zip(path, chunk_size, chunk_overlap, use_ai_chunking)
            return doc_ids[0] if doc_ids else ""
        content = _parse_file(path)
        return self.add_text(
            title=path.stem,
            content=content,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ai_chunking=use_ai_chunking,
        )

    def add_zip(
        self,
        path: Path,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        use_ai_chunking: bool = True,
        work_dir: str | None = None,
    ) -> list[str]:
        """解压 zip 并以内部文件名为标题分别导入。

        Args:
            work_dir: 临时文件解压目录（如 Windows 上 C 盘空间不足时可指定其他盘符）。
        """
        doc_ids: list[str] = []
        logger.debug("解压 zip 导入: path={}", path)
        for filename, content in _extract_zip_texts(path, work_dir=work_dir):
            try:
                title = Path(filename).stem
                doc_id = self.add_text(
                    title=title,
                    content=content,
                    source=str(path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_ai_chunking=use_ai_chunking,
                )
                doc_ids.append(doc_id)
            except (json.JSONDecodeError, KeyError, OSError, zipfile.BadZipFile):
                logger.warning("ZIP 中文件处理失败: filename={}", filename, exc_info=True)
                continue
        return doc_ids

    def query(self, text: str, top_k: int = 5, expansion_threshold: float = 0.6) -> list[Chunk]:
        """语义检索最相关的文本块（支持可选的 rerank 精排和 AI 扩展检索词）。

        Args:
            text: 查询文本。
            top_k: 返回结果数量。
            expansion_threshold: AI 扩展检索词的相似度阈值。
                当最高分低于此值时，用 LLM 生成同义查询重试。
                设为 0 可禁用扩展。
        """
        logger.debug("检索: query={}, top_k={}", text[:50], top_k)
        candidates = self._search_single(text, top_k)
        if not candidates:
            return []

        max_score = candidates[0].score
        if (
            expansion_threshold > 0
            and max_score < expansion_threshold
            and self._expansion_llm is not None
        ):
            logger.info(
                "检索结果分数偏低 (max={:.3f} < {:.2f})，使用 AI 扩展检索词",
                max_score,
                expansion_threshold,
            )
            expanded = self._expand_query(text)
            if len(expanded) > 1:
                for q in expanded[1:]:
                    extra = self._search_single(q, top_k)
                    candidates = self._merge_results(candidates, extra, top_k)
                logger.debug("扩展检索完成: results={}", len(candidates))
        return candidates[:top_k]

    def _search_single(self, text: str, top_k: int) -> list[Chunk]:
        """单次检索（稠密 + 可选 BM25 稀疏 + RRF 融合，含可选 rerank）。"""
        query_emb = self._embedding.embed(text)
        dense_candidates = self._store.search(
            query_emb, top_k=top_k * 2 if (self._reranker or self._sparse_retriever) else top_k
        )

        if self._sparse_retriever is not None:
            sparse_candidates = self._sparse_retriever.search(text, top_k=top_k * 2)
            candidates = rrf_fuse(dense_candidates, sparse_candidates, top_k, k=self._rrf_k)
        else:
            candidates = dense_candidates[:top_k]

        if self._reranker and len(candidates) > 1:
            return self._reranker.rerank(text, candidates, top_k=top_k)
        return candidates[:top_k]

    def _expand_query(self, query: str, num_variants: int = 2) -> list[str]:
        """使用 LLM 生成同义检索词变体，返回包含原始查询的列表。"""
        if self._expansion_llm is None:
            msg = "_expand_query 需要 LLMClient，但未配置"
            raise RuntimeError(msg)
        prompt = QUERY_EXPANSION_PROMPT.format(query=query, num_variants=num_variants)
        try:
            resp: LLMResponse = self._expansion_llm.chat(
                [
                    {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            variants = [line.strip() for line in resp.content.strip().split("\n") if line.strip()]
            result = [query] + variants[:num_variants]
            logger.debug("扩展检索词: {}", result)
            return result
        except (APIError, ValueError, json.JSONDecodeError, KeyError):
            logger.warning("AI 扩展检索词失败，使用原始查询", exc_info=True)
            return [query]

    @staticmethod
    def _merge_results(
        existing: list[Chunk],
        new_results: list[Chunk],
        top_k: int,
    ) -> list[Chunk]:
        """合并两组检索结果，按 id 去重并保留最高分。"""
        seen: dict[str, Chunk] = {}
        for chunk in existing + new_results:
            if chunk.id in seen:
                seen[chunk.id].score = max(seen[chunk.id].score, chunk.score)
            else:
                seen[chunk.id] = chunk
        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return merged[:top_k]

    def list_chunks(self) -> list[Chunk]:
        """列出知识库中所有分块。"""
        chunks = self._store.get_all_chunks()
        logger.debug("列出分块: count={}", len(chunks))
        return chunks

    def list_documents(self) -> list[Document]:
        """列出所有文档。"""
        docs = self._store.list_documents()
        logger.debug("列出文档: count={}", len(docs))
        return docs

    def delete_document(self, document_id: str) -> None:
        """删除指定文档及其分块。"""
        logger.debug("删除文档: doc_id={}", document_id)
        self._store.delete_document(document_id)
