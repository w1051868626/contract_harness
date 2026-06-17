"""知识库模块，支持文档管理、智能分块与语义检索。"""

from __future__ import annotations

import json
import re
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from harness.agent.llm import LLMClient, LLMResponse
from harness.core.config import HarnessConfig, LLMConfig
from harness.core.exceptions import ChunkingError
from harness.rag.docling_parser import DoclingParser
from harness.rag.embedding import EmbeddingProvider, create_embedding_provider
from harness.rag.reranker import Reranker, create_reranker
from harness.rag.vector_store import Chunk, Document, VectorStore, create_vector_store
from harness.utils.log import logger

CHUNK_PROMPT = """你是一个文档分块专家。请将以下文档按逻辑结构拆分成有意义的片段。
每个片段应该是一个完整的主题、章节或逻辑段落，不要切割句子。
直接输出 JSON 数组，每个元素的格式为 {{"content": "..."}}。

文档：
{text}"""

CHUNK_MAX_CHARS = 8000

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
    ):
        """初始化知识库。"""
        self._store = store
        self._embedding = embedding
        self._llm = llm
        self._reranker = reranker
        self._chunk_model = chunk_model
        self._chunk_llm = chunk_llm

    @property
    def store(self) -> VectorStore:
        """返回底层向量存储。"""
        return self._store

    @classmethod
    def from_config(cls, config: HarnessConfig | None = None) -> KnowledgeBase:
        """从 HarnessConfig 创建知识库实例。"""
        cfg = config or HarnessConfig()
        if cfg.use_docling:
            cls.enable_docling()
        store = create_vector_store(cfg.kb_dir)
        embedding = create_embedding_provider(
            provider=cfg.embedding.provider,
            api_key=cfg.embedding.api_key,
            api_base=cfg.embedding.api_base,
            model=cfg.embedding.model,
            proxy=cfg.embedding.proxy,
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
        logger.debug(
            "KnowledgeBase.from_config: kb_dir=%s, embedding=%s, reranker=%s, chunk_llm=%s",
            cfg.kb_dir,
            cfg.embedding.provider,
            cfg.embedding.rerank_provider or "none",
            chunk_cfg.model if chunk_llm else "none",
        )
        return cls(
            store=store,
            embedding=embedding,
            reranker=reranker,
            llm=llm,
            chunk_llm=chunk_llm,
            chunk_model=cfg.llm.chunk_model,
        )

    def add_text(
        self,
        title: str,
        content: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_ai_chunking: bool = True,
    ) -> str:
        """将文本添加到知识库。"""
        doc_id = uuid.uuid4().hex[:12]
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
                return self._chunk_with_ai(content, doc_id)
            except Exception:
                logger.warning("AI 分块失败，回退到传统分块", exc_info=True)
        try:
            md = self._chunk_markdown(content, doc_id, chunk_size, chunk_overlap)
            if md is not None:
                logger.debug("使用 Markdown 结构化分块: chunks={}", len(md))
                return md
            legal = self._chunk_legal_text(content, doc_id, chunk_size, chunk_overlap)
            if legal is not None:
                logger.debug("使用法律条文分块: chunks={}", len(legal))
                return legal
            logger.debug("使用通用文本分块")
            return self._chunk_text(content, doc_id, chunk_size, chunk_overlap)
        except Exception as exc:
            raise ChunkingError(f"所有分块策略均失败: {exc}") from exc

    def _chunk_with_ai(self, text: str, doc_id: str) -> list[Chunk]:
        """使用 LLM 对文本进行智能分块，超长文本自动分段后合并。"""
        if self._chunk_llm is None:
            raise RuntimeError("chunk_llm 未初始化")
        logger.debug("AI 分块开始: text_len={}, model={}", len(text), self._chunk_model)

        all_chunks: list[Chunk] = []
        for i in range(0, len(text), CHUNK_MAX_CHARS):
            segment = text[i : i + CHUNK_MAX_CHARS]
            prompt = CHUNK_PROMPT.format(text=segment)
            resp: LLMResponse = self._chunk_llm.chat(
                [
                    {
                        "role": "system",
                        "content": "你是文档分块专家，严格按 JSON 格式输出。",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self._chunk_model,
                temperature=0.0,
            )
            raw = resp.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("AI chunking did not return a list")
            for item in data:
                content = item.get("content", "").strip()
                if content:
                    all_chunks.append(
                        Chunk(
                            id=uuid.uuid4().hex[:12],
                            document_id=doc_id,
                            content=content,
                            chunk_index=len(all_chunks),
                        )
                    )
        logger.debug("AI 分块完成: chunks={}", len(all_chunks))
        return all_chunks

    def add_file(
        self,
        file_path: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_ai_chunking: bool = True,
    ) -> str:
        """添加文件到知识库。"""
        path = Path(file_path)
        logger.info("添加文件: {}", path.name)
        if path.suffix.lower() == ".zip":
            doc_ids = self.add_zip(path, chunk_size, chunk_overlap, use_ai_chunking)
            return doc_ids[0] if doc_ids else ""
        content = self._parse_file(path)
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
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_ai_chunking: bool = True,
    ) -> list[str]:
        """解压 zip 并以内部文件名为标题分别导入。"""
        supported = {".txt", ".md", ".json", ".pdf", ".docx"}
        doc_ids: list[str] = []
        logger.debug("解压 zip 导入: path={}", path)
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext not in supported:
                    continue
                try:
                    raw = zf.read(info.filename)
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(raw)
                        tmp_path = Path(tmp.name)
                    content = KnowledgeBase._parse_file(tmp_path)
                    tmp_path.unlink(missing_ok=True)
                    title = Path(info.filename).stem
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
                    logger.warning("ZIP 中文件处理失败: filename={}", info.filename, exc_info=True)
                    continue
        return doc_ids

    @staticmethod
    def _parse_file(path: Path) -> str:
        """解析文件内容（支持 txt/md/json/pdf/docx，可选 Docling）。"""
        suffix = path.suffix.lower()
        logger.debug("解析文件: path={}, suffix={}", path.name, suffix)

        # 可选：Docling 结构化解析（支持 PDF/DOCX/PPTX/图片等）
        docling_parser = getattr(KnowledgeBase, "_docling_parser", None)
        if docling_parser and docling_parser.available and docling_parser.supports(path):
            logger.info("Docling 解析: path={}, suffix={}", path.name, suffix)
            try:
                md = docling_parser.parse_to_markdown(path)
                if md.strip():
                    logger.info("Docling 解析成功: path={} ({} 字符)", path.name, len(md))
                    return md
                logger.warning("Docling 返回空内容，回退 path={}", path.name)
            except RuntimeError as e:
                logger.warning("Docling 解析失败，回退 path={}: {}", path.name, e)

        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)
            return str(data)
        if suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except (PdfReadError, KeyError, IndexError):
                logger.warning("PDF 解析失败，按文本读取: {}", path)
                return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".docx":
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        return path.read_text(encoding="utf-8")

    @classmethod
    def enable_docling(cls) -> None:
        """启用 Docling 解析器（在 KnowledgeBase 级共享单例）。"""
        parser = DoclingParser()
        if parser.available:
            cls._docling_parser = parser
            logger.info("Docling 解析器已启用")
        else:
            logger.warning("Docling 解析器不可用（未安装或初始化失败）")
            cls._docling_parser = None

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
        if expansion_threshold > 0 and max_score < expansion_threshold and self._llm is not None:
            logger.info(
                "检索结果分数偏低 (max=%.3f < %.2f)，使用 AI 扩展检索词",
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
        """单次语义检索（含可选的 rerank）。"""
        query_emb = self._embedding.embed(text)
        candidates = self._store.search(query_emb, top_k=top_k * 2 if self._reranker else top_k)
        if self._reranker and len(candidates) > 1:
            return self._reranker.rerank(text, candidates, top_k=top_k)
        return candidates[:top_k]

    def _expand_query(self, query: str, num_variants: int = 2) -> list[str]:
        """使用 LLM 生成同义检索词变体，返回包含原始查询的列表。"""
        assert self._llm is not None
        prompt = QUERY_EXPANSION_PROMPT.format(query=query, num_variants=num_variants)
        try:
            resp: LLMResponse = self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": "你是法律合同检索专家，输出每行一个搜索查询。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            variants = [line.strip() for line in resp.content.strip().split("\n") if line.strip()]
            result = [query] + variants[:num_variants]
            logger.debug("扩展检索词: {}", result)
            return result
        except Exception:
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

    def clear_cache(self) -> None:
        """清除知识库内部缓存（当前无缓存，预留接口）。"""
        pass

    def list_documents(self) -> list[Document]:
        """列出所有文档。"""
        docs = self._store.list_documents()
        logger.debug("列出文档: count={}", len(docs))
        return docs

    def delete_document(self, document_id: str) -> None:
        """删除指定文档及其分块。"""
        logger.debug("删除文档: doc_id={}", document_id)
        self._store.delete_document(document_id)

    @staticmethod
    def _chunk_markdown(
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk] | None:
        """Markdown 结构化分块：以标题为界保持章节完整。

        检测 Markdown 标题行（#/##/###），以此为分割边界；同标题群
        合并到 chunk_size；单段超长回退到段落级分块。非 Markdown 返回 None。
        """
        if not re.search(r'^#{1,6}\s+\S', text, re.MULTILINE):
            return None

        sections = re.split(r'(?=^#{1,6}\s)', text.strip(), flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        chunks: list[Chunk] = []
        idx = 0
        buffer: list[str] = []
        buf_len = 0

        def _flush() -> None:
            nonlocal idx, buffer, buf_len
            if buffer:
                chunks.append(KnowledgeBase._make_chunk(buffer, doc_id, idx))
                idx += 1
                carry = KnowledgeBase._carry_overlap(buffer, overlap)
                buffer = carry
                buf_len = sum(len(s) for s in carry)

        for sec in sections:
            sl = len(sec)
            if buf_len + sl <= chunk_size:
                buffer.append(sec)
                buf_len += sl
            else:
                _flush()
                if sl <= chunk_size:
                    buffer = [sec]
                    buf_len = sl
                else:
                    sub_segments = KnowledgeBase._split_segments(sec)
                    for sub in sub_segments:
                        if buf_len + len(sub) <= chunk_size:
                            buffer.append(sub)
                            buf_len += len(sub)
                        else:
                            _flush()
                            buffer = [sub]
                            buf_len = len(sub)

        _flush()
        logger.debug("Markdown 分块完成: chunks={}", len(chunks))
        return chunks

    @staticmethod
    def _chunk_legal_text(
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk] | None:
        """法律条文专用分块：以「条」为原子单位，不切割条款。

        检测到「第X条」结构时按条分割，每条为完整语义单元；
        多条合并到 chunk_size 内；单条超长时按「（X）」款/项切分。
        非法律文本返回 None，由上层回退到通用分块。
        """
        if not re.search(r"第[一二三四五六七八九十百千零\d]+条", text):
            return None

        split_pat = re.compile(r"(?=\n第[一二三四五六七八九十百千零\d]+[条章节分编])")
        raw_parts = split_pat.split(text.strip())
        parts = [p.strip() for p in raw_parts if p.strip()]

        if not parts:
            return None

        chunks: list[Chunk] = []
        idx = 0

        cur_chapter = ""
        cur_section = ""

        def _get_article_range(buf: list[str]) -> str:
            arts = [p for p in buf if re.match(r"第[一二三四五六七八九十百千零\d]+条", p.strip())]
            if not arts:
                return ""
            first = re.match(r"第([一二三四五六七八九十百千零\d]+)条", arts[0].strip())
            last = re.match(r"第([一二三四五六七八九十百千零\d]+)条", arts[-1].strip())
            if not first or not last:
                return ""
            same = first.group(1) == last.group(1)
            return f"{first.group(0)}—{last.group(0)}" if not same else first.group(0)

        def _flush(buf: list[str]) -> None:
            nonlocal idx
            if buf:
                meta: dict[str, Any] = {}
                if cur_chapter:
                    meta["chapter"] = cur_chapter
                if cur_section:
                    meta["section"] = cur_section
                art_range = _get_article_range(buf)
                if art_range:
                    meta["articles"] = art_range
                chunks.append(KnowledgeBase._make_chunk(buf, doc_id, idx, metadata=meta))
                idx += 1

        def _is_header(p: str) -> bool:
            return bool(re.match(r"第[一二三四五六七八九十百千零\d]+[章节分编]", p.strip()))

        def _split_long_article(article: str) -> list[str]:
            items = re.split(r"(?=\n*（[一二三四五六七八九十百千零\d]+）)", article)
            return [it.strip() for it in items if it.strip()]

        buffer: list[str] = []
        buf_len = 0

        for part in parts:
            part_len = len(part)
            is_hdr = _is_header(part)

            if is_hdr:
                _flush(buffer)
                hdr_match = re.match(
                    r"(第[一二三四五六七八九十百千零\d]+)([章节分编])", part.strip()
                )
                if hdr_match:
                    if hdr_match.group(2) in ("章", "编"):
                        cur_chapter = part.strip()
                        cur_section = ""
                    elif hdr_match.group(2) in ("节", "分编"):
                        cur_section = part.strip()
                buffer = [part]
                buf_len = part_len
            elif buf_len + part_len <= chunk_size:
                buffer.append(part)
                buf_len += part_len
            else:
                _flush(buffer)
                if part_len <= chunk_size:
                    buffer = [part]
                    buf_len = part_len
                else:
                    subs = _split_long_article(part)
                    sub_buf: list[str] = []
                    sub_len = 0
                    for s in subs:
                        sl = len(s)
                        if sub_len + sl <= chunk_size:
                            sub_buf.append(s)
                            sub_len += sl
                        else:
                            _flush(sub_buf)
                            sub_buf = [s]
                            sub_len = sl
                    _flush(sub_buf)
                    buffer = []
                    buf_len = 0

        _flush(buffer)
        logger.debug("法律条文分块完成: chunks={}", len(chunks))
        return chunks

    @staticmethod
    def _chunk_text(
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        """基于段落滑动窗口的传统分块算法。"""
        if not text.strip():
            return []

        segments = KnowledgeBase._split_segments(text)

        if len(segments) == 1 and len(text) <= chunk_size:
            logger.debug("通用分块: text_len={}, single_chunk", len(text))
            return [
                Chunk(
                    id=uuid.uuid4().hex[:12],
                    document_id=doc_id,
                    content=text,
                    chunk_index=0,
                )
            ]

        chunks: list[Chunk] = []
        idx = 0
        buffer: list[str] = []
        buffer_len = 0

        for seg in segments:
            seg_len = len(seg)
            if buffer_len + seg_len <= chunk_size:
                buffer.append(seg)
                buffer_len += seg_len
            else:
                if buffer:
                    chunks.append(KnowledgeBase._make_chunk(buffer, doc_id, idx))
                    idx += 1
                    carry = KnowledgeBase._carry_overlap(buffer, overlap)
                    buffer = carry
                    buffer_len = sum(len(s) for s in carry)

                if seg_len > chunk_size:
                    sub_segments = KnowledgeBase._split_long(seg, chunk_size)
                    for i, sub in enumerate(sub_segments):
                        chunks.append(
                            Chunk(
                                id=uuid.uuid4().hex[:12],
                                document_id=doc_id,
                                content=sub,
                                chunk_index=idx,
                            )
                        )
                        idx += 1
                        if i < len(sub_segments) - 1 and sub_segments[i + 1] == "":
                            break
                    buffer = []
                    buffer_len = 0
                else:
                    buffer = [seg]
                    buffer_len = seg_len

        if buffer:
            chunks.append(KnowledgeBase._make_chunk(buffer, doc_id, idx))

        logger.debug("通用分块完成: text_len={}, chunks={}", len(text), len(chunks))
        return chunks

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        """按空行将文本分割为段落。"""
        raw = re.split(r"\n\s*\n", text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _split_long(text: str, chunk_size: int) -> list[str]:
        """将长文本按句子切分。"""
        sentences = re.split(r"(?<=[。！？；.!?;])\s*", text)
        chunks: list[str] = []
        buf = ""
        for s in sentences:
            if not s.strip():
                continue
            if len(buf) + len(s) <= chunk_size:
                buf += s
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = s
        if buf:
            chunks.append(buf.strip())

        if len(chunks) <= 1 and len(text) > chunk_size:
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks if chunks else [text]

    @staticmethod
    def _carry_overlap(buffer: list[str], overlap_chars: int) -> list[str]:
        """计算窗口重叠部分。"""
        carry: list[str] = []
        carry_len = 0
        for seg in reversed(buffer):
            if carry_len + len(seg) <= overlap_chars:
                carry.insert(0, seg)
                carry_len += len(seg)
            else:
                break
        return carry

    @staticmethod
    def _make_chunk(
        segments: list[str],
        doc_id: str,
        idx: int,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """创建 Chunk 对象。"""
        return Chunk(
            id=uuid.uuid4().hex[:12],
            document_id=doc_id,
            content="\n\n".join(segments),
            chunk_index=idx,
            metadata=metadata or {},
        )
