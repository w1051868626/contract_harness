from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from harness.agent.llm import LLMClient, LLMResponse
from harness.rag.embedding import EmbeddingProvider
from harness.rag.vector_store import Chunk, Document, VectorStore

CHUNK_PROMPT = """你是一个文档分块专家。请将以下文档按逻辑结构拆分成有意义的片段。
每个片段应该是一个完整的主题、章节或逻辑段落，不要切割句子。
直接输出 JSON 数组，每个元素的格式为 {{"content": "..."}}。

文档：
{text}"""


class KnowledgeBase:
    def __init__(
        self,
        store: VectorStore,
        embedding: EmbeddingProvider,
        llm: LLMClient | None = None,
    ):
        self._store = store
        self._embedding = embedding
        self._llm = llm

    @property
    def store(self) -> VectorStore:
        return self._store

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
        return doc_id

    def _resolve_chunks(
        self,
        content: str,
        doc_id: str,
        chunk_size: int,
        chunk_overlap: int,
        use_ai: bool,
    ) -> list[Chunk]:
        if use_ai and self._llm is not None:
            try:
                return self._chunk_with_ai(content, doc_id)
            except Exception:
                pass
        return self._chunk_text(content, doc_id, chunk_size, chunk_overlap)

    def _chunk_with_ai(self, text: str, doc_id: str) -> list[Chunk]:
        assert self._llm is not None
        prompt = CHUNK_PROMPT.format(text=text[:8000])
        resp: LLMResponse = self._llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是文档分块专家，严格按 JSON 格式输出。",
                },
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
        )
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("AI chunking did not return a list")
        return [
            Chunk(
                id=uuid.uuid4().hex[:12],
                document_id=doc_id,
                content=item["content"],
                chunk_index=i,
            )
            for i, item in enumerate(data)
            if item.get("content", "").strip()
        ]

    def add_file(
        self,
        file_path: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_ai_chunking: bool = True,
    ) -> str:
        path = Path(file_path)
        content = self._parse_file(path)
        return self.add_text(
            title=path.stem,
            content=content,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ai_chunking=use_ai_chunking,
        )

    @staticmethod
    def _parse_file(path: Path) -> str:
        suffix = path.suffix.lower()
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
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".docx":
            try:
                from docx import Document as DocxDocument

                doc = DocxDocument(str(path))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return path.read_text(encoding="utf-8", errors="replace")
        return path.read_text(encoding="utf-8")

    def query(self, text: str, top_k: int = 5) -> list[Chunk]:
        query_emb = self._embedding.embed(text)
        return self._store.search(query_emb, top_k=top_k)

    def list_documents(self) -> list[Document]:
        return self._store.list_documents()

    def delete_document(self, document_id: str):
        self._store.delete_document(document_id)

    @staticmethod
    def _chunk_text(
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        segments = KnowledgeBase._split_segments(text)

        if len(segments) == 1 and len(text) <= chunk_size:
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

        return chunks

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _split_long(text: str, chunk_size: int) -> list[str]:
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
    def _make_chunk(segments: list[str], doc_id: str, idx: int) -> Chunk:
        return Chunk(
            id=uuid.uuid4().hex[:12],
            document_id=doc_id,
            content="\n\n".join(segments),
            chunk_index=idx,
        )
