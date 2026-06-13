from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from harness.rag.embedding import EmbeddingProvider
from harness.rag.vector_store import Chunk, Document, VectorStore


class KnowledgeBase:
    def __init__(self, store: VectorStore, embedding: EmbeddingProvider):
        self._store = store
        self._embedding = embedding

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
    ) -> str:
        doc_id = uuid.uuid4().hex[:12]
        doc = Document(
            id=doc_id,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        self._store.add_document(doc)
        chunks = self._chunk_text(content, doc_id, chunk_size, chunk_overlap)
        if chunks:
            texts = [c.content for c in chunks]
            embeddings = self._embedding.embed_batch(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self._store.add_chunks(chunks)
        return doc_id

    def add_file(
        self,
        file_path: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> str:
        path = Path(file_path)
        content = self._parse_file(path)
        return self.add_text(
            title=path.stem,
            content=content,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def _parse_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        if suffix == ".md":
            return path.read_text(encoding="utf-8")
        if suffix == ".json":
            import json

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
