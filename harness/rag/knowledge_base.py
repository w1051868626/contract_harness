from __future__ import annotations

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
        if len(text) <= chunk_size:
            return [
                Chunk(
                    id=uuid.uuid4().hex[:12],
                    document_id=doc_id,
                    content=text,
                    chunk_index=0,
                )
            ]

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(
                Chunk(
                    id=uuid.uuid4().hex[:12],
                    document_id=doc_id,
                    content=text[start:end],
                    chunk_index=idx,
                )
            )
            idx += 1
            if end >= len(text):
                break
            start = end - overlap
        return chunks
