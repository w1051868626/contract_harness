"""向量存储模块，基于 Chroma 后端。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from harness.utils.log import logger


@dataclass
class Chunk:
    """文档分块，包含内容、向量和元数据。"""

    id: str
    document_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    score: float = 0.0


@dataclass
class Document:
    """文档元数据。"""

    id: str
    title: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add_document(self, document: Document) -> str: ...

    @abstractmethod
    def add_chunk(self, chunk: Chunk) -> None: ...

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]: ...

    @abstractmethod
    def list_documents(self) -> list[Document]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class ChromaVectorStore(VectorStore):
    """基于 Chroma 的向量数据库，支持 ANN 近似搜索。"""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "contract_harness",
        embedding_dim: int = 1536,
    ):
        self._embedding_dim = embedding_dim
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "ChromaVectorStore 初始化: persist_dir=%s, collection=%s", persist_path, collection_name
        )

    def add_document(self, document: Document) -> str:
        self._client.get_or_create_collection(
            name=f"doc_meta_{document.id}",
            metadata={"title": document.title, "source": document.source},
        )
        return document.id

    def add_chunk(self, chunk: Chunk) -> None:
        """添加单个 Chunk（已废弃，请使用 add_chunks 批量添加）。"""
        if not chunk.embedding:
            return
        metadata = dict(chunk.metadata)
        metadata["document_id"] = chunk.document_id
        metadata["chunk_index"] = chunk.chunk_index
        self._collection.add(
            ids=[chunk.id],
            embeddings=[chunk.embedding],  # type: ignore[arg-type]
            metadatas=[metadata],  # type: ignore[arg-type]
            documents=[chunk.content],
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            meta = dict(chunk.metadata)
            meta["document_id"] = chunk.document_id
            meta["chunk_index"] = chunk.chunk_index
            metadatas.append(meta)
            documents.append(chunk.content)
        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=metadatas,  # type: ignore[arg-type]
                documents=documents,
            )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        logger.debug("向量检索: top_k={}", top_k)
        raw: Any = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        ids: list[str] = raw.get("ids", [[]])[0] if raw.get("ids") else []
        distances: list[float] = raw.get("distances", [[]])[0] if raw.get("distances") else []
        metadatas_raw = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
        documents = raw.get("documents", [[]])[0] if raw.get("documents") else []

        chunks: list[Chunk] = []
        for i in range(len(ids)):
            meta: dict[str, Any] = (
                dict(metadatas_raw[i]) if metadatas_raw and i < len(metadatas_raw) else {}
            )
            doc_id = str(meta.get("document_id", ""))
            c_idx = int(meta.get("chunk_index", 0))
            chunks.append(
                Chunk(
                    id=ids[i],
                    document_id=doc_id,
                    content=str(documents[i]) if documents and i < len(documents) else "",
                    score=1.0 - distances[i] if distances and i < len(distances) else 0.0,
                    metadata={
                        k: v for k, v in meta.items() if k not in ("document_id", "chunk_index")
                    },
                    chunk_index=c_idx,
                )
            )
        return chunks

    def list_documents(self) -> list[Document]:
        collections = self._client.list_collections()
        docs: list[Document] = []
        for coll in collections:
            if coll.name == self._collection.name:
                continue
            if not coll.name.startswith("doc_meta_"):
                continue
            doc_id = coll.name[len("doc_meta_") :]
            meta = coll.metadata or {}
            docs.append(
                Document(
                    id=doc_id,
                    title=meta.get("title", doc_id),
                    source=meta.get("source", ""),
                )
            )
        return docs

    def delete_document(self, document_id: str) -> None:
        logger.debug("删除文档: document_id={}", document_id)
        meta_key = f"doc_meta_{document_id}"
        try:
            self._client.delete_collection(meta_key)
        except ValueError:
            pass
        results = self._collection.get(where={"document_id": document_id})
        if results and results.get("ids"):
            self._collection.delete(ids=results["ids"])

    def close(self) -> None:
        pass


def create_vector_store(
    path: str | Path,
    **kwargs: Any,
) -> VectorStore:
    """向量存储工厂函数，默认使用 Chroma。"""
    return ChromaVectorStore(path, **kwargs)
