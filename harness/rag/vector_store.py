"""向量存储模块，基于 SQLite 的本地向量数据库。"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _serialize_embedding(emb: list[float]) -> bytes:
    """将向量序列化为二进制。"""
    return json.dumps(emb).encode("utf-8")


def _deserialize_embedding(data: bytes) -> list[float]:
    """从二进制反序列化向量。"""
    return json.loads(data.decode("utf-8"))


class VectorStore:
    """基于 SQLite 的向量存储，支持增删查。"""

    def __init__(self, db_path: str | Path):
        """初始化数据库连接与表结构。"""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        """创建文档表和分块表。"""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id),
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                chunk_index INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
        """)
        self._conn.commit()

    def add_document(self, document: Document) -> str:
        """添加或替换文档记录。"""
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, source, metadata) VALUES (?, ?, ?, ?)",
            (document.id, document.title, document.source, json.dumps(document.metadata)),
        )
        self._conn.commit()
        return document.id

    def add_chunk(self, chunk: Chunk):
        """添加单个分块。"""
        sql = (
            "INSERT OR REPLACE INTO chunks "
            "(id, document_id, content, embedding, metadata, chunk_index) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        self._conn.execute(
            sql,
            (
                chunk.id,
                chunk.document_id,
                chunk.content,
                _serialize_embedding(chunk.embedding) if chunk.embedding else None,
                json.dumps(chunk.metadata),
                chunk.chunk_index,
            ),
        )
        self._conn.commit()

    def add_chunks(self, chunks: list[Chunk]):
        """批量添加分块。"""
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """根据查询向量检索最相似的 top_k 个分块。"""
        rows = self._conn.execute(
            "SELECT id, document_id, content, embedding, metadata, chunk_index FROM chunks "
            "WHERE embedding IS NOT NULL"
        ).fetchall()

        scored: list[Chunk] = []
        for row_id, doc_id, content, emb_bytes, meta_json, idx in rows:
            if not emb_bytes:
                continue
            emb = _deserialize_embedding(emb_bytes)
            score = _cosine_similarity(query_embedding, emb)
            scored.append(
                Chunk(
                    id=row_id,
                    document_id=doc_id,
                    content=content,
                    score=score,
                    metadata=json.loads(meta_json),
                    chunk_index=idx,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def list_documents(self) -> list[Document]:
        """按创建时间倒序列出所有文档。"""
        rows = self._conn.execute(
            "SELECT id, title, source, metadata FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [Document(id=r[0], title=r[1], source=r[2], metadata=json.loads(r[3])) for r in rows]

    def delete_document(self, document_id: str):
        """删除文档及其所有分块。"""
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._conn.commit()

    def close(self):
        """关闭数据库连接。"""
        self._conn.close()
