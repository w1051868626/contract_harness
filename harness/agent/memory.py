"""持久化记忆模块，支持 Agent 存储和检索历史审查案例及修正反馈。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from harness.agent.llm import LLMClient
from harness.core.types import Clause, ComplianceCheck, RiskAssessment
from harness.rag.embedding import EmbeddingProvider, create_embedding_provider
from harness.rag.vector_store import ChromaVectorStore, Chunk
from harness.utils.log import logger

_MEMORY_COLLECTION = "agent_memory"


class MemoryEntry:
    """单条记忆，对应一个条款的历史分析结果。"""

    def __init__(
        self,
        clause_type: str,
        clause_content: str,
        risk_level: str = "",
        risk_reason: str = "",
        risk_suggestion: str = "",
        compliance: list[dict[str, Any]] | None = None,
        session_id: str = "",
        is_correction: bool = False,
        correction_field: str = "",
        correction_value: str = "",
        timestamp: str = "",
    ):
        self.clause_type = clause_type
        self.clause_content = clause_content
        self.risk_level = risk_level
        self.risk_reason = risk_reason
        self.risk_suggestion = risk_suggestion
        self.compliance = compliance or []
        self.session_id = session_id
        self.is_correction = is_correction
        self.correction_field = correction_field
        self.correction_value = correction_value
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_chunk(self) -> Chunk:
        """转为 Chroma 可存储的 Chunk。"""
        meta: dict[str, Any] = {
            "clause_type": self.clause_type,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "risk_suggestion": self.risk_suggestion,
            "compliance": json.dumps(self.compliance, ensure_ascii=False),
            "session_id": self.session_id,
            "is_correction": "true" if self.is_correction else "false",
            "correction_field": self.correction_field,
            "correction_value": self.correction_value,
            "timestamp": self.timestamp,
        }
        return Chunk(
            id=uuid.uuid4().hex[:12],
            document_id="memory",
            content=self.clause_content,
            chunk_index=0,
            metadata=meta,
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> MemoryEntry:
        """从 Chunk 还原记忆条目。"""
        m = chunk.metadata
        compliance_raw = m.get("compliance", "[]")
        try:
            compliance = json.loads(compliance_raw) if isinstance(compliance_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            compliance = []
        return cls(
            clause_type=m.get("clause_type", ""),
            clause_content=chunk.content,
            risk_level=m.get("risk_level", ""),
            risk_reason=m.get("risk_reason", ""),
            risk_suggestion=m.get("risk_suggestion", ""),
            compliance=compliance,
            session_id=m.get("session_id", ""),
            is_correction=m.get("is_correction", "false") == "true",
            correction_field=m.get("correction_field", ""),
            correction_value=m.get("correction_value", ""),
            timestamp=m.get("timestamp", ""),
        )

    def format_reference(self) -> str:
        """格式化为可注入 prompt 的参考文本。"""
        if self.is_correction:
            return (
                f"  【修正案例】条款类型: {self.clause_type}\n"
                f"  修正字段: {self.correction_field}\n"
                f"  正确结果: {self.correction_value}\n"
            )
        lines = [f"  【历史案例】条款类型: {self.clause_type}"]
        if self.risk_level:
            lines.append(f"  风险等级: {self.risk_level}")
            if self.risk_reason:
                lines.append(f"  风险原因: {self.risk_reason}")
            if self.risk_suggestion:
                lines.append(f"  修改建议: {self.risk_suggestion}")
        if self.compliance:
            items = [
                f"{c['regulation']}({'合规' if c.get('status') else '不合规'})"
                for c in self.compliance[:3]
            ]
            lines.append(f"  合规检查: {'; '.join(items)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MemoryEntry(clause_type={self.clause_type}, correction={self.is_correction})"


class MemoryStore:
    """持久化记忆存储，基于 ChromaDB 实现案例记忆与检索。

    每次合同审查完成后调用 ``remember`` 存储条款级分析结果；
    新审查时调用 ``recall`` 获取相似历史案例作为参考。
    评测系统可通过 ``correct`` 注入修正信号，实现自演进。
    """

    def __init__(
        self,
        store_dir: str,
        embedding: EmbeddingProvider | None = None,
        llm: LLMClient | None = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._llm = llm
        if not enabled:
            self._store = None
            self._embedding = None
            return
        self._store = ChromaVectorStore(store_dir, collection_name=_MEMORY_COLLECTION)
        self._embedding = embedding or create_embedding_provider(
            provider="openai",
            api_key="",
            api_base="",
            model="text-embedding-3-small",
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def remember_session(
        self,
        clauses: list[Clause],
        risks: list[RiskAssessment],
        compliance: list[list[ComplianceCheck]],
        session_id: str = "",
    ) -> None:
        """将一次审查的条款级结果存入记忆。"""
        if not self._enabled or not self._store:
            return
        logger.debug("记忆存储: clauses={}, session={}", len(clauses), session_id)
        chunks: list[Chunk] = []
        risk_map: dict[int, RiskAssessment] = {i: r for i, r in enumerate(risks)}
        for i, clause in enumerate(clauses):
            risk = risk_map.get(i)
            compliance_dicts: list[dict[str, Any]] = []
            if i < len(compliance):
                compliance_dicts = [
                    {"regulation": c.regulation, "status": c.status, "detail": c.detail}
                    for c in compliance[i]
                ]
            entry = MemoryEntry(
                clause_type=clause.clause_type,
                clause_content=clause.content,
                risk_level=risk.risk_level.value if risk else "",
                risk_reason=risk.reason if risk else "",
                risk_suggestion=risk.suggestion if risk else "",
                compliance=compliance_dicts,
                session_id=session_id,
            )
            chunks.append(entry.to_chunk())
        if chunks:
            texts = [c.content for c in chunks]
            try:
                embeddings = self._embedding.embed_batch(texts) if self._embedding else []
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                self._store.add_chunks(chunks)
                logger.debug("记忆存储完成: entries={}", len(chunks))
            except Exception:
                logger.warning("记忆存储失败", exc_info=True)

    def recall(self, clause_content: str, top_k: int = 3) -> list[MemoryEntry]:
        """检索与给定条款内容相似的历史记忆。"""
        if not self._enabled or not self._store or not self._embedding:
            return []
        try:
            query_emb = self._embedding.embed(clause_content)
            raw_chunks = self._store.search(query_emb, top_k=top_k + 2)
            entries = [MemoryEntry.from_chunk(c) for c in raw_chunks if c.score > 0.3]
            entries.sort(key=lambda e: 0 if e.is_correction else 1, reverse=False)
            return entries[:top_k]
        except Exception:
            logger.warning("记忆检索失败", exc_info=True)
            return []

    def correct(
        self,
        clause_type: str,
        clause_content: str,
        field: str,
        correct_value: str,
    ) -> None:
        """注入一条修正信号，用于 Agent 自演进。"""
        if not self._enabled or not self._store:
            return
        entry = MemoryEntry(
            clause_type=clause_type,
            clause_content=clause_content,
            is_correction=True,
            correction_field=field,
            correction_value=correct_value,
        )
        chunk = entry.to_chunk()
        try:
            if self._embedding:
                chunk.embedding = self._embedding.embed(clause_content)
            self._store.add_chunks([chunk])
            logger.info("记忆修正已存储: field={}, value={}", field, correct_value)
        except Exception:
            logger.warning("记忆修正存储失败", exc_info=True)

    def format_memory_context(self, memories: list[MemoryEntry]) -> str:
        """将记忆列表格式化为可注入 Prompt 的上下文。"""
        if not memories:
            return ""
        lines = ["\n## 历史参考案例"]
        for m in memories:
            lines.append(m.format_reference())
            lines.append("")
        return "\n".join(lines)
