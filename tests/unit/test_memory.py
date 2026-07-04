"""记忆模块单元测试：MemoryStore 的存储、检索、修正与格式化。"""

from __future__ import annotations

import tempfile

from harness.agent.memory import MemoryEntry, MemoryStore
from harness.core.types import Clause, ComplianceCheck, RiskAssessment, RiskLevel
from harness.rag.embedding import EmbeddingProvider


class _FixedEmbedding(EmbeddingProvider):
    """固定向量嵌入，始终生成 4 维确定性向量。"""

    _DIM = 4

    def embed(self, text: str) -> list[float]:
        chars = [float(ord(c) % 10) / 10.0 for c in text[: self._DIM]]
        while len(chars) < self._DIM:
            chars.append(0.0)
        return chars

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TestMemoryEntry:
    """MemoryEntry 的序列化与格式化测试。"""

    def test_to_chunk_roundtrip(self):
        """应能完整地序列化并还原。"""
        entry = MemoryEntry(
            clause_type="保密",
            clause_content="双方应保密",
            risk_level="medium",
            risk_reason="范围模糊",
            risk_suggestion="建议明确范围",
            compliance=[{"regulation": "民法典", "status": True}],
            session_id="s1",
        )
        chunk = entry.to_chunk()
        assert chunk.content == "双方应保密"
        assert chunk.metadata["clause_type"] == "保密"
        assert chunk.metadata["risk_level"] == "medium"

        restored = MemoryEntry.from_chunk(chunk)
        assert restored.clause_type == "保密"
        assert restored.risk_level == "medium"
        assert restored.compliance[0]["regulation"] == "民法典"

    def test_correction_entry(self):
        """修正条目的格式应包含修正信息。"""
        entry = MemoryEntry(
            clause_type="违约",
            clause_content="违约金过高",
            is_correction=True,
            correction_field="risk_level",
            correction_value="high",
        )
        text = entry.format_reference()
        assert "修正案例" in text
        assert "risk_level" in text
        assert "high" in text

    def test_normal_entry_format(self):
        """普通记忆条目的格式应包含风险详情。"""
        entry = MemoryEntry(
            clause_type="保密",
            clause_content="双方保密",
            risk_level="low",
            risk_reason="标准条款",
            compliance=[{"regulation": "民法典", "status": True, "detail": "合规"}],
        )
        text = entry.format_reference()
        assert "历史案例" in text
        assert "low" in text
        assert "民法典" in text


class TestMemoryStore:
    """MemoryStore 的增删查测测试。"""

    def test_remember_and_recall(self):
        """存储后应能检索到相似内容。"""
        with tempfile.TemporaryDirectory(prefix="memory_test_") as tmpdir:
            emb = _FixedEmbedding()
            store = MemoryStore(tmpdir, embedding=emb)

            clause = Clause(clause_type="保密", content="双方应严格保密")
            store.remember_session(
                clauses=[clause],
                risks=[
                    RiskAssessment(
                        clause=clause,
                        risk_level=RiskLevel.MEDIUM,
                        reason="模糊",
                        suggestion="明确",
                    )
                ],
                compliance=[[ComplianceCheck(regulation="民法典", status=True, detail="合规")]],
                session_id="test_s1",
            )
            results = store.recall("保密", top_k=5)
            assert len(results) >= 1
            assert results[0].clause_type == "保密"
            store.close()

    def test_recall_no_memory(self):
        """空记忆库应返回空列表。"""
        with tempfile.TemporaryDirectory(prefix="memory_test_") as tmpdir:
            store = MemoryStore(tmpdir, embedding=_FixedEmbedding())
            assert store.recall("anything") == []
            store.close()

    def test_disabled_store(self):
        """禁用时应跳过所有操作。"""
        store = MemoryStore("/tmp/nonexistent", enabled=False)
        assert not store.enabled
        store.remember_session([], [], [])
        assert store.recall("test") == []
        store.correct("t", "c", "f", "v")

    def test_correction_storage(self):
        """修正信号应能存储并检索到。"""
        with tempfile.TemporaryDirectory(prefix="memory_test_") as tmpdir:
            emb = _FixedEmbedding()
            store = MemoryStore(tmpdir, embedding=emb)
            store.correct("保密", "保密条款内容", "risk_level", "high")
            results = store.recall("保密", top_k=5)
            assert len(results) == 1
            assert results[0].is_correction
            store.close()

    def test_format_memory_context_empty(self):
        """空记忆列表应返回空字符串。"""
        with tempfile.TemporaryDirectory(prefix="memory_test_") as tmpdir:
            store = MemoryStore(tmpdir, enabled=False)
            assert store.format_memory_context([]) == ""

    def test_format_memory_context_with_entries(self):
        """非空记忆列表应生成格式化文本。"""
        with tempfile.TemporaryDirectory(prefix="memory_test_") as tmpdir:
            store = MemoryStore(tmpdir, enabled=False)
            entries = [
                MemoryEntry(clause_type="保密", clause_content="test", risk_level="medium"),
            ]
            text = store.format_memory_context(entries)
            assert "历史参考案例" in text
            assert "保密" in text
