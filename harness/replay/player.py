"""回放模块，反序列化并逐步骤播放录制会话。"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from typing import Any

from harness.core.types import (
    AgentSession,
    AgentStep,
    Clause,
    ComplianceCheck,
    ContractDocument,
    ReviewReport,
    RiskAssessment,
    RiskLevel,
    ToolCall,
)
from harness.replay.storage import ReplayStorage
from harness.utils.log import logger


class SessionPlayer:
    """会话播放器，加载和遍历录制会话。"""

    def __init__(self, storage: ReplayStorage | None = None):
        """初始化播放器。"""
        self._storage = storage or ReplayStorage()

    def load(self, session_id: str) -> AgentSession | None:
        """加载指定会话。"""
        logger.info("Loading session session_id={}", session_id)
        data = self._storage.load(session_id)
        if data is None:
            return None
        session = self._deserialize(data)
        if session.started_at and session.finished_at:
            start = datetime.fromisoformat(session.started_at)
            end = datetime.fromisoformat(session.finished_at)
            logger.info(
                "Session loaded session_id={} (duration={:.1f}s)",
                session_id,
                (end - start).total_seconds(),
            )
        else:
            logger.info("Session loaded session_id={}", session_id)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有录制会话。"""
        return self._storage.list_sessions()

    def step_through(self, session_id: str) -> Generator[AgentStep, None, None]:
        """逐步骤生成器，遍历会话步骤。"""
        session = self.load(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        for step in session.steps:
            yield step

    def _deserialize(self, data: dict[str, Any]) -> AgentSession:
        """将字典反序列化为 AgentSession。"""
        doc = ContractDocument(
            id=data["document"]["id"],
            title=data["document"]["title"],
            content=data["document"]["content"],
            file_path=data["document"].get("file_path"),
            metadata=data["document"].get("metadata", {}),
        )

        steps = []
        for s in data.get("steps", []):
            tool_calls = [
                ToolCall(
                    tool_name=tc["tool_name"],
                    input=tc["input"],
                    output=tc.get("output"),
                    started_at=tc.get("started_at", ""),
                    finished_at=tc.get("finished_at", ""),
                    duration_ms=tc.get("duration_ms", 0.0),
                )
                for tc in s.get("tool_calls", [])
            ]
            steps.append(
                AgentStep(
                    step_index=s["step_index"],
                    agent_message=s.get("agent_message", ""),
                    tool_calls=tool_calls,
                    timestamp=s.get("timestamp", ""),
                )
            )

        report_data = data.get("report")
        report = self._deserialize_report(report_data) if report_data else None

        return AgentSession(
            session_id=data["session_id"],
            document=doc,
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
            steps=steps,
            report=report,
            metadata=data.get("metadata", {}),
        )

    def _deserialize_report(self, data: dict[str, Any]) -> ReviewReport:
        """将字典反序列化为 ReviewReport。"""
        return ReviewReport(
            document_id=data["document_id"],
            document_title=data["document_title"],
            reviewed_at=data["reviewed_at"],
            summary=data.get("summary", ""),
            clauses=[
                Clause(
                    clause_type=c["clause_type"],
                    content=c["content"],
                    risk=RiskLevel(c.get("risk", "info")),
                    comment=c.get("comment", ""),
                )
                for c in data.get("clauses", [])
            ],
            risks=[
                RiskAssessment(
                    clause=Clause(clause_type="", content=""),
                    risk_level=RiskLevel(r["risk_level"]),
                    reason=r.get("reason", ""),
                    suggestion=r.get("suggestion", ""),
                )
                for r in data.get("risks", [])
            ],
            compliance_checks=[
                ComplianceCheck(
                    regulation=c["regulation"],
                    status=c["status"],
                    detail=c.get("detail", ""),
                )
                for c in data.get("compliance_checks", [])
            ],
            overall_risk=RiskLevel(data.get("overall_risk", "info")),
        )
