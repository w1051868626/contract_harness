"""录制模块，将 Agent 会话序列化为 JSON 文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.core.types import AgentSession, ReviewReport
from harness.utils.io import write_json
from harness.utils.log import logger


class SessionRecorder:
    """会话录制器，将 AgentSession 序列化保存。"""

    def __init__(self, replay_dir: str = ""):
        """初始化录制器。"""
        self._replay_dir = Path(replay_dir or "")

    def record(self, session: AgentSession, output_dir: str | Path | None = None) -> Path:
        """录制会话到 JSON 文件。"""
        logger.info("Recording session session_id={}", session.session_id)
        target = Path(output_dir) if output_dir else self._replay_dir
        target.mkdir(parents=True, exist_ok=True)

        filepath = target / f"session_{session.session_id}.json"
        data = self._serialize(session)
        write_json(filepath, data)
        logger.info("Session recorded to {}", filepath)
        return filepath

    def _serialize(self, session: AgentSession) -> dict[str, Any]:
        """将 AgentSession 序列化为字典。"""
        return {
            "session_id": session.session_id,
            "document": {
                "id": session.document.id,
                "title": session.document.title,
                "content": session.document.content,
                "file_path": session.document.file_path,
                "metadata": session.document.metadata,
            },
            "started_at": session.started_at,
            "finished_at": session.finished_at,
            "steps": [
                {
                    "step_index": s.step_index,
                    "agent_message": s.agent_message,
                    "timestamp": s.timestamp,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "input": tc.input,
                            "output": tc.output,
                            "started_at": tc.started_at,
                            "finished_at": tc.finished_at,
                            "duration_ms": tc.duration_ms,
                        }
                        for tc in s.tool_calls
                    ],
                }
                for s in session.steps
            ],
            "report": self._serialize_report(session.report),
            "metadata": session.metadata,
        }

    def _serialize_report(self, report: ReviewReport | None) -> dict[str, Any] | None:
        """将 ReviewReport 序列化为字典。"""
        if report is None:
            return None
        return {
            "document_id": report.document_id,
            "document_title": report.document_title,
            "reviewed_at": report.reviewed_at,
            "summary": report.summary,
            "overall_risk": report.overall_risk.value,
            "clauses": [
                {
                    "clause_type": c.clause_type,
                    "content": c.content,
                    "risk": c.risk.value,
                    "comment": c.comment,
                }
                for c in report.clauses
            ],
            "risks": [
                {
                    "risk_level": r.risk_level.value,
                    "reason": r.reason,
                    "suggestion": r.suggestion,
                }
                for r in report.risks
            ],
            "compliance_checks": [
                {
                    "regulation": c.regulation,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in report.compliance_checks
            ],
        }
