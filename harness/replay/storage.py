"""回放存储模块，基于文件系统的 JSON 存储管理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.core.config import HarnessConfig
from harness.utils.io import read_json, write_json


class ReplayStorage:
    """回放存储，提供会话的保存、加载、删除和列表功能。"""

    def __init__(self, storage_dir: str | Path | None = None):
        """初始化存储目录。"""
        if storage_dir:
            self._dir = Path(storage_dir)
        else:
            self._dir = Path(HarnessConfig().replay_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict) -> Path:
        """保存会话数据到文件。"""
        filepath = self._filepath(session_id)
        write_json(filepath, data)
        return filepath

    def load(self, session_id: str) -> dict[str, Any] | None:
        """加载会话数据。"""
        filepath = self._filepath(session_id)
        if not filepath.exists():
            return None
        return read_json(filepath)

    def delete(self, session_id: str) -> bool:
        """删除会话文件。"""
        filepath = self._filepath(session_id)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话摘要信息。"""
        sessions = []
        for f in sorted(self._dir.glob("session_*.json"), reverse=True):
            try:
                data = read_json(f)
                sessions.append(
                    {
                        "session_id": data.get("session_id", f.stem),
                        "document_title": data.get("document", {}).get("title", ""),
                        "started_at": data.get("started_at", ""),
                        "finished_at": data.get("finished_at", ""),
                    }
                )
            except (json.JSONDecodeError, Exception):
                continue
        return sessions

    def _filepath(self, session_id: str) -> Path:
        """获取会话文件的完整路径。"""
        return self._dir / f"session_{session_id}.json"
