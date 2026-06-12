from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.utils.io import read_json, write_json


class ReplayStorage:
    def __init__(self, storage_dir: str | Path | None = None):
        if storage_dir:
            self._dir = Path(storage_dir)
        else:
            from harness.core.config import HarnessConfig
            self._dir = Path(HarnessConfig().replay_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict) -> Path:
        filepath = self._filepath(session_id)
        write_json(filepath, data)
        return filepath

    def load(self, session_id: str) -> dict[str, Any] | None:
        filepath = self._filepath(session_id)
        if not filepath.exists():
            return None
        return read_json(filepath)

    def delete(self, session_id: str) -> bool:
        filepath = self._filepath(session_id)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for f in sorted(self._dir.glob("session_*.json"), reverse=True):
            try:
                data = read_json(f)
                sessions.append({
                    "session_id": data.get("session_id", f.stem),
                    "document_title": data.get("document", {}).get("title", ""),
                    "started_at": data.get("started_at", ""),
                    "finished_at": data.get("finished_at", ""),
                })
            except (json.JSONDecodeError, Exception):
                continue
        return sessions

    def _filepath(self, session_id: str) -> Path:
        return self._dir / f"session_{session_id}.json"
