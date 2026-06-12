from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.core.types import ContractDocument, RiskLevel


class EvalDataset:
    def __init__(self, data_dir: str | Path | None = None):
        self._dir = Path(data_dir) if data_dir else Path.cwd() / "examples" / "contracts"
        self._items: list[EvalItem] = []

    def load(self, path: str | Path | None = None) -> None:
        source = Path(path) if path else self._dir
        if source.is_file():
            self._load_file(source)
        elif source.is_dir():
            for f in sorted(source.glob("*.json")):
                self._load_file(f)

    def _load_file(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                self._items.append(EvalItem.from_dict(item))
        else:
            self._items.append(EvalItem.from_dict(data))

    @property
    def items(self) -> list[EvalItem]:
        return list(self._items)

    def add_item(self, item: EvalItem) -> None:
        self._items.append(item)


class EvalItem:
    def __init__(
        self,
        document: ContractDocument,
        expected_clauses: list[dict] | None = None,
        expected_risks: list[dict] | None = None,
        expected_compliance: list[dict] | None = None,
        expected_risk_level: RiskLevel = RiskLevel.INFO,
        metadata: dict[str, Any] | None = None,
    ):
        self.document = document
        self.expected_clauses = expected_clauses or []
        self.expected_risks = expected_risks or []
        self.expected_compliance = expected_compliance or []
        self.expected_risk_level = expected_risk_level
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: dict) -> EvalItem:
        doc = ContractDocument(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
        )
        return cls(
            document=doc,
            expected_clauses=data.get("expected_clauses", []),
            expected_risks=data.get("expected_risks", []),
            expected_compliance=data.get("expected_compliance", []),
            expected_risk_level=RiskLevel(data.get("expected_risk_level", "info")),
            metadata=data.get("metadata", {}),
        )
