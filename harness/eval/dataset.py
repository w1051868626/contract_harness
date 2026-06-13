from __future__ import annotations

"""评测数据集加载与管理。"""

import json
from pathlib import Path
from typing import Any

from harness.core.types import ContractDocument, RiskLevel


class EvalDataset:
    """评测数据集，支持从 JSON 文件或目录加载。"""

    def __init__(self, data_dir: str | Path | None = None):
        self._dir = Path(data_dir) if data_dir else Path.cwd() / "examples" / "contracts"
        self._items: list[EvalItem] = []

    def load(self, path: str | Path | None = None) -> None:
        """从文件或目录加载评测项。"""
        source = Path(path) if path else self._dir
        if source.is_file():
            self._load_file(source)
        elif source.is_dir():
            for f in sorted(source.glob("*.json")):
                self._load_file(f)

    def _load_file(self, path: Path) -> None:
        """加载单个 JSON 文件中的评测项。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                self._items.append(EvalItem.from_dict(item))
        else:
            self._items.append(EvalItem.from_dict(data))

    @property
    def items(self) -> list[EvalItem]:
        """返回所有评测项的副本。"""
        return list(self._items)

    def add_item(self, item: EvalItem) -> None:
        """添加单个评测项。"""
        self._items.append(item)


class EvalItem:
    """单个评测项，包含合同文档与期望结果。"""
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
        """从字典构造评测项。"""
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
