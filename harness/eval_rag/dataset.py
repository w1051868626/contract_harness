from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalRagItem:
    query: str
    expected_chunk_ids: list[str]
    expected_texts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRagResult:
    dataset_name: str
    top_ks: list[int]
    hit_rates: dict[int, float]
    mrr: dict[int, float]
    precisions: dict[int, float]
    recalls: dict[int, float]
    details: list[dict[str, Any]] = field(default_factory=list)


def save_jsonl(path: str, items: list[EvalRagItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "query": item.query,
                        "expected_chunk_ids": item.expected_chunk_ids,
                        "expected_texts": item.expected_texts,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path: str) -> list[EvalRagItem]:
    items: list[EvalRagItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败: {e}") from e
            items.append(
                EvalRagItem(
                    query=data["query"],
                    expected_chunk_ids=data["expected_chunk_ids"],
                    expected_texts=data.get("expected_texts", []),
                    metadata=data.get("metadata", {}),
                )
            )
    return items
