"""文件读写与序列化工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def read_text(path: str | Path) -> str:
    """读取文本文件（UTF-8）。"""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """写入文本文件（UTF-8）。"""
    Path(path).write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    """读取并解析 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    """将数据写入 JSON 文件（含缩进与中文支持）。"""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def serialize_model(model: BaseModel) -> dict[str, Any]:
    """将 Pydantic 模型序列化为字典。"""
    return json.loads(model.model_dump_json())
