"""文件读写与序列化工具函数。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv as _load_dotenv
from pydantic import BaseModel


def make_id() -> str:
    """生成 12 字符短 ID。"""
    return uuid.uuid4().hex[:12]


def normalize_text(text: str) -> str:
    """清理全角空格等非标准空白字符。"""
    return text.replace("\u3000", " ")


def load_dotenv(dotenv_path: str | Path | None = None) -> bool:
    """从 .env 文件加载环境变量。

    从当前目录开始向上搜索 .env 文件，或加载指定路径的文件。
    多次调用安全，已加载后不会重复覆盖已有环境变量。
    """
    if dotenv_path is None:
        return _load_dotenv()
    return _load_dotenv(dotenv_path=Path(dotenv_path))


def read_text(path: str | Path) -> str:
    """读取文本文件（UTF-8），同时清理全角空格等非标准空白字符。"""
    return normalize_text(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, content: str) -> None:
    """写入文本文件（UTF-8）。"""
    Path(path).write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    """读取并解析 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    """将数据写入 JSON 文件（含缩进与中文支持）。"""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def serialize_model(model: BaseModel) -> dict[str, Any]:
    """将 Pydantic 模型序列化为字典。"""
    return json.loads(model.model_dump_json())
