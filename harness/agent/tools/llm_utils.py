"""LLM 工具共享函数，提供 JSON 解析、正则预编译等公共能力。"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.utils.log import logger

_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json_array(text: str) -> list[Any]:
    """从 LLM 响应中提取 JSON 数组，失败返回空列表。"""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        logger.debug("JSON 数组解析失败: text[:100]={}", text[:100])
        return []


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从 LLM 响应中提取 JSON 对象，失败返回 None。"""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        logger.debug("JSON 对象解析失败: text[:100]={}", text[:100])
        return None


def strip_json_fences(text: str) -> str:
    """移除 Markdown JSON 代码块围栏。"""
    return _JSON_FENCE_RE.sub("", text.strip())
