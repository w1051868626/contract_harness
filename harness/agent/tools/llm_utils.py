"""LLM 工具共享函数，提供 JSON 解析、正则预编译等公共能力。"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.utils.log import logger

# 贪婪匹配最外层 [ ... ] / { ... }，配合括号配平校验而非依赖非贪婪。
# 非贪婪 \[.*?\] 会被 JSON 字符串内的 "]" 截断（如 {"reason":"a]b"}），
# 导致解析失败并静默回退到逐条 LLM 调用。
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _balance_bracket(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """从 ``start`` 位置的 ``open_ch`` 开始，按字符串感知的括号配平返回匹配闭合位置。

    跳过 JSON 字符串内部（含转义）的括号，避免被字符串里的 ``]`` / ``}`` 误判。
    找不到配平的闭合位置返回 None。
    """
    depth = 0
    i = start
    in_str = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def extract_json_array(text: str) -> list[Any]:
    """从 LLM 响应中提取 JSON 数组，失败返回空列表。

    解析顺序：
    1. 剥除 ```json``` 围栏后整段尝试 ``json.loads``（最稳健）；
    2. 回退到正则定位首个 ``[``，按括号配平找到闭合 ``]`` 再解析，
       避免非贪婪正则被字符串内的 ``]`` 截断。
    """
    stripped = strip_json_fences(text).strip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    start = text.find("[")
    if start == -1:
        return []
    end = _balance_bracket(text, start, "[", "]")
    if end is None:
        logger.debug("JSON 数组括号未配平: text[:100]={}", text[:100])
        return []
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        logger.debug("JSON 数组解析失败: text[:100]={}", text[:100])
        return []


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从 LLM 响应中提取 JSON 对象，失败返回 None。

    与 ``extract_json_array`` 同样采用"先剥围栏整段解析、再括号配平回退"策略，
    避免非贪婪正则被字符串内的 ``}`` 截断。
    """
    stripped = strip_json_fences(text).strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    start = text.find("{")
    if start == -1:
        return None
    end = _balance_bracket(text, start, "{", "}")
    if end is None:
        logger.debug("JSON 对象括号未配平: text[:100]={}", text[:100])
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        logger.debug("JSON 对象解析失败: text[:100]={}", text[:100])
        return None


def strip_json_fences(text: str) -> str:
    """移除 Markdown JSON 代码块围栏。"""
    return _JSON_FENCE_RE.sub("", text.strip())
