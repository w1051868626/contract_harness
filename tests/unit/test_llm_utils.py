"""LLM 工具共享函数（JSON 解析、围栏剥离）单元测试。

重点覆盖 ``extract_json_array`` / ``extract_json_object`` 对 LLM 输出
各种形态（围栏包裹、字符串内含括号、多对象逗号分隔、单对象）的健壮性。
"""

from __future__ import annotations

from harness.agent.tools.llm_utils import (
    extract_json_array,
    extract_json_object,
    strip_json_fences,
)


class TestStripJsonFences:
    def test_strip_markdown_json_fence(self):
        assert strip_json_fences("```json\n[1,2]\n```") == "[1,2]"

    def test_strip_bare_fence(self):
        assert strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_no_fence_unchanged(self):
        assert strip_json_fences('{"a":1}') == '{"a":1}'


class TestExtractJsonArray:
    def test_plain_array(self):
        assert extract_json_array('[{"index":0}]') == [{"index": 0}]

    def test_fenced_array(self):
        assert extract_json_array('```json\n[{"index":0}]\n```') == [{"index": 0}]

    def test_string_containing_close_bracket(self):
        """字符串内的 ] 不应截断数组解析（R1 回归）。"""
        text = '[{"index":0,"reason":"赔偿上限]约定不明"}]'
        result = extract_json_array(text)
        assert result == [{"index": 0, "reason": "赔偿上限]约定不明"}]

    def test_fenced_array_with_bracket_in_string(self):
        """围栏 + 字符串内 ] 组合（R1 最易失败场景）。"""
        text = '```json\n[{"reason":"a]b"}]\n```'
        result = extract_json_array(text)
        assert result == [{"reason": "a]b"}]

    def test_multiple_brackets_in_strings(self):
        text = '[{"a":"x]y"},{"b":"p]q]r"}]'
        result = extract_json_array(text)
        assert result == [{"a": "x]y"}, {"b": "p]q]r"}]

    def test_escaped_quote_in_string(self):
        text = '[{"reason":"含\\"引号]和括号"}]'
        result = extract_json_array(text)
        assert result == [{"reason": '含"引号]和括号'}]

    def test_empty_array(self):
        assert extract_json_array("[]") == []

    def test_no_array_returns_empty(self):
        assert extract_json_array("没有数组") == []

    def test_malformed_returns_empty(self):
        assert extract_json_array("[{bad json}]") == []


class TestExtractJsonObject:
    def test_plain_object(self):
        assert extract_json_object('{"a":1}') == {"a": 1}

    def test_fenced_object(self):
        assert extract_json_object('```json\n{"a":1}\n```') == {"a": 1}

    def test_string_containing_close_brace(self):
        """字符串内的 } 不应截断对象解析（R1 回归）。"""
        text = '{"reason":"条款}符号"}'
        assert extract_json_object(text) == {"reason": "条款}符号"}

    def test_nested_object_with_brace_in_string(self):
        text = '{"outer":{"inner":"a}b"}}'
        assert extract_json_object(text) == {"outer": {"inner": "a}b"}}

    def test_no_object_returns_none(self):
        assert extract_json_object("没有对象") is None

    def test_malformed_returns_none(self):
        assert extract_json_object("{bad json}") is None
