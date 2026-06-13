from __future__ import annotations

"""测试全局配置，包含 MockLLMClient 与通用 fixture。"""

from typing import Any

import pytest

from harness.agent.llm import LLMResponse


class MockLLMClient:
    """模拟 LLM 客户端，返回预设响应列表。"""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, **kwargs):
        """记录调用并返回下一个预设响应，不足则返回空响应。"""
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return LLMResponse(content="", model="mock")


@pytest.fixture
def clause_extract_response() -> str:
    """返回模拟的条款提取 JSON 响应。"""
    return """[
        {"type": "保密", "content": "双方应对本合同内容保密", "risk": "low"},
        {"type": "违约责任", "content": "违约方应赔偿守约方全部损失", "risk": "medium"}
    ]"""


@pytest.fixture
def risk_analysis_response() -> str:
    """返回模拟的风险分析 JSON 响应。"""
    return '{"risk_level": "medium", "reason": "赔偿范围不明确", "suggestion": "建议明确赔偿上限"}'


@pytest.fixture
def compliance_response() -> str:
    """返回模拟的合规检查 JSON 响应。"""
    return '{"status": true, "detail": "符合相关法律规定"}'


@pytest.fixture
def summary_response() -> str:
    """返回模拟的摘要 JSON 响应。"""
    return "审查完成，共发现2个条款，1个中风险项，0个不合规项。"


@pytest.fixture
def mock_llm(
    clause_extract_response: str,
    risk_analysis_response: str,
    compliance_response: str,
    summary_response: str,
) -> MockLLMClient:
    """每个工具 + 1次summary，共需 1 + 2 + 2*5 + 1 = 14个响应"""
    responses: list[LLMResponse] = [
        LLMResponse(content=clause_extract_response, model="mock"),
    ]
    for _ in range(2):
        responses.append(LLMResponse(content=risk_analysis_response, model="mock"))
    for _ in range(10):
        responses.append(LLMResponse(content=compliance_response, model="mock"))
    responses.append(LLMResponse(content=summary_response, model="mock"))
    return MockLLMClient(responses)
