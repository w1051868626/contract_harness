"""封装 OpenAI API 调用，提供统一的 LLM 交互接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from harness.core.config import LLMConfig


@dataclass
class LLMResponse:
    """LLM 响应数据类，包含内容、模型及用量信息。"""

    content: str
    model: str
    usage: dict[str, int] | None = None


class LLMClient:
    """LLM 客户端封装，管理 API 连接与请求参数。"""

    def __init__(self, config: LLMConfig | None = None):
        """初始化客户端配置，延迟创建 OpenAI 连接。"""
        self.config = config or LLMConfig()
        self._client: OpenAI | None = None

    def _build_http_client(self) -> httpx.Client:
        """构建带有代理配置的 HTTP 客户端。"""
        kwargs: dict[str, Any] = {}
        if self.config.proxy:
            kwargs["proxy"] = self.config.proxy
        return httpx.Client(**kwargs)

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端实例。"""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "未设置 API 密钥。请通过环境变量 OPENAI_API_KEY 或 LLMConfig.api_key 配置。"
                )
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                http_client=self._build_http_client(),
                timeout=self.config.timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """发送聊天请求并返回 LLM 响应。"""
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        resp = self.client.chat.completions.create(**params)
        choice = resp.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )
