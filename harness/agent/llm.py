"""封装 OpenAI API 调用，提供统一的 LLM 交互接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from harness.core.config import LLMConfig
from harness.utils.log import logger


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

    def _mock_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """无 API 密钥时返回模拟响应，确保流水线不中断。"""
        logger.warning("使用模拟 LLM 响应（未设置 API 密钥）")
        combined = " ".join(m.get("content", "") for m in messages)

        if "合同条款提取" in combined:
            clauses = self._mock_clauses(combined)
            return LLMResponse(content=json.dumps(clauses, ensure_ascii=False), model="mock")

        if "风险分析" in combined:
            return LLMResponse(
                content=json.dumps(
                    {
                        "risk_level": "medium",
                        "reason": "条款表述不够明确，存在潜在解释分歧",
                        "suggestion": "建议对关键术语进行明确定义",
                    },
                    ensure_ascii=False,
                ),
                model="mock",
            )

        if "合规" in combined:
            return LLMResponse(
                content=json.dumps(
                    {"status": True, "detail": "符合相关法律规定"}, ensure_ascii=False
                ),
                model="mock",
            )

        return LLMResponse(content="模拟审查完成，未发现异常。", model="mock")

    @staticmethod
    def _mock_clauses(text: str) -> list[dict[str, str]]:
        """从文本中提取模拟条款。"""
        types = ["保密义务", "违约责任", "知识产权", "管辖与法律适用", "付款条件"]
        found: list[dict[str, str]] = []
        for t in types:
            if t in text:
                found.append(
                    {
                        "type": t,
                        "content": f"关于{t}的约定，双方应遵守相关法律法规。",
                        "risk": "medium" if t in ("违约责任",) else "low",
                    }
                )
        if not found:
            found.append(
                {
                    "type": "一般条款",
                    "content": text[:200] if len(text) > 200 else text,
                    "risk": "info",
                }
            )
        return found

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

        try:
            resp = self.client.chat.completions.create(**params)
            choice = resp.choices[0]
            logger.debug(
                "LLM 调用成功: model={}, input_tokens={}",
                resp.model,
                resp.usage.total_tokens if resp.usage else "N/A",
            )
        except ValueError:
            logger.warning("LLM API 密钥缺失，回退到模拟响应")
            return self._mock_chat(messages)

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )
