"""封装 OpenAI API 调用，提供统一的 LLM 交互接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from harness.core.config import LLMConfig
from harness.core.exceptions import AgentError
from harness.utils.log import logger


@dataclass
class LLMResponse:
    """LLM 响应数据类，包含内容、模型及用量信息。"""

    content: str
    model: str
    usage: dict[str, int] | None = None


class LLMClient:
    """LLM 客户端封装，管理 API 连接与请求参数。"""

    def __init__(self, config: LLMConfig | None = None, mock: bool = False):
        """初始化客户端配置，延迟创建 OpenAI 连接。

        Args:
            config: LLM 配置，为 None 时使用默认配置。
            mock: 显式启用 mock 模式，返回预设模拟响应。
                仅用于测试或本地无 API 场景，避免生产环境密钥缺失时
                静默产出假报告。
        """
        self.config = config or LLMConfig()
        self._mock = mock
        self._client: OpenAI | None = None

    def _build_http_client(self) -> httpx.Client:
        """构建带有代理配置的 HTTP 客户端。"""
        kwargs: dict[str, Any] = {}
        if self.config.proxy:
            kwargs["proxy"] = self.config.proxy
        return httpx.Client(**kwargs)

    @property
    def client(self) -> OpenAI | None:
        """懒加载 OpenAI 客户端实例。

        mock 模式下返回 None（不创建真实连接），调用方应在 ``chat`` 入口
        通过 ``self._mock`` 判断，不应直接访问此属性发起请求。
        """
        if self._client is None:
            if not self.config.api_key:
                # 显式 mock 模式下允许无密钥；否则密钥缺失是配置错误，
                # 必须抛 AgentError 而非静默回退，避免产出假审查报告。
                if self._mock:
                    return None
                raise AgentError(
                    "未设置 API 密钥。请通过环境变量 OPENAI_API_KEY 或 LLMConfig.api_key 配置；"
                    "测试场景可显式传入 LLMClient(mock=True) 启用模拟响应。"
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

        # 显式 mock 模式：跳过真实 API，返回模拟响应
        if self._mock:
            return self._mock_chat(messages)

        try:
            resp = self.client.chat.completions.create(**params)
            choice = resp.choices[0]
            logger.debug(
                "LLM 调用成功: model={}, input_tokens={}",
                resp.model,
                resp.usage.total_tokens if resp.usage else "N/A",
            )
        except (APIConnectionError, APITimeoutError) as e:
            # 网络层错误：可重试，向上抛出由调用方决定重试策略
            logger.error("LLM API 网络错误: {}", str(e))
            raise AgentError(f"LLM API 网络错误: {e}") from e
        except RateLimitError as e:
            logger.warning("LLM API 触发限流: {}", str(e))
            raise AgentError(f"LLM API 触发限流，请稍后重试: {e}") from e
        except APIError as e:
            # 兜底所有 OpenAI API 错误（含鉴权失败、请求格式错误等）
            logger.error("LLM API 调用失败: {}", str(e))
            raise AgentError(f"LLM API 调用失败: {e}") from e
        except httpx.HTTPError as e:
            logger.error("LLM HTTP 传输错误: {}", str(e))
            raise AgentError(f"LLM HTTP 传输错误: {e}") from e

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )
