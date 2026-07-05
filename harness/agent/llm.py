"""封装 OpenAI API 调用，提供统一的 LLM 交互接口。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from harness.core.config import LLMConfig
from harness.core.exceptions import AgentError
from harness.utils.log import logger

# 网络层错误重试基线退避秒数，实际等待 = base * 2^attempt
_RETRY_BASE_DELAY: float = 1.0


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
                静默产出假报告。也可通过环境变量 ``LLM_MOCK=1`` 启用，
                便于 CI 在无 API secret 时跑通回归测试。
        """
        self.config = config or LLMConfig()
        # 环境变量 LLM_MOCK=1 自动启用 mock，无需改调用方代码
        # （CI 无 OPENAI_API_KEY secret 时靠此跑通回归测试）
        if not mock and os.getenv("LLM_MOCK", "").lower() in ("1", "true", "yes"):
            mock = True
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

        mock 模式下直接返回 None（不创建真实连接），调用方应在 ``chat`` 入口
        通过 ``self._mock`` 判断，不应直接访问此属性发起请求。
        """
        # mock 模式短路：避免每次访问都重跑密钥检查
        if self._mock:
            return None
        if self._client is None:
            if not self.config.api_key:
                # 密钥缺失是配置错误，必须抛 AgentError 而非静默回退，
                # 避免产出假审查报告。测试场景应用 LLMClient(mock=True)。
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
        *,
        max_retries: int = 3,
        **kwargs,
    ) -> LLMResponse:
        """发送聊天请求并返回 LLM 响应。

        Args:
            messages: OpenAI 消息列表。
            tools: 可选工具定义。
            max_retries: 网络/限流错误的最大重试次数（指数退避）。
                鉴权/请求格式等非瞬时错误不重试，直接抛 ``AgentError``。
        """
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

        # 仅对瞬时错误（网络/限流）重试，鉴权/请求格式错误直接抛出
        # mock 模式已在上面短路返回，此处 client 一定非 None；用显式
        # 检查替代 assert，避免 python -O 剥离后 client 为 None 时抛
        # 无意义的 AttributeError 而非有意义的 AgentError。
        client = self.client
        if client is None:
            raise AgentError("LLM client 未初始化（mock 模式应在上面短路返回）")
        last_error: Exception | None = None
        last_error_msg: str = ""
        for attempt in range(max(max_retries, 1)):
            try:
                resp = client.chat.completions.create(**params)
                choice = resp.choices[0]
                logger.debug(
                    "LLM 调用成功: model={}, input_tokens={}",
                    resp.model,
                    resp.usage.total_tokens if resp.usage else "N/A",
                )
                return LLMResponse(
                    content=choice.message.content or "",
                    model=resp.model,
                    usage=resp.usage.model_dump() if resp.usage else None,
                )
            except (APIConnectionError, APITimeoutError) as e:
                # 网络层错误：可重试（必须在 APIError 之前捕获，
                # 因为 APIConnectionError 是 APIError 子类）
                last_error = e
                last_error_msg = f"LLM API 网络错误: {e}"
            except RateLimitError as e:
                # 限流：可重试（同样是 APIError 子类，需在 APIError 之前）
                last_error = e
                last_error_msg = f"LLM API 触发限流，请稍后重试: {e}"
            except httpx.HTTPError as e:
                last_error = e
                last_error_msg = f"LLM HTTP 传输错误: {e}"
            except APIError as e:
                # 兜底剩余 OpenAI API 错误（含鉴权失败、请求格式错误等）：
                # 非瞬时错误，不重试直接抛出。
                logger.error("LLM API 调用失败: {}", str(e))
                raise AgentError(f"LLM API 调用失败: {e}") from e

            # max_retries<=1：不重试，直接抛原始包装错误（保留具体错误类别字样）
            if max_retries <= 1:
                logger.error(last_error_msg)
                raise AgentError(last_error_msg) from last_error

            # 可重试错误：指数退避后进入下一轮（限流等待时间加倍）
            if isinstance(last_error, RateLimitError):
                delay = _RETRY_BASE_DELAY * (2 ** (attempt + 1))
            else:
                delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "{}（第 {} 次），{:.0f}s 后重试",
                last_error_msg,
                attempt + 1,
                delay,
            )
            time.sleep(delay)

        # 重试耗尽：抛出聚合错误
        logger.error("LLM API 重试 {} 次后仍失败: {}", max_retries, str(last_error))
        raise AgentError(f"LLM API 重试 {max_retries} 次后仍失败: {last_error}") from last_error
