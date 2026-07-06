"""嵌入提供者模块，封装文本向量化接口。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from openai import APIError, AuthenticationError, BadRequestError, InternalServerError, OpenAI

from harness.core.exceptions import EmbeddingError
from harness.rag.constants import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LOCAL_EMBED_MODEL,
    DEFAULT_OPENAI_API_BASE,
    EMBED_MAX_CHARS,
)
from harness.rag.rate_limit import RateLimiter
from harness.utils.log import logger

_SENTENCE_SPLIT = re.compile(r"[。！？；.!?;\n]")


def _estimate_tokens(texts: list[str]) -> int:
    """粗略估算 token 数（中文 ~1 token/字符，英文 ~1 token/4 字符）。"""
    total = 0
    for t in texts:
        ascii_count = sum(1 for c in t if ord(c) < 128)
        non_ascii_count = len(t) - ascii_count
        total += ascii_count // 4 + non_ascii_count
    return total


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """在句子边界截断文本，避免切断语义。"""
    if len(text) <= max_chars:
        return text
    m = _SENTENCE_SPLIT.search(text, max_chars - 1, max_chars)
    if m:
        return text[: m.start() + 1]
    pos = max(text.rfind(c, 0, max_chars) for c in "。！？；.!?;\n")
    if pos > max_chars // 2:
        return text[: pos + 1]
    return text[:max_chars]


class EmbeddingProvider(ABC):
    """嵌入提供者抽象基类。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将多段文本转为向量。"""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 openai 库的嵌入实现，支持速率限制。"""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = DEFAULT_OPENAI_API_BASE,
        model: str = DEFAULT_EMBED_MODEL,
        proxy: str | None = None,
        max_rpm: int = 0,
        max_tpm: int = 0,
    ):
        self.model = model
        http_client = httpx.Client(proxy=proxy) if proxy else None
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            http_client=http_client,
        )
        self._rate_limiter = RateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)

    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量通过 openai 库调用嵌入 API，超长文本自动截断，受速率限制保护。"""
        logger.debug("Embedding batch of {} texts with model={}", len(texts), self.model)
        truncated = [_truncate_at_boundary(t, EMBED_MAX_CHARS) for t in texts]
        estimated_tokens = _estimate_tokens(truncated)
        self._rate_limiter.wait_if_needed(estimated_tokens)
        try:
            resp = self._client.embeddings.create(model=self.model, input=truncated)
            items = sorted(resp.data, key=lambda x: x.index)
            logger.debug("Successfully embedded {} texts", len(items))
            return [item.embedding for item in items]
        except AuthenticationError as exc:
            raise EmbeddingError(f"Embedding API 认证失败: {exc}") from exc
        except BadRequestError as exc:
            raise EmbeddingError(f"Embedding API 请求参数错误: {exc}") from exc
        except (APIError, InternalServerError) as exc:
            raise EmbeddingError(f"Embedding API 服务端错误: {exc}") from exc
        except (httpx.RequestError, OSError, RuntimeError) as exc:
            raise EmbeddingError(f"Embedding API 调用失败: {exc}") from exc


class LocalEmbeddingProvider(EmbeddingProvider):
    """基于本地模型的嵌入实现（sentence-transformers）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """初始化本地嵌入模型名称。"""
        self.model_name = model_name
        self._model: Any = None
        self._loaded = False

    def _load(self) -> None:
        """惰性加载本地嵌入模型。"""
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: I001  # pyright: ignore[reportMissingImports]

            self._model = SentenceTransformer(self.model_name)
            self._loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量使用本地模型编码文本，超长文本自动截断。"""
        logger.debug("Local embedding batch of {} texts with model={}", len(texts), self.model_name)
        self._load()
        if self._model is None:
            raise RuntimeError("本地嵌入模型未加载")
        truncated = [_truncate_at_boundary(t, EMBED_MAX_CHARS) for t in texts]
        return self._model.encode(truncated).tolist()


def create_embedding_provider(
    provider: str = "openai",
    api_key: str = "",
    api_base: str = "",
    model: str = "",
    proxy: str | None = None,
    max_rpm: int = 0,
    max_tpm: int = 0,
) -> EmbeddingProvider:
    """工厂函数，创建嵌入提供者实例。"""
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            api_base=api_base or DEFAULT_OPENAI_API_BASE,
            model=model or DEFAULT_EMBED_MODEL,
            proxy=proxy,
            max_rpm=max_rpm,
            max_tpm=max_tpm,
        )
    if provider == "local":
        return LocalEmbeddingProvider(model_name=model or DEFAULT_LOCAL_EMBED_MODEL)
    raise ValueError(f"Unsupported embedding provider: {provider}")
