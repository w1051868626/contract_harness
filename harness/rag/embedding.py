"""嵌入提供者模块，封装文本向量化接口。"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from openai import APIError, AuthenticationError, BadRequestError, InternalServerError, OpenAI

from harness.core.exceptions import EmbeddingError
from harness.rag.constants import EMBED_MAX_CHARS
from harness.utils.log import logger

_SENTENCE_SPLIT = re.compile(r"[。！？；.!?;\n]")

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
    """基于 openai 库的嵌入实现。"""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        proxy: str | None = None,
    ):
        self.model = model
        http_client = httpx.Client(proxy=proxy) if proxy else None
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
            http_client=http_client,
        )

    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""
        return self.embed_batch([text])[0]

    def _hash_embed(self, text: str) -> list[float]:
        """基于哈希的伪嵌入（离线回退），使用 SHA256 确保跨运行确定性。"""
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return [(int(h[i : i + 2], 16) % 1000) / 1000.0 for i in range(0, min(32, len(h)), 2)] or [
            0.0
        ]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量通过 openai 库调用嵌入 API，超长文本自动截断。"""
        logger.debug("Embedding batch of {} texts with model={}", len(texts), self.model)
        truncated = [_truncate_at_boundary(t, EMBED_MAX_CHARS) for t in texts]
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
        except Exception as exc:
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
) -> EmbeddingProvider:
    """工厂函数，创建嵌入提供者实例。"""
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
            model=model or "text-embedding-3-small",
            proxy=proxy,
        )
    if provider == "local":
        return LocalEmbeddingProvider(model_name=model or "BAAI/bge-small-zh-v1.5")
    raise ValueError(f"Unsupported embedding provider: {provider}")
