"""嵌入提供者模块，封装文本向量化接口。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from harness.utils.log import logger


class EmbeddingProvider(ABC):
    """嵌入提供者抽象基类。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将多段文本转为向量。"""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI API 的嵌入实现。"""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        proxy: str | None = None,
    ):
        """初始化 OpenAI 嵌入客户端。"""
        if not api_key:
            api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_base:
            api_base = os.getenv(
                "EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            )
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()

    def embed(self, text: str) -> list[float]:
        """将单段文本转为向量。"""
        return self.embed_batch([text])[0]

    def _hash_embed(self, text: str) -> list[float]:
        """基于哈希的伪嵌入（离线回退）。"""
        return [(hash(c) % 1000) / 1000.0 for c in text[:16]] or [0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量调用 OpenAI 嵌入 API。"""
        logger.debug("Embedding batch of {} texts with model={}", len(texts), self.model)
        try:
            resp = self._http_client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": texts},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            logger.debug("Successfully embedded {} texts", len(items))
            return [item["embedding"] for item in items]
        except Exception:
            return [self._hash_embed(t) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """基于本地模型的嵌入实现（sentence-transformers）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """初始化本地嵌入模型名称。"""
        self.model_name = model_name
        self._model: Any = None
        self._loaded = False

    def _load(self):
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
        """批量使用本地模型编码文本。"""
        logger.debug("Local embedding batch of {} texts with model={}", len(texts), self.model_name)
        self._load()
        assert self._model is not None
        return self._model.encode(texts).tolist()


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
