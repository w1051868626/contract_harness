from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        proxy: str | None = None,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._http_client.post(
            f"{self.api_base}/embeddings",
            json={"model": self.model, "input": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model: Any = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
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
