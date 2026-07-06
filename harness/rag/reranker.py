"""重排序模块，对粗排结果进行精排，提升检索准确率。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from harness.rag.constants import (
    DEFAULT_LOCAL_RERANK_MODEL,
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_RERANK_MODEL,
)
from harness.rag.rate_limit import RateLimiter
from harness.rag.vector_store import Chunk
from harness.utils.log import logger


def _estimate_rerank_tokens(query: str, candidates: list[Chunk]) -> int:
    """粗略估算 rerank 请求 token 数（query + 各候选文本）。"""
    total = len(query)
    for c in candidates:
        total += len(c.content)
    # 中文 ~1 token/字符，英文 ~1 token/4 字符，取折中 0.5
    ascii_count = sum(1 for c in query if ord(c) < 128)
    return (total - ascii_count) + ascii_count // 4


class Reranker(ABC):
    """重排序器抽象基类。"""

    @abstractmethod
    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """对候选列表按相关性重新排序，返回 top_k 个结果。"""


class OpenAIReranker(Reranker):
    """基于 OpenAI 兼容 API 的重排序实现（/rerank 端点）。

    /rerank 非标准 OpenAI 端点，使用 httpx 直接调用，通过 openai 库提供 proxy 支持。
    支持滑动窗口速率限制（RPM + TPM），避免触发供应商 429。
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        model: str = "rerank-v1",
        proxy: str | None = None,
        max_rpm: int = 0,
        max_tpm: int = 0,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self._rate_limiter = RateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:
        logger.debug(
            "Reranking {} candidates with model={}, top_k={}", len(candidates), self.model, top_k
        )
        if not candidates:
            return []
        self._rate_limiter.wait_if_needed(_estimate_rerank_tokens(query, candidates))
        try:
            resp = self._http_client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [c.content for c in candidates],
                    "top_n": top_k,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            logger.warning("Rerank API 调用失败，返回原始排序", exc_info=True)
            return candidates[:top_k]
        results: list[Chunk] = []
        for item in data.get("results", []):
            idx = item["index"]
            candidates[idx].score = item.get("relevance_score", 0.0)
            results.append(candidates[idx])
        logger.debug("Reranking returned {} results", len(results))
        return results


class LocalReranker(Reranker):
    """基于本地 cross-encoder 模型的重排序实现。"""

    def __init__(self, model_name: str = DEFAULT_LOCAL_RERANK_MODEL):
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # noqa: I001  # pyright: ignore[reportMissingImports]

            self._model = CrossEncoder(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:
        logger.debug(
            "Local reranking {} candidates with model={}, top_k={}",
            len(candidates),
            self.model_name,
            top_k,
        )
        if not candidates:
            return []
        self._load()
        if self._model is None:
            raise RuntimeError("本地重排序模型未加载")
        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)
        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)
        candidates.sort(key=lambda c: c.score, reverse=True)
        result = candidates[:top_k]
        logger.debug("Local reranking returned {} results", len(result))
        return result


def create_reranker(
    provider: str = "",
    api_key: str = "",
    api_base: str = "",
    model: str = "",
    proxy: str | None = None,
    max_rpm: int = 0,
    max_tpm: int = 0,
) -> Reranker | None:
    """工厂函数，创建重排序器实例。provider 为空时返回 None。"""
    if not provider:
        return None
    if provider == "openai":
        return OpenAIReranker(
            api_key=api_key,
            api_base=api_base or DEFAULT_OPENAI_API_BASE,
            model=model or DEFAULT_RERANK_MODEL,
            proxy=proxy,
            max_rpm=max_rpm,
            max_tpm=max_tpm,
        )
    if provider == "local":
        return LocalReranker(model_name=model or "BAAI/bge-reranker-v2-m3")
    raise ValueError(f"Unsupported reranker provider: {provider}")
