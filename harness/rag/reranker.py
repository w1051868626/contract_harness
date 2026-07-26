"""重排序模块，对粗排结果进行精排，提升检索准确率。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]

import httpx

from harness.rag.constants import (
    DEFAULT_LOCAL_RERANK_MODEL,
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_RERANK_MODEL,
)
from harness.rag.rate_limit import RateLimiter
from harness.rag.retry import retry_with_backoff
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
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self._http_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self._rate_limiter = RateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)
        self._max_retries = max_retries

    def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:
        logger.debug(
            "Reranking {} candidates with model={}, top_k={}", len(candidates), self.model, top_k
        )
        if not candidates:
            return []
        self._rate_limiter.wait_if_needed(_estimate_rerank_tokens(query, candidates))

        def _call() -> dict[str, Any] | list[Chunk]:
            resp = self._http_client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [c.content for c in candidates],
                    # 让 API 对全部 pool 候选打分（top_n=pool 大小），客户端再按
                    # relevance_score 截 top_k，给 top-1 判别更大参考面
                    "top_n": len(candidates),
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            # 4xx 错误直接降级不重试（鉴权/请求格式），5xx 抛 HTTPStatusError 走重试
            if 400 <= resp.status_code < 500:
                logger.warning("Rerank API {} 错误，返回原始排序", resp.status_code)
                return candidates[:top_k]
            resp.raise_for_status()
            return resp.json()

        # 可重试：网络错误 + 5xx 服务端错误
        try:
            data = retry_with_backoff(
                _call,
                max_retries=self._max_retries,
                retry_on=(httpx.RequestError, OSError),
                raises=RuntimeError,
                raises_msg="Rerank API 调用失败",
            )
        except RuntimeError:
            logger.warning("Rerank API 重试耗尽，返回原始排序", exc_info=True)
            return candidates[:top_k]

        # _call 内 4xx 已降级返回 list[Chunk]，非 dict 路径直接透传
        if isinstance(data, list):
            return data

        # retry_with_backoff 的 _T 推断会受其他调用点污染，显式 cast 锁定类型
        # API 返回全 pool 打分，按 relevance_score 降序后截 top_k
        scored: list[tuple[int, float]] = []
        for item in cast("dict[str, Any]", data).get("results", []):
            idx = item["index"]
            score = item.get("relevance_score", 0.0)
            candidates[idx].score = score
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [candidates[idx] for idx, _ in scored[:top_k]]
        logger.debug("Reranking returned {} results", len(results))
        return results


class LocalReranker(Reranker):
    """基于本地 cross-encoder 模型的重排序实现。"""

    def __init__(self, model_name: str = DEFAULT_LOCAL_RERANK_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

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
    max_retries: int = 3,
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
            max_retries=max_retries,
        )
    if provider == "local":
        return LocalReranker(model_name=model or "BAAI/bge-reranker-v2-m3")
    raise ValueError(f"Unsupported reranker provider: {provider}")
