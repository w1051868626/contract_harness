"""重排序模块，对粗排结果进行精排，提升检索准确率。

## 优化策略（2026-07-26）

在「API 域唯一可用 rerank 曾型 = BAAI/bge-reranker-v2-m3，无法更换」约束下，
沿调用侧做三步优化（详见 `specs/reranker.md` 与 `docs/optimization_notes.md`）：

1. **候选池扩大** —— `_search_single` dense/sparse 候选池 `top_k*2` → `max(20, top_k*4)`，
   给 Reranker 更大判别空间。
2. **`top_n` 扩大** —— `/rerank` API `top_n` 从 `top_k` 改为 `len(candidates)`，让 API 对
   全 pool �候选打分，客户端按 `relevance_score` 截 top_k。
3. **标题前缀注入**（核心突破）—— `OpenAIReranker._format_for_rerank(chunk)` 给喂给 API
   的 documents 注入 `【{law_name}·{articles}】` 前缀，给 cross-encoder 额外结构信号；
   **不动 `chunk.content`**（其他调用方不受影响），200 样本 A/B top-1 hit_rate +9.5pp。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]

import copy

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

    def rerank_multi(
        self,
        queries: list[str],
        candidates: list[Chunk],
        weights: list[float] | None = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """多 query 加权重排：对每个 query 各跑一次 rerank，按权重融合 relevance_score。

        Args:
            queries: 多个查询视角（如原始 query + 同义改写）。
            candidates: 候选列表（同一份，每次 rerank 都基于它打分）。
            weights: 每个 query 的融合权重，长度需等于 ``queries``；缺省均权。
            top_k: 返回结果数量。

        默认实现循环调用 ``rerank`` 收集每个候选的加权分数；子类可覆盖以
        批量调用 API（少 N-1 次 round-trip）。
        """
        if not candidates or not queries:
            return candidates[:top_k]
        if weights is None:
            weights = [1.0 / len(queries)] * len(queries)
        if len(weights) != len(queries):
            msg = f"weights 长度 {len(weights)} 与 queries 长度 {len(queries)} 不一致"
            raise ValueError(msg)
        total_w = sum(weights)
        if total_w <= 0:
            msg = "weights 之和必须为正"
            raise ValueError(msg)
        weights = [w / total_w for w in weights]

        # 对每个 query 各跑一次 rerank，收集每个候选的加权 relevance_score
        # candidates 是同一份引用，rerank 会改写 score；用 snapshot 隔离避免污染
        accumulated: dict[str, float] = {c.id: 0.0 for c in candidates}
        for q, w in zip(queries, weights, strict=True):
            # 浅拷贝每个 Chunk（dataclass），保留 id；rerank 内会改 score 但不影响原 candidates
            snapshot = [copy.copy(c) for c in candidates]
            ranked = self.rerank(q, snapshot, top_k=len(snapshot))
            # rerank 可能因 4xx 降级返回 candidates[:top_k]，丢失的候选 score 记 0
            seen_in_round: set[str] = set()
            for c in ranked:
                accumulated[c.id] += w * c.score
                seen_in_round.add(c.id)

        # 把加权 score 写回原 candidates 引用，按加权 score 降序截 top_k
        for c in candidates:
            c.score = accumulated.get(c.id, 0.0)
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


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

        # 给 reranker 额外结构信号：注入法条标题前缀（【法律名·第X条】），仅改喂给
        # API 的 documents 文本，不动 chunk.content（其他调用方不受影响）。无 metadata
        # 时降级为裸 content。
        documents = [self._format_for_rerank(c) for c in candidates]

        def _call() -> dict[str, Any] | list[Chunk]:
            resp = self._http_client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
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

    @staticmethod
    def _format_for_rerank(chunk: Chunk) -> str:
        """给 reranker 喂的候选文本：注入法条标题前缀（【法律名·第X条】正文）。

        仅改喂给 API 的文本，不动 chunk.content，其他调用方不受影响。
        无 metadata 或缺 articles/law_name 时降级为裸 content。
        """
        meta = chunk.metadata or {}
        law = meta.get("law_name")
        articles = meta.get("articles")
        if law and articles:
            return f"【{law}·{articles}】{chunk.content}"
        if articles:
            return f"【{articles}】{chunk.content}"
        return chunk.content


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
