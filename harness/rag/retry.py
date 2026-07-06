"""HTTP 调用重试工具，供 Embedding / Reranker 等网络模块复用。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from harness.utils.log import logger

_T = TypeVar("_T")

# 网络层错误重试基线退避秒数，实际等待 = base * 2^attempt
_RETRY_BASE_DELAY: float = 1.0
_DEFAULT_MAX_RETRIES: int = 3


def retry_with_backoff(
    fn: Callable[[], _T],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _RETRY_BASE_DELAY,
    retry_on: tuple[type[BaseException], ...],
    raises: type[BaseException],
    raises_msg: str,
) -> _T:
    """对瞬时错误指数退避重试，非瞬时错误立即抛 ``raises``。

    Args:
        fn: 无参 callable，返回目标值或抛异常。
        max_retries: 最大尝试次数（含首次），<1 时视为 1 不重试。
        base_delay: 首次重试退避秒数，每轮翻倍。
        retry_on: 可重试的瞬时异常类型元组（如网络错误、限流）。
        raises: 重试耗尽或遇到非瞬时错误时抛的异常类型。
        raises_msg: 抛 ``raises`` 时携带的提示前缀，实际消息附原始错误。
    """
    last_error: BaseException | None = None
    for attempt in range(max(max_retries, 1)):
        try:
            return fn()
        except retry_on as e:
            last_error = e
            if max_retries <= 1:
                logger.error("{}: {}", raises_msg, e)
                raise raises(f"{raises_msg}: {e}") from e
            delay = base_delay * (2**attempt)
            logger.warning(
                "{}（{}），{:.0f} 秒后第 {} 次重试...",
                raises_msg,
                e,
                delay,
                attempt + 2,
            )
            time.sleep(delay)
    logger.error("{} 重试 {} 次后仍失败: {}", raises_msg, max_retries, last_error)
    raise raises(f"{raises_msg} 重试 {max_retries} 次后仍失败: {last_error}") from last_error
