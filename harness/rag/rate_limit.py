"""滑动窗口速率限制器，供 Embedding / Reranker 等网络模块复用。"""

from __future__ import annotations

import time
from collections import deque

from harness.utils.log import logger

_WINDOW_SEC = 60


class RateLimiter:
    """滑动窗口速率限制器（RPM + TPM 双约束）。"""

    def __init__(self, max_rpm: int = 0, max_tpm: int = 0):
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """阻塞直到速率限制允许本次请求。"""
        now = time.monotonic()
        cutoff = now - _WINDOW_SEC

        # 清理过期记录
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()

        current_rpm = len(self._requests)
        current_tpm = sum(t for _, t in self._tokens)

        need_wait = False
        if self._max_rpm > 0 and current_rpm >= self._max_rpm:
            need_wait = True
        if self._max_tpm > 0 and current_tpm + estimated_tokens > self._max_tpm:
            need_wait = True

        if need_wait:
            oldest = min(
                self._requests[0] if self._requests else now,
                self._tokens[0][0] if self._tokens else now,
            )
            sleep_for = _WINDOW_SEC - (now - oldest) + 0.1
            if sleep_for > 0:
                logger.info(
                    "速率限制等待 {:.1f}s (RPM={}/{}, TPM={}/{})",
                    sleep_for,
                    current_rpm,
                    self._max_rpm,
                    current_tpm,
                    self._max_tpm,
                )
                time.sleep(sleep_for)

        self._requests.append(now)
        self._tokens.append((now, estimated_tokens))
