"""Loguru 日志配置工具。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(verbose: bool = False, log_dir: str | Path | None = None) -> None:
    """配置 Loguru 日志。

    Args:
        verbose: 是否启用 DEBUG 级别（默认 INFO）。
        log_dir: 日志文件输出目录，为 None 则仅输出到 stderr。
    """
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:^7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path / "harness_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="1 day",
            retention="30 days",
            encoding="utf-8",
        )

    logger.debug("日志系统初始化完成 (level=%s, log_dir=%s)", level, log_dir)


__all__ = ["logger", "setup_logging"]
