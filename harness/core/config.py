"""LLM、嵌入模型与 Harness 全局配置的数据类定义。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM 客户端配置（模型、密钥、代理等）。"""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    proxy: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self):
        """从环境变量自动补充缺失的 API 密钥与地址。"""
        self.provider = os.getenv("LLM_PROVIDER", self.provider)
        if not self.api_key:
            provider_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(provider_key, os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            provider_base = f"{self.provider.upper()}_API_BASE"
            self.api_base = os.getenv(provider_base, "https://api.openai.com/v1")
        if self.proxy is None:
            self.proxy = os.getenv("LLM_PROXY", os.getenv("HTTP_PROXY", "")) or None


@dataclass
class EmbeddingConfig:
    """嵌入模型配置（模型、密钥、代理等）。"""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    proxy: str | None = None

    def __post_init__(self):
        """从环境变量自动补充嵌入 API 密钥与地址。"""
        self.provider = os.getenv("EMBEDDING_PROVIDER", self.provider)
        if not self.api_key:
            self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            self.api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", ""))
        if self.proxy is None:
            self.proxy = os.getenv("EMBEDDING_PROXY", os.getenv("HTTP_PROXY", "")) or None


@dataclass
class HarnessConfig:
    """Harness 全局配置（目录路径、开关等）。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data_dir: str = str(Path.home() / ".harness" / "data")
    kb_dir: str = str(Path.home() / ".harness" / "knowledge")
    replay_dir: str = str(Path.home() / ".harness" / "replays")
    eval_dir: str = str(Path.home() / ".harness" / "evals")
    regression_dir: str = str(Path.home() / ".harness" / "regression")
    report_dir: str = str(Path.home() / ".harness" / "reports")
    verbose: bool = False

    def ensure_dirs(self):
        """确保所有配置中的数据目录存在。"""
        for d in [
            self.data_dir,
            self.kb_dir,
            self.replay_dir,
            self.eval_dir,
            self.regression_dir,
            self.report_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
