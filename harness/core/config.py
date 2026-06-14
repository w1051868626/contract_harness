"""LLM、嵌入模型与 Harness 全局配置的数据类定义。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    """返回项目根目录下的 .harness 目录。"""
    start = Path(__file__).resolve().parent.parent.parent  # up from core/ to project root
    return start / ".harness"


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
    chunk_model: str = "gpt-4o-mini"
    chunk_api_key: str = ""
    chunk_api_base: str = ""

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
        self.chunk_model = os.getenv("CHUNK_MODEL", self.chunk_model)
        if not self.chunk_api_key:
            self.chunk_api_key = os.getenv("CHUNK_API_KEY", self.api_key)
        if not self.chunk_api_base:
            self.chunk_api_base = os.getenv("CHUNK_API_BASE", self.api_base)


@dataclass
class EmbeddingConfig:
    """嵌入模型配置（模型、密钥、代理等）。"""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    proxy: str | None = None

    rerank_provider: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_api_base: str = ""

    def __post_init__(self):
        """从环境变量自动补充嵌入 API 密钥与地址。"""
        self.provider = os.getenv("EMBEDDING_PROVIDER", self.provider)
        if not self.api_key:
            self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            self.api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", ""))
        if self.proxy is None:
            self.proxy = os.getenv("EMBEDDING_PROXY", os.getenv("HTTP_PROXY", "")) or None

        self.rerank_provider = os.getenv("RERANK_PROVIDER", self.rerank_provider)
        if not self.rerank_api_key:
            self.rerank_api_key = os.getenv("RERANK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.rerank_api_base:
            self.rerank_api_base = os.getenv(
                "RERANK_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            )
        if not self.rerank_model:
            self.rerank_model = os.getenv("RERANK_MODEL", "rerank-v1")


@dataclass
class HarnessConfig:
    """Harness 全局配置（目录路径、开关等）。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data_dir: str = ""
    kb_dir: str = ""
    replay_dir: str = ""
    eval_dir: str = ""
    regression_dir: str = ""
    report_dir: str = ""
    verbose: bool = False

    def __post_init__(self):
        """通过环境变量 HARNESS_DATA_DIR 自定义数据根目录，否则使用项目根目录。"""
        root = Path(os.getenv("HARNESS_DATA_DIR", str(_default_data_root())))
        if not self.data_dir:
            self.data_dir = str(root / "data")
        if not self.kb_dir:
            self.kb_dir = str(root / "knowledge")
        if not self.replay_dir:
            self.replay_dir = str(root / "replays")
        if not self.eval_dir:
            self.eval_dir = str(root / "evals")
        if not self.regression_dir:
            self.regression_dir = str(root / "regression")
        if not self.report_dir:
            self.report_dir = str(root / "reports")

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
