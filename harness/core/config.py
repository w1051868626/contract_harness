"""LLM、嵌入模型与 Harness 全局配置的数据类定义。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.types import AgentMode


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
    api_base: str = ""
    proxy: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    chunk_model: str = ""
    chunk_api_key: str = ""
    chunk_api_base: str = ""
    expansion_model: str = ""
    expansion_api_key: str = ""
    expansion_api_base: str = ""

    def __post_init__(self):
        """从环境变量自动补充缺失的 API 密钥与地址，并校验参数范围。"""
        self._load_defaults_from_env()
        self._validate()

    def _load_defaults_from_env(self) -> None:
        """环境变量加载逻辑。"""
        self.provider = os.getenv("LLM_PROVIDER", self.provider)
        if not self.api_key:
            provider_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(provider_key, os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            provider_base = f"{self.provider.upper()}_API_BASE"
            self.api_base = os.getenv(provider_base, "https://api.openai.com/v1")
        if self.proxy is None:
            self.proxy = os.getenv("LLM_PROXY", os.getenv("HTTP_PROXY", "")) or None
        self.model = os.getenv("LLM_MODEL", self.model)
        self.chunk_model = os.getenv("CHUNK_MODEL", self.chunk_model)
        if not self.chunk_api_key:
            self.chunk_api_key = os.getenv("CHUNK_API_KEY", self.api_key)
        if not self.chunk_api_base:
            self.chunk_api_base = os.getenv("CHUNK_API_BASE", self.api_base)
        self.expansion_model = os.getenv("EXPANSION_MODEL", self.expansion_model)
        if not self.expansion_api_key:
            self.expansion_api_key = os.getenv("EXPANSION_API_KEY", self.api_key)
        if not self.expansion_api_base:
            self.expansion_api_base = os.getenv("EXPANSION_API_BASE", self.api_base)

    def _validate(self) -> None:
        """校验配置参数范围。"""
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"temperature 必须在 [0.0, 2.0] 范围内，当前值: {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正整数，当前值: {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout 必须为正整数，当前值: {self.timeout}")


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

    enable_hybrid_search: bool = False
    rrf_k: int = 60
    max_rpm: int = 0
    max_tpm: int = 0

    def __post_init__(self):
        """从环境变量自动补充嵌入 API 密钥与地址。"""
        self._load_defaults_from_env()

    def _load_defaults_from_env(self) -> None:
        """环境变量加载逻辑。"""
        self.provider = os.getenv("EMBEDDING_PROVIDER", self.provider)
        if not self.api_key:
            self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            self.api_base = os.getenv("EMBEDDING_API_BASE", os.getenv("OPENAI_API_BASE", ""))
        if self.proxy is None:
            self.proxy = os.getenv("EMBEDDING_PROXY", os.getenv("HTTP_PROXY", "")) or None
        self.model = os.getenv("EMBEDDING_MODEL", self.model)
        self.max_rpm = int(os.getenv("EMBEDDING_MAX_RPM", str(self.max_rpm)))
        self.max_tpm = int(os.getenv("EMBEDDING_MAX_TPM", str(self.max_tpm)))

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
    log_dir: str = ""
    kb_dir: str = ""
    replay_dir: str = ""
    eval_dir: str = ""
    regression_dir: str = ""
    report_dir: str = ""
    memory_dir: str = ""
    memory_enabled: bool = True
    memory_top_k: int = 3
    agent_mode: AgentMode = AgentMode.PIPELINE
    use_docling: bool = False
    verbose: bool = False

    def __post_init__(self):
        """通过环境变量 HARNESS_DATA_DIR 自定义数据根目录，否则使用项目根目录。"""
        self._load_defaults_from_env()

    def _load_defaults_from_env(self) -> None:
        """环境变量加载逻辑。"""
        root = Path(os.getenv("HARNESS_DATA_DIR", str(_default_data_root())))
        if not self.data_dir:
            self.data_dir = str(root / "data")
        if not self.log_dir:
            self.log_dir = str(root / "logs")
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
        if not self.memory_dir:
            self.memory_dir = str(root / "memory")

    def ensure_dirs(self) -> None:
        """确保所有配置中的数据目录存在。"""
        for d in [
            self.data_dir,
            self.log_dir,
            self.kb_dir,
            self.replay_dir,
            self.eval_dir,
            self.regression_dir,
            self.report_dir,
            self.memory_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
