from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    proxy: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self):
        if not self.api_key:
            provider_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(provider_key, os.getenv("OPENAI_API_KEY", ""))
        if not self.api_base:
            provider_base = f"{self.provider.upper()}_API_BASE"
            self.api_base = os.getenv(provider_base, "https://api.openai.com/v1")


@dataclass
class HarnessConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    data_dir: str = str(Path.home() / ".harness" / "data")
    replay_dir: str = str(Path.home() / ".harness" / "replays")
    eval_dir: str = str(Path.home() / ".harness" / "evals")
    regression_dir: str = str(Path.home() / ".harness" / "regression")
    report_dir: str = str(Path.home() / ".harness" / "reports")
    verbose: bool = False

    def ensure_dirs(self):
        for d in [
            self.data_dir,
            self.replay_dir,
            self.eval_dir,
            self.regression_dir,
            self.report_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
