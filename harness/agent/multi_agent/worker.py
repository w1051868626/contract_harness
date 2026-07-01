"""领域专业 Worker Agent，独立 LLM + 专用 System Prompt。"""

from __future__ import annotations

from typing import Any

from harness.agent.llm import LLMClient
from harness.core.types import WorkerOutput

WORKER_PROMPTS: dict[str, str] = {
    "ClauseExpert": (
        "你是一位专业的合同条款提取专家。你的职责是从合同中准确提取所有关键条款，"
        "包括条款类型、具体内容。请输出结构化 JSON。"
    ),
    "RiskExpert": (
        "你是一位专业的法律风险评估专家。你的职责是对已提取的合同条款进行风险分析，"
        "评估风险等级（critical/high/medium/low/info）并提供理由和建议。请输出结构化 JSON。"
    ),
    "ComplianceExpert": (
        "你是一位专业的合规审查专家。你的职责是检查合同条款是否符合中国法律法规，"
        "包括民法典、劳动合同法、数据安全法、个人信息保护法、反垄断法等。请输出结构化 JSON。"
    ),
}


class WorkerAgent:
    """领域专业 Worker Agent，可独立调用 LLM 执行任务。"""

    def __init__(self, role: str, llm: LLMClient):
        if role not in WORKER_PROMPTS:
            raise ValueError(f"未知 Worker 角色: {role}，可用: {list(WORKER_PROMPTS.keys())}")
        self.role = role
        self._llm = llm

    def execute(
        self,
        content: str,
        peer_results: dict[str, Any] | None = None,
    ) -> WorkerOutput:
        """执行任务，可选传入同行结果供交叉验证。"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": WORKER_PROMPTS[self.role]},
        ]
        if peer_results:
            peer_text = "\n".join(f"## {k} 的分析\n{v}" for k, v in peer_results.items())
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"请审阅以下其他专家的分析意见，检查是否存在分歧或遗漏：\n\n{peer_text}"
                    ),
                }
            )
        messages.append({"role": "user", "content": content})

        resp = self._llm.chat(messages)
        return WorkerOutput(
            worker_role=self.role,
            content=resp.content,
        )
