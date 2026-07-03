"""分歧仲裁器：规则优先，LLM 兜底。"""

from __future__ import annotations

import json
from typing import Any

from harness.agent.llm import LLMClient
from harness.core.types import Disagreement
from harness.utils.log import logger

RISK_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


class CrossValidator:
    """分歧仲裁：按分歧类型分层处理。"""

    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm

    def arbitrate(
        self,
        disagreements: list[Disagreement],
    ) -> list[dict[str, Any]]:
        """对分歧列表逐条仲裁，返回仲裁结果。"""
        results: list[dict[str, Any]] = []
        for d in disagreements:
            result = self._arbitrate_one(d)
            results.append(result)
        return results

    def _arbitrate_one(self, d: Disagreement) -> dict[str, Any]:
        """仲裁单条分歧。"""
        if d.field == "risk_level":
            return self._arbitrate_risk(d)
        if d.field == "compliance_status":
            return {
                "item_id": d.item_id,
                "field": d.field,
                "resolved": d.value_a,
                "needs_human_review": True,
                "explanation": "合规状态分歧，需人工审核确认",
            }
        # 其他分歧（条款类型等）：都保留
        return {
            "item_id": d.item_id,
            "field": d.field,
            "resolved": f"{d.value_a} / {d.value_b}",
            "needs_human_review": True,
            "explanation": f"分歧: {d.worker_a}={d.value_a}, {d.worker_b}={d.value_b}",
        }

    def _arbitrate_risk(self, d: Disagreement) -> dict[str, Any]:
        """仲裁风险等级分歧。"""
        rank_a = RISK_RANK.get(str(d.value_a), 0)
        rank_b = RISK_RANK.get(str(d.value_b), 0)
        diff = abs(rank_a - rank_b)

        if diff <= 1:
            # 差 1 级以内，取更严格侧
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": f"等级差 {diff} 级，取更严格侧: {resolved}",
            }

        # 差 ≥2 级，需要 LLM 仲裁
        return self._llm_arbitrate_risk(d)

    def _llm_arbitrate_risk(self, d: Disagreement) -> dict[str, Any]:
        """用 LLM 仲裁严重分歧的风险等级。"""
        if not self._llm:
            # 无 LLM 时取更严格侧
            rank_a = RISK_RANK.get(str(d.value_a), 0)
            rank_b = RISK_RANK.get(str(d.value_b), 0)
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": "LLM 不可用，取更严格侧",
            }

        prompt = (
            f"两位专家对同一条款的风险等级判断出现严重分歧：\n"
            f"- {d.worker_a} 认为: {d.value_a}\n"
            f"- {d.worker_b} 认为: {d.value_b}\n\n"
            f"请根据风险管理原则，裁定最终风险等级。"
            f'输出 JSON: {{"resolution": "等级", "explanation": "原因"}}'
        )
        try:
            resp = self._llm.chat(
                [
                    {"role": "system", "content": "你是一位风险管理仲裁专家。"},
                    {"role": "user", "content": prompt},
                ]
            )
            text = resp.content.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(text)
            resolved = data.get("resolution", d.value_a)
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": data.get("explanation", "LLM 仲裁"),
            }
        except (ValueError, RuntimeError, json.JSONDecodeError) as e:
            logger.warning("LLM 风险仲裁失败: {}", e)
            rank_a = RISK_RANK.get(str(d.value_a), 0)
            rank_b = RISK_RANK.get(str(d.value_b), 0)
            resolved = d.value_a if rank_a >= rank_b else d.value_b
            return {
                "item_id": d.item_id,
                "field": "risk_level",
                "resolved": resolved,
                "explanation": f"LLM 仲裁失败，取更严格侧: {resolved}",
            }
