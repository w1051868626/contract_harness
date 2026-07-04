from __future__ import annotations

from typing import Any

from json_repair import repair_json
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from harness.agent.llm import LLMClient
from harness.eval_rag.dataset import EvalRagItem
from harness.utils.log import logger


class Questions(BaseModel):
    """由 LLM 生成的用户问题列表。"""

    questions: list[str] = Field(description="用户可能会问的自然语言问题列表")


_parser = PydanticOutputParser(pydantic_object=Questions)

GENERATOR_PROMPT = """你是一位法律知识库的测试数据生成专家。
请根据以下法律文本，生成 {count} 个用户可能会问的自然语言问题。
每个问题应该：
1. 是真实用户会使用的问法（口语化、自然）
2. 答案可以从该文本中找到
3. 各不相同，覆盖不同角度

文本：
{text}

{format_instructions}"""


def _parse_llm_output(content: str, queries_per_chunk: int) -> list[str]:
    """尝试用 PydanticOutputParser 解析，失败则用 repair_json 修复后重试，最终降级为逐行文本。"""
    for text in (content, repair_json(content)):
        if not text:
            continue
        try:
            parsed = _parser.parse(text)
            return parsed.questions[:queries_per_chunk]
        except OutputParserException:
            continue
    logger.warning("LLM 输出解析失败，降级为逐行文本: {}...", content[:80])
    return [q.strip() for q in content.strip().split("\n") if q.strip()][:queries_per_chunk]


class RagDatasetGenerator:
    def generate(
        self,
        kb: Any,
        llm: LLMClient,
        queries_per_chunk: int = 2,
    ) -> list[EvalRagItem]:
        items: list[EvalRagItem] = []
        chunks = kb.list_chunks()
        logger.info("Generating eval dataset from {} chunks", len(chunks))

        for chunk in chunks:
            if not chunk.content.strip():
                continue
            prompt = GENERATOR_PROMPT.format(
                count=queries_per_chunk,
                text=chunk.content[:1000],
                format_instructions=_parser.get_format_instructions(),
            )
            resp = llm.chat(
                [
                    {"role": "system", "content": "你是一个测试数据生成助手。"},
                    {"role": "user", "content": prompt},
                ]
            )
            queries = _parse_llm_output(resp.content, queries_per_chunk)
            for q in queries:
                items.append(
                    EvalRagItem(
                        query=q,
                        expected_chunk_ids=[chunk.id],
                        expected_texts=[chunk.content[:200]],
                        metadata={"source_chunk": chunk.id},
                    )
                )

        logger.info("Generated {} eval items", len(items))
        return items
