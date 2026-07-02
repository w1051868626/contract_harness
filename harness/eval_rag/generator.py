from __future__ import annotations

from typing import Any

from harness.agent.llm import LLMClient
from harness.eval_rag.dataset import EvalRagItem
from harness.utils.log import logger

GENERATOR_PROMPT = """你是一位法律知识库的测试数据生成专家。
请根据以下法律文本，生成 {count} 个用户可能会问的自然语言问题。
每个问题应该：
1. 是真实用户会使用的问法（口语化、自然）
2. 答案可以从该文本中找到
3. 各不相同，覆盖不同角度

文本：
{text}

请直接输出问题，每行一个，不要编号。"""


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
            prompt = GENERATOR_PROMPT.format(count=queries_per_chunk, text=chunk.content[:1000])
            resp = llm.chat(
                [
                    {"role": "system", "content": "你是一个测试数据生成助手。"},
                    {"role": "user", "content": prompt},
                ]
            )
            queries = [
                q.strip().removeprefix(f"{i + 1}.").strip()
                for i, q in enumerate(resp.content.strip().split("\n"))
                if q.strip()
            ]
            for q in queries[:queries_per_chunk]:
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
