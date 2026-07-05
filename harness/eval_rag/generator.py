from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from json_repair import repair_json
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from openai import APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

from harness.agent.llm import LLMClient
from harness.core.exceptions import AgentError, EvalError
from harness.eval_rag.dataset import EvalRagItem
from harness.rag.knowledge_base import KnowledgeBase
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

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


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


def _load_existing_chunk_ids(path: str) -> set[str]:
    """读取已有 JSONL 输出中已处理的 chunk ID 集合（断点恢复用）。"""
    if not Path(path).exists():
        return set()
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                source = data.get("metadata", {}).get("source_chunk")
                if source:
                    ids.add(source)
            except json.JSONDecodeError:
                continue
    return ids


def _append_jsonl(path: str, items: list[EvalRagItem]) -> None:
    """追加 items 到 JSONL 文件（增量写入）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "query": item.query,
                        "expected_chunk_ids": item.expected_chunk_ids,
                        "expected_texts": item.expected_texts,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class RagDatasetGenerator:
    def generate(
        self,
        kb: KnowledgeBase,
        llm: LLMClient,
        queries_per_chunk: int = 2,
        output_path: str | None = None,
    ) -> list[EvalRagItem]:
        """从知识库 chunk 生成评估数据集。

        Args:
            kb: 知识库实例（需有 list_chunks 方法）。
            llm: LLM 客户端。
            queries_per_chunk: 每个 chunk 生成的问题数。
            output_path: JSONL 输出路径。
                提供时启用断点恢复：检测已有条目中已处理的 chunk_id 并跳过；
                同时每处理完一个 chunk 都增量写入，避免中断时数据丢失。
        """
        chunks = kb.list_chunks()
        if not chunks:
            raise EvalError(
                "Missing knowledge base chunks: 知识库为空，请先运行 "
                "`harness kb seed` 或 `harness kb import-file` 导入文档"
            )
        logger.info("Generating eval dataset from {} chunks", len(chunks))

        # 断点恢复：加载已有输出中的 chunk ID，已处理则跳过
        processed: set[str] = set()
        if output_path:
            processed = _load_existing_chunk_ids(output_path)
            if processed:
                logger.info("断点恢复：检测到 {} 个已处理 chunk", len(processed))

        items: list[EvalRagItem] = []
        for chunk in chunks:
            if chunk.id in processed:
                continue
            if not chunk.content.strip():
                continue

            # 组装 prompt，截断 chunk 内容避免超出 LLM 上下文
            prompt = GENERATOR_PROMPT.format(
                count=queries_per_chunk,
                text=chunk.content[:1000],
                format_instructions=_parser.get_format_instructions(),
            )

            queries = self._call_llm_with_retry(llm, prompt, queries_per_chunk)
            if queries is None:
                logger.error("跳过 chunk {}（LLM 调用全部失败）", chunk.id)
                continue

            # 每条 query 对应一个 EvalRagItem，expected_chunk_ids 指向来源 chunk
            chunk_items: list[EvalRagItem] = []
            for q in queries:
                item = EvalRagItem(
                    query=q,
                    expected_chunk_ids=[chunk.id],
                    expected_texts=[chunk.content[:200]],
                    metadata={"source_chunk": chunk.id},
                )
                items.append(item)
                chunk_items.append(item)

            # 增量写入 JSONL，避免中断时数据丢失
            if output_path and chunk_items:
                _append_jsonl(output_path, chunk_items)

        total = len(processed) + len(items)
        logger.info(
            "生成完成：{} 条（新增 {}，复用 {}）来自 {} 个 chunk",
            total,
            len(items),
            len(processed),
            len(chunks),
        )
        return items

    @staticmethod
    def _call_llm_with_retry(
        llm: LLMClient,
        prompt: str,
        queries_per_chunk: int,
    ) -> list[str] | None:
        """调用 LLM 并自动重试，失败时返回 None。"""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = llm.chat(
                    [
                        {"role": "system", "content": "你是一个测试数据生成助手。"},
                        {"role": "user", "content": prompt},
                    ]
                )
                return _parse_llm_output(resp.content, queries_per_chunk)
            except AgentError as e:
                # AgentError 为非瞬时错误（密钥缺失/鉴权失败/重试耗尽等），
                # 不重试直接降级返回 None，避免向上抛打断整个生成流程。
                logger.error("LLM 调用失败（不可重试）: {}", e)
                return None
            except (APIError, APITimeoutError, RateLimitError, httpx.HTTPError) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAY * (2**attempt)
                    logger.warning(
                        "LLM 调用失败（{}），{:.0f} 秒后第 {} 次重试...",
                        e,
                        delay,
                        attempt + 2,
                    )
                    time.sleep(delay)
        logger.error("LLM 重试 {} 次后仍失败: {}", _MAX_RETRIES, last_error)
        return None
