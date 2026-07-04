"""通用文本分块模块，支持 AI 智能分块与滑动窗口传统分块。"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.agent.llm import LLMClient, LLMResponse
from harness.rag.constants import CHUNK_MAX_CHARS
from harness.rag.vector_store import Chunk
from harness.utils.io import make_id
from harness.utils.log import logger

# ===== 分块提示词 =====

CHUNK_SYSTEM_PROMPT = "你是文档分块专家，严格按 JSON 格式输出。"

CHUNK_PROMPT = """你是一个文档分块专家。请将以下文档按逻辑结构拆分成有意义的片段。
每个片段应该是一个完整的主题、章节或逻辑段落，不要切割句子。
直接输出 JSON 数组，每个元素的格式为 {{"content": "..."}}。

文档：
{text}"""


def chunk_with_ai(text: str, doc_id: str, chunk_llm: LLMClient, chunk_model: str) -> list[Chunk]:
    """使用 LLM 对文本进行智能分块，超长文本自动分段后合并。"""
    if chunk_llm is None:
        raise RuntimeError("chunk_llm 未初始化")
    all_chunks: list[Chunk] = []
    for i in range(0, len(text), CHUNK_MAX_CHARS):
        segment = text[i : i + CHUNK_MAX_CHARS]
        prompt = CHUNK_PROMPT.format(text=segment)
        resp: LLMResponse = chunk_llm.chat(
            [
                {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=chunk_model,
            temperature=0.0,
        )
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("AI chunking did not return a list")
        for item in data:
            content = item.get("content", "").strip()
            if content:
                all_chunks.append(
                    Chunk(
                        id=make_id(),
                        document_id=doc_id,
                        content=content,
                        chunk_index=len(all_chunks),
                    )
                )
    logger.debug("AI 分块完成: chunks={}", len(all_chunks))
    return all_chunks


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """基于段落滑动窗口的传统分块算法。"""
    if not text.strip():
        return []

    segments = _split_segments(text)

    if len(segments) == 1 and len(text) <= chunk_size:
        return [
            Chunk(
                id=make_id(),
                document_id=doc_id,
                content=_align_chunk_end(text),
                chunk_index=0,
            )
        ]

    chunks: list[Chunk] = []
    idx = 0
    buffer: list[str] = []
    buffer_len = 0

    for seg in segments:
        seg_len = len(seg)
        if buffer_len + seg_len <= chunk_size:
            buffer.append(seg)
            buffer_len += seg_len
        else:
            if buffer:
                chunks.append(_make_chunk(buffer, doc_id, idx))
                idx += 1
                carry = _carry_overlap(buffer, overlap)
                buffer = carry
                buffer_len = sum(len(s) for s in carry)

            if seg_len > chunk_size:
                for sub in _split_long(seg, chunk_size):
                    chunks.append(
                        Chunk(
                            id=make_id(),
                            document_id=doc_id,
                            content=_align_chunk_end(sub),
                            chunk_index=idx,
                        )
                    )
                    idx += 1
                buffer = []
                buffer_len = 0
            else:
                buffer = [seg]
                buffer_len = seg_len

    if buffer:
        chunks.append(_make_chunk(buffer, doc_id, idx))

    return chunks


def _split_segments(text: str) -> list[str]:
    """按空行将文本分割为段落。"""
    raw = re.split(r"\n\s*\n", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _align_chunk_end(content: str) -> str:
    """将内容结尾对齐到句末标点。"""
    matches = list(re.finditer(r"[。！？；.!?;]", content))
    if matches:
        return content[: matches[-1].end()].rstrip()
    return content


def _split_long(text: str, chunk_size: int) -> list[str]:
    """将长文本按句子切分。"""
    sentences = re.split(r"(?<=[。！？；.!?;])\s*", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if buf:
                chunks.append(buf.strip())
            buf = s
    if buf:
        chunks.append(buf.strip())

    if len(chunks) <= 1 and len(text) > chunk_size:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks if chunks else [text]


def _carry_overlap(buffer: list[str], overlap_chars: int) -> list[str]:
    """计算窗口重叠部分。"""
    carry: list[str] = []
    carry_len = 0
    for seg in reversed(buffer):
        if carry_len + len(seg) <= overlap_chars:
            carry.insert(0, seg)
            carry_len += len(seg)
        else:
            break
    return carry


def _make_chunk(
    segments: list[str],
    doc_id: str,
    idx: int,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """创建 Chunk 对象，自动对齐句子边界。"""
    content = _align_chunk_end("\n\n".join(segments))
    return Chunk(
        id=make_id(),
        document_id=doc_id,
        content=content,
        chunk_index=idx,
        metadata=metadata or {},
    )
