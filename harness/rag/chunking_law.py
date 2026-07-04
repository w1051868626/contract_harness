"""法律/Markdown 文档分块模块，支持结构化层级分块。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from harness.rag.chunking_text import (
    _align_chunk_end,
    _carry_overlap,
    _make_chunk,
    _split_segments,
)
from harness.rag.constants import ART_PAT_RE, DocType, MetaKey
from harness.rag.vector_store import Chunk
from harness.utils.io import make_id

# ===== 预编译正则 =====

_MD_HEADING_RE = re.compile(r"^(?:#{1,6})(?:\s|$|(?=[^\s]))", re.MULTILINE)
_MD_SPLIT_RE = re.compile(r"(?=^(?:#{1,6})(?:\s|$|(?=[^\s])))", re.MULTILINE)

_CN_DIGIT = r"[一二三四五六七八九十百千零\d]"
_CN_CHAPTER = r"[一二三四五六七八九十百千]"
_CN_ARTICLE = r"[一二三四五六七八九十百千零〇\d]"

_CN_NUM: dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
}

_TITLE_RE = re.compile(r"^中华人民共和国.*")
_DATE_RE = re.compile(r"（(\d{4})年(\d+)月(\d+)日.*?通过")

_CHAPTER_LAW_RE = re.compile(r"^(第" + _CN_CHAPTER + r"+章.*)")
_SECTION_LAW_RE = re.compile(r"^(第" + _CN_CHAPTER + r"+节.*)")
_ARTICLE_LAW_RE = re.compile(r"^(第(" + _CN_ARTICLE + r"+)条)")

_CHAPTER_DIVISION_RE = re.compile(r"第" + _CN_DIGIT + r"+[章编]")
_SECTION_HEAD_RE = re.compile(r"第" + _CN_DIGIT + r"+节")
_NUM_LIST_RE = re.compile(_CN_DIGIT + r"+[、．]")
_PAREN_NUM_RE = re.compile(r"[（(]" + _CN_DIGIT + r"+[）)]")
_ART_SEARCH_RE = re.compile(r"第" + _CN_DIGIT + r"+条")

_DECIMAL_RE = re.compile(r"\d+\.(?!\d)")
_DECIMAL_RANGE_RE = re.compile(r"\d+\.\d+")

_LAW_NAME_SCAN_CHARS = 500
_DATE_SCAN_CHARS = 2000
_CASE_SCAN_CHARS = 1000


def _chinese_to_int(text: str) -> int:
    """中文数字转整数。"""
    if text.isdigit():
        return int(text)
    total = 0
    num = 0
    for c in reversed(text):
        if c == "十":
            if num == 0:
                num = 1
            total += num * 10
            num = 0
        elif c == "百":
            if num == 0:
                num = 1
            total += num * 100
            num = 0
        elif c == "千":
            if num == 0:
                num = 1
            total += num * 1000
            num = 0
        else:
            num = _CN_NUM[c]
    total += num
    return total


@dataclass
class _LawContext:
    """法律文本逐行解析的上下文（不变字段），供 _flush_law_article 使用。"""

    doc_id: str
    chunk_size: int
    law_name: str | None = None
    pub_date: str | None = None
    chapter: str | None = None
    section: str | None = None
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class _MdAccumulator:
    """Markdown 分块累加器，封装 buffer/buf_meta/buf_len/idx。"""

    buffer: list[str]
    buf_meta: list[dict[MetaKey, str]]
    buf_len: int
    idx: int
    chunk_size: int
    overlap: int
    doc_id: str
    chunks: list[Chunk]

    def add(self, sec: str, this_meta: dict[MetaKey, str]) -> None:
        sl = len(sec)
        if self.buf_len + sl <= self.chunk_size:
            self.buffer.append(sec)
            self.buf_meta.append(this_meta)
            self.buf_len += sl
        else:
            self.flush()
            if sl <= self.chunk_size:
                self.buffer, self.buf_meta, self.buf_len = [sec], [this_meta], sl
            else:
                for sub in _split_segments(sec):
                    if self.buf_len + len(sub) <= self.chunk_size:
                        self.buffer.append(sub)
                        self.buf_meta.append(this_meta)
                        self.buf_len += len(sub)
                    else:
                        self.flush()
                        self.buffer, self.buf_meta, self.buf_len = [sub], [this_meta], len(sub)

    def flush(self) -> None:
        if not self.buffer:
            return
        meta: dict[str, Any] = {}
        for bm in self.buf_meta:
            for key in (MetaKey.CHAPTER, MetaKey.SECTION):
                if bm.get(key):
                    meta.setdefault(key, bm[key])
        arts = _md_article_range(self.buffer)
        if arts:
            meta[MetaKey.ARTICLES] = arts
        self.chunks.append(_make_chunk(self.buffer, self.doc_id, self.idx, metadata=meta))
        self.idx += 1
        carry = _carry_overlap(self.buffer, self.overlap)
        self.buffer = carry
        self.buf_meta = self.buf_meta[-len(carry) :] if carry else []
        self.buf_len = sum(len(s) for s in carry)


def _detect_md_heading_meta(first_line: str) -> dict[MetaKey, str]:
    """从 Markdown 标题行提取章节元数据。"""
    m: dict[MetaKey, str] = {}
    raw = first_line.lstrip("#").strip()
    if _CHAPTER_DIVISION_RE.match(raw):
        m[MetaKey.CHAPTER] = raw
    elif _NUM_LIST_RE.match(raw):
        m[MetaKey.CHAPTER] = raw
    elif _DECIMAL_RE.match(raw):
        m[MetaKey.CHAPTER] = raw
    if _SECTION_HEAD_RE.match(raw):
        m[MetaKey.SECTION] = raw
    elif _PAREN_NUM_RE.match(raw):
        m[MetaKey.SECTION] = raw
    elif _DECIMAL_RANGE_RE.match(raw):
        m[MetaKey.SECTION] = raw
    return m


def chunk_markdown(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk] | None:
    """以 # 标题为界的结构化分块，无 # 则返回 None。"""
    if not _MD_HEADING_RE.search(text):
        return None

    sections = [s.strip() for s in _MD_SPLIT_RE.split(text.strip()) if s.strip()]
    acc = _MdAccumulator(
        buffer=[],
        buf_meta=[],
        buf_len=0,
        idx=0,
        chunk_size=chunk_size,
        overlap=overlap,
        doc_id=doc_id,
        chunks=[],
    )

    for sec in sections:
        first_line = sec.split("\n")[0].strip()
        this_meta = _detect_md_heading_meta(first_line)
        if not this_meta and acc.buf_meta:
            this_meta = dict(acc.buf_meta[-1])
        elif acc.buf_meta and MetaKey.CHAPTER in acc.buf_meta[-1]:
            this_meta.setdefault(MetaKey.CHAPTER, acc.buf_meta[-1][MetaKey.CHAPTER])

        ch = this_meta.get(MetaKey.CHAPTER)
        if ch and acc.buf_len > 0 and ch != acc.buf_meta[0].get(MetaKey.CHAPTER):
            acc.flush()
            acc.buffer, acc.buf_meta, acc.buf_len = [sec], [this_meta], len(sec)
            continue
        sc = this_meta.get(MetaKey.SECTION)
        if sc and acc.buf_len > 0 and sc != acc.buf_meta[-1].get(MetaKey.SECTION):
            acc.flush()
            acc.buffer, acc.buf_meta, acc.buf_len = [sec], [this_meta], len(sec)
            continue

        acc.add(sec, this_meta)

    acc.flush()
    return acc.chunks


def _split_keep_separator(text: str, pattern: str) -> list[str]:
    """按正则分割，分隔符保留在后续片段开头。"""
    parts = re.split(f"({pattern})", text)
    if len(parts) < 2:
        return [text]
    result: list[str] = []
    buf = parts[0]
    i = 1
    while i < len(parts):
        buf += parts[i]
        if i + 1 < len(parts):
            buf += parts[i + 1]
        result.append(buf)
        buf = ""
        i += 2
    if buf.strip():
        result.append(buf)
    return result


def _hierarchical_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """递归层级文本分割器（模拟 RecursiveCharacterTextSplitter）。

    从最高优先级分隔符开始尝试；若分割后片段仍超 chunk_size，
    用下一级分隔符递归处理。片段不合并——每个片段即为一个 chunk。
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size or not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]

    if sep == r"\n":
        parts = [p for p in text.split("\n") if p.strip()]
        if not parts:
            return [text]
    else:
        parts = _split_keep_separator(text, sep)

    result: list[str] = []
    for part in parts:
        if len(part) > chunk_size and rest:
            result.extend(_hierarchical_split(part, rest, chunk_size))
        else:
            result.append(part)
    return [p for p in result if p.strip()]


def _extract_law_metadata(text: str, filename: str = "") -> dict[str, Any]:
    """从法律文本中提取元数据（法律名称、生效日期等）。"""
    meta: dict[str, Any] = {
        MetaKey.DOC_TYPE: DocType.LAW,
        MetaKey.SOURCE_FILE: filename,
        MetaKey.LAW_NAME: "",
        MetaKey.EFFECTIVE_DATE: "",
    }

    name_pats = [
        r"(中华人民共和国\S+?(?:法典|法|条例|规定|办法|决定|解释))",
        r"(《[^》]+》)",
        r"^(.+(?:法典|法|条例|规定|办法|决定|解释))",
    ]
    for pat in name_pats:
        m = re.search(pat, text[:_LAW_NAME_SCAN_CHARS])
        if m:
            meta[MetaKey.LAW_NAME] = m.group(1).strip("《》")
            break

    if not meta[MetaKey.LAW_NAME] and filename:
        meta[MetaKey.LAW_NAME] = re.sub(r"\.\w+$", "", filename)

    date_pats = [
        r"(\d{4}年\d{1,2}月\d{1,2}日)(?:起)?施行",
        r"自(\d{4}年\d{1,2}月\d{1,2}日)起施行",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pat in date_pats:
        m = re.search(pat, text[:_DATE_SCAN_CHARS])
        if m:
            meta[MetaKey.EFFECTIVE_DATE] = m.group(1)
            break

    return meta


def _extract_case_metadata(text: str, filename: str = "") -> dict[str, Any]:
    """从指导案例文本中提取元数据。"""
    meta: dict[str, Any] = {
        MetaKey.DOC_TYPE: DocType.CASE,
        MetaKey.SOURCE_FILE: filename,
        MetaKey.CASE_NUMBER: "",
        MetaKey.CASE_TITLE: "",
        MetaKey.GUIDING_NUMBER: "",
        MetaKey.KEYWORDS: "",
    }

    g = re.search(r"指导案例(\d+)号", text[:_LAW_NAME_SCAN_CHARS])
    if g:
        meta[MetaKey.GUIDING_NUMBER] = f"指导案例{g.group(1)}号"

    c = re.search(r"[（(]\d{4}[）)][^\s]*\d+号", text[:_CASE_SCAN_CHARS])
    if c:
        meta[MetaKey.CASE_NUMBER] = c.group(0)

    kw = re.search(r"关键词[：:]\s*(.+?)(?:\n|$)", text[:_CASE_SCAN_CHARS])
    if kw:
        keywords = re.split(r"[；;,，\s]+", kw.group(1).strip())
        kw_list = [k.strip() for k in keywords if k.strip()]
        meta[MetaKey.KEYWORDS] = "；".join(kw_list)

    if not meta[MetaKey.CASE_TITLE] and filename:
        meta[MetaKey.CASE_TITLE] = re.sub(r"\.\w+$", "", filename)

    return meta


def _process_law_line(
    line: str,
    ctx: _LawContext,
    state: dict[str, Any],
) -> bool:
    """处理一行法律文本，返回 True 表示触发了条号变更（需要外部 flush）。"""
    if not line:
        return False

    if ctx.law_name is None and _TITLE_RE.match(line):
        ctx.law_name = line
        return False

    if ctx.pub_date is None:
        m = _DATE_RE.search(line)
        if m:
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
            ctx.pub_date = f"{y}-{mo:02d}-{d:02d}"

    m = _CHAPTER_LAW_RE.match(line)
    if m:
        ctx.chapter = m.group(1)
        return False

    m = _SECTION_LAW_RE.match(line)
    if m:
        ctx.section = m.group(1)
        return False

    m = _ARTICLE_LAW_RE.match(line)
    if m:
        state["idx"] = _flush_law_article(
            state["buffer"],
            state["cur_article"],
            state["cur_article_no"],
            ctx,
            state["idx"],
        )
        state["cur_article"] = m.group(1)
        state["cur_article_no"] = _chinese_to_int(m.group(2))
        state["buffer"] = [line]
        return True

    state["buffer"].append(line)
    return False


def chunk_law_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk] | None:
    """法律文本逐行解析分块：按条聚合，超长条递归切分（overlap 参数预留，当前未使用）。

    逐行扫描，识别法律名称、发布日期、章、节、条号；
    每条内容聚合成一个 buffer，超 chunk_size 时按分隔符
    （空行→换行→句号→分号→逗号）递归切分。
    非法律文本返回 None。
    """
    if not _ART_SEARCH_RE.search(text):
        return None

    lines = text.splitlines()
    ctx = _LawContext(doc_id=doc_id, chunk_size=chunk_size)
    state: dict[str, Any] = {
        "cur_article": None,
        "cur_article_no": None,
        "buffer": [],
        "idx": 0,
    }

    for line in lines:
        line = line.strip()
        _process_law_line(line, ctx, state)

    state["idx"] = _flush_law_article(
        state["buffer"],
        state["cur_article"],
        state["cur_article_no"],
        ctx,
        state["idx"],
    )

    return ctx.chunks


def _split_recursive(text: str, chunk_size: int) -> list[str]:
    """递归分割长文本，按空行→换行→句号→分号→逗号 优先级切分。"""
    if len(text) <= chunk_size:
        return [text]
    separators = ["\n\n", "\n", "。", "；", "，"]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            result: list[str] = []
            buf = ""
            for p in parts:
                if not p.strip():
                    if buf:
                        result.append(buf)
                        buf = ""
                    continue
                candidate = (buf + sep + p) if buf else p
                if len(candidate) <= chunk_size:
                    buf = candidate
                else:
                    if buf:
                        result.append(buf)
                    if len(p) <= chunk_size:
                        buf = p
                    else:
                        sub = _split_recursive(p, chunk_size)
                        result.extend(sub)
                        buf = ""
            if buf:
                result.append(buf)
            return result if result else [text]
    return [text]


def _md_article_range(segments: list[str]) -> str:
    """从 Markdown 段列表中提取条文范围。"""
    arts = []
    for s in segments:
        for m in ART_PAT_RE.finditer(s):
            arts.append(int(m.group(1)) if m.group(1).isdigit() else m.group(1))
    if not arts:
        return ""
    return f"第{arts[0]}条" if len(arts) == 1 else f"第{arts[0]}条—第{arts[-1]}条"


def _flush_law_article(
    buffer: list[str],
    cur_article: str | None,
    cur_article_no: int | None,
    ctx: _LawContext,
    idx: int,
) -> int:
    """将法律文本当前条 flush 为一个或多个 Chunk。"""
    if not cur_article:
        return idx
    content = "\n".join(buffer)
    pieces = _split_recursive(content, ctx.chunk_size)
    meta_base: dict[str, Any] = {}
    if ctx.law_name:
        meta_base[MetaKey.LAW_NAME] = ctx.law_name
    if ctx.chapter:
        meta_base[MetaKey.CHAPTER] = ctx.chapter
    if ctx.section:
        meta_base[MetaKey.SECTION] = ctx.section
    meta_base[MetaKey.ARTICLES] = cur_article
    if cur_article_no is not None:
        meta_base[MetaKey.ARTICLE_NO] = cur_article_no
    if ctx.pub_date:
        meta_base[MetaKey.EFFECTIVE_DATE] = ctx.pub_date
    meta_base[MetaKey.CHUNK_TOTAL] = len(pieces)
    for piece in pieces:
        ctx.chunks.append(
            Chunk(
                id=make_id(),
                document_id=ctx.doc_id,
                content=_align_chunk_end(piece),
                chunk_index=idx,
                metadata=dict(meta_base),
            )
        )
        idx += 1
    return idx


def _scan_chunk_meta(
    part: str,
    cur_chapter: str,
    cur_section: str,
) -> tuple[dict[str, Any], str, str]:
    """扫描片段中的章节和条文元数据，返回 (meta, updated_cur_chapter, updated_cur_section)。"""
    meta: dict[str, Any] = {}

    first_ch = ""
    first_sec = ""
    for line in part.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _CHAPTER_DIVISION_RE.match(line) and not first_ch:
            first_ch = line
        if _SECTION_HEAD_RE.match(line) and not first_sec:
            first_sec = line

    if first_ch:
        cur_chapter = first_ch
        cur_section = ""
    if first_sec:
        cur_section = first_sec

    if cur_chapter:
        meta[MetaKey.CHAPTER] = cur_chapter
    if cur_section:
        meta[MetaKey.SECTION] = cur_section

    arts = ART_PAT_RE.findall(part)
    if arts:
        meta[MetaKey.ARTICLES] = (
            f"第{arts[0]}条" if len(arts) == 1 else f"第{arts[0]}条—第{arts[-1]}条"
        )

    return meta, cur_chapter, cur_section


def chunk_legal_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk] | None:
    """法律条文层级回退分块（编→章→节→条→款→项→换行）。

    作为 chunk_law_text 的 fallback，处理非规范格式的法律文本。
    """
    if not _ART_SEARCH_RE.search(text):
        return None

    separators = [
        rf"\n第{_CN_DIGIT}+编",
        rf"\n第{_CN_DIGIT}+章",
        rf"\n第{_CN_DIGIT}+节",
        rf"\n第{_CN_DIGIT}+条",
        rf"\n{_CN_DIGIT}+[、．]",
        rf"\n[（(]{_CN_DIGIT}+[）)]",
        r"\n",
    ]

    parts = _hierarchical_split(text.strip(), separators, chunk_size)
    if not parts:
        return None

    chunks: list[Chunk] = []
    idx = 0
    cur_chapter = ""
    cur_section = ""
    prev_tail = ""

    for part in parts:
        content = prev_tail + part if prev_tail else part
        meta, cur_chapter, cur_section = _scan_chunk_meta(part, cur_chapter, cur_section)

        chunks.append(
            Chunk(
                id=make_id(),
                document_id=doc_id,
                content=_align_chunk_end(content),
                chunk_index=idx,
                metadata=meta,
            )
        )
        idx += 1
        prev_tail = part[-overlap:] if overlap > 0 else ""

    return chunks
