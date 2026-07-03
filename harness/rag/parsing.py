"""文件解析模块，支持 txt/md/json/pdf/docx/zip 格式解析。"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from harness.rag.docling_parser import DoclingParser
from harness.utils.log import logger

_docling_parser: DoclingParser | None = None


def parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
    if isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)
    return str(data)


def parse_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except (PdfReadError, KeyError, IndexError):
        logger.warning("PDF 解析失败，按文本读取: {}", path)
        return path.read_text(encoding="utf-8", errors="replace")


def parse_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


_PARSERS: dict[str, Callable[[Path], str]] = {
    ".txt": parse_txt,
    ".md": parse_txt,
    ".json": parse_json,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    logger.debug("解析文件: path={}, suffix={}", path.name, suffix)

    global _docling_parser
    if _docling_parser and _docling_parser.available and _docling_parser.supports(path):
        logger.info("Docling 解析: path={}, suffix={}", path.name, suffix)
        try:
            md = _docling_parser.parse_to_markdown(path)
            if md.strip():
                logger.info("Docling 解析成功: path={} ({} 字符)", path.name, len(md))
                return md
            logger.warning("Docling 返回空内容，回退 path={}", path.name)
        except RuntimeError as e:
            logger.warning("Docling 解析失败，回退 path={}: {}", path.name, e)

    handler = _PARSERS.get(suffix)
    if handler is not None:
        return handler(path)
    return path.read_text(encoding="utf-8")


def enable_docling() -> None:
    global _docling_parser
    parser = DoclingParser()
    if parser.available:
        _docling_parser = parser
        logger.info("Docling 解析器已启用")
    else:
        logger.warning("Docling 解析器不可用（未安装或初始化失败）")
        _docling_parser = None


def extract_zip_texts(path: Path, work_dir: str | None = None) -> list[tuple[str, str]]:
    supported = {".txt", ".md", ".json", ".pdf", ".docx", ".zip"}
    results: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in supported:
                continue
            try:
                raw = zf.read(info.filename)
                with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
                    tmp_path = Path(tmpdir) / Path(info.filename).name
                    tmp_path.write_bytes(raw)
                    if ext == ".zip":
                        nested = extract_zip_texts(tmp_path, work_dir=work_dir)
                        results.extend(nested)
                    else:
                        content = parse_file(tmp_path)
                        if content.strip():
                            results.append((info.filename, content))
            except (json.JSONDecodeError, KeyError, OSError, zipfile.BadZipFile):
                logger.warning("ZIP 中文件解析失败: {}", info.filename, exc_info=True)
                continue
    return results
