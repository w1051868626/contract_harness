"""Docling 文档解析器封装，提供高质量结构化文本提取（可选依赖）。

Docling 能解析 PDF/DOCX/PPTX/图片等格式，输出保留标题层级、表格、列表
等结构的 Markdown，比 pypdf / python-docx 的平铺文本更适合后续分块处理。

使用方式：
    from harness.rag.docling_parser import DoclingParser
    parser = DoclingParser()
    if parser.available:
        md = parser.parse_to_markdown("合同.pdf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.utils.log import logger

_SUPPORTED_SUFFIXES: set[str] = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
}

DOCLING_NOT_AVAILABLE = "Docling 库未安装，请执行: pip install docling"


class DoclingParser:
    """基于 Docling 的高质量文档解析器。

    自动检测 Docling 是否可用。不可用时所有方法抛出 RuntimeError，
    调用方应检查 available 属性后决定是否回退。
    """

    def __init__(self) -> None:
        self._converter: Any = None
        self._available = False
        self._init_docling()

    def _init_docling(self) -> None:
        """尝试导入并初始化 Docling。"""
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]

            self._converter = DocumentConverter()
            self._available = True
            logger.info("Docling 解析器初始化成功")
        except ImportError:
            logger.info("Docling 未安装，回退到传统解析器")
            self._available = False
        except Exception as exc:
            logger.warning("Docling 初始化失败: {}", exc)
            self._available = False

    @property
    def available(self) -> bool:
        """Docling 是否可用。"""
        return self._available

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """检查文件后缀是否被 Docling 支持。"""
        return Path(path).suffix.lower() in _SUPPORTED_SUFFIXES

    def parse_to_markdown(self, path: str | Path) -> str:
        """解析文档为 Markdown 格式，保留标题层级、表格、列表等结构。

        优先使用 export_to_markdown() 保留结构；若输出过短（<100 字符）
        则说明 markdown 导出遗漏内容，回退到 result.document.text 提取所有页文本。
        """
        if not self._available:
            raise RuntimeError(DOCLING_NOT_AVAILABLE)
        logger.debug("Docling 解析为 Markdown: path={}", path)
        result = self._converter.convert(str(path))
        md = result.document.export_to_markdown()
        if len(md.strip()) >= 100:
            return md
        logger.info("Docling markdown 输出过短 ({} 字符)，回退到全部页文本", len(md.strip()))
        return result.document.text

    def parse_to_text(self, path: str | Path) -> str:
        """解析文档为纯文本，失去结构化信息但内容更简洁。"""
        if not self._available:
            raise RuntimeError(DOCLING_NOT_AVAILABLE)
        logger.debug("Docling 解析为纯文本: path={}", path)
        result = self._converter.convert(str(path))
        return result.document.text
