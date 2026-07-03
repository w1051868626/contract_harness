"""Docling 解析器单元测试。"""

from __future__ import annotations

import pytest

from harness.rag.docling_parser import DoclingParser


class TestDoclingParser:
    """Docling 解析器功能测试（需 docling 库安装后才能完整测试）。"""

    def test_available_false_by_default(self):
        """docling 未安装时 available 应为 False。"""
        parser = DoclingParser()
        assert parser.available is False

    def test_parse_raises_when_not_available(self):
        """docling 不可用时 parse 方法应抛出 RuntimeError。"""
        parser = DoclingParser()
        with pytest.raises(RuntimeError, match="Docling 库未安装"):
            parser.parse_to_markdown("test.pdf")

    def test_parse_text_raises_when_not_available(self):
        """docling 不可用时 parse_to_text 应抛出 RuntimeError。"""
        parser = DoclingParser()
        with pytest.raises(RuntimeError, match="Docling 库未安装"):
            parser.parse_to_text("test.pdf")

    def test_supports_pdf(self):
        """.pdf 后缀应被支持。"""
        assert DoclingParser.supports("test.pdf") is True
        assert DoclingParser.supports("/path/to/document.PDF") is True

    def test_supports_docx(self):
        """.docx 后缀应被支持。"""
        assert DoclingParser.supports("test.docx") is True

    def test_supports_pptx(self):
        """.pptx 后缀应被支持。"""
        assert DoclingParser.supports("test.pptx") is True

    def test_supports_image(self):
        """常见图片后缀应被支持。"""
        assert DoclingParser.supports("test.png") is True
        assert DoclingParser.supports("test.jpg") is True
        assert DoclingParser.supports("test.jpeg") is True
        assert DoclingParser.supports("test.tiff") is True
        assert DoclingParser.supports("test.bmp") is True

    def test_unsupported_suffix(self):
        """不支持的格式应返回 False。"""
        assert DoclingParser.supports("test.txt") is False
        assert DoclingParser.supports("test.md") is False
        assert DoclingParser.supports("test.json") is False
        assert DoclingParser.supports("test.html") is False
        assert DoclingParser.supports("test") is False


class TestKnowledgeBaseDoclingIntegration:
    """KnowledgeBase 与 Docling 集成测试。"""

    def test_enable_docling_when_not_installed(self):
        """docling 未安装时 enable_docling 不应抛出异常。"""
        import harness.rag.parsing as _parsing_mod
        from harness.rag.parsing import enable_docling

        # Reset any previous state
        _parsing_mod._docling_parser = None

        # Should not raise
        enable_docling()
        assert _parsing_mod._docling_parser is None
