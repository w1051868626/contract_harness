"""FastAPI Web 应用单元测试。"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from harness.web.app import app


@pytest.fixture
def client():
    """返回 FastAPI 测试客户端。"""
    return TestClient(app)


class TestWebApp:
    """Web 应用路由测试。"""

    def test_index_returns_html(self, client):
        """GET / 应返回首页 HTML。"""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_review_form_returns_html(self, client):
        """GET /review 应返回审查表单页面。"""
        resp = client.get("/review")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_review_submit_empty(self, client):
        """POST /review 空内容应返回错误提示。"""
        resp = client.post("/review", data={"content": ""})
        assert resp.status_code == 200
        assert "请输入合同内容" in resp.text

    @patch("harness.web.app._run_review")
    def test_review_submit_text(self, mock_run_review, client):
        """POST /review 文本内容应正常处理。"""
        mock_run_review.return_value = {
            "session_id": "test123",
            "summary": "审查完成",
            "overall_risk": "low",
            "clauses": [],
            "risks": [],
            "compliance": [],
        }
        resp = client.post("/review", data={"content": "测试合同内容"})
        assert resp.status_code == 200
        assert "审查完成" in resp.text
        mock_run_review.assert_called_once()

    def test_review_submit_file_oversized(self, client):
        """POST /review 超大文件应返回错误。"""
        resp = client.post(
            "/review",
            files={"file": ("test.txt", b"a" * (10 * 1024 * 1024 + 1))},
        )
        assert resp.status_code == 200
        assert "超过限制" in resp.text

    def test_review_submit_non_utf8(self, client):
        """POST /review 非 UTF-8 文件应返回错误。"""
        resp = client.post(
            "/review",
            files={"file": ("test.txt", b"\xff\xfe\x00\x01")},
        )
        assert resp.status_code == 200
        assert "不是 UTF-8" in resp.text

    def test_sessions_returns_html(self, client):
        """GET /sessions 应返回会话列表页面。"""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_sessions_detail_nonexistent(self, client):
        """GET /sessions/{id} 不存在的会话应返回提示。"""
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 200
        assert "不存在" in resp.text
