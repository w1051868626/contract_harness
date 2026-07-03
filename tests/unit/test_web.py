"""FastAPI Web 应用单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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

    def test_review_submit_text(self, client):
        """POST /review 文本内容应重定向到会话页面。"""
        mock_session = MagicMock()
        mock_session.session_id = "test123"
        with patch("harness.web.app._agent") as mock_agent_fn, patch("harness.web.app._recorder"):
            mock_agent = MagicMock()
            mock_agent.review.return_value = (None, mock_session)
            mock_agent_fn.return_value = mock_agent
            resp = client.post("/review", data={"content": "测试合同内容"}, follow_redirects=False)
        assert resp.status_code == 303, f"Got {resp.status_code}: {resp.text[:200]}"
        assert resp.headers["location"] == "/sessions/test123"

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
        """GET /sessions 应返回会话列表。"""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_sessions_detail_nonexistent(self, client):
        """GET /sessions/{id} 不存在的会话应返回错误。"""
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 200
        assert "不存在" in resp.text

    def test_converse_api_nonexistent(self, client):
        """POST /sessions/{id}/converse 不存在的会话应返回错误。"""
        mock_session = MagicMock()
        mock_session.session_id = "nonexistent"
        mock_session.document.content = "测试合同"
        mock_session.report = None
        mock_session.steps = []
        mock_session.metadata = {}
        with (
            patch("harness.web.app._agent") as mock_agent_fn,
            patch("harness.web.app._player") as mock_player_fn,
        ):
            mock_agent = MagicMock()
            mock_agent.converse.side_effect = ValueError("未找到会话")
            mock_agent_fn.return_value = mock_agent
            mock_player = MagicMock()
            mock_player.load.return_value = mock_session
            mock_player_fn.return_value = mock_player
            resp = client.post("/sessions/nonexistent/converse", data={"query": "追问测试"})
        assert resp.status_code == 200
        assert "未找到会话" in resp.text

    def test_converse_api_empty_query(self, client):
        """POST /sessions/{id}/converse 空查询应返回错误。"""
        mock_session = MagicMock()
        mock_session.session_id = "nonexistent"
        mock_session.document.content = "测试合同"
        mock_session.report = None
        mock_session.steps = []
        mock_session.metadata = {}
        with patch("harness.web.app._player") as mock_player_fn:
            mock_player = MagicMock()
            mock_player.load.return_value = mock_session
            mock_player_fn.return_value = mock_player
            resp = client.post("/sessions/nonexistent/converse", data={"query": ""})
        assert resp.status_code == 200
        assert "请输入问题" in resp.text
