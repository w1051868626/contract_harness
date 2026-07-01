"""FastAPI Web 应用，提供合同审查、会话浏览等功能。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage
from harness.utils.io import load_dotenv, make_id, normalize_text
from harness.utils.log import logger, setup_logging

load_dotenv()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

HERE = Path(__file__).parent

config = HarnessConfig()
config.ensure_dirs()

setup_logging(verbose=config.verbose, log_dir=config.log_dir)

app = FastAPI(title="contract-harness")
logger.info("FastAPI 应用已创建 (config_dir={})", config.data_dir)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))

try:
    PKG_VERSION = version("contract-harness")
except Exception:
    PKG_VERSION = "0.0.0"


def _agent() -> ContractAgent:
    return ContractAgent(LLMClient(config.llm))


def _player() -> SessionPlayer:
    return SessionPlayer(ReplayStorage(config.replay_dir))


def _recorder() -> SessionRecorder:
    return SessionRecorder(config.replay_dir)


def _render(name: str, request: Request, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        name,
        {"version": PKG_VERSION, **context},
    )


def _format_session(s: dict[str, Any]) -> dict[str, Any]:
    """格式化会话数据供模板使用。"""
    ts = s.get("started_at", "")
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            now = datetime.now(timezone.utc)
            delta = now - dt.replace(tzinfo=timezone.utc) if dt.tzinfo else now - dt
            if delta.total_seconds() < 60:
                s["time_ago"] = "刚刚"
            elif delta.total_seconds() < 3600:
                s["time_ago"] = f"{int(delta.total_seconds() // 60)} 分钟前"
            elif delta.total_seconds() < 86400:
                s["time_ago"] = f"{int(delta.total_seconds() // 3600)} 小时前"
            elif delta.total_seconds() < 2592000:
                s["time_ago"] = f"{int(delta.total_seconds() // 86400)} 天前"
            else:
                s["time_ago"] = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            s["time_ago"] = ts
    return s


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    replay_dir = Path(config.replay_dir)
    session_count = len(list(replay_dir.glob("*.json"))) if replay_dir.exists() else 0
    return _render("index.html", request, session_count=session_count)


@app.get("/review", response_class=HTMLResponse)
async def review_form(request: Request):
    return _render("review.html", request)


@app.post("/review")
async def review_submit(
    request: Request,
    content: str = Form(""),
    file: UploadFile | None = None,
):
    try:
        if file and file.filename:
            raw_bytes = await file.read()
            if len(raw_bytes) > MAX_UPLOAD_SIZE:
                return _render("review.html", request, error="文件大小超过限制（最大 10MB）")
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return _render("review.html", request, error="文件编码不是 UTF-8")
            title = file.filename
        else:
            raw = content
            title = "paste.txt"

        raw = normalize_text(raw)

        if not raw.strip():
            return _render("review.html", request, error="请输入合同内容或上传文件")

        doc = ContractDocument(id=make_id(), title=title, content=raw)
        agent = _agent()
        report, session = await asyncio.to_thread(agent.review, doc)
        recorder = _recorder()
        recorder.record(session)
        return RedirectResponse(
            f"/sessions/{session.session_id}",
            status_code=303,
        )
    except Exception as e:
        logger.error("审查失败: {}", str(e))
        return _render("review.html", request, error=f"审查失败: {e}")


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request, q: str = ""):
    player = _player()
    sessions_list = player.list_sessions()
    if q:
        ql = q.lower()
        sessions_list = [
            s
            for s in sessions_list
            if ql in s.get("title", "").lower() or ql in s.get("session_id", "").lower()
        ]
    sessions_list = [_format_session(s) for s in sessions_list]
    return _render("sessions.html", request, sessions=sessions_list, q=q)


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    player = _player()
    session = player.load(session_id)
    if session is None:
        return _render("sessions.html", request, error=f"会话 {session_id} 不存在")
    return _render("session_detail.html", request, session=session)


@app.post("/sessions/{session_id}/converse", response_class=HTMLResponse)
async def session_converse(request: Request, session_id: str, query: str = Form("")):
    if not query.strip():
        player = _player()
        session = player.load(session_id)
        return _render("session_detail.html", request, session=session, error="请输入问题")
    try:
        agent = _agent()
        answer = await asyncio.to_thread(agent.converse, session_id, query)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        player = _player()
        session = player.load(session_id)
        return _render("session_detail.html", request, session=session, error=str(e))
    except Exception as e:
        logger.error("追问失败: {}", str(e))
        player = _player()
        session = player.load(session_id)
        return _render("session_detail.html", request, session=session, error=f"追问失败: {e}")
    player = _player()
    session = player.load(session_id)
    return _render("session_detail.html", request, session=session, answer=answer)
