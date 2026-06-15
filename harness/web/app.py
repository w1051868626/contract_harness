"""FastAPI Web 应用，提供合同审查、会话浏览等功能。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from harness.agent.contract_agent import ContractAgent
from harness.agent.llm import LLMClient
from harness.core.config import HarnessConfig
from harness.core.types import ContractDocument
from harness.replay.player import SessionPlayer
from harness.replay.recorder import SessionRecorder
from harness.replay.storage import ReplayStorage
from harness.utils.io import load_dotenv
from harness.utils.log import logger, setup_logging

load_dotenv()

setup_logging(verbose=True)

HERE = Path(__file__).parent

config = HarnessConfig()
config.ensure_dirs()

app = FastAPI(title="contract-harness")
logger.info("FastAPI 应用已创建 (config_dir=%s)", config.data_dir)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _run_review(content: str, title: str) -> dict[str, Any]:
    """执行合同审查并记录会话，返回结构化结果。"""
    doc = ContractDocument(
        id=uuid.uuid4().hex[:12],
        title=title,
        content=content,
    )
    agent = ContractAgent(LLMClient(config.llm))
    report, session = agent.review(doc)

    recorder = SessionRecorder(config.replay_dir)
    recorder.record(session)

    clauses_data = [
        {"type": c.clause_type, "risk": c.risk.value, "content": c.content, "comment": c.comment}
        for c in report.clauses
    ]
    risks_data = [
        {
            "clause_type": r.clause.clause_type,
            "level": r.risk_level.value,
            "reason": r.reason,
            "suggestion": r.suggestion,
        }
        for r in report.risks
    ]
    compliance_data = [
        {"regulation": c.regulation, "status": c.status, "detail": c.detail}
        for c in report.compliance_checks
    ]

    return {
        "session_id": session.session_id,
        "summary": report.summary,
        "overall_risk": report.overall_risk.value,
        "clauses": clauses_data,
        "risks": risks_data,
        "compliance": compliance_data,
    }


def _render(name: str, request: Request, **context: Any) -> HTMLResponse:
    """渲染 Jinja2 模板并返回 HTML 响应。"""
    return templates.TemplateResponse(request, name, context)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页：显示会话统计数据。"""
    replay_dir = Path(config.replay_dir)
    session_count = len(list(replay_dir.glob("*.json"))) if replay_dir.exists() else 0
    return _render("index.html", request, session_count=session_count)


@app.get("/review", response_class=HTMLResponse)
async def review_form(request: Request):
    """显示合同审查表单页面。"""
    return _render("review.html", request)


@app.post("/review", response_class=HTMLResponse)
async def review_submit(
    request: Request,
    content: str = Form(""),
    file: UploadFile | None = None,
):
    """处理合同审查提交：接受文本或文件上传，返回审查结果。"""
    if file and file.filename:
        raw = (await file.read()).decode("utf-8")
        title = file.filename
    else:
        raw = content
        title = "paste.txt"

    if not raw.strip():
        return _render("review.html", request, error="请输入合同内容或上传文件")

    try:
        result = _run_review(raw, title)
        return _render("review.html", request, result=result)
    except Exception as e:
        return _render("review.html", request, error=str(e))


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request):
    """列出所有回放会话。"""
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    sessions_list = player.list_sessions()
    return _render("sessions.html", request, sessions=sessions_list)


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    """查看指定会话的详细回放信息。"""
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    session = player.load(session_id)
    if session is None:
        return _render("sessions.html", request, error=f"会话 {session_id} 不存在")
    return _render("session_detail.html", request, session=session)
