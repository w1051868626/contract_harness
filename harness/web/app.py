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

HERE = Path(__file__).parent

config = HarnessConfig()
config.ensure_dirs()

app = FastAPI(title="contract-harness")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _run_review(content: str, title: str) -> dict[str, Any]:
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    replay_dir = Path(config.replay_dir)
    session_count = len(list(replay_dir.glob("*.json"))) if replay_dir.exists() else 0
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "session_count": session_count},
    )


@app.get("/review", response_class=HTMLResponse)
async def review_form(request: Request):
    return templates.TemplateResponse("review.html", {"request": request})


@app.post("/review", response_class=HTMLResponse)
async def review_submit(
    request: Request,
    content: str = Form(""),
    file: UploadFile | None = None,
):
    if file and file.filename:
        raw = (await file.read()).decode("utf-8")
        title = file.filename
    else:
        raw = content
        title = "paste.txt"

    if not raw.strip():
        return templates.TemplateResponse(
            "review.html",
            {"request": request, "error": "请输入合同内容或上传文件"},
        )

    try:
        result = _run_review(raw, title)
        return templates.TemplateResponse(
            "review.html",
            {"request": request, "result": result},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "review.html",
            {"request": request, "error": str(e)},
        )


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request):
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    sessions_list = player.list_sessions()
    return templates.TemplateResponse(
        "sessions.html",
        {"request": request, "sessions": sessions_list},
    )


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    player = SessionPlayer(ReplayStorage(config.replay_dir))
    session = player.load(session_id)
    if session is None:
        return templates.TemplateResponse(
            "sessions.html",
            {"request": request, "error": f"会话 {session_id} 不存在"},
        )
    return templates.TemplateResponse(
        "session_detail.html",
        {"request": request, "session": session},
    )
