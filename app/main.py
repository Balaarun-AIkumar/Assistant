"""FastAPI entry point for POST /chat."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

from .agent import run_agent

app = FastAPI(title="Corvanta Helpdesk Assistant", version="1.0.0")
UI_DIR = Path(__file__).resolve().parent / "ui"
app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")
_SESSIONS: dict[str, list[dict[str, str]]] = defaultdict(list)
_SESSION_LOCK = Lock()
MAX_HISTORY_MESSAGES = 20


class ChatRequest(BaseModel):
    message: str
    employee_id: str | None = None
    session_id: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value.strip()

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_flag: str
    tool_calls: list[dict[str, Any]]
    ticket_id: str | None = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    with _SESSION_LOCK:
        history = list(_SESSIONS[request.session_id])
    try:
        result = run_agent(
            message=request.message,
            employee_id=request.employee_id,
            session_id=request.session_id,
            history=history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=502, detail="The assistant could not complete this request. Please try again.") from None

    with _SESSION_LOCK:
        _SESSIONS[request.session_id].extend(
            [{"role": "user", "content": request.message}, {"role": "assistant", "content": result["answer"]}]
        )
        _SESSIONS[request.session_id] = _SESSIONS[request.session_id][-MAX_HISTORY_MESSAGES:]
    return ChatResponse(**result)
