"""
FastAPI application — HTTP API layer over MeetingAssistant.

Routes
------
GET  /health                       → liveness probe
GET  /api/v1/sessions              → list all active sessions
POST /api/v1/sessions              → join a meeting (body: { meeting_url })
GET  /api/v1/sessions/{id}         → session detail + events
DELETE /api/v1/sessions/{id}       → leave / stop bot
GET  /api/v1/sessions/{id}/logs    → SSE stream of live log entries

Run locally
-----------
    uvicorn src.api.app:app --reload --port 8000

CORS is open for dev; tighten ALLOWED_ORIGINS in production via env var.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import secrets

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl, field_validator

from src.api.session_store import store
from src.bot import detect_platform
from src.main import MeetingAssistant

logger = structlog.get_logger()

# ── Allowed CORS origins ──────────────────────────────────────────────────────
_raw = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",")] if _raw != "*" else ["*"]

# ── API key auth ──────────────────────────────────────────────────────────────
_API_KEY = os.getenv("API_SECRET_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(_api_key_header)) -> None:
    """Dependency — rejects requests with a missing or wrong API key.
    If API_SECRET_KEY is not set in env the check is skipped (dev mode).
    """
    if not _API_KEY:
        # No key configured — allow all (local dev only)
        return
    if not key or not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_startup")
    yield
    # Gracefully stop all running sessions on shutdown
    logger.info("api_shutdown", active_sessions=len(store.all()))
    tasks = []
    for session in store.all():
        if session.assistant:
            tasks.append(asyncio.create_task(session.assistant.stop()))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MeetMind AI — Backend API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class JoinRequest(BaseModel):
    meeting_url: str

    @field_validator("meeting_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        allowed = [
            "teams.microsoft.com",
            "teams.live.com",
            "zoom.us",
            "meet.google.com",
        ]
        # Basic host check — rejects obviously wrong input
        from urllib.parse import urlparse
        try:
            host = urlparse(v).hostname or ""
        except Exception:
            raise ValueError("Invalid URL")
        if not any(host.endswith(h) for h in allowed):
            raise ValueError(
                "URL must be a Teams (teams.microsoft.com or teams.live.com), "
                "Zoom, or Google Meet link."
            )
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log_interceptor(session_id: str):
    """
    Returns a structlog processor that captures log events for this session.
    Attach it by binding it into the structlog chain at session start.
    """
    def processor(logger_, method_name: str, event_dict: dict) -> dict:
        level_map = {
            "debug":    "DEBUG",
            "info":     "INFO",
            "warning":  "WARNING",
            "error":    "ERROR",
            "critical": "ERROR",
        }
        level = level_map.get(method_name.lower(), "INFO")

        # Build a human-readable message from the event dict
        event = event_dict.get("event", "")
        extras = {k: v for k, v in event_dict.items()
                  if k not in ("event", "timestamp", "_record", "level", "logger")}
        text = event
        if extras:
            parts = " ".join(f"{k}={v}" for k, v in extras.items())
            text = f"{event} — {parts}"

        store.push_log(session_id, level, text)
        return event_dict  # pass through unchanged

    return processor


async def _run_assistant(session_id: str, meeting_url: str):
    """Background task: starts MeetingAssistant, forwards lifecycle to the store."""
    log = _make_log_interceptor(session_id)

    # Bind our per-session processor into a local structlog context
    bound_logger = structlog.get_logger().bind(_session_id=session_id)
    # Patch structlog globally would affect all sessions; instead we hook into
    # MeetingAssistant's observable callbacks and use push_log directly.

    assistant = MeetingAssistant(meeting_url)
    session = store.get(session_id)
    if not session:
        return
    session.assistant = assistant

    # ── Observable hooks ──────────────────────────────────────────────

    def on_trigger(trigger, context=None):
        store.push_log(session_id, "SUCCESS",
                       f"Trigger fired: {trigger.type.value} (confidence {trigger.confidence:.0%})")
        store.add_event(session_id, "active",
                        f"Trigger: {trigger.type.value} detected")

    def on_response(message, trigger=None):
        store.push_log(session_id, "SUCCESS",
                       f"Response sent to meeting: \"{str(message)[:120]}\"")
        store.add_event(session_id, "success", "Alex sent a message to the meeting")

    assistant.on_trigger_fired  = on_trigger
    assistant.on_response_sent  = on_response

    try:
        store.push_log(session_id, "INFO", "Requesting bot from Recall.ai…")
        store.add_event(session_id, "info", "Requesting bot from Recall.ai…")

        await assistant.start()

        store.set_status(session_id, "JOINED")
        store.push_log(session_id, "SUCCESS",
                       "Bot joined meeting successfully — transcript stream active")
        store.add_event(session_id, "success",
                        "Bot joined successfully — transcript stream active.")
        store.add_event(session_id, "active", "Listening for AI triggers…")

        # Keep running until cancelled (bot runs until meeting ends or /leave called)
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        store.push_log(session_id, "INFO", "Bot leave requested — shutting down…")
        await assistant.stop()
        store.set_status(session_id, "ENDED")
        store.push_log(session_id, "SUCCESS", "Bot removed from meeting successfully.")
        store.add_event(session_id, "success", "Bot removed from meeting successfully.")

    except Exception as exc:
        err_msg = str(exc)
        store.set_status(session_id, "FAILED")
        store.push_log(session_id, "ERROR", f"Join failed: {err_msg}")
        store.add_event(session_id, "error",
                        f"Failed to join — {err_msg[:200]}")
        logger.error("session_failed", session_id=session_id, error=err_msg)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(store.all())}


@app.get("/api/v1/sessions")
async def list_sessions(_: None = Depends(require_api_key)):
    return {"sessions": [s.to_dict() for s in store.all()]}


@app.post("/api/v1/sessions", status_code=201)
async def join_meeting(body: JoinRequest, _: None = Depends(require_api_key)):
    session = store.create(body.meeting_url)
    task = asyncio.create_task(_run_assistant(session.id, body.meeting_url))
    session.task = task
    logger.info("session_created", session_id=session.id, url=body.meeting_url)
    return {"session_id": session.id, "status": session.status}


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str, _: None = Depends(require_api_key)):
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s.to_dict()


@app.delete("/api/v1/sessions/{session_id}")
async def leave_meeting(session_id: str, _: None = Depends(require_api_key)):
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.status not in ("JOINED", "CONNECTING"):
        raise HTTPException(status_code=409, detail=f"Session is already {s.status}")

    store.set_status(session_id, "LEAVING")
    store.add_event(session_id, "info", "Bot leaving meeting…")
    store.push_log(session_id, "INFO", "Leave requested by user")

    if s.task and not s.task.done():
        s.task.cancel()

    logger.info("session_leave_requested", session_id=session_id)
    return {"success": True, "session_id": session_id}


@app.get("/api/v1/sessions/{session_id}/logs")
async def stream_logs(session_id: str, request: Request, _: None = Depends(require_api_key)):
    """Server-Sent Events endpoint — streams log entries in real time."""
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    q = store.subscribe_logs(session_id)
    if q is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {entry.to_sse()}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive ping
                    yield ": ping\n\n"
        finally:
            store.unsubscribe_logs(session_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering for SSE
        },
    )
