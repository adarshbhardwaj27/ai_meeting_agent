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
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from datetime import datetime, timezone

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl, field_validator

from src.api.session_store import store
from src.auth.router import router as auth_router
from src.auth.service import decode_access_token
from src.bot import detect_platform
from src.db.session import init_db, AsyncSessionLocal
from src.main import MeetingAssistant
from src.api.microsoft_router import router as microsoft_router

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
        return
    if not key or not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


async def require_auth(request: Request) -> dict:
    """Dependency — validates the httpOnly JWT cookie and returns the payload.
    Returns: { company_id, user_id, exp }
    Falls back to X-API-Key check for machine-to-machine calls (CI/CD scripts).
    """
    token = request.cookies.get("access_token")
    if token:
        payload = decode_access_token(token)
        if payload:
            return payload
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )

    # Fallback: X-API-Key lets automation scripts still hit the API
    key = request.headers.get("X-API-Key", "")
    if _API_KEY and secrets.compare_digest(key, _API_KEY):
        return {"company_id": "__system__", "user_id": "__system__"}

    if not _API_KEY and not token:
        # Dev mode — no secrets configured, allow all
        return {"company_id": "__dev__", "user_id": "__dev__"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
    )


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    renewal_task = asyncio.create_task(_subscription_renewal_loop())
    logger.info("api_startup")
    yield
    renewal_task.cancel()
    # Gracefully stop all running sessions on shutdown
    logger.info("api_shutdown", active_sessions=len(store.all()))
    tasks = []
    for session in store.all():
        if session.assistant:
            tasks.append(asyncio.create_task(session.assistant.stop()))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _subscription_renewal_loop():
    """Hourly background task: renew Graph subscriptions expiring within 24 h."""
    from datetime import timedelta
    from sqlalchemy import select
    from src.db.models import MicrosoftTenant
    from src.integrations.microsoft.graph_client import get_valid_token, renew_subscription
    while True:
        await asyncio.sleep(3600)
        try:
            async with AsyncSessionLocal() as db:
                cutoff = datetime.now(timezone.utc) + timedelta(hours=24)
                result = await db.execute(
                    select(MicrosoftTenant).where(
                        MicrosoftTenant.subscription_id.isnot(None),
                        MicrosoftTenant.subscription_expires_at < cutoff,
                    )
                )
                for tenant in result.scalars().all():
                    try:
                        token = await get_valid_token(tenant, db)
                        sub = await renew_subscription(token, tenant.subscription_id)
                        expiry = sub.get("expirationDateTime", "")
                        if expiry:
                            tenant.subscription_expires_at = datetime.fromisoformat(
                                expiry.replace("Z", "+00:00")
                            )
                        await db.commit()
                        logger.info("subscription_renewed", tenant_id=tenant.tenant_id)
                    except Exception as e:
                        logger.error("subscription_renewal_failed",
                                     tenant_id=tenant.tenant_id, error=str(e))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("renewal_loop_error", error=str(e))


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

app.include_router(auth_router)
app.include_router(microsoft_router)


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
async def list_sessions(auth: dict = Depends(require_auth)):
    company_id = auth["company_id"]
    return {"sessions": [s.to_dict() for s in store.all(company_id=company_id)]}


@app.post("/api/v1/sessions", status_code=201)
async def join_meeting(body: JoinRequest, auth: dict = Depends(require_auth)):
    session = store.create(body.meeting_url, company_id=auth["company_id"])
    task = asyncio.create_task(_run_assistant(session.id, body.meeting_url))
    session.task = task
    logger.info("session_created", session_id=session.id, url=body.meeting_url)
    return {"session_id": session.id, "status": session.status}


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str, auth: dict = Depends(require_auth)):
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.company_id not in (auth["company_id"], "__system__", "__dev__"):
        raise HTTPException(status_code=404, detail="Session not found")
    return s.to_dict()


@app.delete("/api/v1/sessions/{session_id}")
async def leave_meeting(session_id: str, auth: dict = Depends(require_auth)):
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.company_id not in (auth["company_id"], "__system__", "__dev__"):
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
async def stream_logs(session_id: str, request: Request, auth: dict = Depends(require_auth)):
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
