"""
In-memory session store for active MeetingAssistant instances.

Each session tracks:
  - status  : CONNECTING | JOINED | FAILED | LEAVING | ENDED
  - events  : timeline shown in the dashboard EventFeed
  - log_buf : rolling 500-entry log deque for SSE streaming
  - queues  : per-SSE-subscriber asyncio.Queue (fan-out pattern)
  - task    : the asyncio.Task running MeetingAssistant
"""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SessionEvent:
    id: str
    type: str        # info | success | error | warning | active
    message: str
    time: str


@dataclass
class LogEntry:
    id: str
    timestamp: str   # HH:MM:SS
    level: str       # DEBUG | INFO | SUCCESS | WARNING | ERROR
    text: str

    def to_sse(self) -> str:
        """Format as an SSE data line."""
        import json
        return json.dumps({
            "id": self.id,
            "timestamp": self.timestamp,
            "level": self.level,
            "text": self.text,
        })


@dataclass
class Session:
    id: str
    meeting_url: str
    status: str                       # state machine value
    started_at: datetime
    events: List[SessionEvent] = field(default_factory=list)
    log_buf: deque = field(default_factory=lambda: deque(maxlen=500))
    log_queues: List[asyncio.Queue] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    assistant: Optional[Any] = None   # MeetingAssistant instance

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "meeting_url": self.meeting_url,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "events": [e.__dict__ for e in self.events],
        }


# ── Store ─────────────────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    def create(self, meeting_url: str) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            meeting_url=meeting_url,
            status="CONNECTING",
            started_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def all(self) -> List[Session]:
        return list(self._sessions.values())

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Mutation helpers ──────────────────────────────────────────────

    def set_status(self, session_id: str, status: str) -> None:
        s = self.get(session_id)
        if s:
            s.status = status

    def add_event(self, session_id: str, type_: str, message: str) -> None:
        s = self.get(session_id)
        if not s:
            return
        evt = SessionEvent(
            id=str(uuid.uuid4()),
            type=type_,
            message=message,
            time=datetime.now(timezone.utc).strftime("%H:%M:%S"),
        )
        s.events.append(evt)

    def push_log(self, session_id: str, level: str, text: str) -> None:
        s = self.get(session_id)
        if not s:
            return
        entry = LogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            level=level,
            text=text,
        )
        s.log_buf.append(entry)
        # Fan-out to all SSE subscribers (non-blocking put_nowait)
        dead = []
        for q in s.log_queues:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                s.log_queues.remove(q)
            except ValueError:
                pass

    def subscribe_logs(self, session_id: str) -> Optional[asyncio.Queue]:
        """Create and register a new SSE subscriber queue."""
        s = self.get(session_id)
        if not s:
            return None
        q: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=200)
        # Replay last 50 buffered entries so the client gets context immediately
        for entry in list(s.log_buf)[-50:]:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                break
        s.log_queues.append(q)
        return q

    def unsubscribe_logs(self, session_id: str, q: asyncio.Queue) -> None:
        s = self.get(session_id)
        if s:
            try:
                s.log_queues.remove(q)
            except ValueError:
                pass


# Singleton — imported everywhere
store = SessionStore()
