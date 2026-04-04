"""
Meeting Session Logger.

Appends human-readable logs to a persistent file so you can analyse
what happened in a meeting (or an E2E test) after the fact.

Each session is separated by a header with the start timestamp.
Every line is prefixed with a wall-clock timestamp.

Log file:  logs/meeting_sessions.log  (auto-created, append-mode)
"""

import os
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

# Default log location
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "meeting_sessions.log"
_EVENT_LOG_FILE = _LOG_DIR / "meeting_session_events.jsonl"


class MeetingLogger:
    """
    Append-only file logger for meeting sessions.

    Usage:
        ml = MeetingLogger(meeting_id="my_meeting", source="live")
        ml.log("🔗 Joining meeting: https://...")
        ml.transcript("Adarsh Bhardwaj", "hello", is_final=True, latency="0.2s")
        ml.bot_response("Based on the discussion...")
        ml.close()
    """

    def __init__(
        self,
        meeting_id: str = "",
        source: str = "live",
        log_file: Optional[str] = None,
    ):
        self._log_path = Path(log_file) if log_file else _LOG_FILE
        if log_file:
            self._event_log_path = self._log_path.with_name(f"{self._log_path.stem}_events.jsonl")
        else:
            self._event_log_path = _EVENT_LOG_FILE
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._meeting_id = meeting_id
        self._source = source
        self._start_time = datetime.now()

        # Open in append mode (never overwrite previous sessions)
        self._file = open(self._log_path, "a", encoding="utf-8")

        # Write session header
        self._file.write("\n")
        self._file.write("=" * 80 + "\n")
        self._file.write(
            f"  SESSION START: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self._file.write(f"  Meeting ID:    {meeting_id}\n")
        self._file.write(f"  Source:        {source}\n")
        self._file.write("=" * 80 + "\n")
        self._file.flush()
        self._write_structured_record(
            event_type="session_start",
            payload={
                "started_at": self._start_time.isoformat(),
            },
            occurred_at=self._start_time,
        )

    def _write_structured_record(
        self,
        *,
        event_type: str,
        payload: Optional[dict] = None,
        occurred_at: Optional[datetime] = None,
    ):
        record = {
            "event_type": event_type,
            "timestamp": (occurred_at or datetime.now()).isoformat(),
            "meeting_id": self._meeting_id,
            "source": self._source,
            "session_started_at": self._start_time.isoformat(),
        }
        if payload:
            record.update(payload)

        with self._event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    # ── Core write ───────────────────────────────────────────────────

    def log(self, message: str):
        """Write a timestamped line to the log file."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        self._file.write(f"[{ts}] {message}\n")
        self._file.flush()

    # ── Convenience methods ──────────────────────────────────────────

    def transcript(
        self,
        speaker: str,
        text: str,
        is_final: bool = True,
        latency: str = "",
    ):
        """Log a transcript chunk (interim or final)."""
        label = "🎤" if is_final else "💬"
        lat = f" ⏱️{latency}" if latency else ""
        self.log(f"{label} [{speaker}]: {text}{lat}")
        self._write_structured_record(
            event_type="transcript",
            payload={
                "speaker": speaker,
                "text": text,
                "is_final": is_final,
                "latency": latency,
            },
        )

    def bot_response(self, text: str, trigger_type: str = ""):
        """Log a bot response sent to chat."""
        extra = f" ({trigger_type})" if trigger_type else ""
        self.log(f"🤖 Sent{extra}: {text}")
        self._write_structured_record(
            event_type="bot_response",
            payload={
                "text": text,
                "trigger_type": trigger_type,
            },
        )

    def trigger_fired(
        self,
        trigger_type: str,
        speaker: str = "",
        text: str = "",
        confidence: float = 0.0,
    ):
        """Log a trigger detection event."""
        parts = [f"⚡ Trigger: {trigger_type}"]
        if speaker:
            parts.append(f"speaker={speaker}")
        if text:
            parts.append(f'text="{text[:80]}"')
        if confidence:
            parts.append(f"conf={confidence:.2f}")
        self.log("  ".join(parts))
        self._write_structured_record(
            event_type="trigger_fired",
            payload={
                "trigger_type": trigger_type,
                "speaker": speaker,
                "text": text,
                "confidence": confidence,
            },
        )

    def event(self, emoji: str, message: str):
        """Log a general event (startup, shutdown, etc.)."""
        self.log(f"{emoji} {message}")
        self._write_structured_record(
            event_type="event",
            payload={
                "emoji": emoji,
                "message": message,
            },
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self):
        """Write session footer and close the file."""
        closed_at = datetime.now()
        duration = (closed_at - self._start_time).total_seconds()
        self._file.write("-" * 80 + "\n")
        self._file.write(
            f"  SESSION END: {closed_at.strftime('%Y-%m-%d %H:%M:%S')}"
            f"  (duration: {duration:.1f}s)\n"
        )
        self._file.write("-" * 80 + "\n\n")
        self._file.flush()
        self._write_structured_record(
            event_type="session_end",
            payload={
                "ended_at": closed_at.isoformat(),
                "duration_seconds": round(duration, 3),
            },
            occurred_at=closed_at,
        )
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
