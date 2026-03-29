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
from pathlib import Path
from typing import Optional

# Default log location
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "meeting_sessions.log"


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

    def bot_response(self, text: str, trigger_type: str = ""):
        """Log a bot response sent to chat."""
        extra = f" ({trigger_type})" if trigger_type else ""
        self.log(f"🤖 Sent{extra}: {text}")

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

    def event(self, emoji: str, message: str):
        """Log a general event (startup, shutdown, etc.)."""
        self.log(f"{emoji} {message}")

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self):
        """Write session footer and close the file."""
        duration = (datetime.now() - self._start_time).total_seconds()
        self._file.write("-" * 80 + "\n")
        self._file.write(
            f"  SESSION END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"  (duration: {duration:.1f}s)\n"
        )
        self._file.write("-" * 80 + "\n\n")
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
