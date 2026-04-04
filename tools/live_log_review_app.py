"""Review past meeting trigger logs in a local web UI.

Run with:

    python tools/live_log_review_app.py

Then open http://127.0.0.1:8768 in a browser.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from aiohttp import web


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling import LiveLogDataset  # noqa: E402


TRIGGER_TYPES = [
    "voice_address",
    "explicit_command",
    "repeated_confusion",
    "memory_gap",
    "contradiction",
    "factual_inconsistency",
    "collaboration_opportunity",
    "decision_conflict",
    "topic_drift",
    "circular_discussion",
]


class ReviewStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.state_path = base_path.with_suffix(".json")
        self.jsonl_path = base_path.with_suffix(".jsonl")
        self.csv_path = base_path.with_suffix(".csv")
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self._reviews: Dict[str, Dict[str, Any]] = {}
        self._load()
        if not self.state_path.exists() or not self.jsonl_path.exists() or not self.csv_path.exists():
            self._persist()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._reviews = {review["example_id"]: review for review in payload.get("reviews", [])}

    def all_reviews(self) -> Dict[str, Dict[str, Any]]:
        return dict(sorted(self._reviews.items()))

    def get(self, example_id: str) -> Optional[Dict[str, Any]]:
        return self._reviews.get(example_id)

    def upsert(self, review: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._reviews.get(review["example_id"])
        review["created_at"] = existing.get("created_at") if existing else now
        review["updated_at"] = now
        self._reviews[review["example_id"]] = review
        self._persist()
        return review

    def _persist(self) -> None:
        ordered_reviews = [self._reviews[key] for key in sorted(self._reviews)]
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reviews": ordered_reviews,
        }
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for review in ordered_reviews:
                handle.write(json.dumps(review, ensure_ascii=True) + "\n")

        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "example_id",
                    "meeting_id",
                    "trigger_index",
                    "event_time",
                    "human_label",
                    "should_trigger",
                    "trigger_type",
                    "predicted_trigger_type",
                    "predicted_confidence",
                    "predicted_reason",
                    "relevant_text",
                    "response_text",
                    "notes",
                    "transcript_text",
                    "created_at",
                    "updated_at",
                ],
            )
            writer.writeheader()
            for review in ordered_reviews:
                transcript_text = "\n".join(
                    f"{turn['speaker']}: {turn['text']}" for turn in review.get("transcript_window", [])
                )
                writer.writerow(
                    {
                        "example_id": review.get("example_id"),
                        "meeting_id": review.get("meeting_id"),
                        "trigger_index": review.get("trigger_index"),
                        "event_time": review.get("event_time"),
                        "human_label": review.get("human_label"),
                        "should_trigger": review.get("should_trigger"),
                        "trigger_type": review.get("trigger_type"),
                        "predicted_trigger_type": review.get("predicted_trigger_type"),
                        "predicted_confidence": review.get("predicted_confidence"),
                        "predicted_reason": review.get("predicted_reason"),
                        "relevant_text": review.get("relevant_text"),
                        "response_text": review.get("response_text"),
                        "notes": review.get("notes"),
                        "transcript_text": transcript_text,
                        "created_at": review.get("created_at"),
                        "updated_at": review.get("updated_at"),
                    }
                )


def build_app(log_path: Path, event_log_path: Path, label_base_path: Path) -> web.Application:
    app = web.Application()
    dataset = LiveLogDataset(log_path=log_path, event_log_path=event_log_path)
    review_store = ReviewStore(label_base_path)
    html_path = Path(__file__).with_name("live_log_review_ui.html")

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

    async def sessions(_: web.Request) -> web.Response:
        return web.json_response({"sessions": [session.to_dict() for session in dataset.list_sessions()]})

    async def session_reviews(request: web.Request) -> web.Response:
        meeting_id = request.query.get("meeting_id")
        if not meeting_id:
            raise web.HTTPBadRequest(text="meeting_id is required")

        reviews = dataset.build_reviews(meeting_id)
        labels = {review.example_id: review_store.get(review.example_id) for review in reviews}
        labeled_count = sum(1 for value in labels.values() if value)
        return web.json_response(
            {
                "meeting_id": meeting_id,
                "total_reviews": len(reviews),
                "labeled_count": labeled_count,
                "labels": labels,
                "reviews": [review.to_dict() for review in reviews],
            }
        )

    async def save_review(request: web.Request) -> web.Response:
        payload = await request.json()
        human_label = payload.get("human_label")
        if human_label not in {"should_trigger", "should_not_trigger", "skip"}:
            raise web.HTTPBadRequest(text="human_label must be should_trigger, should_not_trigger, or skip")

        if human_label == "should_trigger" and not payload.get("trigger_type"):
            raise web.HTTPBadRequest(text="trigger_type is required when human_label is should_trigger")

        record = {
            "source": "live_trigger_review",
            "meeting_id": payload.get("meeting_id"),
            "example_id": payload.get("example_id"),
            "trigger_index": payload.get("trigger_index"),
            "event_time": payload.get("event_time"),
            "human_label": human_label,
            "should_trigger": human_label == "should_trigger",
            "trigger_type": payload.get("trigger_type") if human_label == "should_trigger" else "none",
            "predicted_trigger_type": payload.get("predicted_trigger_type"),
            "predicted_confidence": payload.get("predicted_confidence"),
            "predicted_reason": payload.get("predicted_reason"),
            "relevant_text": payload.get("relevant_text"),
            "response_text": payload.get("response_text"),
            "transcript_window": payload.get("transcript_window", []),
            "notes": payload.get("notes", "").strip(),
        }
        review_store.upsert(record)
        return web.json_response({"ok": True, "record": record})

    async def export_jsonl(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(path=review_store.jsonl_path)

    async def export_csv(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(path=review_store.csv_path)

    async def status(_: web.Request) -> web.Response:
        reviews = review_store.all_reviews()
        sessions = dataset.list_sessions()
        return web.json_response(
            {
                "review_count": len(reviews),
                "session_count": len(sessions),
                "log_path": str(log_path),
                "event_log_path": str(event_log_path),
                "state_path": str(review_store.state_path),
                "jsonl_path": str(review_store.jsonl_path),
                "csv_path": str(review_store.csv_path),
                "trigger_types": TRIGGER_TYPES,
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/api/sessions", sessions)
    app.router.add_get("/api/session", session_reviews)
    app.router.add_post("/api/review", save_review)
    app.router.add_get("/api/export/jsonl", export_jsonl)
    app.router.add_get("/api/export/csv", export_csv)
    app.router.add_get("/api/status", status)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the past-session trigger review UI")
    parser.add_argument(
        "--log-path",
        default=str(REPO_ROOT / "logs" / "meeting_sessions.log"),
        help="Path to the human-readable meeting session log.",
    )
    parser.add_argument(
        "--event-log-path",
        default=str(REPO_ROOT / "logs" / "meeting_session_events.jsonl"),
        help="Path to the structured JSONL event log.",
    )
    parser.add_argument(
        "--label-base-path",
        default=str(REPO_ROOT / "labels" / "live_trigger_reviews"),
        help="Base path for saved past-session trigger review labels. The app writes .json, .jsonl, and .csv files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8768, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        log_path=Path(args.log_path),
        event_log_path=Path(args.event_log_path),
        label_base_path=Path(args.label_base_path),
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()