"""Local AMI labeling UI.

Run with:

    python tools/ami_labeling_app.py

Then open http://127.0.0.1:8767 in a browser.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from aiohttp import web


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling import AmiDataset, AmiTurn, AmiWindow, WindowSuggestionEngine  # noqa: E402


TRIGGER_TYPES = [
    "repeated_confusion",
    "memory_gap",
    "contradiction",
    "factual_inconsistency",
    "collaboration_opportunity",
    "decision_conflict",
    "topic_drift",
    "circular_discussion",
]


class LabelStore:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.state_path = base_path.with_suffix(".json")
        self.jsonl_path = base_path.with_suffix(".jsonl")
        self.csv_path = base_path.with_suffix(".csv")
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self._labels: Dict[str, Dict[str, Any]] = {}
        self._load()
        if not self.state_path.exists() or not self.jsonl_path.exists() or not self.csv_path.exists():
            self._persist()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._labels = {label["example_id"]: label for label in payload.get("labels", [])}

    def all_labels(self) -> Dict[str, Dict[str, Any]]:
        return dict(sorted(self._labels.items()))

    def get(self, example_id: str) -> Optional[Dict[str, Any]]:
        return self._labels.get(example_id)

    def upsert(self, label: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._labels.get(label["example_id"])
        label["created_at"] = existing.get("created_at") if existing else now
        label["updated_at"] = now
        self._labels[label["example_id"]] = label
        self._persist()
        return label

    def _persist(self) -> None:
        ordered_labels = [self._labels[key] for key in sorted(self._labels)]
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "labels": ordered_labels,
        }
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for label in ordered_labels:
                handle.write(json.dumps(label, ensure_ascii=True) + "\n")

        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "example_id",
                    "meeting_id",
                    "window_index",
                    "human_label",
                    "should_trigger",
                    "trigger_type",
                    "model_suggestion_label",
                    "model_suggestion_trigger_type",
                    "model_suggestion_confidence",
                    "model_suggestion_reason",
                    "model_suggestion_source",
                    "topic",
                    "start_time",
                    "end_time",
                    "notes",
                    "transcript_text",
                    "created_at",
                    "updated_at",
                ],
            )
            writer.writeheader()
            for label in ordered_labels:
                transcript_text = "\n".join(
                    f"{turn['speaker']}: {turn['text']}" for turn in label.get("transcript_window", [])
                )
                writer.writerow(
                    {
                        "example_id": label.get("example_id"),
                        "meeting_id": label.get("meeting_id"),
                        "window_index": label.get("window_index"),
                        "human_label": label.get("human_label"),
                        "should_trigger": label.get("should_trigger"),
                        "trigger_type": label.get("trigger_type"),
                        "model_suggestion_label": (label.get("model_suggestion") or {}).get("human_label"),
                        "model_suggestion_trigger_type": (label.get("model_suggestion") or {}).get("trigger_type"),
                        "model_suggestion_confidence": (label.get("model_suggestion") or {}).get("confidence"),
                        "model_suggestion_reason": (label.get("model_suggestion") or {}).get("reason"),
                        "model_suggestion_source": (label.get("model_suggestion") or {}).get("source"),
                        "topic": label.get("topic"),
                        "start_time": label.get("start_time"),
                        "end_time": label.get("end_time"),
                        "notes": label.get("notes"),
                        "transcript_text": transcript_text,
                        "created_at": label.get("created_at"),
                        "updated_at": label.get("updated_at"),
                    }
                )


def build_app(dataset_root: Path, label_base_path: Path) -> web.Application:
    app = web.Application()
    dataset = AmiDataset(dataset_root)
    label_store = LabelStore(label_base_path)
    suggestion_engine = WindowSuggestionEngine()
    html_path = Path(__file__).with_name("ami_labeling_ui.html")

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

    async def meetings(_: web.Request) -> web.Response:
        return web.json_response({"meetings": dataset.list_meetings()})

    async def meeting(request: web.Request) -> web.Response:
        meeting_id = request.query.get("meeting_id")
        if not meeting_id:
            raise web.HTTPBadRequest(text="meeting_id is required")

        window_size = max(3, int(request.query.get("window_size", "6")))
        step = max(1, int(request.query.get("step", "3")))
        windows = dataset.build_windows(meeting_id, window_size=window_size, step=step)
        labels = {window.example_id: label_store.get(window.example_id) for window in windows}
        labeled_count = sum(1 for value in labels.values() if value)
        return web.json_response(
            {
                "meeting_id": meeting_id,
                "window_size": window_size,
                "step": step,
                "total_windows": len(windows),
                "labeled_count": labeled_count,
                "labels": labels,
                "windows": [window.to_dict() for window in windows],
            }
        )

    async def save_label(request: web.Request) -> web.Response:
        payload = await request.json()
        human_label = payload.get("human_label")
        if human_label not in {"should_trigger", "should_not_trigger", "skip"}:
            raise web.HTTPBadRequest(text="human_label must be should_trigger, should_not_trigger, or skip")

        if human_label == "should_trigger" and not payload.get("trigger_type"):
            raise web.HTTPBadRequest(text="trigger_type is required when human_label is should_trigger")

        record = {
            "source": "ami_manual_labeler",
            "meeting_id": payload.get("meeting_id"),
            "example_id": payload.get("example_id"),
            "window_index": payload.get("window_index"),
            "human_label": human_label,
            "should_trigger": human_label == "should_trigger",
            "trigger_type": payload.get("trigger_type") if human_label == "should_trigger" else "none",
            "model_suggestion": payload.get("model_suggestion"),
            "topic": payload.get("topic"),
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "weak_signals": payload.get("weak_signals", {}),
            "transcript_window": payload.get("transcript_window", []),
            "notes": payload.get("notes", "").strip(),
        }
        label_store.upsert(record)
        return web.json_response({"ok": True, "record": record})

    async def suggest_label(request: web.Request) -> web.Response:
        payload = await request.json()
        turns = payload.get("transcript_window", [])
        if not turns:
            raise web.HTTPBadRequest(text="transcript_window is required")

        window = AmiWindow(
            meeting_id=str(payload.get("meeting_id")),
            example_id=str(payload.get("example_id")),
            window_index=int(payload.get("window_index", 0)),
            start_time=float(payload.get("start_time", 0.0)),
            end_time=float(payload.get("end_time", 0.0)),
            topic=payload.get("topic"),
            transcript_window=[
                AmiTurn(
                    meeting_id=str(payload.get("meeting_id")),
                    speaker=str(turn.get("speaker", "?")),
                    start_time=float(turn.get("start_time", 0.0)),
                    end_time=float(turn.get("end_time", 0.0)),
                    text=str(turn.get("text", "")),
                    dialogue_acts=list(turn.get("dialogue_acts", [])),
                    adjacency_types=list(turn.get("adjacency_types", [])),
                    addressee=turn.get("addressee"),
                    topic=turn.get("topic"),
                )
                for turn in turns
            ],
            weak_signals=payload.get("weak_signals", {}),
        )
        suggestion = await suggestion_engine.suggest_window(window)
        return web.json_response({"suggestion": suggestion.to_dict()})

    async def export_jsonl(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(path=label_store.jsonl_path)

    async def export_csv(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(path=label_store.csv_path)

    async def status(_: web.Request) -> web.Response:
        labels = label_store.all_labels()
        return web.json_response(
            {
                "label_count": len(labels),
                "state_path": str(label_store.state_path),
                "jsonl_path": str(label_store.jsonl_path),
                "csv_path": str(label_store.csv_path),
                "trigger_types": TRIGGER_TYPES,
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/api/meetings", meetings)
    app.router.add_get("/api/meeting", meeting)
    app.router.add_post("/api/label", save_label)
    app.router.add_post("/api/suggest", suggest_label)
    app.router.add_get("/api/export/jsonl", export_jsonl)
    app.router.add_get("/api/export/csv", export_csv)
    app.router.add_get("/api/status", status)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AMI labeling UI")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "docs" / "ami_public_manual_1.6.2"),
        help="Path to the unpacked AMI XML dataset directory.",
    )
    parser.add_argument(
        "--label-base-path",
        default=str(REPO_ROOT / "labels" / "ami_manual_labels"),
        help="Base path for saved labels. The app writes .json, .jsonl, and .csv files.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(Path(args.dataset_root), Path(args.label_base_path))
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()