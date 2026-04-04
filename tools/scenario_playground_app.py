"""Upload JSON conversations and replay them through the real trigger pipeline."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import uuid

from aiohttp import web


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation import ScenarioPayloadError, ScenarioSimulator, normalize_scenario_payload  # noqa: E402


class SimulationJobStore:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    def create_job(
        self,
        *,
        original_payload: Dict[str, Any],
        normalized_scenario: Dict[str, Any],
        playback_speed: float,
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "meeting_id": normalized_scenario.get("meeting_id"),
            "description": normalized_scenario.get("description"),
            "playback_speed": playback_speed,
            "input_event_count": len(normalized_scenario.get("events") or []),
            "events": [],
            "summary": None,
            "error": None,
            "source_format": _detect_source_format(original_payload),
        }
        self._jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(
            self._run_job(job_id, normalized_scenario, playback_speed),
            name=f"scenario-playground:{job_id}",
        )
        return self._snapshot(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return self._snapshot(job)

    def get_job_events(self, job_id: str, after_index: int = 0) -> Optional[list[Dict[str, Any]]]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job["events"][max(0, after_index):]

    def list_jobs(self) -> list[Dict[str, Any]]:
        jobs = [self._snapshot(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda item: item["created_at"], reverse=True)

    async def _run_job(self, job_id: str, normalized_scenario: Dict[str, Any], playback_speed: float) -> None:
        job = self._jobs[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        simulator = ScenarioSimulator()

        def on_event(event: Dict[str, Any]) -> None:
            event_with_index = {
                "index": len(job["events"]),
                **event,
            }
            job["events"].append(event_with_index)

        result = await simulator.run(
            normalized_scenario,
            playback_speed=playback_speed,
            on_event=on_event,
        )
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        job["summary"] = result.to_dict()
        job["error"] = result.error
        job["status"] = "failed" if result.error else "completed"

        if len(self._jobs) > 20:
            for stale_job_id in list(self._jobs)[:-20]:
                self._jobs.pop(stale_job_id, None)
                task = self._tasks.pop(stale_job_id, None)
                if task and not task.done():
                    task.cancel()

    @staticmethod
    def _snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if key != "events"
        }


def _detect_source_format(payload: Any) -> str:
    if isinstance(payload, list):
        return "conversation_list"
    if isinstance(payload, dict) and "events" in payload:
        return "e2e_style_events"
    if isinstance(payload, dict):
        return "conversation_object"
    return "unknown"


async def _parse_run_request(request: web.Request) -> tuple[Dict[str, Any], int, int, float]:
    payload: Optional[Any] = None
    default_delay_ms = 1200
    final_wait_ms: Optional[int] = None
    playback_speed = 1.0

    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        async for part in reader:
            if part.name == "json_file":
                file_bytes = await part.read()
                if file_bytes:
                    payload = json.loads(file_bytes.decode("utf-8"))
            elif part.name == "json_text":
                text = (await part.text()).strip()
                if text:
                    payload = json.loads(text)
            elif part.name == "default_delay_ms":
                default_delay_ms = int((await part.text()).strip() or "1200")
            elif part.name == "final_wait_ms":
                raw = (await part.text()).strip()
                final_wait_ms = int(raw) if raw else None
            elif part.name == "playback_speed":
                playback_speed = float((await part.text()).strip() or "1.0")

    else:
        body = await request.json()
        payload = body.get("payload", body)
        default_delay_ms = int(body.get("default_delay_ms", 1200))
        if body.get("final_wait_ms") is not None:
            final_wait_ms = int(body["final_wait_ms"])
        playback_speed = float(body.get("playback_speed", 1.0))

    if payload is None:
        raise web.HTTPBadRequest(text="Provide either a JSON file upload or pasted JSON text")

    return payload, default_delay_ms, (final_wait_ms if final_wait_ms is not None else -1), playback_speed


def build_app() -> web.Application:
    app = web.Application()
    job_store = SimulationJobStore()
    html_path = Path(__file__).with_name("scenario_playground_ui.html")

    async def index(_: web.Request) -> web.Response:
        return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")

    async def run_scenario(request: web.Request) -> web.Response:
        try:
            payload, default_delay_ms, final_wait_ms, playback_speed = await _parse_run_request(request)
            normalized = normalize_scenario_payload(payload, default_delay_ms=default_delay_ms)
            if final_wait_ms >= 0:
                normalized["final_wait_ms"] = final_wait_ms
            job = job_store.create_job(
                original_payload=payload if isinstance(payload, dict) else {"payload": payload},
                normalized_scenario=normalized,
                playback_speed=playback_speed,
            )
            return web.json_response({"job": job, "normalized_scenario": normalized})
        except ScenarioPayloadError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(text=f"Invalid JSON: {exc.msg}") from exc

    async def get_job(request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        after_index = int(request.query.get("after_index", "0"))
        job = job_store.get_job(job_id)
        events = job_store.get_job_events(job_id, after_index=after_index)
        if not job or events is None:
            raise web.HTTPNotFound(text="job not found")
        return web.json_response({"job": job, "events": events})

    async def jobs(_: web.Request) -> web.Response:
        return web.json_response({"jobs": job_store.list_jobs()})

    async def status(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "job_count": len(job_store.list_jobs()),
                "hosted_at": "http://127.0.0.1:8769",
                "supported_formats": [
                    "e2e_style_events",
                    "conversation_object",
                    "conversation_list",
                ],
            }
        )

    app.router.add_get("/", index)
    app.router.add_post("/api/run", run_scenario)
    app.router.add_get("/api/jobs", jobs)
    app.router.add_get("/api/jobs/{job_id}", get_job)
    app.router.add_get("/api/status", status)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the scenario playground UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8769, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()