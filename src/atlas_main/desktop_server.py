"""FastAPI bridge exposing Atlas agent capabilities for the desktop app."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import AtlasAgent
from .ollama import OllamaClient, OllamaError

LOGGER = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str


class ResetResponse(BaseModel):
    status: str


@dataclass
class DesktopMetrics:
    """Runtime metrics surfaced to the desktop UI."""

    start_time: float = field(default_factory=time.time)
    turns: int = 0
    total_latency: float = 0.0
    last_latency: float = 0.0
    notifications: List[Dict[str, str]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def register_turn(self, latency: float, message: str) -> None:
        self.turns += 1
        self.total_latency += latency
        self.last_latency = latency
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.timeline.append({
            "time": timestamp,
            "event": message,
            "status": "success",
        })
        self.timeline = self.timeline[-12:]

    def add_notification(self, msg: str, level: str = "info") -> None:
        timestamp = time.strftime("%H:%M", time.localtime())
        self.notifications.append({
            "type": level,
            "msg": msg,
            "time": timestamp,
        })
        self.notifications = self.notifications[-8:]

    def to_payload(self, memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        elapsed_hours = max((time.time() - self.start_time) / 3600.0, 1e-6)
        avg_latency = self.total_latency / self.turns if self.turns else 0.0
        # Lightweight heuristics so the UI looks alive without platform APIs
        cpu_level = min(96, 24 + int(self.turns * 3 + self.last_latency * 100))
        mem_level = min(92, 40 + int(self.turns * 2))
        net_level = min(99, 65 + int(self.last_latency * 120))
        tasks = max(8, 6 + self.turns)
        data_processed = round(0.1 * self.turns, 2)
        return {
            "metrics": {
                "cpu": cpu_level,
                "memory": mem_level,
                "network": net_level,
                "tasks": tasks,
            },
            "quick_stats": {
                "sessions_today": 1,
                "commands_executed": self.turns,
                "data_processed_gb": data_processed,
                "avg_response_time": round(avg_latency, 3),
            },
            "notifications": list(reversed(self.notifications[-3:])),
            "timeline": list(reversed(self.timeline[-6:])),
            "memory": memory_stats,
        }


class DesktopSession:
    """Manages a single Atlas agent instance for desktop interactions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._client: Optional[OllamaClient] = None
        self._agent: Optional[AtlasAgent] = None
        self._metrics = DesktopMetrics()
        self._last_memory_stats: Dict[str, Any] = {}
        self._available = False
        self._startup_error: Optional[str] = None
        self._init_agent()

    def _init_agent(self) -> None:
        try:
            self._client = OllamaClient()
            self._agent = AtlasAgent(self._client)
            self._available = True
            self._metrics.add_notification("Desktop bridge connected to Atlas core", "success")
        except Exception as exc:  # pragma: no cover - depends on local setup
            self._startup_error = str(exc)
            LOGGER.warning("Failed to initialise Atlas agent for desktop bridge: %s", exc)
            self._available = False
            self._metrics.add_notification("Atlas core unavailable", "warn")

    async def send_message(self, message: str) -> Dict[str, Any]:
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        async with self._lock:
            if not self._available or self._agent is None:
                return {
                    "response": "Atlas core is unavailable. Ensure Ollama is running and restart the desktop bridge.",
                    "objective": None,
                    "tags": [],
                    "memory": self._last_memory_stats,
                    "error": self._startup_error or "unavailable",
                }

            started = time.perf_counter()

            def run_turn() -> Dict[str, Any]:
                buffer: List[str] = []

                def stream_chunk(chunk: str) -> None:
                    buffer.append(chunk)

                try:
                    final_text = self._agent.respond(message, stream_callback=stream_chunk)
                except OllamaError as exc:  # pragma: no cover - network/env specific
                    LOGGER.error("Ollama error during desktop turn: %s", exc)
                    return {
                        "response": f"Ollama error: {exc}",
                        "objective": None,
                        "tags": [],
                        "memory": self._last_memory_stats,
                        "error": "ollama_error",
                    }
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.exception("Unexpected error running Atlas turn")
                    return {
                        "response": f"Atlas encountered an unexpected error: {exc}",
                        "objective": None,
                        "tags": [],
                        "memory": self._last_memory_stats,
                        "error": "internal_error",
                    }

                text = final_text or "".join(buffer)
                memory = self._collect_memory_stats()
                objective_raw = getattr(self._agent, "last_objective", None)
                objective = objective_raw() if callable(objective_raw) else objective_raw
                tags_raw = getattr(self._agent, "last_tags", None)
                tags_value = tags_raw() if callable(tags_raw) else tags_raw
                return {
                    "response": text,
                    "objective": objective,
                    "tags": list(tags_value or []),
                    "memory": memory,
                    "error": None,
                }

            result = await asyncio.to_thread(run_turn)
            latency = time.perf_counter() - started
            self._metrics.register_turn(latency, f"Processed user prompt ({len(message)} chars)")
            self._last_memory_stats = result.get("memory", {})
            if result.get("error"):
                self._metrics.add_notification(result["response"], "warn")
            else:
                self._metrics.add_notification("Turn completed successfully", "success")
            return result

    async def reset(self) -> Dict[str, Any]:
        async with self._lock:
            self._close_agent()
            self._init_agent()
            self._last_memory_stats = {}
            self._metrics.add_notification("Desktop session reset", "info")
            return {"status": "reset"}

    def _collect_memory_stats(self) -> Dict[str, Any]:
        agent = self._agent
        if agent is None:
            return {}
        stats: Dict[str, Any] = {}
        try:
            snapshot_fn = getattr(agent, "_memory_stats_snapshot", None)
            if callable(snapshot_fn):
                stats = snapshot_fn()
            else:
                working = getattr(agent, "working_memory", None)
                if working is not None and hasattr(working, "get_stats"):
                    stats["working_memory"] = working.get_stats()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Failed to gather memory stats")
        return stats

    def metrics_payload(self) -> Dict[str, Any]:
        return self._metrics.to_payload(self._last_memory_stats)

    def _close_agent(self) -> None:
        try:
            if self._agent is not None:
                self._agent.close()
        finally:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._agent = None
            self._client = None

    async def shutdown(self) -> None:
        async with self._lock:
            self._close_agent()


session = DesktopSession()

app = FastAPI(title="Atlas Desktop Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    await session.shutdown()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return session.metrics_payload()


@app.post("/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    return await session.send_message(request.message)


@app.post("/session/reset")
async def reset_session() -> ResetResponse:
    await session.reset()
    return ResetResponse(status="reset")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Atlas desktop bridge server")
    parser.add_argument("--host", default=os.getenv("ATLAS_DESKTOP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ATLAS_DESKTOP_PORT", "5175")))
    parser.add_argument("--log-level", default=os.getenv("ATLAS_DESKTOP_LOG", "info"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - runtime path
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = {
        "app": "atlas_main.desktop_server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit("uvicorn is required to run the desktop server") from exc

    def _handle_sigterm(signum, frame) -> None:  # noqa: ANN001 - signal handler signature
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    uvicorn.run(**config)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main(sys.argv[1:])
