from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __package__:
    from .agent import AtlasAgent
    from .ollama import OllamaClient
else:  # pragma: no cover - support `python websocket_server.py`
    import sys
    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))

    from atlas_main.agent import AtlasAgent
    from atlas_main.ollama import OllamaClient

DEFAULT_MEMORY_EVENTS: List[Dict[str, str]] = []

DEFAULT_TOOL_RUNS: List[Dict[str, str]] = []

DEFAULT_TOPIC_DISTRIBUTION: List[Dict[str, int]] = []

DEFAULT_TOOL_USAGE: List[Dict[str, int]] = []

DEFAULT_CONTEXT_USAGE = {"current": 0, "max": 0, "percentage": 0}
DEFAULT_MEMORY_LAYERS = {"episodes": 0, "facts": 0, "insights": 0}
DEFAULT_ATLAS_METRICS = {"tokens": 0, "operations": 0, "inference": 0}
DEFAULT_PROCESSES: List[Dict[str, int]] = []
DEFAULT_FILE_ACCESS: List[Dict[str, str]] = []


class AtlasMetricsCollector:
    """Track agent activity and translate it into UI-facing metrics."""

    def __init__(self, agent: AtlasAgent) -> None:
        self.agent = agent
        self.lock = Lock()
        self.command_count = 0
        self.total_inference_seconds = 0.0
        self.memory_events: Deque[Dict[str, str]] = deque(DEFAULT_MEMORY_EVENTS, maxlen=50)
        self.tool_runs: Deque[Dict[str, str]] = deque(DEFAULT_TOOL_RUNS, maxlen=20)
        self.file_access: Deque[Dict[str, str]] = deque(DEFAULT_FILE_ACCESS, maxlen=20)
        self.tool_usage_counter = Counter({item["tool"]: item["count"] for item in DEFAULT_TOOL_USAGE})
        self.topic_counter = Counter({item["topic"]: item["percentage"] for item in DEFAULT_TOPIC_DISTRIBUTION})
        self.processes = [dict(proc) for proc in DEFAULT_PROCESSES]
        self.last_context_usage = self._initial_context_usage()
        self.last_memory_layers = self._compute_memory_layers()
        self.last_metrics = self._initial_atlas_metrics()
        self._next_tool_id = 6000

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        with self.lock:
            if event == "turn_start":
                objective = payload.get("objective") or "general focus"
                detail = f"Objective: {objective}"
                self._append_event("TURN", detail, timestamp)
                context = payload.get("context_usage")
                if isinstance(context, dict):
                    self._update_context_usage(context)
            elif event == "status":
                message = str(payload.get("message", "")).strip()
                if message:
                    self._append_event("STATUS", message, timestamp)
            elif event == "working_memory_eviction":
                evicted = int(payload.get("evicted_count", 0))
                detail = f"Evicted {evicted} messages from working memory."
                self._append_event("EVICTION", detail, timestamp)
            elif event == "tool_start":
                tools = payload.get("tools") or []
                for tool in tools:
                    name = str(tool.get("name", "tool")).strip() or "tool"
                    self._append_event("TOOL", f"Running {name}", timestamp)
            elif event == "tool_result":
                tool_name = str(payload.get("name", "tool")).strip() or "tool"
                arguments = payload.get("arguments") or {}
                output = str(payload.get("output", "")).strip()
                self._record_tool_usage(tool_name, output, timestamp, arguments)
                self._append_event("TOOL", f"Completed {tool_name}", timestamp)
            elif event == "tool_limit":
                attempted = int(payload.get("attempted", 0))
                maximum = payload.get("max")
                detail = f"Tool limit reached ({attempted}/{maximum})."
                self._append_event("LIMIT", detail, timestamp)
            elif event == "tool_deferred":
                tools = payload.get("tools") or []
                if tools:
                    detail = "Deferred tools: " + ", ".join(str(t) for t in tools)
                    self._append_event("FOCUS", detail, timestamp)
            elif event == "turn_complete":
                summary = str(payload.get("text", "")).strip()
                if summary:
                    preview = summary.splitlines()[0][:140]
                    self._append_event("TURN", f"Reply: {preview}", timestamp)
                memory_stats = payload.get("memory_stats") or {}
                if memory_stats:
                    self._update_memory_layers_from_stats(memory_stats)
                    working = memory_stats.get("working_memory")
                    if isinstance(working, dict):
                        self._update_context_usage(working)
                for raw_tag in payload.get("tags", []) or []:
                    tag = self._normalize_topic(str(raw_tag))
                    self.topic_counter[tag] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            tokens = self._current_token_usage()
            if tokens is not None:
                self.last_metrics["tokens"] = tokens
            self.last_metrics["operations"] = self.command_count
            self.last_metrics["inference"] = self._average_inference_ms()

            self.last_memory_layers = self._compute_memory_layers()

            tool_usage = [
                {"tool": name, "count": int(count)}
                for name, count in self.tool_usage_counter.most_common()
            ] or list(DEFAULT_TOOL_USAGE)

            snapshot = {
                "atlas": dict(self.last_metrics),
                "memoryLayers": dict(self.last_memory_layers),
                "contextUsage": dict(self.last_context_usage),
                "memoryEvents": list(self.memory_events),
                "toolRuns": list(self.tool_runs),
                "topicDistribution": self._topic_distribution(),
                "toolUsage": tool_usage,
                "processes": [dict(proc) for proc in self.processes],
                "fileAccess": list(self.file_access),
            }
            return snapshot

    def record_command_result(
        self,
        command: str,
        response: str,
        duration: float,
        *,
        success: bool,
    ) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        with self.lock:
            self.command_count += 1
            self.total_inference_seconds += max(duration, 0.0)
            status = "OK" if success else "ERR"
            detail = f"[{status}] {command.strip()}"
            self._append_event("COMMAND", detail, timestamp)
            self._maybe_update_processes(duration)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initial_context_usage(self) -> Dict[str, int]:
        working = getattr(self.agent, "working_memory", None)
        if working is not None:
            try:
                stats = working.get_stats()
            except Exception:
                stats = None
            if isinstance(stats, dict):
                return self._context_from_stats(stats)
        budget = self._token_budget()
        max_k = max(0, round(budget / 1000)) if budget else 0
        return {"current": 0, "max": max_k, "percentage": 0}

    def _initial_atlas_metrics(self) -> Dict[str, int]:
        metrics = dict(DEFAULT_ATLAS_METRICS)
        tokens = self._current_token_usage()
        if tokens is not None:
            metrics["tokens"] = tokens
        metrics["operations"] = self.command_count
        metrics["inference"] = self._average_inference_ms()
        return metrics

    def _append_event(self, event_type: str, detail: str, timestamp: Optional[str] = None) -> None:
        if not detail:
            return
        entry = {
            "time": timestamp or datetime.now().strftime("%H:%M"),
            "type": event_type.upper(),
            "detail": detail,
        }
        self.memory_events.appendleft(entry)

    def _update_context_usage(self, snapshot: Dict[str, Any]) -> None:
        self.last_context_usage = self._context_from_stats(snapshot)

    def _context_from_stats(self, stats: Dict[str, Any]) -> Dict[str, int]:
        tokens = int(stats.get("tokens", 0))
        token_budget = int(stats.get("token_budget", 0) or stats.get("token_limit", 0) or 0)
        if token_budget <= 0:
            token_budget = self._token_budget()
        current_k = max(0, round(tokens / 1000))
        max_k = max(0, round(token_budget / 1000)) if token_budget else 0
        pct = stats.get("token_pct")
        if isinstance(pct, (int, float)):
            percentage = max(0, min(100, int(pct)))
        elif token_budget:
            percentage = max(0, min(100, int((tokens / token_budget) * 100)))
        else:
            percentage = 0
        return {"current": current_k, "max": max_k, "percentage": percentage}

    def _update_memory_layers_from_stats(self, stats: Dict[str, Any]) -> None:
        episodes = int(stats.get("episodic_count", 0))
        facts = int(stats.get("semantic_count", 0))
        insights = int(stats.get("reflections_count", 0))
        if episodes or facts or insights:
            self.last_memory_layers = {
                "episodes": episodes,
                "facts": facts,
                "insights": insights,
            }

    def _compute_memory_layers(self) -> Dict[str, int]:
        layers = {"episodes": 0, "facts": 0, "insights": 0}
        layered = getattr(self.agent, "layered_memory", None)
        if layered is None:
            return layers

        episodic = getattr(layered, "episodic", None)
        if episodic is not None:
            try:
                conn = getattr(episodic, "_conn", None)
                if conn is not None:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM episodes")
                    row = cur.fetchone()
                    if row:
                        layers["episodes"] = int(row[0] or 0)
            except Exception:
                pass

        semantic = getattr(layered, "semantic", None)
        if semantic is not None:
            try:
                layers["facts"] = len(getattr(semantic, "_facts", []) or [])
            except Exception:
                pass

        reflections = getattr(layered, "reflections", None)
        if reflections is not None:
            try:
                layers["insights"] = len(reflections.all())
            except Exception:
                pass

        return layers

    def _record_tool_usage(
        self,
        tool_name: str,
        output: str,
        timestamp: str,
        arguments: Dict[str, Any],
    ) -> None:
        clean_name = tool_name.strip() or "tool"
        self.tool_usage_counter[clean_name] += 1
        self._next_tool_id += 1
        summary = output.splitlines()[0] if output else "(tool returned no output)"
        summary = summary[:160]
        label = clean_name.replace("_", " ").title()
        self.tool_runs.appendleft(
            {
                "id": f"tool-{self._next_tool_id}",
                "name": label,
                "summary": summary,
                "time": timestamp,
            }
        )
        self._maybe_record_file_access(clean_name, arguments, timestamp)

    def _maybe_record_file_access(self, tool_name: str, arguments: Dict[str, Any], timestamp: str) -> None:
        if tool_name not in {"read_file", "write_file", "list_directory"}:
            return
        path = str(arguments.get("path") or arguments.get("directory") or "").strip()
        if not path:
            return
        action = "READ"
        if tool_name == "write_file":
            action = "WRITE"
        elif tool_name == "list_directory":
            action = "LIST"
        entry = {"path": path, "action": action, "time": timestamp}
        self.file_access.appendleft(entry)

    def _maybe_update_processes(self, duration: float) -> None:
        if not self.processes:
            self.processes = [dict(proc) for proc in DEFAULT_PROCESSES]
        agent_entry = None
        for proc in self.processes:
            if proc.get("name") == "atlas-agent":
                agent_entry = proc
                break
        if agent_entry is None:
            agent_entry = {"name": "atlas-agent", "cpu": 24, "mem": 512}
            self.processes.insert(0, agent_entry)
        agent_entry["cpu"] = max(5, min(95, agent_entry.get("cpu", 0) + int(duration * 40) + 5))
        agent_entry["mem"] = max(128, min(4096, agent_entry.get("mem", 256) + 16))

    def _topic_distribution(self) -> List[Dict[str, int]]:
        if not self.topic_counter:
            return list(DEFAULT_TOPIC_DISTRIBUTION)
        total = sum(self.topic_counter.values())
        if total <= 0:
            return list(DEFAULT_TOPIC_DISTRIBUTION)
        entries: List[Dict[str, int]] = []
        for name, count in self.topic_counter.most_common(6):
            percentage = int(round((count / total) * 100))
            entries.append({"topic": name, "percentage": max(1, percentage)})
        overflow = sum(item["percentage"] for item in entries) - 100
        if overflow > 0 and entries:
            entries[0]["percentage"] = max(1, entries[0]["percentage"] - overflow)
        return entries

    def _average_inference_ms(self) -> int:
        if self.command_count <= 0:
            return DEFAULT_ATLAS_METRICS["inference"]
        avg = (self.total_inference_seconds / self.command_count) * 1000.0
        return max(1, int(avg))

    def _current_token_usage(self) -> Optional[int]:
        working = getattr(self.agent, "working_memory", None)
        if working is None:
            return None
        try:
            stats = working.get_stats()
        except Exception:
            return None
        tokens = int(stats.get("tokens", 0))
        self._update_context_usage(stats)
        return tokens

    def _token_budget(self) -> int:
        working = getattr(self.agent, "working_memory", None)
        if working is None:
            return DEFAULT_CONTEXT_USAGE["max"] * 1000
        config = getattr(working, "config", None)
        budget = getattr(config, "token_budget", None)
        if isinstance(budget, int) and budget > 0:
            return budget
        return DEFAULT_CONTEXT_USAGE["max"] * 1000

    @staticmethod
    def _normalize_topic(tag: str) -> str:
        cleaned = tag.lower().strip()
        if cleaned.startswith("objective:"):
            cleaned = cleaned.split(":", 1)[1]
        cleaned = cleaned.replace("tool:", "")
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        words = [word for word in cleaned.split() if word]
        return " ".join(words).title() or "General"


class AtlasWebSocketServer:
    """Expose the Atlas agent over a lightweight WebSocket protocol."""

    def __init__(self, host: str = "localhost", port: int = 8765, *, agent: Optional[AtlasAgent] = None) -> None:
        self.host = host
        self.port = port
        self.client = OllamaClient()
        self.agent = agent or AtlasAgent(self.client)
        self.collector = AtlasMetricsCollector(self.agent)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False

    async def handle_message(self, websocket: WebSocketServerProtocol, raw: str) -> None:
        logger.debug(f"Received message: {raw[:100]}...")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            await self._send_error(websocket, "Invalid JSON payload")
            return

        msg_type = data.get("type")
        logger.info(f"Processing message type: {msg_type}")

        if msg_type == "command":
            payload = data.get("payload", "")
            if not isinstance(payload, str):
                await self._send_error(websocket, "Command payload must be a string")
                return
            logger.info(f"Executing command: {payload}")
            await self._execute_command_streaming(websocket, payload)
            metrics = self.collector.snapshot()
            await websocket.send(json.dumps({"type": "metrics", "payload": metrics}))
        elif msg_type == "set_model":
            payload = data.get("payload", "")
            if not isinstance(payload, str):
                await self._send_error(websocket, "Model payload must be a string")
                return
            logger.info(f"Setting model to: {payload}")
            self.agent.set_chat_model(payload)
            await websocket.send(json.dumps({"type": "response", "payload": f"Model switched to {payload}"}))
        elif msg_type == "get_metrics":
            logger.debug("Sending metrics snapshot")
            metrics = self.collector.snapshot()
            await websocket.send(json.dumps({"type": "metrics", "payload": metrics}))
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}" if websocket.remote_address else "unknown"
        logger.info(f"Client connected: {client_info}")

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed from {client_info}: code={e.code}, reason={e.reason}")
            return
        except Exception as e:
            logger.error(f"Error handling connection from {client_info}: {e}", exc_info=True)
            return
        finally:
            logger.info(f"Client disconnected: {client_info}")

    async def start(self) -> None:
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"ATLAS WebSocket server running on ws://{self.host}:{self.port}")
            print(f"ATLAS WebSocket server running on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                logger.info("Server shutdown requested")
                pass

    async def _execute_command(self, command: str) -> str:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()

        def run_agent() -> str:
            return self.agent.respond(command, event_callback=self.collector.handle_event)

        try:
            response = await loop.run_in_executor(self._executor, run_agent)
            success = True
        except Exception as exc:  # noqa: BLE001 - surface agent failures
            response = f"Error: {exc}"
            success = False
        duration = time.perf_counter() - start
        self.collector.record_command_result(command, response, duration, success=success)
        return response

    async def _execute_command_streaming(self, websocket: WebSocketServerProtocol, command: str) -> None:
        """Execute command and stream response chunks with Jarvis prefix."""
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        full_response = []
        sent_prefix = False

        def run_agent_streaming():
            """Generator that yields response chunks from agent."""
            response_text = self.agent.respond(command, event_callback=self.collector.handle_event)
            # Split response into chunks for streaming effect
            words = response_text.split()
            for i in range(0, len(words), 3):  # Send 3 words at a time
                chunk = " ".join(words[i:i+3])
                if i + 3 < len(words):
                    chunk += " "
                yield chunk
            return response_text

        try:
            # Send initial "Jarvis: " prefix
            await websocket.send(json.dumps({
                "type": "response_chunk",
                "payload": "Jarvis: ",
                "is_final": False
            }))
            sent_prefix = True

            # Stream the response in chunks
            for chunk in await loop.run_in_executor(self._executor, lambda: list(run_agent_streaming())):
                full_response.append(chunk)
                await websocket.send(json.dumps({
                    "type": "response_chunk",
                    "payload": chunk,
                    "is_final": False
                }))

            # Send final marker
            await websocket.send(json.dumps({
                "type": "response_chunk",
                "payload": "",
                "is_final": True
            }))
            success = True
        except Exception as exc:  # noqa: BLE001 - surface agent failures
            error_msg = f"Error: {exc}"
            await websocket.send(json.dumps({
                "type": "response_chunk",
                "payload": error_msg if sent_prefix else f"Jarvis: {error_msg}",
                "is_final": True
            }))
            full_response = [error_msg]
            success = False

        duration = time.perf_counter() - start
        response_text = "".join(full_response)
        self.collector.record_command_result(command, response_text, duration, success=success)

    async def _send_error(self, websocket: WebSocketServerProtocol, message: str) -> None:
        logger.error(f"Sending error to client: {message}")
        await websocket.send(json.dumps({"type": "error", "payload": message}))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.agent.close()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass
        self._executor.shutdown(wait=False)


def run(host: str = "localhost", port: int = 8765) -> None:
    server = AtlasWebSocketServer(host=host, port=port)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Stopping Atlas WebSocket server...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    run()
