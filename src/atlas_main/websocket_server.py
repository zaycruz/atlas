from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List, Optional, Callable

import websockets
from websockets.server import WebSocketServerProtocol

from .kg_config import KnowledgeGraphConfig
from .knowledge_graph import SQLiteGraphStore
from .kg_pipeline import KnowledgeGraphPipeline, set_pipeline, clear_pipeline
from .kg_extractor import MemoryGraphExtractor

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
    from atlas_main.kg_config import KnowledgeGraphConfig
    from atlas_main.knowledge_graph import SQLiteGraphStore
    from atlas_main.kg_pipeline import KnowledgeGraphPipeline, set_pipeline, clear_pipeline
    from atlas_main.kg_extractor import MemoryGraphExtractor

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

    def __init__(
        self,
        agent: AtlasAgent,
        *,
        event_emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
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
        self._event_emitter = event_emitter

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        with self.lock:
            if event == "turn_start":
                context = payload.get("context_usage")
                if isinstance(context, dict):
                    self._update_context_usage(context)
            elif event == "status":
                pass
            elif event == "working_memory_eviction":
                pass
            elif event == "tool_start":
                pass
            elif event == "tool_result":
                tool_name = str(payload.get("name", "tool")).strip() or "tool"
                arguments = payload.get("arguments") or {}
                output = str(payload.get("output", "")).strip()
                self._record_tool_usage(tool_name, output, timestamp, arguments)
            elif event == "tool_limit":
                pass
            elif event == "tool_deferred":
                pass
            elif event == "turn_complete":
                summary = str(payload.get("text", "")).strip()
                memory_stats = payload.get("memory_stats") or {}
                if memory_stats:
                    self._update_memory_layers_from_stats(memory_stats, timestamp)
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
        self._emit({"type": "memory_event", "payload": entry})

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

    def _record_memory_event(self, event_type: str, count: int, timestamp: Optional[str] = None) -> None:
        label = "Semantic" if event_type.upper() == "FACT" else "Reflection"
        plural = "memory" if count == 1 else "memories"
        detail = f"Recorded {count} new {label.lower()} {plural}."
        self._append_event(event_type, detail, timestamp)

    def _emit(self, event: Dict[str, Any]) -> None:
        if self._event_emitter is None:
            return
        try:
            self._event_emitter(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to emit event %s: %s", event.get("type"), exc)

    def _update_memory_layers_from_stats(self, stats: Dict[str, Any], timestamp: Optional[str] = None) -> None:
        prev_layers = dict(self.last_memory_layers)
        episodes = int(stats.get("episodic_count", 0))
        facts = int(stats.get("semantic_count", 0))
        insights = int(stats.get("reflections_count", 0))

        new_layers = {
            "episodes": episodes,
            "facts": facts,
            "insights": insights,
        }
        self.last_memory_layers = new_layers

        ts = timestamp or datetime.now().strftime("%H:%M")
        fact_delta = facts - prev_layers.get("facts", 0)
        if fact_delta > 0:
            self._record_memory_event("FACT", fact_delta, ts)

        insight_delta = insights - prev_layers.get("insights", 0)
        if insight_delta > 0:
            self._record_memory_event("INSIGHT", insight_delta, ts)

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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: set[asyncio.Queue] = set()
        self.collector = AtlasMetricsCollector(self.agent, event_emitter=self._queue_event)
        self.graph_config = KnowledgeGraphConfig.from_env()
        self.graph_store: Optional[SQLiteGraphStore] = None
        self.graph_pipeline: Optional[KnowledgeGraphPipeline] = None
        self.graph_extractor = MemoryGraphExtractor()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._closed = False
        self._graph_started = False

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
        elif msg_type == "kg_search":
            await self._handle_kg_search(websocket, data.get("payload", {}))
        elif msg_type == "kg_neighbors":
            await self._handle_kg_neighbors(websocket, data.get("payload", {}))
        elif msg_type == "kg_context":
            await self._handle_kg_context(websocket, data.get("payload", {}))
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}" if websocket.remote_address else "unknown"
        logger.info(f"Client connected: {client_info}")

        subscriber_queue = self._register_subscriber()
        sender_task = asyncio.create_task(self._drain_queue(websocket, subscriber_queue))

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
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task
            self._unregister_subscriber(subscriber_queue)

    async def start(self) -> None:
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            self._loop = asyncio.get_running_loop()
            await self._start_graph_pipeline()
            logger.info(f"ATLAS WebSocket server running on ws://{self.host}:{self.port}")
            print(f"ATLAS WebSocket server running on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                logger.info("Server shutdown requested")
            finally:
                await self._stop_graph_pipeline()

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
        self._queue_event({
            "type": "metrics",
            "payload": self.collector.snapshot()
        })

    async def _send_error(self, websocket: WebSocketServerProtocol, message: str) -> None:
        logger.error(f"Sending error to client: {message}")
        await websocket.send(json.dumps({"type": "error", "payload": message}))

    async def _handle_kg_search(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
        if not self.graph_store:
            await self._send_error(websocket, "Knowledge graph is disabled")
            return
        query = str(payload.get("q") or payload.get("query") or "").strip()
        if not query:
            await self._send_error(websocket, "Missing search query")
            return
        limit = _safe_int(payload.get("limit"), 20)
        records = self.graph_store.search_nodes(query, limit=limit)
        await websocket.send(json.dumps({
            "type": "kg_search",
            "payload": {"nodes": self._serialize_nodes(records)}
        }))

    async def _handle_kg_neighbors(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
        if not self.graph_store:
            await self._send_error(websocket, "Knowledge graph is disabled")
            return
        try:
            node_id = int(payload.get("node_id"))
        except Exception:
            await self._send_error(websocket, "node_id must be provided")
            return
        types = payload.get("types")
        if types and not isinstance(types, list):
            await self._send_error(websocket, "types must be a list")
            return
        limit = _safe_int(payload.get("limit"), 25)
        neighbors, edges = self.graph_store.neighbors(node_id, types=types, limit=limit)
        await websocket.send(json.dumps({
            "type": "kg_neighbors",
            "payload": {
                "nodes": self._serialize_nodes(neighbors),
                "edges": self._serialize_edges(edges),
            },
        }))

    async def _handle_kg_context(self, websocket: WebSocketServerProtocol, payload: Dict[str, Any]) -> None:
        if not self.graph_store:
            await self._send_error(websocket, "Knowledge graph is disabled")
            return
        seeds_raw = payload.get("seeds")
        seeds: List[int] = []
        if isinstance(seeds_raw, list):
            for item in seeds_raw:
                try:
                    seeds.append(int(item))
                except (TypeError, ValueError):
                    continue
        depth = _safe_int(payload.get("depth"), 1)
        limit = _safe_int(payload.get("limit"), 50)
        if not seeds:
            recent = self.graph_store.recent_nodes(limit=limit)
            seeds = [record.id for record in recent]
        nodes, edges = self.graph_store.subgraph(seeds, depth=depth, limit=limit)
        await websocket.send(json.dumps({
            "type": "kg_context",
            "payload": {
                "nodes": self._serialize_nodes(nodes),
                "edges": self._serialize_edges(edges),
            },
        }))

    def _register_subscriber(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def _unregister_subscriber(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def _drain_queue(self, websocket: WebSocketServerProtocol, queue: asyncio.Queue) -> None:
        try:
            while True:
                event = await queue.get()
                await websocket.send(json.dumps(event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("Queue drain stopped for websocket %s: %s", websocket.remote_address, exc)

    def _queue_event(self, event: Dict[str, Any]) -> None:
        if not self._subscribers or self._loop is None:
            return

        for subscriber in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._enqueue_event, subscriber, dict(event))

    @staticmethod
    def _enqueue_event(queue: asyncio.Queue, event: Dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Dropping event %s due to persistent full queue", event.get("type"))

    async def _start_graph_pipeline(self) -> None:
        if self._graph_started or not self.graph_config.enabled or self._loop is None:
            return
        try:
            self.graph_store = SQLiteGraphStore(self.graph_config)
            self.graph_pipeline = KnowledgeGraphPipeline(
                loop=self._loop,
                config=self.graph_config,
                store=self.graph_store,
                extractor=self.graph_extractor,
                event_callback=self._queue_event,
            )
            set_pipeline(self.graph_pipeline)
            self._graph_started = True
        except Exception:
            logger.exception("Failed to initialize knowledge graph pipeline")
            self.graph_store = None
            self.graph_pipeline = None
            self._graph_started = False

    async def _stop_graph_pipeline(self) -> None:
        if self.graph_pipeline:
            try:
                await self.graph_pipeline.shutdown()
            except Exception:
                logger.debug("Graph pipeline shutdown encountered an error", exc_info=True)
        clear_pipeline()
        if self.graph_store:
            try:
                self.graph_store.close()
            except Exception:
                logger.debug("Graph store close encountered an error", exc_info=True)
        self.graph_store = None
        self.graph_pipeline = None
        self._graph_started = False

    def _serialize_nodes(self, nodes: Iterable) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for node in nodes:
            props = getattr(node, "props", {}) or {}
            serialized.append(
                {
                    "id": int(getattr(node, "id")),
                    "type": getattr(node, "type"),
                    "label": getattr(node, "label"),
                    "props": props,
                }
            )
        return serialized

    def _serialize_edges(self, edges: Iterable) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for edge in edges:
            serialized.append(
                {
                    "id": int(getattr(edge, "id")),
                    "from": int(getattr(edge, "src")),
                    "to": int(getattr(edge, "dst")),
                    "type": getattr(edge, "type"),
                }
            )
        return serialized

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
        if self.graph_store:
            try:
                self.graph_store.close()
            except Exception:
                pass
        clear_pipeline()
        self._executor.shutdown(wait=False)


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return max(1, parsed)
    except (TypeError, ValueError):
        return default


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
