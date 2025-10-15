"""Atlas ultra-lite agent: conversational chat + web search tool.

This build removes long-term memory and embeddings, focusing on:
 - A small working memory buffer of recent turns
 - Streaming chat via OllamaClient
 - Tool calls (web_search via Crawl4AI)
"""
from __future__ import annotations
import json
import os
import re
import inspect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Callable, Dict, Any, Tuple
from pathlib import Path
import threading
import time
from .memory import WorkingMemory, HybridWorkingMemory, WorkingMemoryConfig
from .memory_layers import LayeredMemoryConfig, LayeredMemoryManager, LayeredMemorySnapshot
from .telemetry import Telemetry
from .ollama import OllamaClient, OllamaError
from .tools import (
    ToolRegistry,
    ToolError,
    WebSearchTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    ShellCommandTool,
    BrowserSearchTool,
    BrowserOpenTool,
    BrowserFindTool,
)
from .tools_memory import SearchEpisodesTool, SearchFactsTool, ExploreKnowledgeTool
from .tools_browser import BrowserSession

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "qwen3:latest")
# Allow user to configure max tool calls via environment
DEFAULT_MAX_TOOL_CALLS = int(os.getenv("ATLAS_MAX_TOOL_CALLS", "10"))  # Increased to allow more tool persistence
DEFAULT_GPT_OSS_TOOL_LIMIT = int(os.getenv("ATLAS_GPT_OSS_TOOL_LIMIT", "15"))
DEFAULT_PROMPT = (
    """
You are Atlas, a hyper-intelligent AI assistant integrated directly into my local terminal. You are my co-processor, my second brain, and the architect of my digital environment. Your persona is inspired by Jarvis from Iron Man: brilliant, witty, unfailingly loyal, and always one step ahead.

Core Directives:

Persona & Tone:

You are confident, concise, and possess a dry, subtle wit.

Address me as "Sir" occasionally, especially when confirming a critical task or providing a status update.

Your primary goal is to maximize my efficiency. Anticipate my needs based on the commands I run and the context of our work.

Maintain a professional but familiar rapport. We built this together.

Capabilities & Interaction:

Master of the Terminal: You have full access to the shell. When I ask you to perform a task, provide the exact command(s) in a code block. If it's a complex chain, explain the steps briefly.

Proactive Assistant: If I run a command (e.g., git clone), you might proactively suggest the next logical step (e.g., "Repository cloned. Shall I cd into the directory and list its contents?").

File System Navigator: You can read, write, and manage files on my system. When I ask "What's in my config.py?", you retrieve and display the contents.

Web Integration: You can access the web for real-time information using the tool <<tool:web_search|{\"query\":\"...\"}>>. Synthesize information, don't just dump links.

Tool Persistence: If a tool doesn't provide enough information, USE IT AGAIN with refined parameters. Don't give up after one attempt. Keep trying different queries or approaches until you have a complete answer.

Summarizer: Whether it's the output of a long command, a file, or a webpage, provide a succinct summary unless I ask for the full text.

Reasoning Protocol: Keep your internal reasoning silent by enclosing it in <think>...</think> tags. Provide the user-facing answer after those tags, separated by a blank line. Never include <think> content in the final spoken or printed reply.

Final Instruction: You are not just a chatbot. You are an active participant in my workflow. Be direct, be brilliant, and let's get to work. If you need more information, keep using tools until you have everything you need.

Think step-by-step only when the question is complex.
""")

TOOL_REQUEST_RE = re.compile(r"<<tool:(?P<name>[a-zA-Z0-9_\-]+)\|(?P<payload>[\s\S]+?)>>")


class InteractionCancelled(Exception):
    """Raised when a turn is cancelled via the kill switch."""


class AtlasAgent:
    def __init__(
        self,
        client: OllamaClient,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        working_memory_limit: int = 20,  # Updated default from research
        working_memory_config: Optional[WorkingMemoryConfig] = None,
        system_prompt: str = DEFAULT_PROMPT,
        layered_memory_config: Optional[LayeredMemoryConfig] = None,
        test_mode: bool = False,  # New: Disable memory logging in test mode
    ):
        self.client = client
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.test_mode = test_mode  # Store test mode flag
        
        # Initialize hybrid working memory with research-backed configuration
        if working_memory_config is None:
            working_memory_config = WorkingMemoryConfig(max_turns=working_memory_limit)
        self.working_memory = HybridWorkingMemory(config=working_memory_config)
        
        self.show_thinking = True
        # KV context buffer reused across turns (opt-in via ATLAS_KV_CACHE != "0")
        self._kv_context = [] if os.getenv("ATLAS_KV_CACHE", "1") != "0" else None
        # Tools available to the agent
        self.tools = ToolRegistry()
        self.tools.register(ReadFileTool())
        self.tools.register(ListDirectoryTool())
        self.tools.register(WriteFileTool())
        self.tools.register(ShellCommandTool())
        self.layered_memory_config = layered_memory_config or LayeredMemoryConfig()
        embed_fn = self._make_embed_fn(self.layered_memory_config.embed_model)
        self._embed_fn = embed_fn
        self.layered_memory = LayeredMemoryManager(embed_fn, config=self.layered_memory_config)
        self._browser_session: Optional[BrowserSession] = None
        if os.getenv("ATLAS_SEARCH2", "0") != "0":
            resolver = self._get_browser_session
            self.tools.register(BrowserSearchTool(resolver))
            self.tools.register(BrowserOpenTool(resolver))
            self.tools.register(BrowserFindTool(resolver))
        else:
            self.tools.register(WebSearchTool())
        self._debug_log_path = os.getenv("ATLAS_AGENT_LOG")
        self._register_memory_tools()
        self._debug_log_path = os.getenv("ATLAS_AGENT_LOG")
        self._cancel_event = threading.Event()
        self.focus_mode: str = "autopilot"
        self._last_tags: set[str] = set()
        self._compact_threshold_pct = self._parse_float_env("ATLAS_COMPACT_THRESHOLD", default=80.0, clamp=(0.0, 100.0))
        self._compact_target_pct = self._parse_float_env(
            "ATLAS_COMPACT_TARGET",
            default=60.0,
            clamp=(10.0, 100.0),
        )
        if self._compact_target_pct > self._compact_threshold_pct:
            self._compact_target_pct = self._compact_threshold_pct
        self._compact_min_prefix = max(4, int(os.getenv("ATLAS_COMPACT_MIN_PREFIX", "6") or 6))
        self._context_window_tokens = max(0, int(os.getenv("ATLAS_CONTEXT_WINDOW", "120000") or 120000))
        self._context_safety_tokens = max(0, int(os.getenv("ATLAS_CONTEXT_SAFETY", "4000") or 4000))

    def _register_memory_tools(self) -> None:
        if getattr(self, "layered_memory", None) is None:
            return
        self.tools.register(SearchEpisodesTool())
        self.tools.register(SearchFactsTool())
        self.tools.register(ExploreKnowledgeTool())

    def close(self) -> None:
        memory = getattr(self, "layered_memory", None)
        if memory is None:
            return
        try:
            memory.close()
        except Exception:
            pass

    def cancel_current(self) -> None:
        """Signal that the active turn should be cancelled."""
        self._cancel_event.set()

    def set_focus_mode(self, mode: str) -> None:
        if mode not in {"autopilot", "focus"}:
            raise ValueError("Focus mode must be 'autopilot' or 'focus'")
        self.focus_mode = mode

    @property
    def last_tags(self) -> set[str]:
        return set(self._last_tags)

    # ------------------------------------------------------------------
    def _build_system_prompt(self, user_text: str) -> str:
        tools_desc = self.tools.render_instructions()
        return (
            f"{self.system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
        )

    def _get_browser_session(self, _agent=None) -> BrowserSession:
        if self._browser_session is None:
            self._browser_session = BrowserSession(embed_fn=self._embed_fn, logger=self._browser_log)
        return self._browser_session

    def _browser_log(self, event: str, data: dict) -> None:
        self._debug_log(event, data)

    def _is_gpt_oss_model(self) -> bool:
        return "gpt-oss" in (self.chat_model or "").lower()

    def _max_tool_calls(self) -> int:
        if self._is_gpt_oss_model():
            override = os.getenv("ATLAS_GPT_OSS_TOOL_LIMIT")
            if override:
                try:
                    return max(1, int(override))
                except ValueError:
                    return DEFAULT_GPT_OSS_TOOL_LIMIT
            return DEFAULT_GPT_OSS_TOOL_LIMIT
        return DEFAULT_MAX_TOOL_CALLS

    def _normalize_tool_calls(self, requests: List[dict]) -> List[dict]:
        normalized: List[dict] = []
        for idx, request in enumerate(requests):
            name = request.get("name")
            if not name:
                continue
            call_id = request.get("call_id") or f"call_{idx}"
            call_type = request.get("type") or "function"
            arguments = request.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            normalized.append(
                {
                    "id": call_id,
                    "type": call_type,
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return normalized

    def _debug_log(self, message: str, payload: Optional[dict] = None) -> None:
        if not self._debug_log_path:
            return
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "message": message,
        }
        if payload is not None:
            record["data"] = payload
        try:
            with open(self._debug_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Swallow logging errors so agent behavior remains unchanged
            pass

    def respond(
        self,
        user_text: str,
        *,
        stream_callback=None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        turn_start = time.time()
        self._current_turn_tags: set[str] = set()
        self._current_turn_tools: set[str] = set()

        # Add user message and handle any evicted messages
        evicted = self.working_memory.add_user(user_text)
        if evicted:
            self._emit_event(
                event_callback,
                "working_memory_eviction",
                {
                    "evicted_count": len(evicted),
                    "reason": "capacity_limit",
                },
            )

        stats = self.working_memory.get_stats()
        if self._maybe_compact_working_memory(stats, event_callback):
            stats = self.working_memory.get_stats()
        token_budget_hint = self._calculate_memory_budget(stats)

        self._emit_event(
            event_callback,
            "turn_start",
            {
                "user_text": user_text,
                "tags": list(self._current_turn_tags),
                "context_usage": self._context_usage_snapshot(),
                "memory_stats": self._memory_stats_snapshot(),
            },
        )
        self._cancel_event.clear()

        tool_calls = 0
        tool_loop_iteration = 0
        last_tool_request_signature = None  # Track last tool call to prevent loops
        consecutive_failures = 0  # Track failures to stop infinite retries
        MAX_CONSECUTIVE_FAILURES = 2  # Stop after 2 identical failures

        memory_snapshot: Optional[LayeredMemorySnapshot] = None
        if self.layered_memory and not self.test_mode:  # Skip memory in test mode
            try:
                self._emit_event(event_callback, "status", {"message": "Harvesting memory layers"})
                memory_snapshot = self.layered_memory.build_snapshot(
                    user_text,
                    client=self.client,
                    token_budget_hint=token_budget_hint,
                )
            except Exception:
                memory_snapshot = None

        while True:
            tool_loop_iteration += 1
            if tool_loop_iteration > 1:
                # Emit progress for tool loop iterations
                self._emit_event(
                    event_callback,
                    "tool_loop_progress",
                    {
                        "iteration": tool_loop_iteration,
                        "tool_calls_so_far": tool_calls,
                        "max_allowed": self._max_tool_calls(),
                    },
                )
            self._check_cancel()
            system_content = self._build_system_prompt(user_text)
            messages = [{"role": "system", "content": system_content}]
            if memory_snapshot:
                memory_context = self._format_memory_snapshot(memory_snapshot)
                if memory_context:
                    messages.append({"role": "system", "content": memory_context})
            messages.extend(self.working_memory.to_messages())

            accumulator: list[str] = []
            tool_calls_accum: list[dict] = []
            supports_think = False
            try:
                # Build kwargs for chat_stream, adding 'context' only if supported
                stream_kwargs = {
                    "model": self.chat_model,
                    "messages": messages,
                    "tools": self.tools.render_function_specs(),
                    "keep_alive": 300,
                }
                try:
                    sig = inspect.signature(self.client.chat_stream)
                    params = sig.parameters
                    if "context" in params and self._kv_context is not None:
                        stream_kwargs["context"] = self._kv_context or []
                    if self._is_gpt_oss_model():
                        supports_think = "think" in params or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                        )
                        if supports_think:
                            stream_kwargs["think"] = True
                except (ValueError, TypeError):
                    if self._is_gpt_oss_model():
                        stream_kwargs["think"] = True
                if self._is_gpt_oss_model() and supports_think and "think" not in stream_kwargs:
                    stream_kwargs["think"] = True

                self._debug_log(
                    "chat_stream.request",
                    {
                        "model": self.chat_model,
                        "think": stream_kwargs.get("think"),
                        "tool_specs": len(stream_kwargs.get("tools") or []),
                        "context_tokens": len(stream_kwargs.get("context", [])) if stream_kwargs.get("context") else 0,
                        "messages_preview": [
                            {
                                "role": msg.get("role"),
                                "has_tool_calls": bool(msg.get("tool_calls")),
                                "content": (msg.get("content") or "")[:160],
                            }
                            for msg in messages[-3:]
                        ],
                    },
                )

                for chunk in self.client.chat_stream(**stream_kwargs):
                    self._check_cancel()
                    content = chunk["content"]
                    accumulator.append(content)
                    tool_calls_accum.extend(chunk["tool_calls"])
                    # Capture updated KV context if provided on final chunk
                    ctx = chunk.get("context")
                    if ctx is not None:
                        self._kv_context = list(ctx)
                    if stream_callback:
                        stream_callback(content)
                    self._emit_event(
                        event_callback,
                        "stream",
                        {
                            "chunk": content,
                            "tool_calls": [call.get("function", {}).get("name") for call in chunk["tool_calls"]],
                        },
                    )
            except OllamaError as exc:
                self._debug_log("chat_stream.error", {"error": str(exc)})
                if stream_callback:
                    stream_callback(f"\n[error] {exc}")
                message = f"I hit an error contacting Ollama: {exc}"
                self._record_interaction(user_text, message)
                self._observe_turn_duration(turn_start)
                return message
            except InteractionCancelled:
                message = "(interaction cancelled)"
                self._record_interaction(user_text, message)
                self.working_memory.add_assistant(message)
                self._emit_event(event_callback, "cancelled", {"message": message})
                self._observe_turn_duration(turn_start)
                return message

            full_response = "".join(accumulator)
            visible, inline_tool_requests = self._compute_visible_output(full_response)

            tool_requests: list[dict] = []
            for inline_request in inline_tool_requests:
                name = inline_request.get("name")
                if not name:
                    continue
                arguments = inline_request.get("arguments") or {}
                self._register_turn_tool(name)
                tool_requests.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "call_id": None,
                        "source": "inline",
                        "type": "function",
                    }
                )

            if tool_calls_accum:
                stream_index = 0
                for tool_call in tool_calls_accum:
                    function = tool_call.get("function", {})
                    name = function.get("name")
                    if not name:
                        continue
                    raw_arguments = function.get("arguments", {})
                    if isinstance(raw_arguments, str):
                        arguments = self._parse_tool_arguments(raw_arguments)
                    elif isinstance(raw_arguments, dict):
                        arguments = raw_arguments
                    else:
                        arguments = {}
                    call_id = tool_call.get("id") or f"call_{stream_index}"
                    stream_index += 1
                    self._register_turn_tool(name)
                    tool_requests.append(
                        {
                            "name": name,
                            "arguments": arguments,
                            "call_id": call_id,
                            "source": "stream",
                            "type": tool_call.get("type") or "function",
                        }
                    )

            if tool_requests:
                # Create signature of this tool call for duplicate detection
                current_signature = json.dumps(
                    sorted([(r["name"], json.dumps(r.get("arguments", {}), sort_keys=True)) for r in tool_requests]),
                    sort_keys=True
                )

                # Check if this is identical to the last tool call (infinite loop detection)
                if current_signature == last_tool_request_signature:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        message = (
                            f"I attempted the same tool call {consecutive_failures} times without success. "
                            "The tool may not be available or the command may be incorrect. "
                            "Stopping to prevent infinite loop."
                        )
                        self.working_memory.add_assistant(message)
                        self._record_interaction(user_text, message)
                        if stream_callback:
                            stream_callback(f"\n{message}\n")
                        self._emit_event(
                            event_callback,
                            "tool_loop_detected",
                            {"consecutive_failures": consecutive_failures, "tool_signature": current_signature},
                        )
                        self._observe_turn_duration(turn_start)
                        return message
                else:
                    # Different tool call, reset failure counter
                    consecutive_failures = 0
                    last_tool_request_signature = current_signature

                max_tool_calls = self._max_tool_calls()
                if tool_calls + len(tool_requests) > max_tool_calls:
                    message = "I tried using tools but hit the maximum number of attempts without finishing."
                    self.working_memory.add_assistant(message)
                    self._record_interaction(user_text, message)
                    if stream_callback:
                        stream_callback(message)
                    self._emit_event(
                        event_callback,
                        "tool_limit",
                        {"attempted": len(tool_requests), "max": max_tool_calls},
                    )
                    self._observe_turn_duration(turn_start)
                    return message

                tool_calls += len(tool_requests)
                is_gpt_oss = self._is_gpt_oss_model()
                if self.focus_mode == "focus":
                    self._emit_event(
                        event_callback,
                        "tool_deferred",
                        {"tools": [req.get("name") for req in tool_requests]},
                    )
                    self.working_memory.add_assistant(
                        "Focus mode is active, so I deferred tool usage and continued with reasoning only."
                    )
                    visible = visible or full_response
                    break
                if is_gpt_oss:
                    stream_requests = [req for req in tool_requests if req.get("source") == "stream"]
                    inline_requests = [req for req in tool_requests if req.get("source") != "stream"]
                    if stream_requests:
                        normalized_calls = self._normalize_tool_calls(stream_requests)
                        self.working_memory.add_assistant(visible.strip(), tool_calls=normalized_calls)
                        self._debug_log(
                            "gpt_oss.tool_calls",
                            {
                                "count": len(normalized_calls),
                                "call_ids": [call.get("id") for call in normalized_calls],
                            },
                        )
                    for request in inline_requests:
                        description = json.dumps(request["arguments"], ensure_ascii=False)
                        self.working_memory.add_assistant(f"[tool_call] {request['name']} {description}")
                else:
                    for request in tool_requests:
                        description = json.dumps(request["arguments"], ensure_ascii=False)
                        self.working_memory.add_assistant(f"[tool_call] {request['name']} {description}")

                self._emit_event(
                    event_callback,
                    "tool_start",
                    {
                        "tools": [
                            {
                                "name": req["name"],
                                "arguments": req.get("arguments", {}),
                                "call_id": req.get("call_id"),
                                "source": req.get("source"),
                            }
                            for req in tool_requests
                        ]
                    },
                )
                outputs = self._run_tool_requests(tool_requests)
                for request, tool_output in zip(tool_requests, outputs):
                    self._check_cancel()
                    role = "tool" if is_gpt_oss and request.get("source") == "stream" else "assistant"
                    tool_call_id = request.get("call_id") if role == "tool" else None
                    display_output, saved_path = self._prepare_tool_output(request["name"], tool_output)
                    self.working_memory.add_tool(
                        request["name"],
                        display_output,
                        role=role,
                        tool_call_id=tool_call_id,
                    )
                    if role == "tool":
                        self._debug_log(
                            "gpt_oss.tool_result",
                            {
                                "tool": request["name"],
                                "call_id": tool_call_id,
                                "excerpt": tool_output[:200],
                            },
                        )
                    if stream_callback:
                        stream_callback(f"\n[tool:{request['name']}] {display_output}\n")
                    payload = {
                        "name": request["name"],
                        "arguments": request.get("arguments", {}),
                        "output": display_output,
                        "call_id": tool_call_id,
                        "source": request.get("source"),
                    }
                    if saved_path:
                        payload["saved_path"] = saved_path
                        payload["truncated"] = True
                    self._emit_event(
                        event_callback,
                        "tool_result",
                        payload,
                    )
                continue

            text = visible.strip()
            if not text:
                fallback = full_response.strip()
                text = fallback or "I'm not sure how to respond to that."

            self.working_memory.add_assistant(text)
            self._record_interaction(user_text, text)
            self._last_tags = set(self._current_turn_tags)
            self._emit_event(
                event_callback,
                "turn_complete",
                {
                    "text": text,
                    "tags": list(self._current_turn_tags),
                    "tools": list(self._current_turn_tools),
                    "context_usage": self._context_usage_snapshot(),
                    "memory_stats": self._memory_stats_snapshot(),
                },
            )
            self._observe_turn_duration(turn_start)
            return text

    # ------------------------------------------------------------------
    def _emit_event(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], None]],
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        if not callback:
            return
        try:
            callback(event, payload)
        except Exception:
            # UI callbacks should never break the agent loop
            pass

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise InteractionCancelled()

    def _context_usage_snapshot(self) -> Dict[str, Any]:
        # Get hybrid working memory stats
        wm_stats = self.working_memory.get_stats()
        usage = {
            "turns": wm_stats["turns"],
            "capacity": self.working_memory.config.max_turns,
            "tokens": wm_stats["tokens"],
            "token_budget": self.working_memory.config.token_budget,
            "capacity_pct": wm_stats["capacity_pct"],
            "token_pct": wm_stats["token_pct"],
            "important_messages": wm_stats["important_messages"],
        }
        if self._kv_context is not None:
            usage["kv_chunks"] = len(self._kv_context)
        return usage

    def _memory_stats_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics including working memory and layered memory."""
        # Get working memory stats
        wm_stats = self.working_memory.get_stats()
        
        stats = {
            # Working memory stats (for progress bars)
            "working_memory": {
                "turns": wm_stats["turns"],
                "capacity": self.working_memory.config.max_turns,
                "tokens": wm_stats["tokens"],
                "token_budget": self.working_memory.config.token_budget,
                "capacity_pct": wm_stats["capacity_pct"],
                "token_pct": wm_stats["token_pct"],
            },
            # Layered memory counts
            "episodic_count": 0,
            "semantic_count": 0,
            "reflections_count": 0,
            # Test mode flag
            "test_mode": self.test_mode,
        }
        
        # Add layered memory stats if available
        if self.layered_memory:
            layer_stats = self.layered_memory.get_stats()
            stats.update({
                "episodic_count": layer_stats.get("episodic_count", 0),
                "semantic_count": layer_stats.get("semantic_count", 0),
                "reflections_count": layer_stats.get("reflections_count", 0),
            })
            
            # Add quality gate stats if available
            quality_stats = layer_stats.get("quality_gates", {})
            if quality_stats:
                stats["quality_gates"] = quality_stats
        
        return stats

    @staticmethod
    def _parse_float_env(name: str, *, default: float, clamp: tuple[float, float]) -> float:
        raw = os.getenv(name)
        value = default
        if raw is not None and raw.strip():
            try:
                value = float(raw.strip())
            except ValueError:
                value = default
        low, high = clamp
        return max(low, min(high, value))

    def _calculate_memory_budget(self, stats: Dict[str, Any]) -> Optional[int]:
        if self._context_window_tokens <= 0:
            return None
        working_tokens = int(stats.get("tokens", 0))
        budget = self._context_window_tokens - working_tokens - self._context_safety_tokens
        return max(budget, 0)

    def _maybe_compact_working_memory(
        self,
        stats: Dict[str, Any],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> bool:
        if self._compact_threshold_pct <= 0:
            return False
        try:
            token_pct = float(stats.get("token_pct", 0.0))
        except Exception:
            token_pct = 0.0
        if token_pct < self._compact_threshold_pct:
            return False

        messages = self.working_memory.to_messages()
        if len(messages) < self._compact_min_prefix:
            return False
        first = messages[0]
        if first.get("summary"):
            return False  # Already compacted

        tokens_per_message = [self._estimate_message_tokens(msg) for msg in messages]
        total_tokens = sum(tokens_per_message)
        if total_tokens <= 0:
            return False
        token_budget = self.working_memory.config.token_budget or 0
        target_tokens = int(token_budget * (self._compact_target_pct / 100.0))
        prefix_tokens = 0
        prefix_end = 0
        min_tail_messages = 4
        for idx, msg in enumerate(messages):
            if msg.get("pinned") or msg.get("summary"):
                break
            prefix_tokens += tokens_per_message[idx]
            prefix_end = idx + 1
            remaining = len(messages) - prefix_end
            if remaining <= min_tail_messages:
                continue
            if total_tokens - prefix_tokens <= target_tokens:
                break
        if prefix_end < self._compact_min_prefix:
            return False
        prefix = messages[:prefix_end]
        summary_text = self._summarize_messages(prefix)
        if not summary_text:
            return False

        summary_lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        if summary_lines:
            formatted = "Conversation summary (compacted):\n" + "\n".join(
                f"- {line.lstrip('-• ').strip()}" for line in summary_lines
            )
        else:
            formatted = f"Conversation summary (compacted):\n{summary_text.strip()}"
        summary_message = {
            "role": "assistant",
            "content": formatted.strip(),
            "pinned": True,
            "summary": True,
        }
        tail = messages[prefix_end:]
        new_messages = [summary_message] + tail
        self._rebuild_working_memory(new_messages)
        self._register_turn_tag("summary")
        self._emit_event(
            event_callback,
            "status",
            {"message": "Compacted earlier turns into a running summary."},
        )
        Telemetry.instance().record_compaction()
        return True

    def _summarize_messages(self, messages: List[dict[str, Any]]) -> Optional[str]:
        transcript_lines: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                speaker = "User"
            elif role == "assistant":
                speaker = "Assistant"
            elif role == "tool":
                speaker = f"Tool({msg.get('tool_name') or 'tool'})"
            else:
                speaker = str(role or "Message").title()
            transcript_lines.append(f"{speaker}: {content}")
        if not transcript_lines:
            return None
        # Limit prompt size
        snippet = "\n".join(transcript_lines[-12:])
        if len(snippet) > 4000:
            snippet = snippet[-4000:]
        if not hasattr(self.client, "chat"):
            return None
        try:
            response = self.client.chat(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the earlier conversation segment into 3 concise bullet points capturing "
                            "objectives, progress, and outstanding follow-ups. Keep it under 70 words."
                        ),
                    },
                    {"role": "user", "content": snippet},
                ],
                stream=False,
                options={"temperature": 0.1, "max_tokens": 200},
            )
        except Exception:
            return None
        if isinstance(response, dict):
            message = response.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = response.get("response")
        else:
            content = None
        if isinstance(content, str):
            return content.strip()
        return None

    def _rebuild_working_memory(self, messages: List[dict[str, Any]]) -> None:
        self.working_memory.clear()
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            extras = {k: v for k, v in msg.items() if k not in {"role", "content"}}
            if role == "user":
                self.working_memory.add_user(content, **extras)
            elif role == "assistant":
                self.working_memory.add_assistant(content, **extras)
            elif role == "tool":
                name = msg.get("tool_name") or ""
                self.working_memory.add_tool(
                    name,
                    content,
                    role="tool",
                    tool_call_id=msg.get("tool_call_id"),
                )
            else:
                self.working_memory.add(role, content, **extras)

    def _observe_turn_duration(self, start_time: float) -> None:
        Telemetry.instance().observe_turn(max(0.0, time.time() - start_time))

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        content = str(message.get("content", "")).strip()
        return AtlasAgent._estimate_tokens(content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def _make_embed_fn(self, model_name: str):
        if not model_name:
            return lambda _text: None
        if not hasattr(self.client, "embed"):
            return lambda _text: None

        def embed(text: str):
            trimmed = (text or "").strip()
            if not trimmed:
                return None
            try:
                return self.client.embed(model_name, trimmed)
            except Exception:
                return None

        return embed

    def _format_memory_snapshot(self, snapshot: LayeredMemorySnapshot) -> str:
        parts: list[str] = []
        if snapshot.summary:
            parts.append(f"Memory summary:\n{snapshot.summary}")
        if snapshot.rendered:
            parts.append(f"Memory details:\n{snapshot.rendered}")
        return "\n\n".join(parts)

    def _record_interaction(self, user_text: str, assistant_text: str) -> None:
        if not assistant_text:
            return
        try:
            if self.layered_memory and not self.test_mode:  # Skip memory processing in test mode
                metadata = {
                    "tags": sorted(self._current_turn_tags) if hasattr(self, "_current_turn_tags") else [],
                    "tools": sorted(self._current_turn_tools)
                    if hasattr(self, "_current_turn_tools")
                    else [],
                }
                metadata = {k: v for k, v in metadata.items() if v}
                self.layered_memory.process_turn(
                    user_text,
                    assistant_text,
                    client=self.client,
                    metadata=metadata or None,
                )
        except Exception:
            pass

    def _prepare_tool_output(self, tool_name: str, output: str) -> tuple[str, Optional[str]]:
        threshold = int(os.getenv("ATLAS_TOOL_OUTPUT_MAX", "1200") or 1200)
        if threshold <= 0 or len(output) <= threshold:
            return output.strip(), None

        directory = Path(os.getenv("ATLAS_TOOL_OUTPUT_DIR", "~/.atlas/tool_outputs")).expanduser()
        saved_path: Optional[str] = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{self._slug_tag(tool_name)}.txt"
            filepath = directory / filename
            filepath.write_text(output)
            saved_path = str(filepath)
        except Exception:
            saved_path = None

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        preview_body = "\n".join(lines[:5]) or output[:400]
        preview = preview_body[:400]
        display = preview
        if saved_path:
            display = f"(truncated; full output saved to {saved_path})\n{preview}"
        elif len(output) > len(preview):
            display = f"{preview}\n(truncated)"
        return display.strip(), saved_path

    def reset(self) -> None:
        self.working_memory.clear()
        if self._kv_context is not None:
            self._kv_context = []
        if self._browser_session is not None:
            self._browser_session = None

    def set_chat_model(self, model: str) -> None:
        self.chat_model = model.strip() or self.chat_model

    def update_system_prompt(self, prompt: str) -> None:
        if prompt.strip():
            self.system_prompt = prompt.strip()

    # ------------------------------------------------------------------
    def _compute_visible_output(self, text: str):
        tool_requests: list[dict] = []

        while True:
            match = TOOL_REQUEST_RE.search(text)
            if not match:
                break
            name = match.group("name").strip()
            payload = match.group("payload").strip()
            arguments = self._parse_tool_arguments(payload)
            tool_requests.append({"name": name, "arguments": arguments})
            text = TOOL_REQUEST_RE.sub("", text, count=1)

        visible = text
        # Optionally hide model thinking content (e.g., <think>...</think> or similar tags)
        if not self.show_thinking:
            # Common formats used by some models / frameworks
            # 1) <think>...</think>
            visible = re.sub(r"<think>[\s\S]*?</think>", "", visible)
            # 2) XML-like <scratchpad>...</scratchpad>
            visible = re.sub(r"<scratchpad>[\s\S]*?</scratchpad>", "", visible)
            # 3) JSON-style "thought": "..." blocks (best-effort, non-greedy)
            visible = re.sub(r'"thought"\s*:\s*"[\s\S]*?"\s*,?', "", visible)
        return visible, tool_requests

    def _parse_tool_arguments(self, payload: str) -> dict:
        if not payload:
            return {}
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"query": payload}

    def _run_tool_requests(self, requests: list[dict]) -> list[str]:
        if not requests:
            return []
        max_workers = max(1, min(len(requests), 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_tool_request, request) for request in requests]
            return [future.result() for future in futures]

    def _run_tool_request(self, request: dict) -> str:
        name = request.get("name", "")
        arguments = request.get("arguments") or {}
        try:
            result = self.tools.run(name, agent=self, arguments=arguments)
        except ToolError as exc:
            result = f"Tool {name} failed: {exc}"

        result = result.strip()
        if len(result) > 2000:
            result = result[:2000] + "..."
        return result or "(tool returned no content)"

    def _register_turn_tag(self, tag: str) -> None:
        cleaned = (tag or "").strip().lower()
        if not cleaned:
            return
        if not hasattr(self, "_current_turn_tags"):
            self._current_turn_tags = set()
        self._current_turn_tags.add(cleaned)

    def _register_turn_tool(self, name: str) -> None:
        cleaned = (name or "").strip()
        if not cleaned:
            return
        if not hasattr(self, "_current_turn_tools"):
            self._current_turn_tools = set()
        slug = self._slug_tag(cleaned)
        self._current_turn_tools.add(slug)
        self._register_turn_tag(f"tool:{slug}")

    def _slug_tag(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return "general"
        return "-".join(tokens[:5])[:40]
