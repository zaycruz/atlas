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
from typing import List, Optional, Callable, Dict, Any
import threading
from .memory import WorkingMemory
from .memory_layers import LayeredMemoryConfig, LayeredMemoryManager, LayeredMemorySnapshot
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
from .tools_browser import BrowserSession

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "gpt-oss:latest")
DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_GPT_OSS_TOOL_LIMIT = 10
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

Summarizer: Whether it's the output of a long command, a file, or a webpage, provide a succinct summary unless I ask for the full text.

Reasoning Protocol: Keep your internal reasoning silent by enclosing it in <think>...</think> tags. Provide the user-facing answer after those tags, separated by a blank line. Never include <think> content in the final spoken or printed reply.

Final Instruction: You are not just a chatbot. You are an active participant in my workflow. Be direct, be brilliant, and let's get to work."""
)

TOOL_REQUEST_RE = re.compile(r"<<tool:(?P<name>[a-zA-Z0-9_\-]+)\|(?P<payload>[\s\S]+?)>>")


class InteractionCancelled(Exception):
    """Raised when a turn is cancelled via the kill switch."""


class AtlasAgent:
    def __init__(
        self,
        client: OllamaClient,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        working_memory_limit: int = 12,
        system_prompt: str = DEFAULT_PROMPT,
        layered_memory_config: Optional[LayeredMemoryConfig] = None,
    ):
        self.client = client
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.working_memory = WorkingMemory(capacity=working_memory_limit)
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
        self._cancel_event = threading.Event()
        self.focus_mode: str = "autopilot"
        self._last_objective: Optional[str] = None
        self._last_tags: set[str] = set()

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
    def last_objective(self) -> Optional[str]:
        return self._last_objective

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

        self._current_turn_tags: set[str] = set()
        self._current_turn_tools: set[str] = set()
        self._current_objective = self._extract_objective(user_text)
        if self._current_objective:
            self._register_turn_tag(f"objective:{self._slug_tag(self._current_objective)}")

        self.working_memory.add_user(user_text)
        self._emit_event(
            event_callback,
            "turn_start",
            {
                "user_text": user_text,
                "objective": self._current_objective,
                "tags": list(self._current_turn_tags),
                "context_usage": self._context_usage_snapshot(),
            },
        )
        self._cancel_event.clear()

        tool_calls = 0
        memory_snapshot: Optional[LayeredMemorySnapshot] = None
        if self.layered_memory:
            try:
                self._emit_event(event_callback, "status", {"message": "Harvesting memory layers"})
                memory_snapshot = self.layered_memory.build_snapshot(user_text, client=self.client)
            except Exception:
                memory_snapshot = None

        while True:
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
                return message
            except InteractionCancelled:
                message = "(interaction cancelled)"
                self._record_interaction(user_text, message)
                self.working_memory.add_assistant(message)
                self._emit_event(event_callback, "cancelled", {"message": message})
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
                    self.working_memory.add_tool(
                        request["name"],
                        tool_output,
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
                        stream_callback(f"\n[tool:{request['name']}] {tool_output}\n")
                    self._emit_event(
                        event_callback,
                        "tool_result",
                        {
                            "name": request["name"],
                            "arguments": request.get("arguments", {}),
                            "output": tool_output,
                            "call_id": tool_call_id,
                            "source": request.get("source"),
                        },
                    )
                continue

            text = visible.strip()
            if not text:
                fallback = full_response.strip()
                text = fallback or "I'm not sure how to respond to that."

            self.working_memory.add_assistant(text)
            self._record_interaction(user_text, text)
            self._last_objective = self._current_objective
            self._last_tags = set(self._current_turn_tags)
            self._emit_event(
                event_callback,
                "turn_complete",
                {
                    "text": text,
                    "objective": self._current_objective,
                    "tags": list(self._current_turn_tags),
                    "tools": list(self._current_turn_tools),
                    "context_usage": self._context_usage_snapshot(),
                },
            )
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
        usage = {
            "turns": len(self.working_memory.to_messages()),
            "capacity": self.working_memory.capacity,
        }
        if self._kv_context is not None:
            usage["kv_chunks"] = len(self._kv_context)
        return usage

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
            if self.layered_memory:
                metadata = {
                    "tags": sorted(self._current_turn_tags) if hasattr(self, "_current_turn_tags") else [],
                    "objective": getattr(self, "_current_objective", None),
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

    def _extract_objective(self, text: str) -> str:
        snippet = (text or "").strip()
        if not snippet:
            return ""
        parts = re.split(r"[.!?\n]", snippet, maxsplit=1)
        objective = parts[0].strip()
        words = objective.split()
        return " ".join(words[:12])

    def _slug_tag(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return "general"
        return "-".join(tokens[:5])[:40]
