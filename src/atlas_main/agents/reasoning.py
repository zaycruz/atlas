"""Adapter for reasoning-focused planner models (DeepSeek-R1, Sonnet, GPT-5)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from ..ollama import OllamaClient, OllamaError
from ..orchestrator.types import Artifact, StepResult, StepSpec
from .base import AgentAdapter, ConversationMessage, StreamingAgentSession

__all__ = ["ReasoningAgentAdapter"]

LOGGER = logging.getLogger(__name__)

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are a technical planner creating execution plans for coding agents. "
    "Analyze objectives, understand repository context, and propose actionable, "
    "parallelizable plans. Provide clear reasoning for your decisions."
)

_DEFAULT_AGENT_DESCRIPTIONS = [
    "codex: GitHub Codex CLI (general coding)",
    "claude-code: Claude Code CLI (complex refactors)",
    "droid: Factory Droid (Python specialist)",
]


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _join_lines(label: str, value: Any) -> str:
    formatted = _format_value(value)
    return f"- {label}: {formatted}"


def _extract_reasoning_blocks(text: str) -> Tuple[str, str]:
    """Return (reasoning_trace, remainder_without_reasoning)."""
    if not text:
        return "", ""

    reasoning_fragments: List[str] = []
    remainder = text

    # Extract <think> blocks (DeepSeek-R1)
    def _strip_pattern(pattern: re.Pattern[str], source: str) -> str:
        matches = list(pattern.finditer(source))
        if not matches:
            return source
        cleaned = source
        for match in matches:
            reasoning_fragments.append(match.group(1).strip())
            cleaned = cleaned.replace(match.group(0), "")
        return cleaned

    remainder = _strip_pattern(re.compile(r"<think>(.*?)</think>", re.DOTALL), remainder)
    remainder = _strip_pattern(re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL), remainder)
    remainder = _strip_pattern(
        re.compile(r"```thinking\s*(.*?)```", re.DOTALL | re.IGNORECASE), remainder
    )

    reasoning = "\n\n".join(fragment for fragment in reasoning_fragments if fragment).strip()
    return reasoning, remainder.strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Attempt to parse a JSON object from free-form text."""
    if not text:
        raise ValueError("Empty response.")

    text = text.strip()
    candidates: List[str] = []

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    candidates.append(text)

    first_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as exc:  # noqa: BLE001 - collect first exception for diagnostics
            if first_error is None:
                first_error = exc

    raise ValueError(f"Failed to parse planner JSON: {first_error}") from first_error


class ReasoningAgentAdapter(AgentAdapter):
    """Adapter bridging planner-facing reasoning models into Atlas."""

    def __init__(
        self,
        *,
        agent_id: str,
        model: str,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        max_tokens: int = 8000,
        ollama_client: Optional[OllamaClient] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key or self._resolve_default_api_key(self.provider)

        self._ollama_client = ollama_client
        self._owns_ollama_client = False
        self._http_client = http_client
        self._owns_http_client = False

        if self.provider == "ollama":
            if self._ollama_client is None:
                self._ollama_client = OllamaClient()
                self._owns_ollama_client = True
        elif self.provider in {"anthropic", "openai"}:
            if self._http_client is None:
                timeout_config = httpx.Timeout(self.timeout, read=self.timeout)
                self._http_client = httpx.AsyncClient(timeout=timeout_config)
                self._owns_http_client = True
        else:
            raise ValueError(f"Unsupported reasoning provider: {self.provider}")

    async def execute_step(
        self,
        *,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> StepResult:
        """Execute a single planning step and return a structured plan."""
        messages = self._build_messages_for_step(step=step, shared_context=shared_context)

        try:
            assistant_text, reasoning_trace, provider_meta = await self._complete_chat(messages)
        except Exception as exc:  # noqa: BLE001 - surface full error details in result
            LOGGER.exception("Reasoning agent %s failed: %s", self.agent_id, exc)
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=f"{self.provider} planner call failed: {exc}",
                logs=[str(exc)],
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.model,
                    "provider": self.provider,
                },
            )

        try:
            plan_dict = _extract_json_object(assistant_text)
        except ValueError as exc:
            LOGGER.warning("Planner response parsing failed: %s", exc)
            logs = [assistant_text]
            if reasoning_trace:
                logs.insert(0, f"reasoning:\n{reasoning_trace}")
            return StepResult(
                step_id=step.id,
                status="failed",
                summary="Planner returned invalid plan payload.",
                logs=logs,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.model,
                    "provider": self.provider,
                    "reasoning_trace": reasoning_trace,
                },
            )

        artifact_content = json.dumps(plan_dict, indent=2, sort_keys=True)
        plan_steps = plan_dict.get("steps") or []
        summary = plan_dict.get("reasoning") or f"Generated plan with {len(plan_steps)} steps."

        metadata = {
            "agent_id": self.agent_id,
            "model": self.model,
            "provider": self.provider,
            "reasoning_trace": reasoning_trace or plan_dict.get("reasoning", ""),
        }
        metadata.update(provider_meta)

        logs: List[str] = []
        if reasoning_trace:
            logs.append(f"reasoning:\n{reasoning_trace}")
        logs.append(f"plan:\n{artifact_content}")

        return StepResult(
            step_id=step.id,
            status="succeeded",
            summary=summary,
            artifacts=[
                Artifact(kind="plan", content=artifact_content, metadata={"format": "json"}),
            ],
            logs=logs,
            metadata=metadata,
        )

    async def open_session(
        self,
        *,
        task: str,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> StreamingAgentSession:
        """Open a multi-turn planning conversation with the reasoning model."""
        initial_messages = self._build_initial_conversation(task, step, shared_context)
        session = _ReasoningStreamingSession(
            adapter=self,
            initial_messages=initial_messages,
            timeout=self.timeout,
        )
        await session.start()
        return session

    async def close(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._owns_ollama_client and self._ollama_client is not None:
            self._ollama_client.close()
            self._ollama_client = None

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def _build_messages_for_step(
        self,
        *,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> List[Dict[str, str]]:
        system_prompt = _SYSTEM_PROMPT
        prompt_body = self._render_prompt(step=step, shared_context=shared_context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_body},
        ]

    def _build_initial_conversation(
        self,
        task: str,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> List[ConversationMessage]:
        prompt_body = self._render_prompt(step=step, shared_context=shared_context, task=task)
        return [
            ConversationMessage(role="system", content=_SYSTEM_PROMPT, metadata={}),
            ConversationMessage(role="user", content=prompt_body, metadata={"task": task}),
        ]

    def _render_prompt(
        self,
        *,
        step: StepSpec,
        shared_context: Mapping[str, Any],
        task: Optional[str] = None,
    ) -> str:
        planning_context = self._extract_planning_context(step, shared_context)
        objective = planning_context.get("objective") or step.description or task or ""
        codebase = planning_context.get("codebase_context") or {}

        structure = (
            codebase.get("structure")
            or planning_context.get("structure")
            or planning_context.get("codebase_structure")
        )
        languages = codebase.get("languages") or planning_context.get("languages")
        frameworks = codebase.get("frameworks") or planning_context.get("frameworks")
        entry_points = codebase.get("entry_points") or planning_context.get("entry_points")
        test_framework = codebase.get("test_framework") or planning_context.get("test_framework")
        git_status = codebase.get("git_status") or planning_context.get("git_status")
        recent_commits = codebase.get("recent_commits") or planning_context.get("recent_commits")
        dependencies = codebase.get("dependencies") or planning_context.get("dependencies")
        constraints = planning_context.get("constraints")

        available_agents = planning_context.get("available_agents") or _DEFAULT_AGENT_DESCRIPTIONS

        sections: List[str] = [
            "OBJECTIVE:",
            objective,
            "",
            "CODEBASE CONTEXT:",
            _join_lines("Structure", structure),
            _join_lines("Languages", languages),
            _join_lines("Frameworks", frameworks),
            _join_lines("Entry Points", entry_points),
            _join_lines("Test Framework", test_framework),
            _join_lines("Git Status", git_status),
            _join_lines("Recent Commits", recent_commits),
            _join_lines("Dependencies", dependencies),
        ]

        if constraints:
            sections.extend(["", "CONSTRAINTS:", _format_value(constraints)])

        sections.extend(
            [
                "",
                "AVAILABLE AGENTS:",
                "\n".join(f"- {agent}" for agent in available_agents),
                "",
                "TASK:",
                "Create a detailed execution plan with:",
                "1. Steps (id, description, agent_id, dependencies)",
                "2. Parallel groups (steps that can run concurrently)",
                "3. Branch strategy details",
                "4. Estimated duration per step",
                "",
                "RULES:",
                "- Steps must be granular and outcome-driven.",
                "- Identify independent steps for parallel execution.",
                "- Match agents to step requirements based on expertise.",
                "- Include dependencies using depends_on with step IDs.",
                "- Provide reasoning for plan decisions.",
                "",
                "OUTPUT FORMAT (JSON):",
                '{',
                '  "objective": "...",',
                '  "steps": [...],',
                '  "parallel_groups": [...],',
                '  "reasoning": "..."',
                '}',
            ]
        )

        return "\n".join(sections)

    def _extract_planning_context(
        self,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context = {}
        if "planning_context" in shared_context:
            context = dict(shared_context["planning_context"])
        elif "planning_context" in step.inputs:
            context = dict(step.inputs["planning_context"])

        # Merge legacy keys if present
        context.setdefault("objective", step.inputs.get("objective") or step.description)
        return context

    # ------------------------------------------------------------------
    # Provider integration
    # ------------------------------------------------------------------
    async def _complete_chat(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if self.provider == "ollama":
            return await self._call_ollama(messages)
        if self.provider == "anthropic":
            return await self._call_anthropic(messages)
        if self.provider == "openai":
            return await self._call_openai(messages)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    async def _call_ollama(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        assert self._ollama_client is not None  # For mypy
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "think": True,
        }

        try:
            response = await asyncio.to_thread(self._ollama_client.chat, **payload)
        except OllamaError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        message = response.get("message", {})
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""
        reasoning, remainder = _extract_reasoning_blocks(thinking + "\n" + content)
        if not remainder:
            remainder = content.strip()
        metadata: Dict[str, Any] = {}
        if "context" in response:
            metadata["context"] = response["context"]
        if reasoning and not thinking:
            metadata["reasoning_source"] = "content"
        elif thinking:
            metadata["reasoning_source"] = "thinking_field"
        return remainder, reasoning, metadata

    async def _call_anthropic(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("Anthropic API key not configured.")
        assert self._http_client is not None

        system_segments = [msg["content"] for msg in messages if msg["role"] == "system"]
        system_prompt = "\n\n".join(system_segments) if system_segments else None

        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            converted.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": msg["content"]}],
                }
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = await self._http_client.post(
            _ANTHROPIC_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Anthropic API error {response.status_code}: {response.text}")

        body = response.json()
        content_blocks = body.get("content", [])
        text_segments = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(segment for segment in text_segments if segment).strip()
        reasoning, remainder = _extract_reasoning_blocks(text)
        metadata = {"usage": body.get("usage", {})}
        return remainder or text, reasoning, metadata

    async def _call_openai(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured.")
        assert self._http_client is not None

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }

        response = await self._http_client.post(
            _OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI API returned no choices.")
        message = choices[0].get("message") or {}
        text = message.get("content", "") or ""
        reasoning, remainder = _extract_reasoning_blocks(text)
        metadata = {"usage": body.get("usage", {}), "finish_reason": choices[0].get("finish_reason")}
        return remainder or text, reasoning, metadata

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_default_api_key(provider: str) -> Optional[str]:
        if provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return None


class _ReasoningStreamingSession(StreamingAgentSession):
    """Streaming session built on top of repeated reasoning model calls."""

    def __init__(
        self,
        *,
        adapter: ReasoningAgentAdapter,
        initial_messages: List[ConversationMessage],
        timeout: float,
    ) -> None:
        self._adapter = adapter
        self._timeout = timeout
        self._history: List[Dict[str, str]] = []
        self._queue: asyncio.Queue[ConversationMessage] = asyncio.Queue()
        self._generation_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._started = False
        self._initial_messages = initial_messages
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for message in self._initial_messages:
            await self.send(message)

    async def send(self, message: ConversationMessage) -> None:
        if self._closed:
            raise RuntimeError("Session already closed.")
        record = {"role": message.role, "content": message.content}
        self._history.append(record)
        if message.role == "user":
            await self._schedule_generation()

    async def receive(self, *, timeout: Optional[float] = None) -> Optional[ConversationMessage]:
        if self._closed:
            return None
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._generation_task:
            self._generation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._generation_task

    async def _schedule_generation(self) -> None:
        async with self._lock:
            if self._generation_task and not self._generation_task.done():
                return
            self._generation_task = asyncio.create_task(self._produce_reply())

    async def _produce_reply(self) -> None:
        try:
            assistant_text, reasoning_trace, provider_meta = await self._adapter._complete_chat(self._history)
        except Exception as exc:  # noqa: BLE001 - surface to caller
            LOGGER.exception("Streaming reasoning session failed: %s", exc)
            await self._queue.put(
                ConversationMessage(
                    role="system",
                    content=f"Planner error: {exc}",
                    metadata={"error": True},
                )
            )
            return

        reply = ConversationMessage(
            role="assistant",
            content=assistant_text,
            metadata={
                "provider": self._adapter.provider,
                "model": self._adapter.model,
                "reasoning_trace": reasoning_trace,
                "provider_meta": provider_meta,
            },
        )
        self._history.append({"role": "assistant", "content": assistant_text})
        await self._queue.put(reply)
