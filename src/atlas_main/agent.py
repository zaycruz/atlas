"""Atlas Lite agent: minimal working + simple episodic memory and streaming chat.

Removed subsystems: enhanced episodic (chroma), semantic profile, journal, identity,
desires, tools, critic/controller. This keeps only:
 - WorkingMemory (recent turns)
 - Simple episodic memory (vector recall if embed succeeds; otherwise skipped)
 - Streaming chat via OllamaClient
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from .memory import WorkingMemory, EpisodicMemory, render_memory_snippets
from .ollama import OllamaClient, OllamaError

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "qwen3:latest")
# Recommended: align with E5 used by enhanced memory (set ATLAS_EMBED_MODEL to override)
DEFAULT_EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "e5-large")
DEFAULT_MEMORY_PATH = Path(os.getenv("ATLAS_MEMORY_PATH", "~/.local/share/atlas/episodic.json"))
AVAILABLE_TOOLS = "(tooling disabled in lite build)"

DEFAULT_PROMPT = (
    "You are Atlas Lite — a minimal local model REPL companion.\n"
    "Respond concisely and helpfully. No external tools are available.\n"
    "If information is uncertain, say so briefly. Keep answers grounded and minimally verbose."
)



# DEFAULT_PROMPT is defined above as the primary system prompt.
# (Legacy "free will" prompt removed to avoid unsafe/no‑rules behavior.)

TOOL_REQUEST_PATTERN = None  # tooling disabled


class AtlasAgent:
    def __init__(
        self,
        client: OllamaClient,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        working_memory_limit: int = 12,
        episodic_limit: int = 240,
        system_prompt: str = DEFAULT_PROMPT,
        semantic_model: Optional[str] = None,
        policies_path: Optional[Path] = None,
    ):
        self.client = client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.system_prompt = system_prompt
        self.working_memory = WorkingMemory(capacity=working_memory_limit)
        self.show_thinking: bool = False
        self.episodic_memory = EpisodicMemory(
            DEFAULT_MEMORY_PATH,
            embedding_fn=self._embed_text,
            max_records=episodic_limit,
        )
        self._pending_tool_request = None  # retained for API compatibility

    def _load_policies(self) -> dict:  # compatibility stub
        return {}

    # ------------------------------------------------------------------
    def _embed_text(self, text: str):
        try:
            return self.client.embed(self.embed_model, text)
        except OllamaError:
            return None

    def _build_system_prompt(self, user_text: str) -> str:
        episodic = self.episodic_memory.recall(user_text, top_k=4)
        episodic_snippets = render_memory_snippets(episodic) or "(no episodic memories yet)"

        return (
            f"{self.system_prompt}\n\n"
            f"{AVAILABLE_TOOLS}\n\n"
            f"Relevant memories:\n{episodic_snippets}\n\n"
        )

    def respond(self, user_text: str, *, stream_callback=None) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        self.working_memory.add_user(user_text)
        system_content = self._build_system_prompt(user_text)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.working_memory.to_messages())

        self._pending_tool_request = None
        accumulator: list[str] = []
        sent_len = 0
        try:
            for chunk in self.client.chat_stream(model=self.chat_model, messages=messages):
                accumulator.append(chunk)
                visible, _ = self._compute_visible_output("".join(accumulator))
                if stream_callback and sent_len < len(visible):
                    stream_callback(visible[sent_len:])
                    sent_len = len(visible)
        except OllamaError as exc:
            if stream_callback:
                stream_callback(f"\n[error] {exc}")
            return f"I hit an error contacting Ollama: {exc}"

        full_response = "".join(accumulator)
        visible, tool_request = self._compute_visible_output(full_response)
        if stream_callback and sent_len < len(visible):
            stream_callback(visible[sent_len:])

        text = visible.strip()
        if not text:
            text = "I'm not sure how to respond to that."

        self.working_memory.add_assistant(text)
        self.episodic_memory.remember(user_text, text)
        # semantic + journaling disabled
        self._pending_tool_request = None
        return text

    def reset(self) -> None:
        self.working_memory.clear()
        self.episodic_memory.clear()
        self._pending_tool_request = None

    def pop_tool_request(self) -> Optional[dict]:
        request = self._pending_tool_request
        self._pending_tool_request = None
        return request

    def set_chat_model(self, model: str) -> None:
        self.chat_model = model.strip() or self.chat_model

    def update_system_prompt(self, prompt: str) -> None:
        if prompt.strip():
            self.system_prompt = prompt.strip()

    # ------------------------------------------------------------------
    def _maybe_journal(self, *args, **kwargs):  # stub
        return None

    # ------------------------------------------------------------------
    def _compute_visible_output(self, text: str) -> tuple[str, Optional[dict]]:
        tool_request = None
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
        return visible, tool_request
