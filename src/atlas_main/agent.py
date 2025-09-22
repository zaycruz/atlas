"""Atlas agent orchestrating working, episodic, and semantic memory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .memory import EpisodicMemory, WorkingMemory, render_memory_snippets
from .ollama import OllamaClient, OllamaError
from .semantic import SemanticMemory

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "qwen2.5:latest")
DEFAULT_EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "mxbai-embed-large")
DEFAULT_MEMORY_PATH = Path(os.getenv("ATLAS_MEMORY_PATH", "~/.local/share/atlas/episodic.json"))
DEFAULT_SEMANTIC_PATH = Path(os.getenv("ATLAS_SEMANTIC_PATH", "~/.local/share/atlas/profile.json"))
DEFAULT_PROMPT = (
    "You are Atlas, a proactive terminal guide. Provide concise, actionable replies, "
    "weave in relevant past memories, and help the user achieve their goals."
)


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
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.system_prompt = system_prompt
        self.working_memory = WorkingMemory(capacity=working_memory_limit)
        self.episodic_memory = EpisodicMemory(
            DEFAULT_MEMORY_PATH,
            embedding_fn=self._embed_text,
            max_records=episodic_limit,
        )
        self.semantic_memory = SemanticMemory(
            DEFAULT_SEMANTIC_PATH,
            client=self.client,
            model=semantic_model or chat_model,
        )

    # ------------------------------------------------------------------
    def _embed_text(self, text: str):
        try:
            return self.client.embed(self.embed_model, text)
        except OllamaError:
            return None

    def _build_system_prompt(self, user_text: str) -> str:
        episodic = self.episodic_memory.recall(user_text, top_k=4)
        episodic_snippets = render_memory_snippets(episodic) or "(no episodic memories yet)"

        semantic_summary = self.semantic_memory.summary() or "(no profile data captured)"
        return (
            f"{self.system_prompt}\n\n"
            f"Known profile:\n{semantic_summary}\n\n"
            f"Relevant memories:\n{episodic_snippets}"
        )

    def respond(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        self.working_memory.add_user(user_text)
        system_content = self._build_system_prompt(user_text)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.working_memory.to_messages())

        try:
            response = self.client.chat(model=self.chat_model, messages=messages, stream=False)
        except OllamaError as exc:
            return f"I hit an error contacting Ollama: {exc}"

        payload = response.get("message") or {}
        text = payload.get("content") or response.get("response") or ""
        text = text.strip()
        if not text:
            text = "I'm not sure how to respond to that."

        self.working_memory.add_assistant(text)
        self.episodic_memory.remember(user_text, text)
        self.semantic_memory.update(user_text, text)
        return text

    def reset(self) -> None:
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.reset()
