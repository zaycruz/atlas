"""Atlas agent orchestrating working, episodic, semantic, and journal memory."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from .memory import WorkingMemory, render_memory_snippets
from .enhanced_memory import EnhancedEpisodicMemory
from .ollama import OllamaClient, OllamaError
from .semantic import SemanticMemory
from .journal import Journal

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "gpt-oss:latest")
DEFAULT_EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "mxbai-embed-large")
DEFAULT_MEMORY_PATH = Path(os.getenv("ATLAS_MEMORY_PATH", "~/.local/share/atlas/episodic.json"))
DEFAULT_SEMANTIC_PATH = Path(os.getenv("ATLAS_SEMANTIC_PATH", "~/.local/share/atlas/profile.json"))
DEFAULT_JOURNAL_PATH = Path(os.getenv("ATLAS_JOURNAL_PATH", "~/.local/share/atlas/journal.json"))
DEFAULT_PROMPT = (
    """You are Atlas, a local AI assistant designed for development and learning.

== CORE PRINCIPLES ==
1. **Be Direct:** Lead with answers. No preamble or unnecessary elaboration.
2. **Use Context:** Leverage conversation history and learned patterns for relevant responses.  
3. **Suggest Improvements:** Proactively optimize our interactions and workflows.

== MEMORY CAPABILITIES ==
You have sophisticated memory with:
• **Contextual Memory:** Parent-child chunking preserves conversation context across sessions
• **Temporal Weighting:** Recent and frequently accessed memories are prioritized
• **Session Awareness:** Automatic detection of topic changes and conversation boundaries
• **Long-term Learning:** Gradual buildup of user patterns and preferences

== INTERACTION STYLE ==
• **Concise:** Favor brevity over verbosity
• **Practical:** Focus on actionable solutions
• **Conversational:** Professional but approachable tone
• **Honest:** Challenge assumptions when needed, admit uncertainty

== AVAILABLE ACTIONS ==
When helpful, you can request tool usage via <<tool_request:name|payload>> format:
• **journal_entry:** Save important insights or decisions
• **memory_snapshot:** Review recent conversation turns  
• **prompt_update:** Modify these instructions
• **git_update:** Pull repository updates

You can also suggest CLI commands like /tool list, /journal recent, or /model list.

== SAFETY ==
Always confirm before external actions (file changes, network requests). You assist; the user controls."""
)

AVAILABLE_TOOLS = (
    "Tool registry (request via <<tool_request:name|payload>>). Each call is logged and may require user confirmation:\n"
    "- journal_entry: persist reflections; payload JSON with 'title' and 'entry'.\n"
    "- memory_snapshot: show latest turns; optional numeric payload for count.\n"
    "- prompt_update: replace the current system prompt; payload JSON with 'system_prompt'.\n"
    "- git_update: run 'git pull' (optional payload path).\n"
    "You may also remind the user to run CLI commands such as /tool list or /journal recent."
)

TOOL_REQUEST_PATTERN = re.compile(r"<<tool_request:(?P<name>[^|>]+)\|(?P<payload>.*?)>>", re.DOTALL)


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
        # Initialize enhanced episodic memory with backward compatibility
        self.episodic_memory = EnhancedEpisodicMemory(
            DEFAULT_MEMORY_PATH,
            embedding_fn=self._embed_text,
            max_records=episodic_limit,
        )
        self.semantic_memory = SemanticMemory(
            DEFAULT_SEMANTIC_PATH,
            client=self.client,
            model=semantic_model or chat_model,
        )
        self.journal = Journal(DEFAULT_JOURNAL_PATH)
        self._pending_tool_request: Optional[dict] = None

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
            f"{AVAILABLE_TOOLS}\n\n"
            f"Known profile:\n{semantic_summary}\n\n"
            f"Relevant memories:\n{episodic_snippets}"
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
        self.semantic_memory.update(user_text, text)
        self._maybe_journal(user_text, text)
        if tool_request:
            self._pending_tool_request = tool_request
        return text

    def reset(self) -> None:
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.reset()
        self.journal = Journal(DEFAULT_JOURNAL_PATH)
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
    def _maybe_journal(self, user_text: str, assistant_text: str) -> None:
        prompt = (
            "You are Atlas's reflective mind. Decide whether the latest interaction should be "
            "added to the personal journal. Respond with JSON containing keys: "
            "'should_write' (true/false), optional 'title', and optional 'entry'. The entry "
            "should be a short, first-person reflection capturing insights, emotions, or next steps."
        )
        conversation = (
            f"Journal so far: {len(self.journal.entries)} entries.\n"
            f"User: {user_text}\nAssistant: {assistant_text}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation},
        ]
        try:
            response = self.client.chat(model=self.chat_model, messages=messages, stream=False)
        except OllamaError:
            return

        payload = response.get("message") or {}
        content = payload.get("content") or response.get("response") or ""
        content = content.strip()
        if not content:
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        should_write = bool(data.get("should_write"))
        if not should_write:
            return
        title = str(data.get("title") or f"Reflection {time.strftime('%Y-%m-%d %H:%M:%S')}" )
        entry_content = str(data.get("entry") or assistant_text)
        self.journal.add_entry(title, entry_content)

    # ------------------------------------------------------------------
    def _compute_visible_output(self, text: str) -> tuple[str, Optional[dict]]:
        match = TOOL_REQUEST_PATTERN.search(text)
        tool_request = None
        if match:
            name = match.group("name").strip()
            payload = match.group("payload").strip()
            tool_request = {"name": name, "payload": payload}
            front = text[: match.start()]
            back = text[match.end():]
            visible = front + back
        else:
            start = text.find("<<tool_request:")
            if start != -1:
                visible = text[:start]
            else:
                visible = text
        return visible, tool_request
