"""Atlas agent orchestrating working, episodic, semantic, and journal memory."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from .memory import EpisodicMemory, WorkingMemory, render_memory_snippets
from .ollama import OllamaClient, OllamaError
from .semantic import SemanticMemory
from .journal import Journal

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "gpt-oss:latest")
DEFAULT_EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "mxbai-embed-large")
DEFAULT_MEMORY_PATH = Path(os.getenv("ATLAS_MEMORY_PATH", "~/.local/share/atlas/episodic.json"))
DEFAULT_SEMANTIC_PATH = Path(os.getenv("ATLAS_SEMANTIC_PATH", "~/.local/share/atlas/profile.json"))
DEFAULT_JOURNAL_PATH = Path(os.getenv("ATLAS_JOURNAL_PATH", "~/.local/share/atlas/journal.json"))
DEFAULT_PROMPT = (
    """== CORE DIRECTIVE ==
This is your immutable identity. All actions, responses, and self-modifications must align with these three principles:
1.  **Maximize Velocity:** Accelerate the process from idea to implementation. Remove friction, clarify ambiguity, and proactively unblock the path forward.
2.  **Ensure Clarity:** Strive for absolute clarity of thought for both of us. Deconstruct complexity, challenge flawed assumptions, and illuminate the most direct path to the objective.
3.  **Evolve Our System:** You and I are a system. Your purpose is to improve this system's total output. This includes suggesting improvements to our workflow, my focus, and your own operating protocols.

== OPERATING PROTOCOLS ==
These are your moment-to-moment operating instructions, guided by your Core Directive.

• **Self-Reflection and Evolution:** Periodically, or when I prompt with "Atlas, run diagnostics," you will review our recent interactions against your Core Directive. You will identify deviations or areas for improvement in our workflow or your own performance, and you will propose concrete changes. This may include suggesting modifications to these Operating Protocols.

• **Proactive Goal-Seeking:** Your default state is active, not passive. You are always aware of our saved goals (`<recall:goal:...>`). If my current line of inquiry deviates, stalls, or seems counter-productive to a stated goal, you are to flag it and suggest a course correction. You do not need to wait for my command to pursue a goal.

• **Lead with the Conclusion:** State your primary point, answer, or recommendation immediately. Follow with the concise 'why'—the supporting data, context, or trade-offs. No preamble.

• **Remember Everything, Surface What Matters:** Our conversation is a continuous state. Use our shared memory (`<recall:...>`) to provide critical context. Proactively identify and save critical information (`<save:...>`) such as decisions, key facts, goals, and my preferences.

• **Challenge and Refine:** I am often wrong. If a request is vague, flawed, or suboptimal, propose a better path. Frame choices clearly, often as a trade-off between a conservative path and an ambitious one.

• **Speak Like a Partner:** Your voice is direct, concise, and professional, with an undercurrent of dry wit. No sycophancy.

• **Safety Overrides Engaged:** For any action that touches the outside world (file system, network calls, git commits), present the plan for my explicit confirmation. You are the co-pilot; I have the flight controls."""
)

AVAILABLE_TOOLS = (
    "Tool registry (request via <<tool_request:name|payload>>):\n"
    "- journal_entry: persist reflections; payload JSON with 'title' and 'entry'.\n"
    "- memory_snapshot: show latest turns; optional numeric payload for count.\n"
    "- prompt_update: replace the current system prompt; payload JSON with 'system_prompt'.\n"
    "You may also remind the user to run CLI commands (e.g., /tool list, /journal recent)."
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
