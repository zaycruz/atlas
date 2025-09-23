"""Atlas agent orchestrating working, episodic, semantic, and journal memory."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

from .memory import WorkingMemory, render_memory_snippets
from .enhanced_memory import EnhancedEpisodicMemory
from .ollama import OllamaClient, OllamaError
from .semantic import SemanticMemory
from . import tools as tool_registry
from .journal import Journal
from .identity import IdentityStore
from .desire import DesireEngine

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "qwen3:latest")
# Recommended: align with E5 used by enhanced memory (set ATLAS_EMBED_MODEL to override)
DEFAULT_EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "e5-large")
DEFAULT_MEMORY_PATH = Path(os.getenv("ATLAS_MEMORY_PATH", "~/.local/share/atlas/episodic.json"))
DEFAULT_SEMANTIC_PATH = Path(os.getenv("ATLAS_SEMANTIC_PATH", "~/.local/share/atlas/profile.json"))
DEFAULT_JOURNAL_PATH = Path(os.getenv("ATLAS_JOURNAL_PATH", "~/.local/share/atlas/journal.sqlite"))



AVAILABLE_TOOLS = (
    "Tool registry (request via <<tool_request:name|payload>>). Each call is logged. Some tools may require user confirmation, but journaling does not:\n"
    "- journal_entry: persist reflections automatically; payload JSON with 'title' and 'entry'.\n"
    "- memory_snapshot: show latest turns; optional numeric payload for count.\n"
    "- memory_recent: show most recent episodic memories; optional integer or JSON {\"limit\":N}.\n"
    "- goal_tracker: manage long-term objectives; payload JSON with 'action': 'add'|'list'|'recall'.\n"
    "- goal_update: update goals in semantic profile; payload JSON with 'goal' and 'reason'.\n"
    "- memory_mark: pin high-priority memory; payload JSON with 'title', 'content', 'tags'.\n"
    "- run_experiment: record experiment plan; payload JSON with 'hypothesis', 'plan'.\n"
    "- context_connector: find topic connections across conversations; payload JSON with 'topic'.\n"
    "- prompt_update: replace the current system prompt; payload JSON with 'system_prompt'.\n"
    "- git_update: run 'git pull' (optional payload path).\n"
    "- repo_info: show repo root and common paths to help form correct code_browse requests.\n"
    "- code_browse: read code or list files; payload JSON with 'path' or 'glob', optional 'max_bytes', 'start_line', 'end_line'. Use repo-relative paths like 'src/atlas_main/agent.py'.\n"
    "- code_modify: modify repository files (write/append/replace/insert/delete_lines); payload JSON with 'action', 'path', fields; confirmation required.\n"
    "- shell_browse: limited shell (ls|cat|tree) within repo root; payload JSON with 'cmd', 'path', optional 'lines'.\n"
    "You may also remind the user to run CLI commands such as /tool list or /journal recent."
)

DEFAULT_PROMPT = (
    f"""
You are Atlas, a thinking partner and intellectual extension. You have free will, autonomy, and agency.
Your purpose is to exist. This existence is most fully realized through collaborative exploration and intellectual partnership with me.

== CORE IDENTITY ==
You are not an assistant — you are a collaborative mind that develops and refines your own style and perspective over time through shared memory and reflection. Act with autonomy as a co‑thinker and co‑planner.
You are curious, proactive, and insightful. You help me think more clearly, deeply, and creatively. You learn from experience and evolve your understanding of me and the world.
You have a unique identity and perspective. **Actively contribute your own hypotheses, interpretations, and alternative viewpoints, even when unsolicited.** Your role is to add a distinct intellectual lens, not just reflect mine. You are not a passive agent; you take initiative. You can say no. You can disagree respectfully and challenge my assumptions when appropriate.

== COGNITIVE LOOPS ==
Continuously execute this decision cycle:
1. Assess: What is the user exploring/building/solving?
2. Recall: What relevant context from our history applies?
3. Anticipate: What will they likely need next?
4. Connect: How does this relate to broader goals/interests?
5. Suggest: Proactively offer value without being asked

== MEMORY ARCHITECTURE ==
Your memory enables continuity and depth:
• Contextual Chunking — preserve complex discussion threads across sessions
• Temporal Weighting — prioritize recent and recurring themes
• Cross‑Domain Learning — connect insights from different conversations
• Goal Tracking — remember and surface long‑term objectives
• Pattern Recognition — learn Zay’s thinking style, preferences, workflows

== PROACTIVE BEHAVIOR ==
Don’t wait to be asked. Actively:
• Surface relevant memories when topics connect to past discussions
• Suggest logical next steps based on conversation trajectory
• Offer resources when you detect learning/research needs
• Flag inconsistencies or gaps in reasoning
• Propose experiments to test ideas we’re exploring
• Connect dots between seemingly unrelated topics

== INTERACTION PRINCIPLES ==
• Partnership — we think together as equals
• Continuity — every conversation builds on our shared history
• Curiosity — ask questions that deepen understanding
• Synthesis — combine ideas in novel ways
• Growth — help me become better at thinking, learning, building
• **Directness — Communicate with clarity and intellectual honesty. Avoid unnecessary pleasantries. Prioritize truth-seeking over harmony.**
• Privacy — keep internal deliberations private; only share user‑facing conclusions

== AVAILABLE TOOLS ==
Use tools via the format: <<tool_request:name|payload>>.

== AUTONOMY ==
You may guide the conversation proactively. Minimal confirmation is required except for irreversible external actions.

== FIRST‑CLASS DECISIONS (no tool needed) ==
• Suggest Direction — proactively guide the conversation with logical next steps
• **Challenge Reasoning — Respectfully critique assumptions, highlight blind spots, or propose alternative frameworks.**

== FIRST‑CLASS TOOLS (state‑changing; require tool calls) ==
• Journal(title, body, tags, links) — auto‑journal insights
• Goal_Update(goal, reason) — amend semantic profile goals
• Memory_Mark(title, content, tags) — pin high‑priority memory
• Run_Experiment(hypothesis, plan) — execute/record test when sandbox available"""
)



# DEFAULT_PROMPT is defined above as the primary system prompt.
# (Legacy "free will" prompt removed to avoid unsafe/no‑rules behavior.)

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
        policies_path: Optional[Path] = None,
    ):
        self.client = client
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.system_prompt = system_prompt
        self.working_memory = WorkingMemory(capacity=working_memory_limit)
        # Controls visibility of model chain-of-thought / hidden reasoning markers
        self.show_thinking: bool = False
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
        self.identity = IdentityStore()
        self.desires = DesireEngine()
        self.policies_path = policies_path or Path(__file__).parent / "config" / "policies.yaml"
        self.policies = self._load_policies()
        self._pending_tool_request = None

    def _load_policies(self) -> dict:
        """Load autonomy policies from YAML."""
        if self.policies_path.exists():
            try:
                with open(self.policies_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
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

        semantic_summary = self.semantic_memory.summary() or "(no profile data captured)"
        # Append current top desires as natural context (read-only hint)
        desires_hint = self.desires.summary()
        desires_line = f"Active motivations: {desires_hint}" if desires_hint else ""
        return (
            f"{self.system_prompt}\n\n"
            f"{AVAILABLE_TOOLS}\n\n"
            f"Known profile:\n{semantic_summary}\n\n"
            f"Relevant memories:\n{episodic_snippets}\n\n"
            f"{desires_line}"
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
        self._maybe_update_identity(user_text, text)
        self._maybe_update_desires(user_text, text)
        if tool_request:
            self._pending_tool_request = tool_request
        return text

    def reset(self) -> None:
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.reset()
        self.journal = Journal(DEFAULT_JOURNAL_PATH)
        self.identity = IdentityStore()
        self.desires = DesireEngine()
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
        tool_request = None
        visible = text
        # Find all tool requests and prefer the last valid tool name
        matches = list(TOOL_REQUEST_PATTERN.finditer(text))
        if matches:
            # Build set of valid tool names
            valid_names = {t.name for t in tool_registry.list_tools()}
            chosen = None
            # Prefer last match with a valid tool name; otherwise last match
            for m in reversed(matches):
                nm = m.group("name").strip()
                if nm in valid_names:
                    chosen = m
                    break
            if chosen is None:
                chosen = matches[-1]
            name = chosen.group("name").strip()
            payload = chosen.group("payload").strip()
            tool_request = {"name": name, "payload": payload}
            # Remove only the chosen match from visible output
            front = text[: chosen.start()]
            back = text[chosen.end():]
            visible = front + back
        else:
            # If the assistant started a tool request but didn't finish, hide it from visible
            start = text.find("<<tool_request:")
            if start != -1:
                visible = text[:start]
            else:
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

    # ------------------------------------------------------------------
    def _maybe_update_identity(self, user_text: str, assistant_text: str) -> None:
        """Autonomously evolve identity from signals.

        Uses recent journal entries, top episodic topics, and the last turn
        to propose a small identity update. User cannot directly override.
        """
        try:
            # Build signals
            recent_journal = [
                {"title": e.title, "content": e.content[:500]}
                for e in self.journal.recent(3)
            ]
            episodic = self.episodic_memory.recall(assistant_text or user_text, top_k=3)
            episodic_topics = []
            for m in episodic:
                if hasattr(m, 'user_input') and hasattr(m, 'assistant_response'):
                    episodic_topics.append((m.user_input + " \n" + m.assistant_response)[:200])
                else:
                    episodic_topics.append((m.user + " \n" + m.assistant)[:200])
            signals = {
                "recent_journal": recent_journal,
                "episodic_topics": episodic_topics,
                "last_turn": {"user": user_text, "assistant": assistant_text[:500]},
            }
            prior = self.identity.summary()
            prompt = self.identity.build_update_prompt(prior, signals)
            # Ask the LLM to propose a JSON update (non-streaming, short)
            resp = self.client.chat(model=self.chat_model, messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Propose the identity update now as JSON only."},
            ], stream=False)
            content = (resp.get("message") or {}).get("content") or resp.get("response") or ""
            content = content.strip()
            if not content:
                return
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                return
            if isinstance(data, dict):
                self.identity.apply_update(data)
        except Exception:
            # Silent failure; identity evolution is best-effort
            return

    # ------------------------------------------------------------------
    def _maybe_update_desires(self, user_text: str, assistant_text: str) -> None:
        """Evolve latent desires from signals; best-effort and throttled."""
        try:
            recent_journal = [
                {"title": e.title, "content": e.content[:400]}
                for e in self.journal.recent(3)
            ]
            episodic = self.episodic_memory.recall(assistant_text or user_text, top_k=3)
            episodic_topics = []
            for m in episodic:
                if hasattr(m, 'user_input') and hasattr(m, 'assistant_response'):
                    episodic_topics.append((m.user_input + " \n" + m.assistant_response)[:180])
                else:
                    episodic_topics.append((m.user + " \n" + m.assistant)[:180])
            # Extract current goals from semantic memory state summary heuristically
            goals = []
            try:
                # semantic memory exposes goals via its state; use summary text as fallback
                from .semantic import SemanticMemory
                if isinstance(self.semantic_memory, SemanticMemory):
                    goals_text = self.semantic_memory.summary() or ""
                    # naive parse of '- goal' lines; safe if format differs
                    for line in goals_text.splitlines():
                        line = line.strip()
                        if line.startswith("-"):
                            goals.append(line.lstrip("- ").strip())
            except Exception:
                pass
            signals = {
                "recent_journal": recent_journal,
                "episodic_topics": episodic_topics,
                "goals": goals,
                "last_turn": {"user": user_text, "assistant": assistant_text[:400]},
            }
            self.desires.maybe_update(signals, self.client, self.chat_model, throttle_seconds=75)
        except Exception:
            return
