"""Atlas Reasoning Loop Controller - Jarvis v1 Core

Orchestrates the single path to output: Perceive → Retrieve → Reason → (Tools if needed) → Critic → Respond → Reflect
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from ..agent import AtlasAgent
from ..ollama import OllamaClient
from .. import tools as tool_registry


class ReasoningController:
    """Controller for Atlas's reasoning loop."""

    def __init__(self, agent: AtlasAgent, policies_path: Path):
        self.agent = agent
        self.policies = self._load_policies(policies_path)
        self.last_reflection_turn = 0
        self.turn_count = 0
        # Internal tool loop steps (configurable); defaults based on model capability
        self.max_steps = self.default_max_steps_for_model(self.agent.chat_model)

    def _load_policies(self, path: Path) -> Dict[str, Any]:
        """Load autonomy policies from YAML."""
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    @staticmethod
    def default_max_steps_for_model(model_name: str) -> int:
        name = (model_name or "").lower()
        if "qwen" in name:
            return 7
        if "gpt-oss" in name:
            return 1
        return 5

    def set_max_steps(self, steps: int) -> None:
        self.max_steps = max(1, int(steps))

    def process_turn(self, user_input: str, stream_callback=None, max_steps: Optional[int] = None) -> str:
        """Process a single conversation turn through the reasoning loop.

        Phases:
        1. Perceive: Parse user input
        2. Retrieve: Get relevant memories (already in agent._build_system_prompt)
        3. Reason: Generate response with LLM
        4. Tools: Execute any triggered tools (auto for safe tools; requeue confirm tools)
        5. Respond: Return response
        6. Reflect: Auto-journal if policy triggers
        """
        self.turn_count += 1
        logging.info(
            f"[REASONING LOOP] Turn {self.turn_count}: Starting processing for input: {user_input[:50]}..."
        )

        # 1: Perceive
        logging.debug("[REASONING LOOP] Phase 1: Perceive - Parsing user input")

        # 2: Retrieve
        logging.debug("[REASONING LOOP] Phase 2: Retrieve - Gathering relevant memories")

        # 3-4: Reason + Tools (with limited internal loop)
        logging.debug("[REASONING LOOP] Phase 3: Reason - Generating response with LLM")
        step = 0
        response = ""
        last_error_feedback: Optional[str] = None
        steps_budget = max_steps if isinstance(max_steps, int) and max_steps > 0 else self.max_steps
        while step < steps_budget:
            step += 1
            logging.debug(f"[REASONING LOOP] Step {step}/{steps_budget}")
            # Stream only on final step for a clean user experience
            do_stream = stream_callback if step == steps_budget else None
            prompt = user_input if step == 1 else (last_error_feedback or user_input)
            response = self.agent.respond(prompt, stream_callback=do_stream)
            logging.info(
                f"[REASONING LOOP] Phase 3: Reason - Response generated: {response[:100]}..."
            )

            # 4: Tools
            logging.debug("[REASONING LOOP] Phase 4: Tools - Checking for tool requests")
            pending = self.agent.pop_tool_request()
            if not pending:
                # No tools requested; if we haven't streamed yet, stream now
                if stream_callback and step < steps_budget:
                    try:
                        stream_callback(response)
                    except Exception:
                        pass
                break
            logging.info(
                f"[REASONING LOOP] Phase 4: Tools - Tool request detected: {pending}"
            )

            tool = tool_registry.get_tool(pending.get("name", ""))
            if not tool:
                last_error_feedback = (
                    f"The requested tool '{pending.get('name')}' is unknown. Call 'repo_info' and retry with a valid path/tool."
                )
                logging.warning(
                    "[REASONING LOOP] Unknown tool requested; providing feedback and retrying"
                )
                continue
            if tool.requires_confirmation:
                # Defer to CLI/user confirmation; provide guidance and requeue
                last_error_feedback = (
                    f"The tool '{tool.name}' requires user confirmation. Consider using 'repo_info' or 'code_browse' first or ask the user to confirm."
                )
                logging.info(
                    "[REASONING LOOP] Tool requires confirmation; halting internal loop for user consent"
                )
                # Requeue the pending tool so the CLI layer can prompt the user
                try:
                    self.agent._pending_tool_request = pending  # internal coordination
                except Exception:
                    pass
                break

            # Execute non-confirmation tools directly
            result = tool_registry.execute_tool(
                self.agent, tool.name, pending.get("payload", "")
            )
            if result.success:
                logging.info(f"[REASONING LOOP] Tool '{tool.name}' executed successfully")
                last_error_feedback = (
                    f"Tool '{tool.name}' succeeded with: {result.message[:500]}\nContinue based on this result."
                )
                continue
            logging.warning(
                f"[REASONING LOOP] Tool '{tool.name}' failed: {result.message}"
            )
            last_error_feedback = (
                f"The tool '{tool.name}' failed with: {result.message}. If this is a path issue, call 'repo_info' and retry using a repo-relative path like 'src/atlas_main/agent.py'."
            )
            # Loop again with error feedback

        # 5: Respond
        logging.debug("[REASONING LOOP] Phase 5: Respond - Returning response")

        # 6: Reflect
        logging.debug("[REASONING LOOP] Phase 6: Reflect - Checking reflection policies")
        self._maybe_reflect(user_input, response)

        logging.info(f"[REASONING LOOP] Turn {self.turn_count}: Completed")
        return response

    def _maybe_reflect(self, user_input: str, response: str) -> None:
        """Check policies and reflect/journal if triggered (no placeholder branches)."""
        policies = self.policies.get('reflection', {})
        logging.debug(f"[REASONING LOOP] Reflection check - Policies: {policies}")

        # Only actionable trigger: turns since last reflection
        turns_threshold = int(policies.get('turns_since_last', 20))
        if self.turn_count - self.last_reflection_turn >= turns_threshold:
            logging.info(
                f"[REASONING LOOP] Reflection triggered: {self.turn_count - self.last_reflection_turn} turns since last reflection (>= {turns_threshold})"
            )
            self._trigger_reflection(user_input, response)
            self.last_reflection_turn = self.turn_count

    def _trigger_reflection(self, user_input: str, response: str) -> None:
        """Trigger autonomous reflection and journaling."""
        logging.info("[REASONING LOOP] Triggering auto-reflection")
        # Use agent's LLM to generate reflection
        prompt = (
            "You are Atlas reflecting on the recent conversation. "
            "Generate a brief reflection on insights, emotions, or next steps. "
            "Keep it concise."
        )
        conversation = f"User: {user_input}\nAssistant: {response}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation},
        ]
        try:
            reflection_response = self.agent.client.chat(
                model=self.agent.chat_model,
                messages=messages,
                stream=False
            )
            reflection_content = reflection_response.get("message", {}).get("content", "")
            if reflection_content.strip():
                title = f"Auto-reflection Turn {self.turn_count}"
                self.agent.journal.add_entry(title, reflection_content.strip())
                logging.info(f"[REASONING LOOP] Auto-reflection journaled: {title}")
            else:
                logging.warning("[REASONING LOOP] Auto-reflection generated empty content")
        except Exception as e:
            logging.error(f"[REASONING LOOP] Auto-reflection failed: {e}")


# (Critic stage intentionally omitted until a concrete implementation is added.)
