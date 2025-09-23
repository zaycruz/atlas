"""Tool registry for Atlas."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .agent import AtlasAgent


@dataclass
class ToolResult:
    success: bool
    message: str


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[["AtlasAgent", str], ToolResult]
    requires_confirmation: bool = True


_REGISTRY: Dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def list_tools() -> List[Tool]:
    return list(_REGISTRY.values())


def execute_tool(agent: "AtlasAgent", name: str, payload: str) -> ToolResult:
    tool = get_tool(name)
    if not tool:
        return ToolResult(False, f"Unknown tool '{name}'.")
    try:
        return tool.handler(agent, payload)
    except Exception as exc:
        return ToolResult(False, f"Tool '{name}' failed: {exc}")


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _tool_journal_entry(agent: "AtlasAgent", payload: str) -> ToolResult:
    if not payload:
        return ToolResult(False, "journal_entry requires JSON payload with 'title' and 'entry'.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for journal_entry.")
    title = str(data.get("title") or "Reflection")
    entry = str(data.get("entry") or "")
    if not entry.strip():
        return ToolResult(False, "journal_entry payload missing 'entry' content.")
    journal_entry = agent.journal.add_entry(title, entry)
    timestamp = journal_entry.created_at
    return ToolResult(True, f"Journal entry saved as '{journal_entry.title}'.")


def _tool_memory_snapshot(agent: "AtlasAgent", payload: str) -> ToolResult:
    limit = 3
    if payload.strip():
        try:
            limit = max(1, int(payload.strip()))
        except ValueError:
            pass
    recent = agent.working_memory.to_messages()[-limit:]
    if not recent:
        return ToolResult(True, "No recent turns available.")
    lines = []
    for item in recent:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return ToolResult(True, "\n".join(lines))


register_tool(
    Tool(
        name="journal_entry",
        description="Persist a reflective journal entry. Payload JSON with 'title' and 'entry'.",
        handler=_tool_journal_entry,
        requires_confirmation=True,
    )
)

register_tool(
    Tool(
        name="memory_snapshot",
        description="Display the most recent conversation turns (payload optional integer).",
        handler=_tool_memory_snapshot,
        requires_confirmation=False,
    )
)


def _tool_update_prompt(agent: "AtlasAgent", payload: str) -> ToolResult:
    if not payload:
        return ToolResult(False, "prompt_update requires JSON payload with 'system_prompt'.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for prompt_update.")
    prompt = str(data.get("system_prompt") or "").strip()
    if not prompt:
        return ToolResult(False, "'system_prompt' must be a non-empty string.")
    agent.update_system_prompt(prompt)
    return ToolResult(True, "System prompt updated.")


register_tool(
    Tool(
        name="prompt_update",
        description="Update the agent's system prompt. Payload JSON with 'system_prompt'.",
        handler=_tool_update_prompt,
        requires_confirmation=True,
    )
)


__all__ = [
    "Tool",
    "ToolResult",
    "register_tool",
    "get_tool",
    "list_tools",
    "execute_tool",
]
