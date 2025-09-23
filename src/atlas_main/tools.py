"""Tool registry for Atlas."""
from __future__ import annotations

import json
import subprocess
import time
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
        description=(
            "Update the agent's system prompt via the tool registry. "
            "Payload JSON with 'system_prompt'."
        ),
        handler=_tool_update_prompt,
        requires_confirmation=True,
    )
)


def _tool_git_update(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Pull latest git changes and optionally install requirements."""
    repo_dir = payload.strip() or "."
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ToolResult(False, "git command not found on PATH.")
    if result.returncode != 0:
        return ToolResult(False, f"git pull failed: {result.stderr.strip() or result.stdout.strip()}")

    message = result.stdout.strip() or "Repository updated."
    return ToolResult(True, message)


register_tool(
    Tool(
        name="git_update",
        description="Run 'git pull' (payload optional path). Requires confirmation.",
        handler=_tool_git_update,
        requires_confirmation=True,
    )
)


def _tool_goal_tracker(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Track and manage long-term goals and objectives."""
    if not payload:
        return ToolResult(False, "goal_tracker requires JSON payload with 'action' and optional 'goal', 'context'.")
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for goal_tracker.")
    
    action = data.get("action", "").lower()
    goal_text = data.get("goal", "")
    context = data.get("context", "")
    
    if action == "add":
        if not goal_text.strip():
            return ToolResult(False, "Adding a goal requires 'goal' text.")
        
        # Store as special journal entry with goal prefix
        goal_entry = f"GOAL: {goal_text.strip()}"
        if context:
            goal_entry += f"\nContext: {context.strip()}"
        
        entry = agent.journal.add_entry(f"Goal: {goal_text[:50]}...", goal_entry)
        return ToolResult(True, f"Goal added and tracked: '{goal_text[:50]}...'")
    
    elif action == "list":
        # Find all goal entries in journal
        goal_entries = [e for e in agent.journal.entries if e.content.startswith("GOAL:")]
        if not goal_entries:
            return ToolResult(True, "No goals currently tracked.")
        
        goals_list = []
        for entry in goal_entries[-10:]:  # Show last 10 goals
            goal_line = entry.content.split('\n')[0].replace("GOAL: ", "")
            timestamp = time.strftime("%Y-%m-%d", time.localtime(entry.created_at))
            goals_list.append(f"• {goal_line} ({timestamp})")
        
        return ToolResult(True, f"Current goals:\n" + "\n".join(goals_list))
    
    elif action == "recall":
        topic = goal_text or context
        if not topic:
            return ToolResult(False, "Recall requires 'goal' or 'context' to search for.")
        
        # Search journal for goal-related entries
        relevant_goals = agent.journal.find_by_keyword(topic)
        goal_matches = [e for e in relevant_goals if "GOAL:" in e.content]
        
        if not goal_matches:
            return ToolResult(True, f"No goals found related to '{topic}'.")
        
        matches = []
        for entry in goal_matches[:5]:
            goal_line = entry.content.split('\n')[0].replace("GOAL: ", "")
            matches.append(f"• {goal_line}")
        
        return ToolResult(True, f"Goals related to '{topic}':\n" + "\n".join(matches))
    
    else:
        return ToolResult(False, "Action must be 'add', 'list', or 'recall'.")


register_tool(
    Tool(
        name="goal_tracker",
        description="Track and manage long-term goals. Payload JSON with 'action': 'add'|'list'|'recall', optional 'goal', 'context'.",
        handler=_tool_goal_tracker,
        requires_confirmation=False,
    )
)


def _tool_context_connector(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Connect current topic to relevant past conversations and memories."""
    if not payload:
        return ToolResult(False, "context_connector requires JSON payload with 'topic' and optional 'depth'.")
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for context_connector.")
    
    topic = data.get("topic", "").strip()
    depth = data.get("depth", "medium").lower()
    
    if not topic:
        return ToolResult(False, "context_connector requires 'topic' to search for connections.")
    
    # Determine search scope based on depth
    if depth == "shallow":
        top_k = 3
        max_results = 3
    elif depth == "deep":
        top_k = 10
        max_results = 8
    else:  # medium
        top_k = 6
        max_results = 5
    
    connections = []
    
    # Search episodic memory for related conversations
    try:
        memory_results = agent.episodic_memory.recall(topic, top_k=top_k)
        if memory_results:
            connections.append("== RELATED CONVERSATIONS ==")
            for i, memory in enumerate(memory_results[:max_results]):
                # Get timestamp for context
                timestamp = time.strftime("%Y-%m-%d", time.localtime(memory.created_at))
                
                # Extract key snippet from conversation
                if hasattr(memory, 'user_input') and hasattr(memory, 'assistant_response'):
                    # Enhanced memory format
                    snippet = memory.user_input[:100] + "..." if len(memory.user_input) > 100 else memory.user_input
                    response_snippet = memory.assistant_response[:150] + "..." if len(memory.assistant_response) > 150 else memory.assistant_response
                    connections.append(f"{i+1}. ({timestamp}) User: {snippet}")
                    connections.append(f"   Atlas: {response_snippet}")
                else:
                    # Legacy memory format
                    snippet = memory.user[:100] + "..." if len(memory.user) > 100 else memory.user
                    response_snippet = memory.assistant[:150] + "..." if len(memory.assistant) > 150 else memory.assistant
                    connections.append(f"{i+1}. ({timestamp}) User: {snippet}")
                    connections.append(f"   Atlas: {response_snippet}")
                
                if i < len(memory_results) - 1:
                    connections.append("")
    except Exception as e:
        connections.append(f"Memory search error: {str(e)}")
    
    # Search journal for related insights
    try:
        journal_results = agent.journal.find_by_keyword(topic)
        if journal_results:
            connections.append("\n== RELATED INSIGHTS ==")
            for i, entry in enumerate(journal_results[:3]):
                timestamp = time.strftime("%Y-%m-%d", time.localtime(entry.created_at))
                title = entry.title[:60] + "..." if len(entry.title) > 60 else entry.title
                content_snippet = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
                connections.append(f"{i+1}. ({timestamp}) {title}")
                connections.append(f"   {content_snippet}")
                if i < len(journal_results) - 1:
                    connections.append("")
    except Exception as e:
        connections.append(f"\nJournal search error: {str(e)}")
    
    if not connections or len(connections) <= 2:
        return ToolResult(True, f"No significant connections found for '{topic}'. This appears to be a new topic area.")
    
    result = f"Connections found for '{topic}':\n\n" + "\n".join(connections)
    return ToolResult(True, result)


register_tool(
    Tool(
        name="context_connector",
        description="Find connections between current topic and past conversations. Payload JSON with 'topic', optional 'depth': 'shallow'|'medium'|'deep'.",
        handler=_tool_context_connector,
        requires_confirmation=False,
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
