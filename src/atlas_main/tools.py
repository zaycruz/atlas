"""Tool registry for Atlas."""
from __future__ import annotations

import json
import subprocess
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

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


def _tool_memory_status(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Show memory stores: paths and basic stats."""
    # Episodic memory stats
    try:
        stats = agent.episodic_memory.get_memory_stats()
    except Exception as exc:
        stats = {"error": str(exc)}
    # Semantic path
    sem_path = str(agent.semantic_memory.storage_path.expanduser())
    # Journal path
    journal_path = str(agent.journal.storage_path.expanduser())
    # Vector store path
    try:
        vec_path = str(agent.episodic_memory.storage_path.expanduser().parent / "vector_store")
    except Exception:
        vec_path = "(unknown)"
    lines = [
        "Memory Stores:",
        f"- Episodic (Chroma vectors): {vec_path}",
        f"- Episodic (legacy/json for backup): {agent.episodic_memory.storage_path}",
        f"- Semantic (profile JSON): {sem_path}",
        f"- Journal (SQLite): {journal_path}",
        "",
        "Episodic Stats:",
        json.dumps(stats, indent=2),
    ]
    return ToolResult(True, "\n".join(lines))


register_tool(
    Tool(
        name="journal_entry",
        description="Persist a reflective journal entry. Payload JSON with 'title' and 'entry'.",
        handler=_tool_journal_entry,
    requires_confirmation=False,
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

register_tool(
    Tool(
        name="memory_status",
        description="Show memory store locations and basic episodic stats.",
        handler=_tool_memory_status,
        requires_confirmation=False,
    )
)

def _tool_memory_recent(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Show most recent episodic memories (long-term), default 5.

    Payload can be either a plain integer string (e.g., "5") or JSON like {"limit":5}.
    """
    limit = 5
    p = (payload or "").strip()
    if p:
        # Try JSON first
        if p.startswith("{"):
            try:
                data = json.loads(p)
                limit = int(data.get("limit", limit))
            except Exception:
                pass
        else:
            try:
                limit = int(p)
            except ValueError:
                pass
    limit = max(1, min(50, limit))

    try:
        recents = agent.episodic_memory.get_recent(limit)
    except Exception as exc:
        return ToolResult(False, f"Failed to fetch recent memories: {exc}")

    if not recents:
        return ToolResult(True, "(no episodic memories)")

    lines: List[str] = [f"Most recent {min(limit, len(recents))} episodic memories:"]
    for i, r in enumerate(reversed(recents), 1):  # oldest->newest in list; show newest first
        try:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(getattr(r, "timestamp", time.time())))
        except Exception:
            ts = "(unknown time)"
        user_snip = (r.user or "").strip()
        asst_snip = (r.assistant or "").strip()
        if len(user_snip) > 80:
            user_snip = user_snip[:77] + "..."
        if len(asst_snip) > 120:
            asst_snip = asst_snip[:117] + "..."
        lines.append(f"{i}. [{ts}] U: {user_snip}\n   A: {asst_snip}")

    return ToolResult(True, "\n".join(lines))


register_tool(
    Tool(
        name="memory_recent",
        description="Show most recent episodic memories (payload optional integer or {\"limit\":N}).",
        handler=_tool_memory_recent,
        requires_confirmation=False,
    )
)


def _tool_repo_info(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Return repository root and common source locations to guide code tools."""
    root = _repo_root()
    top = [p.name + ("/" if p.is_dir() else "") for p in sorted(root.iterdir()) if not p.name.startswith(".")]
    common = [
        "src/atlas_main/agent.py",
        "src/atlas_main/tools.py",
        "src/atlas_main/enhanced_memory.py",
        "src/atlas_main/cli.py",
        "tests/",
        "docs/",
    ]
    msg = (
        f"Repo root: {root}\n"
        f"Top-level entries:\n- " + "\n- ".join(top[:50]) + "\n\n"
        f"Common paths (use repo-relative):\n- " + "\n- ".join(common)
    )
    return ToolResult(True, msg)


register_tool(
    Tool(
        name="repo_info",
        description="Show repo root and common paths to help form correct code_browse requests.",
        handler=_tool_repo_info,
        requires_confirmation=False,
    )
)


def _tool_memory_reindex(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Recompute embeddings with current embed model and rebuild Chroma."""
    try:
        data = json.loads(payload) if payload.strip() else {}
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for memory_reindex.")
    batch_size = int(data.get("batch_size", 128))
    summary = agent.episodic_memory.reindex(batch_size=batch_size)
    return ToolResult(True, f"Reindex complete: {json.dumps(summary)}")


register_tool(
    Tool(
        name="memory_reindex",
        description="Recompute embeddings and rebuild Chroma (uses current embed model). Optional payload {'batch_size':N}.",
        handler=_tool_memory_reindex,
        requires_confirmation=True,
    )
)


def _run_safe_shell(cmd: List[str], cwd: Path) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=10)
        out = res.stdout.strip() or res.stderr.strip()
        return res.returncode == 0, out
    except Exception as exc:
        return False, str(exc)


def _tool_shell_browse(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Limited shell for browsing within repo root: ls, cat (head), tree (if available).

    Payload JSON:
      - cmd: one of ["ls", "cat", "tree"]
      - path: target path (repo-relative)
      - lines: for cat, max lines (default 200)
    """
    if not payload:
        return ToolResult(False, "shell_browse requires JSON payload.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for shell_browse.")
    cmd = (data.get("cmd") or "").lower()
    path = data.get("path") or "."
    lines = int(data.get("lines", 200))
    try:
        target = _safe_resolve(path)
    except Exception as exc:
        return ToolResult(False, f"shell_browse error: {exc}")
    root = _repo_root()
    if cmd == "ls":
        args = ["ls", "-la", str(target)]
        ok, out = _run_safe_shell(args, root)
        return ToolResult(ok, out if out else "(no output)")
    if cmd == "cat":
        if not target.is_file():
            return ToolResult(False, f"Not a file: {target}")
        # Use head to cap lines
        args = ["bash", "-lc", f"head -n {max(1,lines)} '{target}'"]
        ok, out = _run_safe_shell(args, root)
        return ToolResult(ok, out if out else "(no output)")
    if cmd == "tree":
        args = ["bash", "-lc", f"command -v tree >/dev/null 2>&1 && tree -L 3 '{target}' || ls -la '{target}'"]
        ok, out = _run_safe_shell(args, root)
        return ToolResult(ok, out if out else "(no output)")
    return ToolResult(False, "Unsupported cmd. Use ls|cat|tree.")


register_tool(
    Tool(
        name="shell_browse",
        description=(
            "Limited shell for browsing within repo root (ls|cat|tree). Payload JSON with 'cmd', 'path', optional 'lines'."
        ),
        handler=_tool_shell_browse,
        requires_confirmation=True,
    )
)


# ---------------------------------------------------------------------------
# Code access and modification tools (guarded)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # src/atlas_main/tools.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _safe_resolve(path_str: str) -> Path:
    root = _repo_root()
    p = Path(path_str)
    rp = (root / p).resolve() if not p.is_absolute() else p.resolve()
    root_res = root.resolve()
    if not str(rp).startswith(str(root_res)):
        raise ValueError("Path escapes repository root")
    return rp


def _truncate_text(s: str, max_bytes: int) -> Tuple[str, bool]:
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s, False
    truncated = encoded[:max_bytes]
    # avoid cutting a multibyte char
    while (truncated and (truncated[-1] & 0xC0) == 0x80):
        truncated = truncated[:-1]
    return truncated.decode("utf-8", errors="ignore") + "\n...[truncated]", True


def _tool_code_browse(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Read code or list files within the repository.

    Payload JSON supports:
      - path: file or directory (relative to repo root)
      - glob: glob pattern relative to repo root (e.g., "src/atlas_main/**/*.py")
      - max_bytes: limit output bytes (default 20000)
      - start_line, end_line: line range for file reads (1-based, inclusive)
    """
    try:
        data = json.loads(payload) if payload.strip() else {}
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for code_browse.")

    max_bytes = int(data.get("max_bytes", 20000))
    path = data.get("path")
    pattern = data.get("glob")
    start_line = int(data.get("start_line", 1))
    end_line = data.get("end_line")
    if end_line is not None:
        end_line = int(end_line)

    root = _repo_root()
    try:
        if pattern:
            matches = [str(p) for p in root.glob(pattern)]
            if not matches:
                return ToolResult(True, f"No files match pattern: {pattern}")
            listing = "\n".join(matches)
            listing, did_trunc = _truncate_text(listing, max_bytes)
            suffix = "\n...[truncated]" if did_trunc else ""
            return ToolResult(True, f"Matches for '{pattern}':\n{listing}{suffix}")

        if not path:
            return ToolResult(False, "code_browse requires 'path' or 'glob'.")
        target = _safe_resolve(path)
        if target.is_dir():
            items = sorted([str(p) for p in target.rglob("*") if p.is_file()])
            if not items:
                return ToolResult(True, f"(empty directory) {target}")
            listing = "\n".join(items)
            listing, did_trunc = _truncate_text(listing, max_bytes)
            suffix = "\n...[truncated]" if did_trunc else ""
            return ToolResult(True, f"Files under {target}:\n{listing}{suffix}")
        if not target.exists():
            return ToolResult(False, f"Path not found: {target}")
        # Read file
        text = target.read_text(encoding="utf-8", errors="ignore")
        if end_line is None:
            end_line = len(text.splitlines())
        lines = text.splitlines()
        start_idx = max(1, start_line) - 1
        end_idx = max(start_idx + 1, min(len(lines), end_line))
        slice_text = "\n".join(lines[start_idx:end_idx])
        header = f"{target} (lines {start_idx+1}-{end_idx}/{len(lines)})\n"
        out, did_trunc = _truncate_text(header + slice_text, max_bytes)
        return ToolResult(True, out)
    except Exception as exc:
        return ToolResult(False, f"code_browse error: {exc}")


def _tool_code_modify(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Modify repository files (guarded). Requires confirmation.

    Payload JSON supports one of the following actions:
      - {"action":"write", "path":"...", "content":"..."} (overwrite file)
      - {"action":"append", "path":"...", "content":"..."}
      - {"action":"replace", "path":"...", "find":"...", "replace":"...", "count":1}
      - {"action":"insert", "path":"...", "line":N, "content":"..."}
      - {"action":"delete_lines", "path":"...", "start":A, "end":B}
    """
    if not payload:
        return ToolResult(False, "code_modify requires JSON payload.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for code_modify.")

    action = (data.get("action") or "").lower()
    path = data.get("path")
    if not action or not path:
        return ToolResult(False, "'action' and 'path' are required.")
    try:
        target = _safe_resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if action == "write":
            content = data.get("content", "")
            target.write_text(str(content), encoding="utf-8")
            return ToolResult(True, f"Wrote {target} ({len(str(content))} bytes). Restart to apply.")
        if action == "append":
            content = data.get("content", "")
            with target.open("a", encoding="utf-8") as f:
                f.write(str(content))
            return ToolResult(True, f"Appended to {target}.")
        if action == "replace":
            find = str(data.get("find", ""))
            repl = str(data.get("replace", ""))
            count = data.get("count")
            text = target.read_text(encoding="utf-8", errors="ignore")
            new_text = text.replace(find, repl, count if isinstance(count, int) else -1)
            target.write_text(new_text, encoding="utf-8")
            return ToolResult(True, f"Replaced occurrences in {target}.")
        if action == "insert":
            line = int(data.get("line", 1))
            content = str(data.get("content", ""))
            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
            idx = max(0, min(len(lines), line - 1))
            lines[idx:idx] = [content]
            target.write_text("\n".join(lines), encoding="utf-8")
            return ToolResult(True, f"Inserted at line {line} in {target}.")
        if action == "delete_lines":
            start = int(data.get("start", 1))
            end = int(data.get("end", start))
            lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
            s = max(0, min(len(lines), start - 1))
            e = max(s, min(len(lines), end))
            del lines[s:e]
            target.write_text("\n".join(lines), encoding="utf-8")
            return ToolResult(True, f"Deleted lines {start}-{end} in {target}.")
        return ToolResult(False, f"Unsupported action: {action}")
    except Exception as exc:
        return ToolResult(False, f"code_modify error: {exc}")


register_tool(
    Tool(
        name="code_browse",
        description=(
            "Read code or list files within the repository. Payload JSON with 'path' (file/dir), "
            "or 'glob', optional 'max_bytes', 'start_line', 'end_line'."
        ),
        handler=_tool_code_browse,
        requires_confirmation=False,
    )
)


register_tool(
    Tool(
        name="code_modify",
        description=(
            "Modify repository files (guarded). Actions: write|append|replace|insert|delete_lines. "
            "Payload JSON includes 'action', 'path', and action-specific fields. Requires confirmation."
        ),
        handler=_tool_code_modify,
        requires_confirmation=True,
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


def _tool_goal_update(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Update or amend goals in semantic profile."""
    if not payload:
        return ToolResult(False, "goal_update requires JSON payload with 'goal' and 'reason'.")
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for goal_update.")
    
    goal = data.get("goal", "").strip()
    reason = data.get("reason", "").strip()
    
    if not goal:
        return ToolResult(False, "'goal' must be provided.")
    
    # Update semantic memory with goal
    goal_fact = f"Goal: {goal}"
    if reason:
        goal_fact += f" (Reason: {reason})"
    
    agent.semantic_memory.update("", goal_fact)  # Empty user input for system update
    
    return ToolResult(True, f"Goal updated in profile: {goal}")


register_tool(
    Tool(
        name="goal_update",
        description="Update goals in semantic profile. Payload JSON with 'goal' and 'reason'.",
        handler=_tool_goal_update,
        requires_confirmation=False,
    )
)


def _tool_memory_mark(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Mark a high-priority memory item."""
    if not payload:
        return ToolResult(False, "memory_mark requires JSON payload with 'title', 'content', and optional 'tags'.")
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for memory_mark.")
    
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    tags = data.get("tags", [])
    
    if not title or not content:
        return ToolResult(False, "'title' and 'content' required.")
    
    # Add to episodic memory with high importance
    # For now, just remember as user/assistant
    agent.episodic_memory.remember(f"[MARKED] {title}", content)
    
    return ToolResult(True, f"Memory marked: {title}")


register_tool(
    Tool(
        name="memory_mark",
        description="Pin high-priority memory item. Payload JSON with 'title', 'content', 'tags'.",
        handler=_tool_memory_mark,
        requires_confirmation=False,
    )
)


def _tool_run_experiment(agent: "AtlasAgent", payload: str) -> ToolResult:
    """Run an experiment (placeholder for future sandbox)."""
    if not payload:
        return ToolResult(False, "run_experiment requires JSON payload with 'hypothesis' and 'plan'.")
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolResult(False, "Invalid JSON payload for run_experiment.")
    
    hypothesis = data.get("hypothesis", "").strip()
    plan = data.get("plan", "").strip()
    
    if not hypothesis or not plan:
        return ToolResult(False, "'hypothesis' and 'plan' required.")
    
    # Placeholder: just journal the experiment
    experiment_entry = f"Experiment: {hypothesis}\nPlan: {plan}\nStatus: Planned"
    agent.journal.add_entry(f"Experiment: {hypothesis[:50]}", experiment_entry)
    
    return ToolResult(True, f"Experiment recorded: {hypothesis}")


register_tool(
    Tool(
        name="run_experiment",
        description="Record and plan an experiment. Payload JSON with 'hypothesis', 'plan'.",
        handler=_tool_run_experiment,
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
