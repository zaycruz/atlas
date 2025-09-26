"""Atlas CLI with journaling commands."""
from __future__ import annotations

import json
import textwrap
import time
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import AtlasAgent
from .ollama import OllamaClient
from .atlas_core.controller import ReasoningController
from . import tools as tool_registry

ASCII_ATLAS = r"""
    █████╗ ████████╗██╗      █████╗ ███████╗
   ██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔════╝
   ███████║   ██║   ██║     ███████║███████╗
   ██╔══██║   ██║   ██║     ██╔══██║╚════██║
   ██║  ██║   ██║   ███████╗██║  ██║███████║
   ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝
"""

console = Console()


def main() -> None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    console.print(Panel.fit(ASCII_ATLAS, style="cyan"))
    console.print("[bold cyan]Make of this what you will.[/bold cyan]")
    console.print(textwrap.fill("Atlas ready. Type your prompt and press Enter. Use Ctrl+D or /quit to exit.", width=72))
    console.print(textwrap.fill("use /model <model> to switch the active model.", width=72))
    console.print(textwrap.fill("use /model list to view available models.", width=72))
    client = OllamaClient()
    agent = AtlasAgent(client)
    controller = ReasoningController(agent, agent.policies_path)

    try:
        while True:
            try:
                user_text = console.input("[bold green]you: [/bold green]")
            except EOFError:
                console.print("\n[bold yellow]Goodbye.[/bold yellow]")
                break
            except KeyboardInterrupt:
                console.print("\n[bold yellow](Interrupted. Type /quit to exit.)[/bold yellow]")
                continue

            stripped = user_text.strip()
            if not stripped:
                continue

            lowered = stripped.lower()
            if lowered in {"/quit", "/exit"}:
                console.print("[bold yellow]Exiting.[/bold yellow]")
                break

            if stripped.startswith("/"):
                if _handle_command(agent, controller, stripped):
                    continue

            console.print("[bold cyan]atlas:[/bold cyan]")
            buffer: list[str] = []

            def stream_chunk(chunk: str) -> None:
                buffer.append(chunk)
                console.print(chunk, end="", highlight=False, soft_wrap=True)

            controller.process_turn(user_text, stream_callback=stream_chunk)
            console.print("")  # Final newline after streaming

            pending_tool = agent.pop_tool_request()
            if pending_tool:
                _process_tool_request(agent, pending_tool)
    finally:
        client.close()


def _handle_command(agent: AtlasAgent, controller: ReasoningController, command_line: str) -> bool:
    parts = command_line[1:].strip().split()
    if not parts:
        return True
    cmd, *rest = parts
    if cmd == "help":
        _print_help()
        return True
    if cmd == "journal":
        _handle_journal(agent, rest)
        return True
    if cmd == "tool":
        _handle_tool(agent, rest)
        return True
    if cmd == "model":
        _handle_model(agent, rest, controller)
        return True
    if cmd == "memory":
        _handle_memory(agent, rest)
        return True
    if cmd == "snapshot":
        _handle_snapshot(agent, rest)
        return True
    if cmd == "log":
        _handle_log(rest)
        return True
    if cmd == "thinking":
        _handle_thinking(agent, rest)
        return True
    if cmd == "identity":
        _handle_identity(agent, rest)
        return True
    console.print(f"Unknown command: {cmd}. Type /help for options.", style="yellow")
    return True


def _print_help() -> None:
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Commands:[/bold]")
    table.add_row("  /journal recent", "show the latest journal reflections")
    table.add_row("  /journal search <keyword>", "search the journal for a word or phrase")
    table.add_row("  /tool list", "list available tools")
    table.add_row("  /tool run <name> [args]", "execute a tool manually")
    table.add_row("  /model <name>", "switch the active Ollama model")
    table.add_row("  /memory status", "show memory store paths and stats")
    table.add_row("  /memory recent [N]", "show the most recent episodic memories (default 5)")
    table.add_row("  /snapshot [N]", "write an internal state snapshot (optional recent messages N)")
    table.add_row("  /identity summary", "show Atlas's current identity summary")
    table.add_row("  /identity history [N]", "show recent identity changes (default 5)")
    table.add_row("  /desires summary", "show current active motivations")
    table.add_row("  /desires history [N]", "show recent desire updates (default 5)")
    table.add_row("  /quit", "exit the chat")
    console.print(table)


def _handle_memory(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        console.print("Usage: /memory <status|reindex>", style="yellow")
        return
    sub, *rest = args
    if sub == "status":
        result = tool_registry.execute_tool(agent, "memory_status", "")
        style = "green" if result.success else "red"
        console.print(result.message, style=style)
        return
    if sub == "recent":
        n = 5
        if rest:
            try:
                n = max(1, int(rest[0]))
            except ValueError:
                n = 5
        try:
            records = agent.episodic_memory.get_recent(n)
        except Exception as exc:
            console.print(f"Failed to fetch recent memories: {exc}", style="red")
            return
        if not records:
            console.print("(no episodic memories)", style="dim")
            return
        console.print(f"[bold]Most recent {len(records)} episodic memories:[/bold]")
        for rec in records:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.timestamp)) if getattr(rec, 'timestamp', None) else ""
            user_snip = (rec.user or "").replace("\n", " ")[:120]
            asst_snip = (rec.assistant or "").replace("\n", " ")[:120]
            console.print(f"- [cyan]{ts}[/cyan] | user: {user_snip}")
            if asst_snip:
                console.print(f"  assistant: {asst_snip}")
        return
    if sub == "reindex":
        bs = None
        if rest:
            try:
                bs = int(rest[0])
            except ValueError:
                bs = None
        payload = json.dumps({"batch_size": bs}) if bs else ""
        tool = tool_registry.get_tool("memory_reindex")
        if tool and tool.requires_confirmation:
            answer = console.input("Reindex episodic memory now? This will rebuild the vector store. [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                console.print("Skipped reindex.", style="yellow")
                return
        result = tool_registry.execute_tool(agent, "memory_reindex", payload)
        style = "green" if result.success else "red"
        console.print(result.message, style=style)
        return
    console.print(f"Unknown memory command: {sub}", style="yellow")


def _handle_log(args: list[str]) -> None:
    if not args:
        console.print("Usage: /log <off|error|warn|info|debug>", style="yellow")
        return
    level = args[0].lower()
    root = logging.getLogger()
    if level == "off":
        logging.disable(logging.CRITICAL)
        console.print("Logging disabled (off)", style="green")
        return
    # Re-enable if previously disabled
    logging.disable(logging.NOTSET)
    mapping = {
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    selected = mapping.get(level)
    if selected is None:
        console.print("Usage: /log <off|error|warn|info|debug>", style="yellow")
        return
    root.setLevel(selected)
    console.print(f"Logging level set to {level.upper()}", style="green")


def _handle_thinking(agent: AtlasAgent, args: list[str]) -> None:
    if not args or args[0].lower() not in {"on", "off"}:
        console.print("Usage: /thinking <on|off>", style="yellow")
        return
    agent.show_thinking = (args[0].lower() == "on")
    state = "ON" if agent.show_thinking else "OFF"
    console.print(f"Thinking visibility set to {state}", style="green")


def _handle_journal(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        console.print("Usage: /journal <recent|search>", style="yellow")
        return
    sub, *rest = args
    if sub == "recent":
        entries = agent.journal.recent(5)
        if not entries:
            console.print("[dim](journal is empty)[/dim]")
            return
        for entry in entries:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.created_at))
            console.print(f"[bold]{timestamp}[/] | [cyan]{entry.title}[/]")
            console.print(textwrap.fill(entry.content, width=72))
            console.print("")
        return
    if sub == "search":
        if not rest:
            console.print("Usage: /journal search <keyword>", style="yellow")
            return
        keyword = " ".join(rest)
        entries = agent.journal.find_by_keyword(keyword)
        if not entries:
            console.print("No matching journal entries.", style="yellow")
            return
        for entry in entries:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.created_at))


        def _handle_critic(controller: ReasoningController, args: list[str]) -> None:
            if not args or args[0].lower() in {"status"}:
                state = "ON" if controller.critic_enabled else "OFF"
                console.print(
                    f"Critic: {state} | model={controller.critic_model} | max_rounds={controller.max_critique_rounds}",
                    style="green" if controller.critic_enabled else "yellow",
                )
                return
            sub = args[0].lower()
            if sub in {"on", "off"}:
                controller.critic_enabled = (sub == "on")
                console.print(f"Critic turned {'ON' if controller.critic_enabled else 'OFF'}", style="green")
                return
            if sub == "model":
                if len(args) < 2:
                    console.print("Usage: /critic model <name>", style="yellow")
                    return
                controller.critic_model = args[1]
                console.print(f"Critic model set to {controller.critic_model}", style="green")
                return
            if sub == "rounds":
                if len(args) < 2:
                    console.print("Usage: /critic rounds <N>", style="yellow")
                    return
                try:
                    val = int(args[1])
                except ValueError:
                    console.print("Rounds must be an integer", style="yellow")
                    return
                controller.max_critique_rounds = max(0, val)
                console.print(f"Critic max rounds set to {controller.max_critique_rounds}", style="green")
                return
            console.print("Usage: /critic <on|off|model <name>|rounds <N>|status>", style="yellow")
            console.print(f"[bold]{timestamp}[/] | [cyan]{entry.title}[/]")
            console.print(textwrap.fill(entry.content, width=72))
            console.print("")
        return
    console.print(f"Unknown journal command: {sub}", style="yellow")


def _handle_snapshot(agent: AtlasAgent, args: list[str]) -> None:
    n = None
    if args:
        try:
            n = int(args[0])
        except ValueError:
            n = None
    payload = json.dumps({"recent_limit": n}) if n else ""
    result = tool_registry.execute_tool(agent, "state_snapshot", payload)
    style = "green" if result.success else "red"
    console.print(result.message, style=style)


def _process_tool_request(agent: AtlasAgent, request: dict) -> None:
    name = request.get("name", "?")
    payload = request.get("payload", "")
    tool = tool_registry.get_tool(name)
    console.print("[bold magenta]Tool request detected.[/bold magenta]")
    if not tool:
        console.print(f"[yellow]Unknown tool '{name}'. Skipping.[/yellow]")
        return
    console.print(f"  Tool: [cyan]{tool.name}[/cyan]")
    console.print(f"  Description: {tool.description}")
    if payload:
        console.print(f"  Payload: {payload}")
    execute = True
    if tool.requires_confirmation:
        answer = console.input("Run this tool? [y/N] ").strip().lower()
        execute = answer in {"y", "yes"}
    if not execute:
        console.print("Tool request skipped.", style="yellow")
        return
    result = tool_registry.execute_tool(agent, tool.name, payload)
    if result.success:
        console.print(f"✔ {result.message}", style="green")
        return
    # Auto-retry: feed error back into the agent to let it correct the tool request once
    console.print(f"✖ {result.message}", style="red")
    hint = (
        f"The tool '{tool.name}' failed with: {result.message}. "
        f"If this is a path issue, call 'repo_info' first and then retry with a repo-relative path."
    )
    # Let the agent reason and possibly emit a corrected tool request
    followup = agent.respond(hint)
    # Process one more potential tool request
    pending = agent.pop_tool_request()
    if pending:
        console.print("[bold magenta]Retrying tool with corrected request...[/bold magenta]")
        _process_tool_request(agent, pending)


def _handle_tool(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        console.print("Usage: /tool <list|run>", style="yellow")
        return
    sub, *rest = args
    if sub == "list":
        tools = tool_registry.list_tools()
        if not tools:
            console.print("(no tools registered)", style="dim")
            return
        for tool in tools:
            flag = "(confirm)" if tool.requires_confirmation else "(auto)"
            console.print(f"- [cyan]{tool.name}[/cyan] {flag}: {tool.description}")
        return
    if sub == "run":
        if not rest:
            console.print("Usage: /tool run <name> [payload]", style="yellow")
            return
        name = rest[0]
        payload = " ".join(rest[1:]) if len(rest) > 1 else ""
        tool = tool_registry.get_tool(name)
        if not tool:
            console.print(f"Unknown tool '{name}'. Use /tool list to view options.", style="yellow")
            return
        if tool.requires_confirmation:
            answer = console.input("Run this tool? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                console.print("Skipped.", style="yellow")
                return
        result = tool_registry.execute_tool(agent, name, payload)
        status = "✔" if result.success else "✖"
        style = "green" if result.success else "red"
        console.print(f"{status} {result.message}", style=style)
        return
    console.print(f"Unknown tool command: {sub}", style="yellow")


def _handle_identity(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        console.print("Usage: /identity <summary|history [N]>", style="yellow")
        return
    sub, *rest = args
    if sub == "summary":
        console.print("[bold]Identity summary:[/bold]")
        console.print(agent.identity.summary(), soft_wrap=True)
        return
    if sub == "history":
        n = 5
        if rest:
            try:
                n = max(1, int(rest[0]))
            except ValueError:
                pass
        entries = agent.identity.list_history(limit=n)
        if not entries:
            console.print("(no identity changes yet)", style="dim")
            return
        console.print(f"[bold]Last {len(entries)} identity change sets:[/bold]")
        for h in entries:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h.get("timestamp", 0)))
            changes = h.get("changes", {})
            added = changes.get("added", [])
            adjusted = changes.get("adjusted", [])
            removed = changes.get("removed", [])
            console.print(f"- [cyan]{ts}[/cyan]")
            if added:
                console.print("  added: " + ", ".join([f"{a.get('category')}: {a.get('text')} ({a.get('confidence', '')})" for a in added]))
            if adjusted:
                console.print("  adjusted: " + ", ".join([f"{a.get('category')}: {a.get('text')} -> {a.get('confidence', '')}" for a in adjusted]))
            if removed:
                console.print("  removed: " + ", ".join([f"{a.get('category')}: {a.get('text')}" for a in removed]))
        return
    console.print(f"Unknown identity command: {sub}", style="yellow")


def _handle_desires(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        console.print("Usage: /desires <summary|history [N]>", style="yellow")
        return
    sub, *rest = args
    if sub == "summary":
        console.print("[bold]Active motivations:[/bold]")
        console.print(agent.desires.summary(), soft_wrap=True)
        return
    if sub == "history":
        n = 5
        if rest:
            try:
                n = max(1, int(rest[0]))
            except ValueError:
                pass
        entries = agent.desires.list_history(limit=n)
        if not entries:
            console.print("(no desire updates yet)", style="dim")
            return
        console.print(f"[bold]Last {len(entries)} desire updates:[/bold]")
        for h in entries:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h.get("timestamp", 0)))
            changes = h.get("changes", {})
            added = changes.get("added", [])
            boosted = changes.get("boosted", [])
            dampened = changes.get("dampened", [])
            removed = changes.get("removed", [])
            console.print(f"- [cyan]{ts}[/cyan]")
            if added:
                console.print("  added: " + ", ".join([f"{a.get('name')}: {a.get('intensity', '')}" for a in added]))
            if boosted:
                console.print("  boosted: " + ", ".join([f"{a.get('name')}: {a.get('intensity', '')}" for a in boosted]))
            if dampened:
                console.print("  dampened: " + ", ".join([f"{a.get('name')}: {a.get('intensity', '')}" for a in dampened]))
            if removed:
                console.print("  removed: " + ", ".join([f"{a.get('name')}" for a in removed]))
        return
    console.print(f"Unknown desires command: {sub}", style="yellow")


def _handle_model(agent: AtlasAgent, args: list[str], controller: ReasoningController | None = None) -> None:
    if not args:
        console.print(f"Current model: [cyan]{agent.chat_model}[/cyan]", style="bold")
        console.print("Usage: /model <name> | /model list", style="yellow")
        return
    sub = args[0]
    if sub == "list":
        try:
            models = agent.client.list_models()
        except Exception as exc:
            console.print(f"Failed to list models: {exc}", style="red")
            return
        if not models:
            console.print("No models returned by Ollama.", style="yellow")
            return
        console.print("[bold]Available models:[/bold]")
        for name in models:
            marker = "*" if name == agent.chat_model else "-"
            console.print(f"  {marker} [cyan]{name}[/cyan]")
        return

    new_model = " ".join(args).strip()
    if not new_model:
        console.print("Usage: /model <name>", style="yellow")
        return
    agent.set_chat_model(new_model)
    # Reset controller loop steps based on model defaults, if available
    if controller is not None:
        controller.set_max_steps(ReasoningController.default_max_steps_for_model(new_model))
    console.print(f"Switched to model [cyan]{new_model}[/cyan].")


def _handle_loopsteps(controller: ReasoningController, args: list[str]) -> None:
    if not args:
        console.print(f"Current loop steps: {controller.max_steps}", style="bold")
        console.print("Usage: /loopsteps <N>", style="yellow")
        return
    try:
        n = int(args[0])
        if n < 1:
            raise ValueError
    except ValueError:
        console.print("/loopsteps requires a positive integer.", style="yellow")
        return
    controller.set_max_steps(n)
    console.print(f"Internal loop steps set to {n}", style="green")


if __name__ == "__main__":  # pragma: no cover
    main()
