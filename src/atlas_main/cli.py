"""Atlas CLI with journaling commands."""
from __future__ import annotations

import json
import textwrap
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .agent import AtlasAgent
from .ollama import OllamaClient
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
    console.print(Panel.fit(ASCII_ATLAS, style="cyan"))
    console.print("[bold cyan]Your autistic pal in the terminal.[/bold cyan]")
    console.print(textwrap.fill("Atlas ready. Type your prompt and press Enter. Use Ctrl+D or /quit to exit.", width=72))
    client = OllamaClient()
    agent = AtlasAgent(client)

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
                if _handle_command(agent, stripped):
                    continue

            console.print("[bold cyan]atlas:[/bold cyan]")
            buffer: list[str] = []

            with Live(Markdown(""), console=console, refresh_per_second=8) as live:
                def live_stream(chunk: str) -> None:
                    buffer.append(chunk)
                    live.update(Markdown("".join(buffer)))

                reply = agent.respond(user_text, stream_callback=live_stream)

            if buffer:
                console.print(Markdown("".join(buffer)))

            pending_tool = agent.pop_tool_request()
            if pending_tool:
                _process_tool_request(agent, pending_tool)
    finally:
        client.close()


def _handle_command(agent: AtlasAgent, command_line: str) -> bool:
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
        _handle_model(agent, rest)
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
    table.add_row("  /quit", "exit the chat")
    console.print(table)


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
            console.print(f"[bold]{timestamp}[/] | [cyan]{entry.title}[/]")
            console.print(textwrap.fill(entry.content, width=72))
            console.print("")
        return
    console.print(f"Unknown journal command: {sub}", style="yellow")


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
    status = "✔" if result.success else "✖"
    style = "green" if result.success else "red"
    console.print(f"{status} {result.message}", style=style)


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


def _handle_model(agent: AtlasAgent, args: list[str]) -> None:
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
    console.print(f"Switched to model [cyan]{new_model}[/cyan].")


if __name__ == "__main__":  # pragma: no cover
    main()
