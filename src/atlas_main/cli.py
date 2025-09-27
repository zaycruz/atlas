"""Atlas CLI (ultra-lite): chat + web search tool.

Supports:
 - chatting with streaming output
 - switching/listing models
 - toggling thinking visibility
 - adjusting log level
"""
from __future__ import annotations
import textwrap
import time
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import AtlasAgent
from .ollama import OllamaClient

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
    console.print(textwrap.fill("Atlas may call tools automatically (e.g. web_search) when extra information is needed.", width=72))
    
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

                def stream_chunk(chunk: str) -> None:
                    buffer.append(chunk)
                    console.print(chunk, end="", highlight=False, soft_wrap=True)

                agent.respond(user_text, stream_callback=stream_chunk)
                console.print("")  # Final newline after streaming
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
    if cmd == "model":
        _handle_model(agent, rest)
        return True
    if cmd == "log":
        _handle_log(rest)
        return True
    if cmd == "thinking":
        _handle_thinking(agent, rest)
        return True
    console.print(f"Unknown command: {cmd}. Type /help for options.", style="yellow")
    return True


def _print_help() -> None:
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Commands:[/bold]")
    table.add_row("  /model <name>", "switch the active Ollama model")
    table.add_row("  /model list", "list available models")
    table.add_row("  /thinking <on|off>", "show or hide model thinking content")
    table.add_row("  /log <off|error|warn|info|debug>", "adjust logging level")
    table.add_row("  /quit", "exit the chat")
    console.print(table)


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


# Lite build: no tools, journal, critic, identity, desires, or snapshots.


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


    # Loop steps feature is not available in the lite build.


if __name__ == "__main__":  # pragma: no cover
    main()
