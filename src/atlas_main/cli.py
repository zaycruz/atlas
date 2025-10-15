"""Atlas CLI (ultra-lite): chat + web search tool.

Supports:
 - chatting with streaming output
 - switching/listing models
 - toggling thinking visibility
 - adjusting log level
 - managing memory and tools
"""
from __future__ import annotations
import json
import textwrap
import time
import logging
from pathlib import Path
import threading
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import box
from typing import Any, Dict, Optional, List, Set

from .agent import AtlasAgent
from .ollama import OllamaClient
from .stt import Microphone, VadSegmenter, WhisperTranscriber
from .watchers import ClipboardWatcher, FileWatcher
from .ui import ConversationShell

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

    console.print(Panel.fit(ASCII_ATLAS, style="cyan", border_style="bright_cyan"))
    console.print("[bold cyan]Make of this what you will.[/bold cyan]\n")

    # Quick start info
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dim")
    info_table.add_column(style="white")

    info_table.add_row("📝", "Type your message and press Enter to chat")
    info_table.add_row("🔧", "Atlas can automatically use tools like web search")
    info_table.add_row("⚙️", "Type [cyan]/help[/cyan] to see all available commands")
    info_table.add_row("🚪", "Press [cyan]Ctrl+D[/cyan] or type [cyan]/quit[/cyan] to exit")

    console.print(Panel(info_table, title="Quick Start", border_style="green", box=box.ROUNDED))
    console.print("")

    client = OllamaClient()
    agent = AtlasAgent(client)
    ui = ConversationShell(console)

    # Show initial context info
    console.print(f"[dim]Token capacity: {agent.working_memory.config.token_budget} tokens | "
                  f"Model: {agent.chat_model} | Focus: {agent.focus_mode}[/dim]\n")
    runtime: Dict[str, Any] = {
        "voice_mode": "off",
        "voice_thread": None,
        "voice_stop": threading.Event(),
        "microphone": None,
        "vad": None,
        "transcriber": None,
        "agent_lock": threading.Lock(),
        "favorites": set(),
        "ui": ui,
        "active_turn": None,
        "watchers": [],
    }
    runtime["watchers"] = _start_watchers(agent)

    try:
        # No Live during normal operation - use it only during agent processing
        runtime["live"] = None
        ui.register_live(None)

        while True:
            try:
                # Show a clear, unobstructed input prompt
                user_text = console.input("\n[bold cyan]You[/bold cyan] [dim]›[/dim] ")
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
                handled = _handle_command(agent, stripped, runtime)
                if handled:
                    continue
                # If command wasn't handled, show error message
                # (already printed by _handle_command)
                continue

            # Run normal agent turn for non-command input
            _run_agent_turn(agent, runtime, user_text)
    finally:
        for watcher in runtime.get("watchers", []):
            try:
                watcher.stop()
            except Exception:
                pass
            try:
                watcher.join(timeout=1.0)
            except Exception:
                pass
        try:
            agent.close()
        except Exception:
            pass
        client.close()


def _start_watchers(agent: AtlasAgent) -> List[threading.Thread]:
    watchers: List[threading.Thread] = []
    if getattr(agent, "layered_memory", None) is None:
        return watchers
    dirs_env = os.getenv("ATLAS_WATCH_DIRS", "")
    directories = [Path(part.strip()).expanduser() for part in dirs_env.split(",") if part.strip()]
    if directories:
        ext_env = os.getenv("ATLAS_WATCH_EXT", "txt,md,py")
        extensions = [segment.strip() for segment in ext_env.split(",") if segment.strip()]
        interval = float(os.getenv("ATLAS_WATCH_INTERVAL", "5"))
        watcher = FileWatcher(agent, directories, interval=interval, extensions=extensions)
        watcher.start()
        watchers.append(watcher)
    if os.getenv("ATLAS_CLIPBOARD", "0").strip().lower() in {"1", "true", "on"}:
        interval = float(os.getenv("ATLAS_CLIPBOARD_INTERVAL", "4"))
        minimum = int(os.getenv("ATLAS_CLIPBOARD_MIN", "64"))
        watcher = ClipboardWatcher(agent, interval=interval, minimum_length=minimum)
        if getattr(watcher, "available", False):
            watcher.start()
            watchers.append(watcher)
        else:
            logging.getLogger(__name__).info("Clipboard watcher requested but dependency not available.")
    return watchers

def _handle_command(agent: AtlasAgent, command_line: str, runtime: Dict[str, Any]) -> bool:
    parts = command_line[1:].strip().split()
    if not parts:
        return True
    cmd, *rest = parts
    ui: ConversationShell = runtime.get("ui")
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
    if cmd == "memory":
        _handle_memory(agent, rest)
        return True
    if cmd == "voice":
        _handle_voice(agent, rest, runtime)
        return True
    if cmd == "scroll":
        _handle_scroll(ui, rest)
        return True
    if cmd == "up":
        _handle_scroll(ui, ["up"])
        return True
    if cmd == "down":
        _handle_scroll(ui, ["down"])
        return True
    if cmd == "top":
        _handle_scroll(ui, ["top"])
        return True
    if cmd == "bottom":
        _handle_scroll(ui, ["bottom"])
        return True
    if cmd == "test":
        _handle_test_mode(agent, rest, runtime)
        return True
    if cmd == "expand":
        _handle_expand(ui, rest, True)
        return True
    if cmd == "collapse":
        _handle_expand(ui, rest, False)
        return True
    if cmd == "pin":
        _handle_pin(ui, rest, pin=True)
        return True
    if cmd == "unpin":
        _handle_pin(ui, rest, pin=False)
        return True
    if cmd == "pins":
        pinned = ui.pinned_turns if ui else []
        if not pinned:
            console.print("(no pinned turns yet)", style="yellow")
        else:
            console.print(f"Pinned turns: {', '.join(str(t) for t in pinned)}", style="cyan")
        return True
    if cmd == "focus":
        _handle_focus(agent, ui, rest)
        return True
    if cmd == "rerun":
        if ui and rest and rest[0].isdigit():
            turn_id = int(rest[0])
            turn = ui.get_turn(turn_id)
            if turn:
                console.print(f"Re-running turn {turn_id}", style="cyan")
                _run_agent_turn(agent, runtime, turn.user_text)
            else:
                console.print(f"No turn #{turn_id} recorded yet.", style="yellow")
        else:
            console.print("Usage: /rerun <turn_id>", style="yellow")
        return True
    if cmd == "kill":
        agent.cancel_current()
        console.print("Kill switch engaged. Attempting to cancel current turn...", style="yellow")
        return True
    if cmd == "tool":
        _handle_tool_command(agent, rest, runtime)
        return True
    if cmd == "feedback":
        _handle_feedback(ui, rest)
        return True
    if cmd == "adjust":
        _handle_adjust(agent, ui, rest)
        return True
    if cmd == "status":
        _handle_status(agent, ui)
        return True
    if cmd == "stats":
        _handle_stats(agent)
        return True
    console.print(f"Unknown command: {cmd}. Type /help for options.", style="yellow")
    return True


def _handle_status(agent: AtlasAgent, ui: Optional[ConversationShell]) -> None:
    """Show current agent status and context usage."""
    table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    # Model info
    table.add_row("Model", agent.chat_model)
    table.add_row("Focus Mode", agent.focus_mode)

    # Memory info
    wm_stats = agent.working_memory.get_stats()
    turns = wm_stats.get("turns", 0)
    capacity = agent.working_memory.config.max_turns
    tokens = wm_stats.get("tokens", 0)
    token_budget = agent.working_memory.config.token_budget

    table.add_row("Token Usage", f"{tokens}/{token_budget} tokens ({wm_stats.get('token_pct', 0):.0f}%)")
    table.add_row("Messages Stored", str(turns))

    # Objective
    if ui and ui.objective:
        table.add_row("Objective", ui.objective)

    # Memory stats
    if ui:
        stats = ui.memory_stats
        table.add_row("Episodic Memory", str(stats.get("episodic_count", 0)))
        table.add_row("Semantic Facts", str(stats.get("semantic_count", 0)))
        table.add_row("Reflections", str(stats.get("reflections_count", 0)))
        if ui.test_mode:
            table.add_row("Test Mode", "[yellow]ACTIVE (no memory logging)[/yellow]")

    console.print(Panel(table, title="Atlas Status", border_style="cyan"))


def _handle_stats(agent: AtlasAgent) -> None:
    telemetry = Telemetry.instance().stats()
    usage = agent._context_usage_snapshot()

    console.print("[bold cyan]Atlas Metrics[/bold cyan]")
    turn_stats = telemetry["turns"]
    console.print(
        f"Turn latency — count: {turn_stats['count']}  avg: {turn_stats['avg']:.2f}s  "
        f"p50: {turn_stats['p50']:.2f}s  p95: {turn_stats['p95']:.2f}s",
        style="dim",
    )
    console.print(
        f"Compactions: {telemetry['compactions']['count']}  "
        f"Snapshot avg saved tokens: {telemetry['snapshots']['avg_saved']:.1f}",
        style="dim",
    )

    if telemetry["tools"]:
        table = Table(box=box.ROUNDED, show_header=True, border_style="green")
        table.add_column("Tool", style="bold")
        table.add_column("Runs", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("p50 (s)", justify="right")
        table.add_column("p95 (s)", justify="right")
        for name, data in sorted(telemetry["tools"].items()):
            table.add_row(
                name,
                str(data["runs"]),
                str(data["errors"]),
                f"{data['p50']:.2f}",
                f"{data['p95']:.2f}",
            )
        console.print(table)

    console.print(
        f"Working memory tokens: {usage.get('tokens', 0)} / {usage.get('token_budget', 0)} "
        f"({usage.get('token_pct', 0):.1f}% of budget)",
        style="dim",
    )


def _print_help() -> None:
    console.print("\n[bold cyan]Atlas CLI Commands[/bold cyan]\n")

    # Core Commands
    table1 = Table(show_header=True, box=box.ROUNDED, border_style="cyan")
    table1.add_column("Command", style="bold yellow")
    table1.add_column("Description", style="white")

    table1.add_row("/status", "Show current agent status and context")
    table1.add_row("/model <name>", "Switch the active Ollama model")
    table1.add_row("/model list", "List all available models")
    table1.add_row("/thinking <on|off>", "Toggle model thinking visibility")
    table1.add_row("/test <on|off>", "Toggle test mode (disables memory)")
    table1.add_row("/quit or Ctrl+D", "Exit Atlas")

    console.print(Panel(table1, title="Core Commands", border_style="cyan"))

    # Navigation Commands
    table2 = Table(show_header=True, box=box.ROUNDED, border_style="blue")
    table2.add_column("Command", style="bold yellow")
    table2.add_column("Description", style="white")

    table2.add_row("/scroll <up|down|top|bottom|#>", "Navigate conversation history")
    table2.add_row("/up / /down", "Quick scroll shortcuts")
    table2.add_row("/expand <id>", "Show detailed info for a turn")
    table2.add_row("/collapse <id>", "Hide detailed info for a turn")
    table2.add_row("/pin <id> / /unpin <id>", "Pin/unpin important turns")
    table2.add_row("/rerun <id>", "Re-execute a previous prompt")

    console.print(Panel(table2, title="Navigation", border_style="blue"))

    # Memory Commands
    table3 = Table(show_header=True, box=box.ROUNDED, border_style="magenta")
    table3.add_column("Command", style="bold yellow")
    table3.add_column("Description", style="white")

    table3.add_row("/memory episodic [limit]", "Show episodic memories")
    table3.add_row("/memory semantic [limit]", "Show semantic facts")
    table3.add_row("/memory reflections [limit]", "Show reflections")
    table3.add_row("/memory snapshot <query>", "Get memory snapshot for query")
    table3.add_row("/memory stats", "Show memory statistics")
    table3.add_row("/memory path", "Show memory storage paths")

    console.print(Panel(table3, title="Memory Management", border_style="magenta"))

    # Tool Commands
    table4 = Table(show_header=True, box=box.ROUNDED, border_style="green")
    table4.add_column("Command", style="bold yellow")
    table4.add_column("Description", style="white")

    table4.add_row("/tool list", "List all available tools")
    table4.add_row("/tool run <name> [args]", "Execute a tool manually")
    table4.add_row("/tool sandbox", "Interactive tool testing")
    table4.add_row("/focus <autopilot|focus>", "Change tool execution mode")

    console.print(Panel(table4, title="Tool Management", border_style="green"))

    # Advanced Commands
    table5 = Table(show_header=True, box=box.ROUNDED, border_style="yellow")
    table5.add_column("Command", style="bold yellow")
    table5.add_column("Description", style="white")

    table5.add_row("/voice <on|off|ptt>", "Voice input control")
    table5.add_row("/log <level>", "Set logging level (off/error/warn/info/debug)")
    table5.add_row("/feedback <works|issue>", "Provide feedback on responses")
    table5.add_row("/adjust <style>", "Adjust response style (shorter/longer/formal/casual)")

    console.print(Panel(table5, title="Advanced", border_style="yellow"))

    console.print("\n[dim]Tip: Use Ctrl+C to cancel the current response[/dim]\n")


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


def _handle_voice(agent: AtlasAgent, args: list[str], runtime: Dict[str, Any]) -> None:
    if not args:
        console.print("Usage: /voice <on|off|ptt>", style="yellow")
        return
    action = args[0].lower()
    if action == "on":
        _start_voice_listener(agent, runtime)
    elif action == "off":
        _stop_voice_listener(runtime)
    elif action == "ptt":
        _push_to_talk(agent, runtime)
    else:
        console.print("Usage: /voice <on|off|ptt>", style="yellow")


def _handle_scroll(ui: Optional[ConversationShell], args: list[str]) -> None:
    """Handle conversation scrolling commands."""
    if not ui:
        console.print("UI not initialized yet.", style="yellow")
        return
    
    if not args:
        console.print("Usage: /scroll <up|down|top|bottom|#>", style="yellow")
        return
    
    action = args[0].lower()
    total_turns = len(ui.turns)
    
    if total_turns <= ui.max_visible_turns:
        console.print("All conversation turns are visible.", style="dim")
        return
    
    if action == "up":
        # Scroll up (show older messages)
        max_offset = max(0, total_turns - ui.max_visible_turns)
        new_offset = min(ui.scroll_offset + 3, max_offset)
        if new_offset == ui.scroll_offset:
            console.print("Already at oldest messages", style="dim")
        else:
            ui.scroll_offset = new_offset
            ui.auto_scroll = False
            console.print(f"Scrolled up (offset: {ui.scroll_offset}/{max_offset})", style="cyan")

    elif action == "down":
        # Scroll down (show newer messages)
        new_offset = max(ui.scroll_offset - 3, 0)
        if new_offset == ui.scroll_offset and ui.scroll_offset == 0:
            console.print("Already at latest messages", style="dim")
        else:
            ui.scroll_offset = new_offset
            if ui.scroll_offset == 0:
                ui.auto_scroll = True
            console.print(f"Scrolled down (offset: {ui.scroll_offset})", style="cyan")
        
    elif action == "top":
        # Jump to oldest messages
        ui.scroll_offset = total_turns - ui.max_visible_turns
        ui.auto_scroll = False
        console.print("Jumped to oldest conversation turns", style="cyan")
        
    elif action == "bottom":
        # Jump to latest messages
        ui.scroll_offset = 0
        ui.auto_scroll = True
        console.print("Jumped to latest conversation turns", style="cyan")
        
    elif action.isdigit():
        # Jump to specific turn number
        turn_num = int(action)
        if turn_num < 1 or turn_num > total_turns:
            console.print(f"Turn {turn_num} doesn't exist (1-{total_turns})", style="yellow")
            return
        
        # Calculate offset to show the requested turn
        target_offset = max(0, total_turns - turn_num - ui.max_visible_turns // 2)
        ui.scroll_offset = min(target_offset, total_turns - ui.max_visible_turns)
        ui.auto_scroll = False
        console.print(f"Jumped to turn {turn_num}", style="cyan")
        
    else:
        console.print("Usage: /scroll <up|down|top|bottom|#>", style="yellow")


def _handle_test_mode(agent: AtlasAgent, args: list[str], runtime: Dict[str, Any]) -> None:
    """Handle test mode toggle - disables memory logging."""
    ui = runtime.get("ui")
    
    if not args:
        # Show current state
        mode = "ON" if agent.test_mode else "OFF"
        style = "yellow" if agent.test_mode else "green"
        console.print(f"Test mode: {mode}", style=style)
        if agent.test_mode:
            console.print("Memory logging is DISABLED - no memories will be saved", style="yellow")
        else:
            console.print("Memory logging is ENABLED - memories will be saved normally", style="green")
        console.print("Usage: /test <on|off>", style="dim")
        return
    
    action = args[0].lower()
    
    if action == "on":
        agent.test_mode = True
        if ui:
            ui.set_test_mode(True)
        console.print("✅ Test mode ENABLED", style="yellow")
        console.print("   Memory logging is now DISABLED", style="dim yellow")
        console.print("   Conversations will not be saved to episodic, semantic, or reflection memory", style="dim")
        
    elif action == "off":
        agent.test_mode = False
        if ui:
            ui.set_test_mode(False)
        console.print("✅ Test mode DISABLED", style="green")
        console.print("   Memory logging is now ENABLED", style="dim green")
        console.print("   Conversations will be saved to memory normally", style="dim")
        
    else:
        console.print("Usage: /test <on|off>", style="yellow")
        console.print("  /test on  - Enable test mode (disable memory logging)", style="dim")
        console.print("  /test off - Disable test mode (enable memory logging)", style="dim")


def _handle_expand(ui: Optional[ConversationShell], args: list[str], expanded: bool) -> None:
    if not ui:
        console.print("UI not initialized yet.", style="yellow")
        return
    if not args or not args[0].isdigit():
        console.print("Usage: /expand <turn_id>" if expanded else "Usage: /collapse <turn_id>", style="yellow")
        return
    turn_id = int(args[0])
    if not ui.toggle_expand(turn_id, expanded=expanded):
        console.print(f"No turn #{turn_id} found.", style="yellow")


def _handle_pin(ui: Optional[ConversationShell], args: list[str], *, pin: bool) -> None:
    if not ui:
        console.print("UI not initialized yet.", style="yellow")
        return
    if not args or not args[0].isdigit():
        console.print("Usage: /pin <turn_id>" if pin else "Usage: /unpin <turn_id>", style="yellow")
        return
    turn_id = int(args[0])
    action = ui.pin_turn if pin else ui.unpin_turn
    if not action(turn_id):
        console.print(f"No turn #{turn_id} found.", style="yellow")
    else:
        console.print(("Pinned" if pin else "Unpinned") + f" turn {turn_id}.", style="green")


def _handle_focus(agent: AtlasAgent, ui: Optional[ConversationShell], args: list[str]) -> None:
    if not args or args[0].lower() not in {"autopilot", "focus"}:
        console.print("Usage: /focus <autopilot|focus>", style="yellow")
        return
    mode = args[0].lower()
    try:
        agent.set_focus_mode(mode)
    except ValueError as exc:
        console.print(str(exc), style="red")
        return
    if ui:
        ui.set_focus_mode(mode)
    console.print(f"Focus mode set to {mode}.", style="green")


def _handle_tool_command(agent: AtlasAgent, args: list[str], runtime: Dict[str, Any]) -> None:
    if not args:
        console.print("Usage: /tool <sandbox|run|list|favorite|unfavorite>", style="yellow")
        return
    action = args[0].lower()
    ui: Optional[ConversationShell] = runtime.get("ui")
    favorites: Set[str] = runtime.setdefault("favorites", set())
    if action == "list":
        names = list(agent.tools.list_names())
        if not names:
            console.print("No tools registered.", style="yellow")
        else:
            console.print("Tools: " + ", ".join(sorted(names)), style="cyan")
        return
    if action == "sandbox":
        _tool_sandbox(agent, runtime)
        return
    if action == "favorite" and len(args) >= 2:
        favorites.add(args[1])
        if ui:
            ui.add_quick_action(f"/tool run {args[1]}")
        console.print(f"Favorited tool '{args[1]}'.", style="green")
        return
    if action == "unfavorite" and len(args) >= 2:
        favorites.discard(args[1])
        console.print(f"Removed favorite '{args[1]}'.", style="green")
        return
    if action == "run" and len(args) >= 2:
        name = args[1]
        payload = " ".join(args[2:])
        arguments: Dict[str, Any] = {}
        if payload:
            try:
                arguments = json.loads(payload)
            except json.JSONDecodeError as exc:
                console.print(f"Invalid JSON payload: {exc}", style="red")
                return
        try:
            output = agent.tools.run(name, agent=agent, arguments=arguments)
        except Exception as exc:
            console.print(f"Tool '{name}' failed: {exc}", style="red")
            return
        console.print(f"[tool:{name}] {output}")
        if ui:
            ui.record_tool_card(name=name, output=output, arguments=arguments)
            ui.add_quick_action(f"/tool run {name}")
        return
    console.print("Usage: /tool <sandbox|run|list|favorite|unfavorite>", style="yellow")


def _handle_feedback(ui: Optional[ConversationShell], args: list[str]) -> None:
    if not args:
        console.print("Usage: /feedback <works|issue>", style="yellow")
        return
    rating = args[0].lower()
    if rating not in {"works", "issue"}:
        console.print("Usage: /feedback <works|issue>", style="yellow")
        return
    message = "Thanks for the feedback!" if rating == "works" else "Logged that you need adjustments."
    console.print(message, style="green")
    if ui:
        ui.set_status(message)


def _handle_adjust(agent: AtlasAgent, ui: Optional[ConversationShell], args: list[str]) -> None:
    if not args:
        console.print("Usage: /adjust <shorter|longer|formal|casual>", style="yellow")
        return
    choice = args[0].lower()
    tweaks = {
        "shorter": "Make it more concise.",
        "longer": "Add more detail.",
        "formal": "Adopt a more formal tone.",
        "casual": "Relax the tone.",
    }
    directive = tweaks.get(choice)
    if not directive:
        console.print("Usage: /adjust <shorter|longer|formal|casual>", style="yellow")
        return
    console.print(f"Adjustment noted: {directive}", style="green")
    if ui:
        ui.set_status(f"Adjustment queued: {choice}")


def _tool_sandbox(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    ui: Optional[ConversationShell] = runtime.get("ui")
    favorites: Set[str] = runtime.setdefault("favorites", set())
    console.print("[bold cyan]Tool sandbox[/bold cyan]")
    instructions = agent.tools.render_instructions()
    console.print(instructions or "(no tools registered)")
    name = console.input("Tool to run (blank to cancel): ").strip()
    if not name:
        console.print("Sandbox cancelled.", style="yellow")
        return
    payload_text = console.input("Arguments JSON (blank for {}): ").strip()
    arguments: Dict[str, Any] = {}
    if payload_text:
        try:
            arguments = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            console.print(f"Invalid JSON payload: {exc}", style="red")
            return
    try:
        output = agent.tools.run(name, agent=agent, arguments=arguments)
    except Exception as exc:
        console.print(f"Tool '{name}' failed: {exc}", style="red")
        return
    console.print(f"[tool:{name}] {output}")
    if ui:
        ui.record_tool_card(name=name, output=output, arguments=arguments)
        ui.add_quick_action(f"/tool run {name}")
    mark_favorite = console.input("Mark as favorite? (y/N): ").strip().lower()
    if mark_favorite == "y":
        favorites.add(name)
        console.print(f"Added '{name}' to favorites.", style="green")

def _ensure_transcriber(runtime: Dict[str, Any]) -> WhisperTranscriber:
    if runtime.get("transcriber") is None:
        model = os.getenv("ATLAS_STT_MODEL", "medium.en")
        device = os.getenv("ATLAS_STT_DEVICE", "cpu")
        runtime["transcriber"] = WhisperTranscriber(model_name=model, device=device)
    return runtime["transcriber"]


def _start_voice_listener(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    if runtime.get("voice_mode") == "on":
        console.print("Voice input already enabled.", style="yellow")
        return
    try:
        transcriber = _ensure_transcriber(runtime)
        vad_mode = int(os.getenv("ATLAS_VAD_MODE", "2"))
        microphone = Microphone()
        vad = VadSegmenter(aggressiveness=vad_mode)
    except RuntimeError as exc:
        console.print(f"Voice setup failed: {exc}", style="red")
        return

    runtime["microphone"] = microphone
    runtime["vad"] = vad
    runtime["voice_stop"].clear()

    def worker() -> None:
        stop_event = runtime["voice_stop"]
        mic: Microphone = runtime["microphone"]
        vad_inst: VadSegmenter = runtime["vad"]
        mic.start()
        try:
            while not stop_event.is_set():
                chunk = mic.read(timeout=0.2)
                if chunk is None:
                    continue
                segments = vad_inst.feed(chunk)
                for segment in segments:
                    _process_voice_segment(agent, runtime, segment)
        finally:
            leftover = vad_inst.flush()
            for segment in leftover:
                _process_voice_segment(agent, runtime, segment)
            mic.stop()
            runtime["voice_mode"] = "off"
            runtime["voice_thread"] = None
            runtime["microphone"] = None
            runtime["vad"] = None

    thread = threading.Thread(target=worker, daemon=True)
    runtime["voice_thread"] = thread
    runtime["voice_mode"] = "on"
    thread.start()
    console.print("Voice listening enabled. Say /voice off to stop.", style="green")


def _stop_voice_listener(runtime: Dict[str, Any]) -> None:
    if runtime.get("voice_mode") != "on":
        console.print("Voice input is not active.", style="yellow")
        return
    runtime["voice_stop"].set()
    thread = runtime.get("voice_thread")
    if thread:
        thread.join(timeout=2)
    mic: Optional[Microphone] = runtime.get("microphone")
    if mic:
        mic.stop()
    runtime["voice_mode"] = "off"
    runtime["voice_thread"] = None
    runtime["microphone"] = None
    runtime["vad"] = None
    console.print("Voice listening disabled", style="green")


def _push_to_talk(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    try:
        transcriber = _ensure_transcriber(runtime)
        microphone = Microphone()
    except RuntimeError as exc:
        console.print(f"Push-to-talk unavailable: {exc}", style="red")
        return

    console.print("[cyan]Recording...[/cyan]", highlight=False)
    microphone.start()
    from numpy import ndarray  # local import to keep dependency optional

    frames: List[ndarray] = []
    start = time.time()
    while time.time() - start < 8:
        chunk = microphone.read(timeout=0.2)
        if chunk is not None:
            frames.append(chunk.copy())
    microphone.stop()

    if not frames:
        console.print("(no audio captured)", style="yellow")
        return

    audio_bytes = b"".join(frame.tobytes() for frame in frames)
    try:
        result = transcriber.transcribe(audio_bytes)
    except RuntimeError as exc:
        console.print(f"Transcription failed: {exc}", style="red")
        return
    text = (result.get("text") or "").strip()
    if not text:
        console.print("(unable to transcribe speech)", style="yellow")
        return
    console.print(f"[bold green]you (voice):[/bold green] {text}")
    _run_agent_turn(agent, runtime, text)


def _process_voice_segment(agent: AtlasAgent, runtime: Dict[str, Any], segment: bytes) -> None:
    if not segment:
        return

    # Don't process voice input while another turn is active
    if not runtime["agent_lock"].acquire(blocking=False):
        console.print("(voice input ignored - agent is busy)", style="yellow")
        return

    try:
        transcriber = _ensure_transcriber(runtime)
        result = transcriber.transcribe(segment)
    except RuntimeError as exc:
        console.print(f"(voice error: {exc})", style="red")
        _stop_voice_listener(runtime)
        return
    except Exception as exc:
        console.print(f"(unexpected voice error: {exc})", style="red")
        return
    finally:
        runtime["agent_lock"].release()

    text = (result.get("text") or "").strip()
    if not text:
        return
    console.print(f"[bold green]you (voice):[/bold green] {text}")
    _run_agent_turn(agent, runtime, text)
def _run_agent_turn(agent: AtlasAgent, runtime: Dict[str, Any], prompt: str) -> None:
    ui: Optional[ConversationShell] = runtime.get("ui")

    # Track turn history
    turn = ui.add_turn(prompt) if ui else None
    runtime["active_turn"] = turn
    cancelled = False

    # Show objective if one exists
    if ui and ui.objective:
        console.print(f"[dim]Objective: {ui.objective}[/dim]")

    # Start streaming directly to terminal
    console.print("[bold cyan]Atlas[/bold cyan] › ", end="", highlight=False)

    def stream_chunk(chunk: str) -> None:
        # Stream directly to terminal
        console.print(chunk, end="", highlight=False)
        # Also update turn for history
        if turn:
            turn.assistant_text += chunk
            turn.raw_stream += chunk

    def handle_event(event: str, payload: Dict[str, Any]) -> None:
        nonlocal cancelled

        if event == "turn_start":
            # Update objective and context silently (for history)
            if ui:
                ui.set_objective(payload.get("objective"), payload.get("tags", []))
                ui.set_context_usage(payload.get("context_usage", {}))
                if turn:
                    ui.set_turn_tags(turn, payload.get("tags", []))
                memory_stats = payload.get("memory_stats")
                if memory_stats:
                    ui.update_memory_stats(memory_stats)

        elif event == "status":
            # Print status updates inline (Claude Code style)
            message = payload.get("message")
            if message:
                console.print(f"\n[dim cyan]{message}[/dim cyan]")

        elif event == "tool_start":
            # Print tool invocation inline
            names = [tool.get("name") for tool in payload.get("tools", []) if tool.get("name")]
            if names:
                console.print(f"\n[dim]→ Using tool: {', '.join(names)}[/dim]")

        elif event == "tool_result":
            # Print tool result inline
            name = payload.get("name", "tool")
            output = payload.get("output", "")
            # Truncate long output
            display_output = output[:150] + "..." if len(output) > 150 else output
            console.print(f"[dim]  {name}: {display_output}[/dim]")
            # Store in turn for history
            if ui and turn:
                ui.add_tool_detail(
                    turn,
                    name=name,
                    arguments=payload.get("arguments", {}),
                    output=output,
                    call_id=payload.get("call_id"),
                    source=payload.get("source"),
                )

        elif event == "tool_loop_progress":
            # Show progress when agent is iterating through tool calls
            iteration = payload.get("iteration", 0)
            tool_calls = payload.get("tool_calls_so_far", 0)
            max_allowed = payload.get("max_allowed", 0)
            console.print(f"[dim cyan]  ⟳ Iteration {iteration} ({tool_calls}/{max_allowed} tools used)[/dim cyan]")

        elif event == "tool_loop_detected":
            # Infinite loop detected - same tool called repeatedly
            failures = payload.get("consecutive_failures", 0)
            console.print(f"\n[yellow]⚠ Infinite loop detected: same tool called {failures} times. Stopping.[/yellow]")

        elif event == "tool_limit":
            attempted = payload.get("attempted", 0)
            max_calls = payload.get("max")
            console.print(f"\n[yellow]Tool limit reached ({attempted}/{max_calls})[/yellow]")

        elif event == "turn_complete":
            # Update metadata silently
            if ui:
                ui.set_objective(payload.get("objective"), payload.get("tags", []))
                ui.set_context_usage(payload.get("context_usage", {}))
                if turn:
                    ui.set_turn_tags(turn, payload.get("tags", []))
                memory_stats = payload.get("memory_stats")
                if memory_stats:
                    ui.update_memory_stats(memory_stats)

        elif event == "cancelled":
            cancelled = True

    final_text = ""
    try:
        with runtime["agent_lock"]:
            final_text = agent.respond(
                prompt,
                stream_callback=stream_chunk,
                event_callback=handle_event,
            )
    except KeyboardInterrupt:
        agent.cancel_current()
        cancelled = True
        console.print("\n[yellow](Cancelled)[/yellow]")
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        cancelled = True
    finally:
        runtime["active_turn"] = None
        console.print("")  # Newline after response

    # Update turn history
    if ui and turn:
        if final_text and not turn.assistant_text.strip():
            turn.assistant_text = final_text
            turn.raw_stream = final_text
        ui.mark_turn_complete(turn, cancelled=cancelled)
    
    # Update memory stats after turn completion
    try:
        memory_stats = {}

        # Working memory stats (hybrid)
        if hasattr(agent, 'working_memory') and agent.working_memory:
            wm_stats = agent.working_memory.get_stats()
            memory_stats["working_memory"] = wm_stats

        # Layered memory stats - use public API instead of private attributes
        if hasattr(agent, 'layered_memory') and agent.layered_memory:
            stats = agent.layered_memory.get_stats()

            # Use get_stats() which should provide counts
            memory_stats.update({
                "episodic_count": stats.get("episodic_count", 0),
                "semantic_count": stats.get("semantic_count", 0),
                "reflections_count": stats.get("reflections_count", 0),
            })

            # Quality gate statistics
            harvest_stats = stats.get("harvest", {})
            if harvest_stats:
                memory_stats["quality_gates"] = {
                    "facts_accepted": harvest_stats.get("accepted_facts", 0),
                    "reflections_accepted": harvest_stats.get("accepted_reflections", 0),
                }

            # Add memory event if this was a harvest turn (every few turns)
            if harvest_stats.get("attempts", 0) > 0:
                ui.add_memory_event("harvest", f"Processed turn {turn.turn_id}", harvest_stats)

        # Update UI with all memory stats
        ui.update_memory_stats(memory_stats)

    except Exception as e:
        # Log the error for debugging but don't break the UI
        import logging
        logging.debug(f"Failed to update memory stats: {e}")


def _handle_memory(agent: AtlasAgent, args: list[str]) -> None:
    lm = getattr(agent, "layered_memory", None)
    if lm is None:
        console.print("Layered memory is disabled in this build.", style="yellow")
        return

    if not args:
        console.print(
            "Usage: /memory <episodic|semantic|reflections|snapshot|path|prune|stats> [options]",
            style="yellow",
        )
        return

    section = args[0].lower()
    rest = args[1:]
    limit = 5
    if rest and rest[0].isdigit():
        limit = max(1, int(rest[0]))
        rest = rest[1:]
    query = " ".join(rest).strip()

    if section == "path":
        config = getattr(agent, "layered_memory_config", getattr(lm, "config", None))
        if not config:
            console.print("Memory configuration unavailable.", style="yellow")
            return
        console.print("[bold]Memory storage paths[/bold]")
        console.print(f"Base: {config.base_dir}")
        console.print(f"Episodic DB: {config.episodic_path}")
        console.print(f"Semantic JSON: {config.semantic_path}")
        console.print(f"Reflections JSON: {config.reflections_path}")
        return

    if section == "episodic":
        _print_episodic_memory(lm, limit=limit, query=query)
        return
    if section == "semantic":
        _print_semantic_memory(lm, limit=limit, query=query)
        return
    if section == "reflections":
        _print_reflections(lm, limit=limit)
        return
    if section == "snapshot":
        if not query:
            console.print("Usage: /memory snapshot <query>", style="yellow")
            return
        assembled = lm.assemble(query)
        rendered = lm.render(assembled)
        if rendered:
            console.print(f"[bold]Snapshot for '{query}':[/bold]", highlight=False)
            console.print(rendered)
        else:
            console.print("(no memory retrieved for that query)", style="yellow")
        return
    if section == "stats":
        _print_memory_stats(lm)
        return
    if section == "prune":
        _run_memory_prune(agent, rest)
        return

    console.print(
        "Unknown memory command. Use episodic, semantic, reflections, snapshot, path, prune, or stats.",
        style="yellow",
    )


def _print_episodic_memory(lm, *, limit: int, query: str) -> None:
    items = []
    header = "Recent episodes"
    if query:
        hits = lm.episodic.recall(query, top_k=limit)  # type: ignore[attr-defined]
        if hits:
            header = f"Episodic recall for '{query}'"
            for score, rec in hits:
                text = (rec.get("assistant") or rec.get("user") or "").strip()
                if text:
                    items.append(f"{score:.3f} • {text}")
        if not items:
            header = f"Recent episodes (no vector hits for '{query}')"
    if not items:
        recent = lm.episodic.recent(limit)  # type: ignore[attr-defined]
        for rec in recent:
            text = (rec.get("assistant") or rec.get("user") or "").strip()
            if text:
                items.append(text)
    if not items:
        console.print("(no episodic entries recorded yet)", style="yellow")
        return
    console.print(f"[bold]{header}[/bold]", highlight=False)
    for entry in items:
        console.print(f"- {entry}", highlight=False)


def _print_semantic_memory(lm, *, limit: int, query: str) -> None:
    items = []
    header = "Semantic facts"
    if query:
        hits = lm.semantic.recall(query, top_k=limit)  # type: ignore[attr-defined]
        if hits:
            header = f"Semantic recall for '{query}'"
            for score, fact in hits:
                text = str(fact.get("text", "")).strip()
                if text:
                    items.append(f"{score:.3f} • {text}")
        if not items:
            header = f"Semantic head entries (no vector hits for '{query}')"
    if not items:
        head = lm.semantic.head(limit)  # type: ignore[attr-defined]
        for fact in head:
            text = str(fact.get("text", "")).strip()
            if text:
                items.append(text)
    if not items:
        console.print("(semantic memory is empty)", style="yellow")
        return
    console.print(f"[bold]{header}[/bold]", highlight=False)
    for entry in items:
        console.print(f"- {entry}", highlight=False)


def _print_reflections(lm, *, limit: int) -> None:
    lessons = lm.reflections.recent(limit)  # type: ignore[attr-defined]
    if not lessons:
        console.print("(no reflections logged yet)", style="yellow")
        return
    console.print("[bold]Recent reflections[/bold]", highlight=False)
    for item in lessons:
        text = str(item.get("text", "")).strip()
        if text:
            console.print(f"- {text}", highlight=False)


def _print_memory_stats(lm) -> None:
    stats = lm.get_stats()  # type: ignore[attr-defined]
    harvest = stats.get("harvest", {})
    prune = stats.get("prune", {})
    console.print("[bold]Harvest stats[/bold]", highlight=False)
    if harvest:
        for key, value in harvest.items():
            console.print(f"- {key.replace('_', ' ')}: {int(value)}")
    else:
        console.print("(no harvest activity yet)", style="yellow")
    console.print("[bold]Prune stats[/bold]", highlight=False)
    if prune:
        for key, value in prune.items():
            console.print(f"- {key.replace('_', ' ')}: {int(value)}")
    else:
        console.print("(no pruning yet)", style="yellow")


def _run_memory_prune(agent: AtlasAgent, args: list[str]) -> None:
    lm = agent.layered_memory
    if not args:
        console.print(
            "Usage: /memory prune <semantic|reflections|all> [limit] [--review]",
            style="yellow",
        )
        return
    target = args[0].lower()
    review = "--review" in args
    numeric_args = [part for part in args[1:] if part.isdigit()]
    limit = int(numeric_args[0]) if numeric_args else None
    review_client = agent.client if review else None

    semantic_limit = -1
    reflections_limit = -1
    if target in {"semantic", "all"}:
        semantic_limit = limit if limit is not None else lm.config.prune_semantic_max_items  # type: ignore[attr-defined]
    if target in {"reflections", "all"}:
        reflections_limit = limit if limit is not None else lm.config.prune_reflections_max_items  # type: ignore[attr-defined]
    if semantic_limit == -1 and reflections_limit == -1:
        console.print("Select semantic, reflections, or all to prune.", style="yellow")
        return

    result = lm.prune_long_term(  # type: ignore[attr-defined]
        semantic_limit=semantic_limit,
        reflections_limit=reflections_limit,
        review_client=review_client,
    )
    console.print(
        f"Pruned semantic={result['semantic_removed']} reflections={result['reflections_removed']}.",
        style="green",
    )


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
