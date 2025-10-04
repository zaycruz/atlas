"""Atlas CLI (ultra-lite): chat + web search tool.

Supports:
 - chatting with streaming output
 - switching/listing models
 - toggling thinking visibility
 - adjusting log level
"""
from __future__ import annotations
import json
import textwrap
import time
import logging
import threading
import os
from collections import deque

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Any, Dict, Optional, List

from .agent import AtlasAgent
from .ollama import OllamaClient
from .stt import Microphone, VadSegmenter, WhisperTranscriber

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
    runtime: Dict[str, Any] = {
        "voice_mode": "off",
        "voice_thread": None,
        "voice_stop": threading.Event(),
        "microphone": None,
        "vad": None,
        "transcriber": None,
        "agent_lock": threading.Lock(),
        "turns": [],
        "turn_counter": 0,
        "pinned": [],
        "focus_mode": False,
        "abort_event": threading.Event(),
        "tool_drawer": deque(maxlen=6),
        "timeline": [],
        "timeline_interval": 5,
    }
    agent.focus_mode = runtime["focus_mode"]

    try:
        while True:
            _render_session_shell(agent, runtime)
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
                if _handle_command(agent, stripped, runtime):
                    continue

            _run_agent_turn(agent, runtime, user_text)
    finally:
        try:
            agent.close()
        except Exception:
            pass
        client.close()


def _handle_command(agent: AtlasAgent, command_line: str, runtime: Dict[str, Any]) -> bool:
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
    if cmd == "memory":
        _handle_memory(agent, rest)
        return True
    if cmd == "voice":
        _handle_voice(agent, rest, runtime)
        return True
    if cmd == "detail":
        _handle_detail(runtime, rest)
        return True
    if cmd == "rerun":
        _handle_rerun(agent, runtime, rest)
        return True
    if cmd == "pin":
        _handle_pin(agent, runtime, rest)
        return True
    if cmd == "unpin":
        _handle_unpin(agent, runtime, rest)
        return True
    if cmd == "pins":
        _show_pins(runtime)
        return True
    if cmd == "drawer":
        _show_tool_drawer(runtime)
        return True
    if cmd == "toolbox":
        _handle_toolbox(agent, runtime, rest)
        return True
    if cmd == "focus":
        _handle_focus(agent, runtime, rest)
        return True
    if cmd == "objective":
        _handle_objective(runtime, rest)
        return True
    if cmd == "timeline":
        _show_timeline(runtime)
        return True
    if cmd == "context":
        _show_context(runtime, agent)
        return True
    if cmd == "feedback":
        _handle_feedback(runtime, rest)
        return True
    if cmd == "kill":
        abort_event = runtime.get("abort_event")
        if isinstance(abort_event, threading.Event):
            abort_event.set()
        console.print("[yellow]Kill switch triggered. Pending tools will cancel ASAP.[/yellow]")
        return True
    if cmd == "qa":
        _handle_quick_action(rest)
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
    table.add_row("  /voice <on|off|ptt>", "control voice input (always-on or push-to-talk)")
    table.add_row("  /memory ...", "inspect episodic, semantic, or reflection memory")
    table.add_row("  /detail <turn>", "expand a prior reply with tool traces")
    table.add_row("  /rerun <turn>", "re-run a previous prompt")
    table.add_row("  /pin <turn> [user|assistant]", "pin a message to keep it in context")
    table.add_row("  /unpin <pin_id>", "remove a pinned chip")
    table.add_row("  /pins", "list pinned items")
    table.add_row("  /drawer", "show recent tool cards")
    table.add_row("  /toolbox [run <tool> ...]", "manually trigger tools")
    table.add_row("  /focus <on|off>", "toggle focus mode to defer tools")
    table.add_row("  /objective <set|clear> ...", "override the active objective")
    table.add_row("  /timeline", "view 5-turn session snapshots")
    table.add_row("  /context", "inspect context retention vs rollover")
    table.add_row("  /feedback <good|adjust> <turn>", "log quick feedback on a turn")
    table.add_row("  /kill", "abort the active tool run")
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
    try:
        transcriber = _ensure_transcriber(runtime)
        result = transcriber.transcribe(segment)
    except RuntimeError as exc:
        console.print(f"(voice error: {exc})", style="red")
        _stop_voice_listener(runtime)
        return
    text = (result.get("text") or "").strip()
    if not text:
        return
    console.print(f"[bold green]you (voice):[/bold green] {text}")
    _run_agent_turn(agent, runtime, text)
def _run_agent_turn(agent: AtlasAgent, runtime: Dict[str, Any], prompt: str) -> None:
    console.print("[bold cyan]atlas:[/bold cyan]")
    buffer: List[str] = []
    turn_meta: Dict[str, Any] = {"tool_events": []}
    runtime["abort_event"].clear()
    status_holder: Dict[str, Any] = {"status": None}

    def stream_chunk(chunk: str) -> None:
        buffer.append(chunk)
        console.print(chunk, end="", highlight=False, soft_wrap=True)

    def handle_event(event: Dict[str, Any]) -> None:
        etype = event.get("type")
        status = status_holder.get("status")
        if etype == "thinking_start":
            if status:
                status.update("[dim cyan]assistant is thinking…[/dim cyan]")
            return
        if etype == "tool_start":
            request = event.get("request") or {}
            name = request.get("name", "tool")
            console.print(f"[dim magenta]◴ {name}: running…[/dim magenta]")
            if status:
                status.update(f"[dim cyan]running {name}…[/dim cyan]")
            turn_meta.setdefault("tool_events", []).append({
                "type": "start",
                "name": name,
                "payload": request,
                "time": time.time(),
            })
            return
        if etype == "tool_result":
            request = event.get("request") or {}
            name = request.get("name", "tool")
            output = event.get("output", "")
            console.print(f"[dim magenta]✔ {name}: completed[/dim magenta]")
            if status:
                status.update("[dim cyan]assistant is thinking…[/dim cyan]")
            record = {
                "type": "result",
                "name": name,
                "payload": request,
                "output": output,
                "time": time.time(),
            }
            turn_meta.setdefault("tool_events", []).append(record)
            drawer = runtime.get("tool_drawer")
            if isinstance(drawer, deque):
                drawer.appendleft(
                    {
                        "name": name,
                        "output": output,
                        "time": time.time(),
                        "summary": (output or "").strip().splitlines()[0][:80] if output else "(no output)",
                    }
                )
            return
        if etype == "tool_cancelled":
            request = event.get("request") or {}
            name = request.get("name", "tool")
            console.print(f"[yellow]⚠ {name}: cancelled[/yellow]")
            turn_meta.setdefault("tool_events", []).append({
                "type": "cancelled",
                "name": name,
                "payload": request,
                "time": time.time(),
            })
            if status:
                status.update("[yellow]operation cancelled[/yellow]")
            return
        if etype == "tool_limit":
            console.print(
                f"[yellow]Tool limit reached ({event.get('count')}/{event.get('limit')}).[/yellow]",
            )
            return
        if etype == "tool_skipped":
            requests = event.get("requests") or []
            names = ", ".join(req.get("name", "tool") for req in requests) or "tools"
            console.print(
                f"[dim]Focus mode: deferred {names}. Use /focus off to re-enable automatic tools.[/dim]"
            )
            turn_meta.setdefault("tool_events", []).append({
                "type": "skipped",
                "names": names,
                "time": time.time(),
            })
            return
        if etype == "turn_complete" and status:
            status.update("[dim cyan]response finalized[/dim cyan]")

    with runtime["agent_lock"]:
        with console.status("[dim cyan]assistant is thinking…[/dim cyan]", spinner="dots") as status:
            status_holder["status"] = status
            final_text = agent.respond(
                prompt,
                stream_callback=stream_chunk,
                event_callback=handle_event,
                abort_event=runtime.get("abort_event"),
            )
    console.print("")

    report = getattr(agent, "last_turn_report", {}) or {}
    turn = _record_turn(runtime, prompt, final_text, buffer, report, turn_meta)
    _render_turn_summary(turn)


def _record_turn(
    runtime: Dict[str, Any],
    prompt: str,
    final_text: str,
    chunks: List[str],
    report: Dict[str, Any],
    turn_meta: Dict[str, Any],
) -> Dict[str, Any]:
    turn_id = runtime.get("turn_counter", 0) + 1
    runtime["turn_counter"] = turn_id
    raw_text = "".join(chunks)
    summary = (final_text or "").strip().splitlines()[0] if final_text else ""
    objective = runtime.get("objective_override") or report.get("objective")
    turn = {
        "id": turn_id,
        "prompt": prompt,
        "response": final_text,
        "raw": raw_text,
        "summary": summary,
        "tool_events": turn_meta.get("tool_events", []),
        "tool_results": report.get("executed_tools") or [],
        "tags": report.get("tags") or [],
        "objective": objective,
        "timestamp": time.time(),
        "expanded": False,
        "memory_snapshot": report.get("memory_snapshot") or {},
        "kv_context": report.get("kv_context"),
    }
    runtime.setdefault("turns", []).append(turn)
    _update_timeline(runtime, turn)
    return turn


def _update_timeline(runtime: Dict[str, Any], turn: Dict[str, Any]) -> None:
    interval = runtime.get("timeline_interval", 5) or 0
    if interval <= 0:
        return
    if turn["id"] % interval != 0:
        return
    runtime.setdefault("timeline", []).append(
        {
            "turn_id": turn["id"],
            "objective": turn.get("objective"),
            "summary": turn.get("summary"),
            "timestamp": turn.get("timestamp"),
        }
    )


def _render_turn_summary(turn: Dict[str, Any]) -> None:
    turn_id = turn.get("id")
    summary = turn.get("summary") or "(no assistant reply captured)"
    chips: List[str] = []
    if turn.get("tool_results"):
        chips.append(f"{len(turn['tool_results'])} tool(s)")
    if turn.get("tags"):
        chips.append("tags: " + ", ".join(turn["tags"]))
    if turn.get("objective"):
        chips.append("obj: " + turn["objective"])
    if turn.get("kv_context"):
        chips.append(f"kv ctx: {turn['kv_context']}")
    meta = " • ".join(chips)
    body_lines = [summary]
    if meta:
        body_lines.append(f"[dim]{meta}[/dim]")
    body_lines.append(
        f"[dim]/detail {turn_id}  •  /rerun {turn_id}  •  /pin {turn_id}[/dim]"
    )
    body_lines.append(
        f"[dim]React: 👍 /feedback good {turn_id}   ⚙ /feedback adjust {turn_id}[/dim]"
    )
    console.print(
        Panel(
            "\n".join(body_lines),
            title=f"Turn {turn_id}",
            style="cyan" if not turn.get("tool_results") else "magenta",
        )
    )


def _render_session_shell(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    header_parts: List[str] = []
    objective = runtime.get("objective_override") or _latest_objective(runtime)
    if objective:
        header_parts.append(f"[bold cyan]Objective:[/bold cyan] {objective}")
    else:
        header_parts.append("[bold cyan]Objective:[/bold cyan] (set with /objective set <text>)")
    mode = "Focus" if runtime.get("focus_mode") else "Autopilot"
    header_parts.append(f"Mode: {mode}")
    tags = _latest_tags(runtime)
    if tags:
        header_parts.append("Tags: " + ", ".join(tags))
    console.print(" | ".join(header_parts))
    _render_context_meter(agent, runtime)
    _render_pins(runtime)
    console.print(
        "[dim]Quick actions:[/dim] /qa summarize <text>  •  /qa draft <goal>  •  /qa follow <topic>"
    )


def _render_context_meter(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    pins = agent.working_memory.get_pins() if hasattr(agent.working_memory, "get_pins") else []
    recent_total = len(agent.working_memory.to_messages()) - len(pins)
    capacity = getattr(agent.working_memory, "capacity", 1)
    ratio = min(1.0, recent_total / max(1, capacity))
    if ratio < 0.34:
        stage = "Calm"
    elif ratio < 0.67:
        stage = "Aware"
    else:
        stage = "Critical"
    filled = int(ratio * 12)
    bar = "█" * filled + "░" * (12 - filled)
    console.print(
        f"[dim]Context {stage}: {bar} ({recent_total}/{capacity} recent • {len(pins)} pinned) — /context for details[/dim]"
    )


def _render_pins(runtime: Dict[str, Any]) -> None:
    pins = runtime.get("pinned") or []
    if not pins:
        return
    chips = []
    for pin in pins:
        label = pin.get("label") or pin.get("role") or "pin"
        chips.append(f"[cyan]{pin.get('id')}[/cyan] {label}")
    console.print("[dim]Pinned:[/dim] " + "  ".join(chips))


def _latest_objective(runtime: Dict[str, Any]) -> Optional[str]:
    for turn in reversed(runtime.get("turns", [])):
        if turn.get("objective"):
            return turn["objective"]
    return None


def _latest_tags(runtime: Dict[str, Any]) -> List[str]:
    for turn in reversed(runtime.get("turns", [])):
        tags = turn.get("tags") or []
        if tags:
            return tags
    return []


def _get_turn_by_id(runtime: Dict[str, Any], turn_id: int) -> Optional[Dict[str, Any]]:
    for turn in runtime.get("turns", []):
        if turn.get("id") == turn_id:
            return turn
    return None


def _handle_detail(runtime: Dict[str, Any], args: List[str]) -> None:
    if not runtime.get("turns"):
        console.print("[yellow](no turns recorded yet)[/yellow]")
        return
    if not args:
        console.print("Usage: /detail <turn>", style="yellow")
        return
    try:
        turn_id = int(args[0])
    except ValueError:
        console.print("Turn id must be an integer.", style="yellow")
        return
    turn = _get_turn_by_id(runtime, turn_id)
    if not turn:
        console.print(f"[yellow]No turn {turn_id} recorded.[/yellow]")
        return
    tool_table = Table(title="Tool outputs", show_lines=True)
    tool_table.add_column("Name", style="magenta")
    tool_table.add_column("Call ID")
    tool_table.add_column("Arguments")
    tool_table.add_column("Excerpt")
    for tool in turn.get("tool_results", []):
        arguments = json.dumps(tool.get("arguments") or {}, ensure_ascii=False)[:120]
        excerpt = (tool.get("output") or "").strip().replace("\n", " ")[:120]
        tool_table.add_row(
            tool.get("name") or "(unnamed)",
            tool.get("call_id") or "-",
            arguments,
            excerpt or "(no output)",
        )
    events_table = Table(title="Tool timeline", show_lines=True)
    events_table.add_column("Event")
    events_table.add_column("Details")
    for event in turn.get("tool_events", []):
        label = event.get("type", "?")
        if label in {"start", "cancelled"}:
            detail = event.get("name") or "tool"
        elif label == "skipped":
            detail = event.get("names") or "tool"
        else:
            detail = event.get("name") or "tool"
        events_table.add_row(label, detail)
    snapshot = turn.get("memory_snapshot") or {}
    memory_panel = Panel(
        (snapshot.get("summary") or "(no episodic summary)")
        + "\n\n"
        + (snapshot.get("rendered") or "(no layered details)"),
        title="Memory snapshot",
    )
    console.print(
        Panel(
            f"[bold]Prompt:[/bold] {turn.get('prompt')}\n[bold]Reply:[/bold] {turn.get('response')}",
            title=f"Turn {turn_id}"
        )
    )
    if turn.get("tool_results"):
        console.print(tool_table)
    else:
        console.print("[dim](no tool outputs recorded)[/dim]")
    if turn.get("tool_events"):
        console.print(events_table)
    console.print(memory_panel)
    console.print(
        Panel(
            turn.get("raw") or "(no raw stream captured)",
            title="Raw stream",
            subtitle=f"Use /rerun {turn_id} to re-run",
        )
    )


def _handle_rerun(agent: AtlasAgent, runtime: Dict[str, Any], args: List[str]) -> None:
    if not args:
        console.print("Usage: /rerun <turn>", style="yellow")
        return
    try:
        turn_id = int(args[0])
    except ValueError:
        console.print("Turn id must be an integer.", style="yellow")
        return
    turn = _get_turn_by_id(runtime, turn_id)
    if not turn:
        console.print(f"[yellow]No turn {turn_id} recorded.[/yellow]")
        return
    console.print(f"[dim]Re-running turn {turn_id}…[/dim]")
    _run_agent_turn(agent, runtime, turn.get("prompt") or "")


def _handle_pin(agent: AtlasAgent, runtime: Dict[str, Any], args: List[str]) -> None:
    if not runtime.get("turns"):
        console.print("[yellow](no turns to pin yet)[/yellow]")
        return
    target_id = runtime.get("turns", [])[-1]["id"]
    role = "assistant"
    if args:
        try:
            target_id = int(args[0])
        except ValueError:
            console.print("Usage: /pin <turn> [user|assistant]", style="yellow")
            return
        if len(args) > 1:
            role = args[1].lower()
    elif len(args) == 0:
        role = "assistant"
    if role not in {"assistant", "user"}:
        console.print("Role must be 'user' or 'assistant'", style="yellow")
        return
    turn = _get_turn_by_id(runtime, target_id)
    if not turn:
        console.print(f"[yellow]No turn {target_id} recorded.[/yellow]")
        return
    content = turn.get("response") if role == "assistant" else turn.get("prompt")
    if not content:
        console.print("[yellow]Nothing to pin for that selection.[/yellow]")
        return
    pin_id = f"P{len(runtime.get('pinned', [])) + 1}"
    label = f"T{target_id}-{role[:1]}"
    runtime.setdefault("pinned", []).append(
        {"id": pin_id, "turn_id": target_id, "role": role, "content": content, "label": label}
    )
    _refresh_pins(agent, runtime)
    console.print(f"[cyan]Pinned {role} message from turn {target_id} as {pin_id}.[/cyan]")


def _refresh_pins(agent: AtlasAgent, runtime: Dict[str, Any]) -> None:
    pins = runtime.get("pinned") or []
    messages = []
    for pin in pins:
        content = (pin.get("content") or "").strip()
        if not content:
            continue
        role = pin.get("role") or "assistant"
        messages.append({"role": role, "content": content})
    agent.working_memory.set_pins(messages)


def _handle_unpin(agent: AtlasAgent, runtime: Dict[str, Any], args: List[str]) -> None:
    if not args:
        console.print("Usage: /unpin <pin_id>", style="yellow")
        return
    pin_id = args[0].upper()
    pins = runtime.get("pinned") or []
    remaining = [pin for pin in pins if (pin.get("id") or "").upper() != pin_id]
    if len(remaining) == len(pins):
        console.print(f"[yellow]No pin {pin_id} found.[/yellow]")
        return
    runtime["pinned"] = remaining
    _refresh_pins(agent, runtime)
    console.print(f"[cyan]Removed pin {pin_id}.[/cyan]")


def _show_pins(runtime: Dict[str, Any]) -> None:
    pins = runtime.get("pinned") or []
    if not pins:
        console.print("[dim](no pinned messages yet)[/dim]")
        return
    table = Table(title="Pinned", show_header=True)
    table.add_column("ID")
    table.add_column("Turn")
    table.add_column("Role")
    table.add_column("Preview")
    for pin in pins:
        preview = (pin.get("content") or "").strip().splitlines()[0][:80]
        table.add_row(pin.get("id") or "?", str(pin.get("turn_id")), pin.get("role") or "-", preview)
    console.print(table)


def _show_tool_drawer(runtime: Dict[str, Any]) -> None:
    drawer = runtime.get("tool_drawer")
    if not drawer:
        console.print("[dim](no tool runs yet)[/dim]")
        return
    if isinstance(drawer, deque):
        cards = list(drawer)
    else:
        cards = list(drawer)
    if not cards:
        console.print("[dim](no tool runs yet)[/dim]")
        return
    for idx, card in enumerate(cards[:6], start=1):
        summary = card.get("summary") or "(no output)"
        console.print(
            Panel(
                summary,
                title=f"{idx}. {card.get('name')} @ {time.strftime('%H:%M:%S', time.localtime(card.get('time', 0)))}",
                subtitle="/drawer to refresh • /toolbox to rerun",
            )
        )


def _handle_toolbox(agent: AtlasAgent, runtime: Dict[str, Any], args: List[str]) -> None:
    if not args:
        table = Table(title="Tool sandbox", show_header=True)
        table.add_column("Name", style="magenta")
        table.add_column("Description")
        for name, tool in sorted(agent.tools._tools.items()):  # type: ignore[attr-defined]
            table.add_row(name, getattr(tool, "description", ""))
        console.print(table)
        console.print("Use /toolbox run <name> key=value ... to invoke a tool manually.")
        return
    if args[0] != "run" or len(args) < 2:
        console.print("Usage: /toolbox run <name> key=value ...", style="yellow")
        return
    tool_name = args[1]
    kv_pairs = _parse_kv_args(args[2:])
    try:
        result = agent.tools.run(tool_name, agent=agent, arguments=kv_pairs)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Tool {tool_name} failed: {exc}[/red]")
        return
    console.print(Panel(result or "(no output)", title=f"Tool {tool_name}"))
    drawer = runtime.get("tool_drawer")
    if isinstance(drawer, deque):
        drawer.appendleft(
            {
                "name": tool_name,
                "output": result,
                "time": time.time(),
                "summary": (result or "").strip().splitlines()[0][:80] if result else "(no output)",
            }
        )


def _parse_kv_args(pairs: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def _handle_focus(agent: AtlasAgent, runtime: Dict[str, Any], args: List[str]) -> None:
    if not args or args[0].lower() not in {"on", "off"}:
        console.print("Usage: /focus <on|off>", style="yellow")
        return
    enabled = args[0].lower() == "on"
    runtime["focus_mode"] = enabled
    setattr(agent, "focus_mode", enabled)
    state = "ON" if enabled else "OFF"
    console.print(f"[cyan]Focus mode {state}. Atlas will {'avoid' if enabled else 'resume'} tool calls.[/cyan]")


def _handle_objective(runtime: Dict[str, Any], args: List[str]) -> None:
    if not args:
        current = runtime.get("objective_override") or _latest_objective(runtime)
        console.print(f"Current objective: {current or '(none)'}")
        return
    action = args[0].lower()
    if action == "clear":
        runtime["objective_override"] = None
        console.print("[cyan]Cleared manual objective override.[/cyan]")
        return
    if action == "set":
        text = " ".join(args[1:]).strip()
        if not text:
            console.print("Usage: /objective set <text>", style="yellow")
            return
        runtime["objective_override"] = text
        console.print(f"[cyan]Objective pinned to: {text}[/cyan]")
        return
    console.print("Usage: /objective <set|clear> ...", style="yellow")


def _show_timeline(runtime: Dict[str, Any]) -> None:
    timeline = runtime.get("timeline") or []
    if not timeline:
        console.print("[dim](timeline will populate every 5 turns)[/dim]")
        return
    table = Table(title="Session timeline", show_header=True)
    table.add_column("Turn")
    table.add_column("Objective")
    table.add_column("Summary")
    table.add_column("Time")
    for entry in timeline:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        table.add_row(
            str(entry.get("turn_id")),
            entry.get("objective") or "-",
            (entry.get("summary") or "").strip()[:80],
            ts,
        )
    console.print(table)


def _show_context(runtime: Dict[str, Any], agent: AtlasAgent) -> None:
    messages = agent.working_memory.to_messages()
    table = Table(title="Context window", show_header=True)
    table.add_column("Idx")
    table.add_column("Role")
    table.add_column("Pinned")
    table.add_column("Preview")
    for idx, message in enumerate(messages, start=1):
        preview = (message.get("content") or "").strip().splitlines()[0][:80]
        table.add_row(str(idx), message.get("role", ""), "★" if message.get("pinned") else "", preview)
    console.print(table)


def _handle_feedback(runtime: Dict[str, Any], args: List[str]) -> None:
    if len(args) < 2:
        console.print("Usage: /feedback <good|adjust> <turn>", style="yellow")
        return
    kind = args[0].lower()
    if kind not in {"good", "adjust"}:
        console.print("Feedback must be 'good' or 'adjust'", style="yellow")
        return
    try:
        turn_id = int(args[1])
    except ValueError:
        console.print("Turn id must be an integer.", style="yellow")
        return
    feedback_log = runtime.setdefault("feedback", [])
    feedback_log.append({"turn": turn_id, "type": kind, "time": time.time()})
    if kind == "adjust":
        console.print("[cyan]Noted. Consider rephrasing with /qa follow <hint> to steer the retry.[/cyan]")
    else:
        console.print("[cyan]Appreciated![/cyan]")


def _handle_quick_action(args: List[str]) -> None:
    if not args:
        console.print("Quick action templates:\n - summarize <text>\n - draft <goal>\n - follow <topic>")
        return
    action = args[0].lower()
    remainder = " ".join(args[1:]).strip() or "(provide details)"
    if action == "summarize":
        template = f"Summarize this for me focusing on next steps: {remainder}"
    elif action == "draft":
        template = f"Draft a command plan to accomplish: {remainder}"
    elif action == "follow":
        template = f"Ask a concise follow-up about: {remainder}"
    else:
        console.print("Supported quick actions: summarize, draft, follow", style="yellow")
        return
    console.print(Panel(template, title="Quick action"))
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
