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
import threading
import os

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
    }

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

    def stream_chunk(chunk: str) -> None:
        buffer.append(chunk)
        console.print(chunk, end="", highlight=False, soft_wrap=True)

    with runtime["agent_lock"]:
        final_text = agent.respond(prompt, stream_callback=stream_chunk)
    console.print("")


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
