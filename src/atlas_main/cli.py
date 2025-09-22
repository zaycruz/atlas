"""Atlas CLI with journaling commands."""
from __future__ import annotations

import textwrap
import time

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

TAGLINE = "Your autistic pal in the terminal."
BANNER_COLOR = "\033[1;96m"
TAGLINE_COLOR = "\033[96m"
RESET_COLOR = "\033[0m"


def main() -> None:
    print(f"{BANNER_COLOR}{ASCII_ATLAS}{RESET_COLOR}")
    print(f"{TAGLINE_COLOR}{TAGLINE}{RESET_COLOR}")
    print(textwrap.fill("Atlas ready. Type your prompt and press Enter. Use Ctrl+D or /quit to exit.", width=72))
    client = OllamaClient()
    agent = AtlasAgent(client)

    try:
        while True:
            try:
                user_text = input("you> ")
            except EOFError:
                print("\nGoodbye.")
                break
            except KeyboardInterrupt:
                print("\n(Interrupted. Type /quit to exit.)")
                continue

            stripped = user_text.strip()
            if not stripped:
                continue

            lowered = stripped.lower()
            if lowered in {"/quit", "/exit"}:
                print("Exiting.")
                break

            if stripped.startswith("/"):
                if _handle_command(agent, stripped):
                    continue

            print("atlas> ", end="", flush=True)
            def stream_print(chunk: str) -> None:
                print(chunk, end="", flush=True)

            reply = agent.respond(user_text, stream_callback=stream_print)
            pending_tool = agent.pop_tool_request()
            if reply and not reply.endswith("\n"):
                print("", flush=True)
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
    print(f"Unknown command: {cmd}. Type /help for options.")
    return True


def _print_help() -> None:
    print("Commands:")
    print("  /journal recent            - show the latest journal reflections")
    print("  /journal search <keyword>  - search the journal for a word or phrase")
    print("  /tool list                - list available tools")
    print("  /tool run <name> [args]   - execute a tool manually")
    print("  /quit                      - exit the chat")


def _handle_journal(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        print("Usage: /journal <recent|search>")
        return
    sub, *rest = args
    if sub == "recent":
        entries = agent.journal.recent(5)
        if not entries:
            print("(journal is empty)")
            return
        for entry in entries:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.created_at))
            print(f"[{timestamp}] {entry.title}")
            print(textwrap.fill(entry.content, width=72))
            print("")
        return
    if sub == "search":
        if not rest:
            print("Usage: /journal search <keyword>")
            return
        keyword = " ".join(rest)
        entries = agent.journal.find_by_keyword(keyword)
        if not entries:
            print("No matching journal entries.")
            return
        for entry in entries:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.created_at))
            print(f"[{timestamp}] {entry.title}")
            print(textwrap.fill(entry.content, width=72))
            print("")
        return
    print(f"Unknown journal command: {sub}")


def _process_tool_request(agent: AtlasAgent, request: dict) -> None:
    name = request.get("name", "?")
    payload = request.get("payload", "")
    tool = tool_registry.get_tool(name)
    print("Tool request detected.")
    if not tool:
        print(f"[warn] Unknown tool '{name}'. Skipping.")
        return
    print(f"  Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    if payload:
        print(f"  Payload: {payload}")
    execute = True
    if tool.requires_confirmation:
        answer = input("Run this tool? [y/N] ").strip().lower()
        execute = answer in {"y", "yes"}
    if not execute:
        print("Tool request skipped.")
        return
    result = tool_registry.execute_tool(agent, tool.name, payload)
    status = "✔" if result.success else "✖"
    print(f"{status} {result.message}")


def _handle_tool(agent: AtlasAgent, args: list[str]) -> None:
    if not args:
        print("Usage: /tool <list|run>")
        return
    sub, *rest = args
    if sub == "list":
        tools = tool_registry.list_tools()
        if not tools:
            print("(no tools registered)")
            return
        for tool in tools:
            flag = "(confirm)" if tool.requires_confirmation else "(auto)"
            print(f"- {tool.name} {flag}: {tool.description}")
        return
    if sub == "run":
        if not rest:
            print("Usage: /tool run <name> [payload]")
            return
        name = rest[0]
        payload = " ".join(rest[1:]) if len(rest) > 1 else ""
        tool = tool_registry.get_tool(name)
        if not tool:
            print(f"Unknown tool '{name}'. Use /tool list to view options.")
            return
        if tool.requires_confirmation:
            answer = input("Run this tool? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Skipped.")
                return
        result = tool_registry.execute_tool(agent, name, payload)
        status = "✔" if result.success else "✖"
        print(f"{status} {result.message}")
        return
    print(f"Unknown tool command: {sub}")


if __name__ == "__main__":  # pragma: no cover
    main()
