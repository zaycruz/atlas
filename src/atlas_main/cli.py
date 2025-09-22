"""Minimal Atlas CLI harnessing the agent with layered memory."""
from __future__ import annotations

import textwrap

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

            if user_text.strip().lower() in {"/quit", "/exit"}:
                print("Exiting.")
                break

            reply = agent.respond(user_text)
            if reply:
                print(f"atlas> {reply}")
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover
    main()
