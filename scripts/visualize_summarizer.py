#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from atlas_main.memory import MemoryRecord, summarize_memories_abstractive
from atlas_main.ollama import OllamaClient


def main() -> None:
    # Demo episodes (replace with actual recalls)
    episodes = [
        MemoryRecord(id="1", user="", assistant="Set up repo and initialized dev branch.", timestamp=1.0),
        MemoryRecord(id="2", user="", assistant="Implemented web_search tool using Crawl4AI.", timestamp=2.0),
        MemoryRecord(id="3", user="", assistant="Added multi-tool loop and tests for native + inline calls.", timestamp=3.0),
        MemoryRecord(id="4", user="", assistant="Integrated KV context reuse with optional passing.", timestamp=4.0),
    ]

    client = OllamaClient()
    model = os.getenv("ATLAS_SUMMARY_MODEL", "phi3:latest")

    print("== Episodes ==")
    for ep in episodes:
        print(f"- {ep.assistant}")

    print("\n== Abstractive Summary (model: %s) ==" % model)
    summary = summarize_memories_abstractive(episodes, client, model=model, style="bullets")
    print(summary)


if __name__ == "__main__":
    main()
