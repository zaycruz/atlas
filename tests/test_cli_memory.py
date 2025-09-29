from __future__ import annotations

import json
import os

from atlas_main.cli import _handle_memory, console
from atlas_main.memory_layers import LayeredMemoryConfig, LayeredMemoryManager


class _StubAgent:
    def __init__(self, manager: LayeredMemoryManager, config: LayeredMemoryConfig) -> None:
        self.layered_memory = manager
        self.layered_memory_config = config


def test_memory_cli_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
    config = LayeredMemoryConfig(base_dir=tmp_path)

    config.semantic_path.write_text(
        json.dumps({"facts": [{"text": "Zay builds Prometheus Studios."}]}),
        encoding="utf-8",
    )

    manager = LayeredMemoryManager(embed_fn=lambda _text: None, config=config)
    manager.log_interaction("What did I say?", "You discussed balcony gardens.")
    manager.reflections.add("Mention hardware specifics when discussing garden automations.")

    agent = _StubAgent(manager, config)

    with console.capture() as capture:
        _handle_memory(agent, ["path"])
    output = capture.get()
    assert "episodes.sqlite3" in output

    with console.capture() as capture:
        _handle_memory(agent, ["episodic", "3"])
    output = capture.get()
    assert "balcony gardens" in output

    with console.capture() as capture:
        _handle_memory(agent, ["semantic"])
    output = capture.get()
    assert "Prometheus Studios" in output

    with console.capture() as capture:
        _handle_memory(agent, ["reflections"])
    output = capture.get()
    assert "hardware specifics" in output


class _FakeMemoryClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def chat(self, *, model, messages, stream=False, options=None, tools=None, context=None, keep_alive=None, request_timeout=None):
        return {"message": {"content": json.dumps(self.payload)}}


def test_layered_memory_process_turn_updates_long_term(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("ATLAS_MEMORY_MODEL", "fake-model")
    config = LayeredMemoryConfig(base_dir=tmp_path)
    manager = LayeredMemoryManager(embed_fn=lambda _text: None, config=config)
    client = _FakeMemoryClient(
        {
            "facts": ["User enjoys coffee brewing."],
            "reflections": ["Remember to suggest coffee equipment."],
        }
    )

    manager.process_turn("I love experimenting with coffee", "I'll keep that in mind.", client=client)

    facts = manager.semantic.head(1)
    assert facts and "coffee" in facts[0]["text"].lower()

    reflections = manager.reflections.recent(1)
    assert reflections and "coffee" in reflections[0]["text"].lower()
