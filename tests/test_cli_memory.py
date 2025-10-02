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
        json.dumps({"facts": [{"text": "Atlas maintains Skyline Labs."}]}),
        encoding="utf-8",
    )

    manager = LayeredMemoryManager(embed_fn=lambda _text: None, config=config)
    manager.log_interaction("What did I say?", "You discussed balcony gardens.")
    manager.reflections.add("Mention hardware specifics when discussing garden automations.")
    manager.reflections.add("Track moisture sensors for balcony beds.")
    manager.semantic.extend_facts(
        [
            "Atlas monitors irrigation schedules.",
            "Atlas records sunshine totals for balcony beds.",
        ]
    )

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
    assert "Skyline Labs" in output

    with console.capture() as capture:
        _handle_memory(agent, ["reflections"])
    output = capture.get()
    assert "hardware specifics" in output

    with console.capture() as capture:
        _handle_memory(agent, ["stats"])
    output = capture.get()
    assert "Harvest stats" in output

    with console.capture() as capture:
        _handle_memory(agent, ["prune", "semantic", "1"])
    output = capture.get()
    assert "Pruned semantic=" in output
    assert len(manager.semantic.head(10)) == 1

    with console.capture() as capture:
        _handle_memory(agent, ["prune", "reflections", "1"])
    output = capture.get()
    assert "reflections=" in output
    assert len(manager.reflections.recent(10)) == 1


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


def test_harvest_confidence_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
    config = LayeredMemoryConfig(
        base_dir=tmp_path,
        min_fact_confidence=0.6,
        min_reflection_confidence=0.6,
    )
    manager = LayeredMemoryManager(embed_fn=lambda _text: None, config=config)
    client = _FakeMemoryClient(
        {
            "facts": [
                {"text": "User loves espresso", "confidence": 0.9},
                {"text": "Discard this", "confidence": 0.2},
            ],
            "reflections": [
                {"text": "Offer new brew techniques", "confidence": 0.7},
                {"text": "Low confidence", "confidence": 0.1},
            ],
        }
    )

    manager.process_turn("Any coffee updates?", "Let's log your preferences.", client=client)

    facts = manager.semantic.head(5)
    assert len(facts) == 1
    assert "espresso" in facts[0]["text"].lower()

    lessons = manager.reflections.recent(5)
    assert len(lessons) == 1
    assert "brew" in lessons[0]["text"].lower()

    stats = manager.get_stats()
    assert stats["harvest"]["accepted_facts"] == 1
    assert stats["harvest"]["accepted_reflections"] == 1
    assert stats["harvest"]["rejected_low_confidence"] >= 2
