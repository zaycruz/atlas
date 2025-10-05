"""Targeted tests for HybridWorkingMemory eviction rules."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atlas_main.memory import HybridWorkingMemory, WorkingMemoryConfig


def test_eviction_preserves_important_turns():
    config = WorkingMemoryConfig(max_turns=2, preserve_important=True)
    memory = HybridWorkingMemory(config)

    memory.add_user("keep", pinned=True)
    memory.add_user("discard")
    evicted = memory.add_user("new")

    assert [msg["content"] for msg in evicted] == ["discard"]
    remaining = [msg["content"] for msg in memory.to_messages()]
    assert "keep" in remaining and "discard" not in remaining
