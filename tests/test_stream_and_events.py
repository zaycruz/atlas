"""Basic tests to verify agent stream + event emissions."""
import sys
import os
from typing import Dict, Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas_main.agent import AtlasAgent


class _StubClient:
    def __init__(self):
        self._ctx = []

    def list_models(self):
        return ["test-model"]

    def chat_stream(self, **kwargs):
        # minimal stream of two chunks then done
        yield {"content": "Hello", "tool_calls": [], "context": None}
        yield {"content": " world", "tool_calls": [], "context": self._ctx}

    def chat(self, **kwargs):
        # not used in this test
        return {"message": {"content": "ok"}}


def test_stream_and_events_collects_callbacks():
    events = []
    chunks = []

    def ev(kind: str, payload: Dict[str, Any]):
        events.append((kind, payload))

    agent = AtlasAgent(client=_StubClient(), test_mode=True)

    text = agent.respond("hi", stream_callback=chunks.append, event_callback=ev)

    # stream delivered chunks
    assert "Hello" in "".join(chunks)
    assert "world" in "".join(chunks)

    # final text contains both parts
    assert "Hello world" in text

    # event sequence includes turn_start, at least one stream, and turn_complete
    kinds = [k for k, _ in events]
    assert "turn_start" in kinds
    assert "stream" in kinds
    assert "turn_complete" in kinds
