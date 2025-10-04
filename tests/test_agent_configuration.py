"""Additional unit tests for AtlasAgent configuration helpers."""
from unittest.mock import Mock

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atlas_main.agent import (
    AtlasAgent,
    DEFAULT_GPT_OSS_TOOL_LIMIT,
    DEFAULT_MAX_TOOL_CALLS,
)


@pytest.fixture
def agent(monkeypatch):
    """Create an AtlasAgent with a mocked Ollama client."""
    client = Mock()
    client.chat = Mock()
    client.chat_stream = Mock()
    client.list_models.return_value = ["stub-model"]
    monkeypatch.delenv("ATLAS_GPT_OSS_TOOL_LIMIT", raising=False)
    return AtlasAgent(client=client, test_mode=True)


def test_cancel_current_sets_flag(agent):
    """Calling cancel_current should flip the internal cancel event."""
    assert agent._cancel_event.is_set() is False
    agent.cancel_current()
    assert agent._cancel_event.is_set() is True


def test_focus_mode_validation(agent):
    """Focus mode accepts only the documented choices."""
    agent.set_focus_mode("focus")
    assert agent.focus_mode == "focus"
    agent.set_focus_mode("autopilot")
    assert agent.focus_mode == "autopilot"
    with pytest.raises(ValueError):
        agent.set_focus_mode("extreme")


def test_max_tool_calls_default(agent):
    """Non GPT-OSS models should use the standard tool call limit."""
    agent.chat_model = "llama3"
    assert agent._max_tool_calls() == DEFAULT_MAX_TOOL_CALLS


def test_max_tool_calls_for_gpt_oss(agent, monkeypatch):
    """GPT-OSS models respect overrides and guard invalid values."""
    agent.chat_model = "gpt-oss-mini"
    assert agent._max_tool_calls() == DEFAULT_GPT_OSS_TOOL_LIMIT

    monkeypatch.setenv("ATLAS_GPT_OSS_TOOL_LIMIT", "7")
    assert agent._max_tool_calls() == 7

    monkeypatch.setenv("ATLAS_GPT_OSS_TOOL_LIMIT", "not-an-int")
    assert agent._max_tool_calls() == DEFAULT_GPT_OSS_TOOL_LIMIT


def test_normalize_tool_calls(agent):
    """Tool call normalization should coerce IDs, types, and arguments."""
    requests = [
        {"name": "", "arguments": {}},
        {"name": "weather", "arguments": "{\"city\": \"Paris\"}"},
        {
            "name": "search",
            "arguments": {"query": "atlas testing"},
            "call_id": "custom-id",
            "type": "function",
        },
        {"name": "broken", "arguments": "{invalid json"},
        {"arguments": {"noop": True}},
    ]

    normalized = agent._normalize_tool_calls(requests)
    assert len(normalized) == 3

    first = normalized[0]
    assert first["id"].startswith("call_")
    assert first["type"] == "function"
    assert first["function"] == {"name": "weather", "arguments": {"city": "Paris"}}

    second = normalized[1]
    assert second["id"] == "custom-id"
    assert second["function"]["arguments"] == {"query": "atlas testing"}

    third = normalized[2]
    assert third["function"] == {"name": "broken", "arguments": {}}
