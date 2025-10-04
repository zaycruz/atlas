"""Tests for LLM-based objective extraction functionality."""
import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atlas_main.agent import AtlasAgent
from atlas_main.ollama import OllamaClient


@pytest.fixture
def mock_client():
    client = Mock(spec=OllamaClient)
    client.list_models.return_value = ["test-model"]
    client.chat = Mock()
    return client


@pytest.fixture
def agent(mock_client):
    return AtlasAgent(client=mock_client, test_mode=True)


def test_command_doesnt_extract_objective(agent):
    result, should_update = agent._extract_objective_with_llm("/help", None)
    assert result is None
    assert should_update is False


def test_empty_input_doesnt_extract_objective(agent):
    result, should_update = agent._extract_objective_with_llm("   ", None)
    assert result is None
    assert should_update is False


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_with_keep_response(mock_fallback, agent):
    agent.client.chat.return_value = {"message": {"content": "KEEP"}}
    result, should_update = agent._extract_objective_with_llm("continue please", "existing objective")
    assert result == "existing objective"
    assert should_update is False
    mock_fallback.assert_not_called()


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_with_none_response(mock_fallback, agent):
    agent.client.chat.return_value = {"message": {"content": "NONE"}}
    result, should_update = agent._extract_objective_with_llm("thanks", "existing objective")
    assert result is None
    assert should_update is True
    mock_fallback.assert_not_called()


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_with_new_objective(mock_fallback, agent):
    agent.client.chat.return_value = {"message": {"content": "debug auth issue"}}
    result, should_update = agent._extract_objective_with_llm("Help me debug auth issue", None)
    assert result == "debug auth issue"
    assert should_update is True
    mock_fallback.assert_not_called()


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_fallback_on_exception(mock_fallback, agent):
    agent.client.chat.side_effect = Exception("boom")
    mock_fallback.return_value = "fallback objective"
    result, should_update = agent._extract_objective_with_llm("Help me with docker build", None)
    assert result == "fallback objective"
    assert should_update is True
    mock_fallback.assert_called_once_with("Help me with docker build")


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_handles_missing_content(mock_fallback, agent):
    agent.client.chat.return_value = {}
    mock_fallback.return_value = "infer failing tests"
    result, should_update = agent._extract_objective_with_llm("Help me figure out failing tests", None)
    assert result == "infer failing tests"
    assert should_update is True
    mock_fallback.assert_called_once_with("Help me figure out failing tests")


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_preserves_objective_when_fallback_none(mock_fallback, agent):
    agent.client.chat.return_value = {"message": {}}
    mock_fallback.return_value = None
    result, should_update = agent._extract_objective_with_llm("Sounds good, thanks", "existing objective")
    assert result == "existing objective"
    assert should_update is False
    mock_fallback.assert_called_once_with("Sounds good, thanks")


@patch.object(AtlasAgent, '_extract_objective_fallback')
def test_llm_extraction_exception_preserves_when_fallback_none(mock_fallback, agent):
    agent.client.chat.side_effect = Exception("boom")
    mock_fallback.return_value = None
    result, should_update = agent._extract_objective_with_llm("Thanks again", "existing objective")
    assert result == "existing objective"
    assert should_update is False
    mock_fallback.assert_called_once_with("Thanks again")


def test_fallback_extraction_with_help_phrases(agent):
    result = agent._extract_objective_fallback("help me debug this nasty issue")
    assert result.startswith("help me debug") or result.startswith("help me")


def test_fallback_extraction_filters_short_input(agent):
    result = agent._extract_objective_fallback("yo")
    assert result is None


def test_manual_objective_management(agent):
    agent.set_manual_objective("improve test coverage")
    assert agent._current_objective == "improve test coverage"
    agent.clear_objective()
    assert agent._current_objective is None
