"""Tests for instant command handling functionality."""
import pytest
from unittest.mock import Mock
import sys
import os

# Add src to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas_main.cli import _handle_command_in_chat, _handle_objective_in_chat, _get_help_text
from atlas_main.agent import AtlasAgent
from atlas_main.ui import ConversationTurn


class TestInstantCommands:
    @pytest.fixture
    def mock_agent(self):
        agent = Mock(spec=AtlasAgent)
        agent.chat_model = "test-model"
        agent.show_thinking = False
        agent.test_mode = False
        agent.client = Mock()
        agent.client.list_models.return_value = ["model1", "model2", "model3"]
        return agent

    @pytest.fixture
    def mock_turn(self):
        turn = Mock(spec=ConversationTurn)
        turn.assistant_text = ""
        turn.status = "pending"
        return turn

    @pytest.fixture
    def mock_runtime(self):
        ui = Mock()
        return {"ui": ui}

    def test_help_command_instant_response(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/help", mock_runtime, mock_turn) is True
        assert mock_turn.status == "done"
        assert "Commands:" in mock_turn.assistant_text
        assert "/model" in mock_turn.assistant_text
        assert "/objective" in mock_turn.assistant_text
        mock_runtime["ui"].refresh.assert_called_once()

    def test_model_show_current_command(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/model", mock_runtime, mock_turn) is True
        assert mock_turn.status == "done"
        assert mock_turn.assistant_text == "Current model: test-model"
        mock_runtime["ui"].refresh.assert_called_once()

    def test_model_list_command(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/model list", mock_runtime, mock_turn) is True
        assert mock_turn.status == "done"
        assert "Available models:" in mock_turn.assistant_text
        assert "• model1" in mock_turn.assistant_text
        assert "• model2" in mock_turn.assistant_text
        assert "• model3" in mock_turn.assistant_text

    def test_model_switch_command(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/model new-model", mock_runtime, mock_turn) is True
        assert mock_turn.status == "done"
        assert mock_agent.chat_model == "new-model"
        assert mock_turn.assistant_text == "Switched to model: new-model"

    def test_thinking_toggle_on_off_and_invalid(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/thinking on", mock_runtime, mock_turn) is True
        assert mock_agent.show_thinking is True
        assert "enabled" in mock_turn.assistant_text

        # reset UI mock call count
        mock_runtime["ui"].refresh.reset_mock()
        assert _handle_command_in_chat(mock_agent, "/thinking off", mock_runtime, mock_turn) is True
        assert mock_agent.show_thinking is False
        assert "disabled" in mock_turn.assistant_text

        mock_runtime["ui"].refresh.reset_mock()
        assert _handle_command_in_chat(mock_agent, "/thinking maybe", mock_runtime, mock_turn) is True
        assert mock_turn.assistant_text == "Usage: /thinking <on|off>"

    def test_test_mode_toggle(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/test", mock_runtime, mock_turn) is True
        assert "Test mode:" in mock_turn.assistant_text

        mock_runtime["ui"].refresh.reset_mock()
        assert _handle_command_in_chat(mock_agent, "/test on", mock_runtime, mock_turn) is True
        assert mock_agent.test_mode is True
        assert "now DISABLED" in mock_turn.assistant_text

        mock_runtime["ui"].refresh.reset_mock()
        assert _handle_command_in_chat(mock_agent, "/test off", mock_runtime, mock_turn) is True
        assert mock_agent.test_mode is False
        assert "now ENABLED" in mock_turn.assistant_text

    def test_empty_and_unknown_command(self, mock_agent, mock_turn, mock_runtime):
        assert _handle_command_in_chat(mock_agent, "/", mock_runtime, mock_turn) is True
        assert mock_turn.assistant_text == "Empty command."

        mock_runtime["ui"].refresh.reset_mock()
        assert _handle_command_in_chat(mock_agent, "/unknown", mock_runtime, mock_turn) is True
        assert "Unknown command" in mock_turn.assistant_text


class TestObjectiveCommandHandling:
    @pytest.fixture
    def mock_agent_with_objective(self):
        agent = Mock()
        agent._current_objective = "test objective"
        agent._last_objective = "test objective"
        agent.clear_objective = Mock()
        agent.set_manual_objective = Mock()
        return agent

    @pytest.fixture
    def mock_runtime_with_ui(self):
        ui = Mock()
        ui.clear_objective = Mock()
        ui.set_objective = Mock()
        return {"ui": ui}

    def test_objective_show_current(self, mock_agent_with_objective, mock_runtime_with_ui):
        text = _handle_objective_in_chat(mock_agent_with_objective, [], mock_runtime_with_ui)
        assert text.startswith("Current objective:")

    def test_objective_clear(self, mock_agent_with_objective, mock_runtime_with_ui):
        text = _handle_objective_in_chat(mock_agent_with_objective, ["clear"], mock_runtime_with_ui)
        assert text == "Objective cleared"
        mock_agent_with_objective.clear_objective.assert_called_once()

    def test_objective_set(self, mock_agent_with_objective, mock_runtime_with_ui):
        text = _handle_objective_in_chat(mock_agent_with_objective, ["set", "ship", "feature"], mock_runtime_with_ui)
        assert text.startswith("Objective set to:")
        mock_agent_with_objective.set_manual_objective.assert_called_once()

    def test_objective_usage(self, mock_agent_with_objective, mock_runtime_with_ui):
        text = _handle_objective_in_chat(mock_agent_with_objective, ["unknown"], mock_runtime_with_ui)
        assert text.startswith("Usage: /objective")
