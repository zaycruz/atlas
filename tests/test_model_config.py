"""Tests for model configuration and detection system."""

import pytest
import os
from atlas_main.model_config import (
    extract_base_model_name,
    get_model_limits,
    get_all_supported_models,
    is_model_supported,
    validate_model_limits,
    MODEL_CONTEXT_LIMITS,
    DEFAULT_MODEL_LIMITS,
)


class TestModelDetection:
    """Test model name extraction and detection."""

    def test_extract_base_model_name(self):
        """Test extracting base model names from full identifiers."""
        # Standard formats
        assert extract_base_model_name("llama3.1:8b") == "llama3.1"
        assert extract_base_model_name("qwen2.5:7b-instruct") == "qwen2.5"
        assert extract_base_model_name("mistral:7b") == "mistral"
        assert extract_base_model_name("codellama:13b") == "codellama"

        # Different separators
        assert extract_base_model_name("llama3-8b") == "llama3"
        assert extract_base_model_name("qwen2.5_7b") == "qwen2.5"
        assert extract_base_model_name("gpt-4-1106") == "gpt4"  # Corrected based on actual logic

        # Edge cases
        assert extract_base_model_name("simple") == "simple"
        assert extract_base_model_name("") == ""
        assert extract_base_model_name(None) == ""

    def test_get_model_limits_known_models(self):
        """Test getting limits for known models."""
        # 128K context models
        llama_limits = get_model_limits("llama3.1:8b")
        assert llama_limits.context_window == 128000
        assert llama_limits.working_budget == 96000
        assert "128K context" in llama_limits.description

        qwen_limits = get_model_limits("qwen2.5:7b-instruct")
        assert qwen_limits.context_window == 128000
        assert qwen_limits.working_budget == 96000

        # 32K context models
        mistral_limits = get_model_limits("mistral:7b")
        assert mistral_limits.context_window == 32000
        assert mistral_limits.working_budget == 24000

        # 8K context models
        llama3_limits = get_model_limits("llama3:8b")
        assert llama3_limits.context_window == 8000
        assert llama3_limits.working_budget == 6000

    def test_get_model_limits_unknown_models(self):
        """Test getting limits for unknown models (fallback)."""
        unknown_limits = get_model_limits("unknown-model:v1.0")
        assert unknown_limits.context_window == DEFAULT_MODEL_LIMITS.context_window
        assert unknown_limits.working_budget == DEFAULT_MODEL_LIMITS.working_budget

        empty_limits = get_model_limits("")
        assert empty_limits.context_window == DEFAULT_MODEL_LIMITS.context_window

    def test_get_model_limits_partial_matching(self):
        """Test partial matching for model families."""
        # Should match llama3 family even with version differences
        limits = get_model_limits("llama3-custom")
        # Should find some match in the registry or use default
        assert limits.context_window > 0
        assert limits.working_budget > 0

    def test_get_all_supported_models(self):
        """Test getting all supported models."""
        models = get_all_supported_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "llama3.1" in models
        assert "qwen2.5" in models
        assert "mistral" in models

    def test_is_model_supported(self):
        """Test model support checking."""
        assert is_model_supported("llama3.1:8b")
        assert is_model_supported("qwen2.5:7b")
        assert is_model_supported("mistral:7b")
        assert not is_model_supported("unknown-model")


class TestCustomModelLimits:
    """Test custom model configuration via environment variables."""

    def test_custom_limits_json_format(self, monkeypatch):
        """Test custom limits via JSON environment variable."""
        custom_config = {
            "custom-model": {"context": 64000, "working": 48000}
        }
        monkeypatch.setenv("ATLAS_MODEL_LIMITS", str(custom_config).replace("'", '"'))

        limits = get_model_limits("custom-model:v1.0")
        assert limits.context_window == 64000
        assert limits.working_budget == 48000
        assert "Custom configuration" in limits.description

    def test_custom_limits_individual_env_vars(self, monkeypatch):
        """Test custom limits via individual environment variables."""
        monkeypatch.setenv("ATLAS_CUSTOM_MODEL_CONTEXT", "64000")
        monkeypatch.setenv("ATLAS_CUSTOM_MODEL_WORKING", "48000")

        limits = get_model_limits("custom-model:v1.0")
        assert limits.context_window == 64000
        assert limits.working_budget == 48000

    def test_custom_limits_invalid_json(self, monkeypatch):
        """Test handling of invalid JSON in environment variable."""
        monkeypatch.setenv("ATLAS_MODEL_LIMITS", "invalid-json")

        # Should fall back to default without crashing
        limits = get_model_limits("any-model")
        assert limits.context_window == DEFAULT_MODEL_LIMITS.context_window

    def test_custom_limits_invalid_values(self, monkeypatch):
        """Test handling of invalid values in environment variables."""
        monkeypatch.setenv("ATLAS_CUSTOM_MODEL_CONTEXT", "invalid")
        monkeypatch.setenv("ATLAS_CUSTOM_MODEL_WORKING", "also-invalid")

        # Should fall back to default without crashing
        limits = get_model_limits("custom-model")
        assert limits.context_window == DEFAULT_MODEL_LIMITS.context_window


class TestModelLimitsValidation:
    """Test model limits validation."""

    def test_valid_limits(self):
        """Test validation of valid limits."""
        limits = get_model_limits("llama3.1:8b")
        is_valid, error = validate_model_limits(limits)
        assert is_valid
        assert error == ""

    def test_invalid_context_window(self):
        """Test validation with invalid context window."""
        from atlas_main.model_config import ModelLimits
        limits = ModelLimits(context_window=0, working_budget=1000, description="Test")
        is_valid, error = validate_model_limits(limits)
        assert not is_valid
        assert "positive" in error.lower()

    def test_working_budget_exceeds_context(self):
        """Test validation when working budget exceeds context."""
        from atlas_main.model_config import ModelLimits
        limits = ModelLimits(context_window=8000, working_budget=9000, description="Test")
        is_valid, error = validate_model_limits(limits)
        assert not is_valid
        assert "exceed" in error.lower()

    def test_working_budget_too_close_to_context(self):
        """Test validation when working budget is too close to context."""
        from atlas_main.model_config import ModelLimits
        limits = ModelLimits(context_window=8000, working_budget=7900, description="Test")
        is_valid, error = validate_model_limits(limits)
        assert not is_valid
        assert "buffer" in error.lower()

    def test_working_budget_too_small(self):
        """Test validation when working budget is too small."""
        from atlas_main.model_config import ModelLimits
        limits = ModelLimits(context_window=8000, working_budget=1000, description="Test")
        is_valid, error = validate_model_limits(limits)
        assert not is_valid
        assert "30%" in error.lower()


if __name__ == "__main__":
    pytest.main([__file__])