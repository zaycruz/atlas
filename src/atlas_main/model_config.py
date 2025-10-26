"""Model-specific configuration for Atlas token limits and context windows.

This module provides intelligent detection of model context limits and
appropriate token budgeting based on the specific model being used.
"""

import os
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelLimits:
    """Token limits for a specific model."""
    context_window: int  # Maximum context window for the model
    working_budget: int  # Recommended working memory budget (75% of context)
    description: str     # Human-readable description


# Model context registry based on common Ollama models
MODEL_CONTEXT_LIMITS: Dict[str, ModelLimits] = {
    # 128K context models
    "llama3.1": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="Llama 3.1 (128K context)"
    ),
    "llama3.2": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="Llama 3.2 (128K context)"
    ),
    "qwen2.5": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="Qwen 2.5 (128K context)"
    ),
    "qwen3": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="Qwen 3 (128K context)"
    ),
    "gpt-oss": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="GPT-OSS (128K context)"
    ),
    "deepseek": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="DeepSeek (128K context)"
    ),
    "gpt4": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="GPT-4 (128K context)"
    ),
    "gpt-4": ModelLimits(
        context_window=128000,
        working_budget=96000,
        description="GPT-4 (128K context)"
    ),

    # 32K context models
    "mistral": ModelLimits(
        context_window=32000,
        working_budget=24000,
        description="Mistral (32K context)"
    ),
    "codellama": ModelLimits(
        context_window=32000,
        working_budget=24000,
        description="CodeLlama (32K context)"
    ),
    "mixtral": ModelLimits(
        context_window=32000,
        working_budget=24000,
        description="Mixtral (32K context)"
    ),

    # 8K context models
    "llama3": ModelLimits(
        context_window=8000,
        working_budget=6000,
        description="Llama 3 (8K context)"
    ),

    # 4K context models (older models)
    "llama2": ModelLimits(
        context_window=4096,
        working_budget=3000,
        description="Llama 2 (4K context)"
    ),

    # Other common models
    "phi": ModelLimits(
        context_window=32000,
        working_budget=24000,
        description="Phi (32K context)"
    ),
    "gemma": ModelLimits(
        context_window=8000,
        working_budget=6000,
        description="Gemma (8K context)"
    ),
    "yi": ModelLimits(
        context_window=200000,  # Yi models often have 200K context
        working_budget=150000,
        description="Yi (200K context)"
    ),
}

# Default fallback for unknown models (conservative estimate)
DEFAULT_MODEL_LIMITS = ModelLimits(
    context_window=8000,
    working_budget=6000,
    description="Unknown model (conservative 8K estimate)"
)


def extract_base_model_name(model_name: str) -> str:
    """Extract the base model name from a full model identifier.

    Examples:
        "llama3.1:8b" -> "llama3.1"
        "qwen2.5:7b-instruct" -> "qwen2.5"
        "mistral:7b" -> "mistral"
        "custom-model:v1.0" -> "custom-model"
    """
    if not model_name:
        return ""

    # Remove any leading/trailing whitespace
    model_name = model_name.strip().lower()

    # Split on common separators to get the base name
    # Handle formats like: "model:tag", "model-size", "model-version", etc.
    for separator in [":", "-", "_", "."]:
        if separator in model_name:
            base_name = model_name.split(separator)[0]
            # Special handling for common model families
            if base_name in ["llama", "qwen", "gpt"]:
                # Look ahead to see if there's a version number
                parts = model_name.split(separator)
                if len(parts) > 1 and parts[1].replace(".", "").isdigit():
                    return f"{base_name}{parts[1]}"
            return base_name

    return model_name


def get_model_limits(model_name: str) -> ModelLimits:
    """Get token limits for a specific model.

    Args:
        model_name: Full model name (e.g., "llama3.1:8b", "qwen2.5:7b")

    Returns:
        ModelLimits object with context window and working budget
    """
    if not model_name:
        return DEFAULT_MODEL_LIMITS

    base_name = extract_base_model_name(model_name)

    # Try to find exact match first
    if base_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[base_name]

    # Try partial matching for model families
    for known_model, limits in MODEL_CONTEXT_LIMITS.items():
        if known_model in base_name or base_name in known_model:
            return limits

    # Check environment variable for custom limits
    custom_limits = _get_custom_model_limits(model_name)
    if custom_limits:
        return custom_limits

    # Fallback to default
    return DEFAULT_MODEL_LIMITS


def _get_custom_model_limits(model_name: str) -> Optional[ModelLimits]:
    """Get custom model limits from environment variables.

    Environment variable format:
    ATLAS_MODEL_LIMITS={"llama3.1": {"context": 128000, "working": 96000}}

    Or per-model:
    ATLAS_LLAMA3_1_CONTEXT=128000
    ATLAS_LLAMA3_1_WORKING=96000
    """
    # Try JSON format first
    custom_json = os.getenv("ATLAS_MODEL_LIMITS")
    if custom_json:
        try:
            import json
            custom_config = json.loads(custom_json)
            base_name = extract_base_model_name(model_name)
            if base_name in custom_config:
                config = custom_config[base_name]
                return ModelLimits(
                    context_window=config.get("context", DEFAULT_MODEL_LIMITS.context_window),
                    working_budget=config.get("working", DEFAULT_MODEL_LIMITS.working_budget),
                    description=f"Custom configuration for {base_name}"
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Fall through to individual environment variables

    # Try individual environment variables
    base_name = extract_base_model_name(model_name).upper().replace("-", "_").replace(".", "_")

    context_env = f"ATLAS_{base_name}_CONTEXT"
    working_env = f"ATLAS_{base_name}_WORKING"

    context_val = os.getenv(context_env)
    working_val = os.getenv(working_env)

    if context_val or working_val:
        try:
            context = int(context_val) if context_val else DEFAULT_MODEL_LIMITS.context_window
            working = int(working_val) if working_val else DEFAULT_MODEL_LIMITS.working_budget
            return ModelLimits(
                context_window=context,
                working_budget=working,
                description=f"Custom configuration for {model_name}"
            )
        except ValueError:
            pass  # Invalid values, fall through to default

    return None


def validate_model_limits(limits: ModelLimits) -> Tuple[bool, str]:
    """Validate that model limits are sensible.

    Args:
        limits: ModelLimits to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if limits.context_window <= 0:
        return False, "Context window must be positive"

    if limits.working_budget <= 0:
        return False, "Working budget must be positive"

    if limits.working_budget > limits.context_window:
        return False, "Working budget cannot exceed context window"

    # Working budget should be reasonable (not too close to context window)
    utilization = limits.working_budget / limits.context_window
    if utilization > 0.95:
        return False, "Working budget too close to context window (leave more buffer)"

    if utilization < 0.3:
        return False, "Working budget too small (use at least 30% of context)"

    return True, ""


def get_all_supported_models() -> Dict[str, str]:
    """Get all supported models and their descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    return {name: limits.description for name, limits in MODEL_CONTEXT_LIMITS.items()}


def is_model_supported(model_name: str) -> bool:
    """Check if a model is in the supported model registry.

    Args:
        model_name: Model name to check

    Returns:
        True if model is supported, False otherwise
    """
    base_name = extract_base_model_name(model_name)
    return base_name in MODEL_CONTEXT_LIMITS