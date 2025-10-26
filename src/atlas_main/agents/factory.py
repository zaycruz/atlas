"""Agent factory that builds adapters from configuration."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml

from .process import ProcessAgentAdapter
from .codex import CodexAgentAdapter
from .claude import ClaudeAgentAdapter
from .droid import DroidAgentAdapter
from .reasoning import ReasoningAgentAdapter


class AgentFactory:
    """Loads agent definitions and instantiates adapters."""

    def __init__(self, *, config_path: Optional[Path] = None) -> None:
        self._config_path = config_path or _default_config_path()
        self._raw_config = self._load()

    async def create(self, agent_id: str):
        config = self._raw_config.get(agent_id)
        if config is None:
            raise KeyError(f"Unknown agent '{agent_id}'")

        agent_type = config.get("type")
        if agent_type == "process":
            command = config.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"Agent '{agent_id}' missing process command")
            timeout = float(config.get("timeout", 120))
            cwd = config.get("cwd")
            env = config.get("env") or {}
            normalized_command = [_resolve_command_part(part) for part in command]
            return ProcessAgentAdapter(agent_id=agent_id, command=normalized_command, cwd=cwd, env=env, timeout=timeout)
        if agent_type == "codex":
            binary = str(config.get("binary", "codex"))
            timeout = float(config.get("timeout", 600))
            extra_args = _ensure_string_list(config.get("args")) or ["--full-auto"]
            env = _string_mapping(config.get("env"))
            return CodexAgentAdapter(
                agent_id=agent_id,
                binary=binary,
                timeout=timeout,
                extra_args=extra_args,
                env=env,
            )
        if agent_type == "claude":
            binary = str(config.get("binary", "claude"))
            timeout = float(config.get("timeout", 600))
            permission_mode = str(config.get("permission_mode", "acceptEdits"))
            extra_args = _ensure_string_list(config.get("args"))
            env = _string_mapping(config.get("env"))
            return ClaudeAgentAdapter(
                agent_id=agent_id,
                binary=binary,
                timeout=timeout,
                permission_mode=permission_mode,
                extra_args=extra_args,
                env=env,
            )
        if agent_type == "droid":
            binary = str(config.get("binary", "droid"))
            timeout = float(config.get("timeout", 900))
            auto_level = str(config.get("auto_level", "medium"))
            env = _string_mapping(config.get("env"))
            extra_args = _ensure_string_list(config.get("args"))
            return DroidAgentAdapter(
                agent_id=agent_id,
                binary=binary,
                timeout=timeout,
                auto_level=auto_level,
                env=env,
                extra_args=extra_args,
            )
        if agent_type == "reasoning":
            model = str(config.get("model", "deepseek-r1"))
            provider = str(config.get("provider", "ollama"))
            timeout = float(config.get("timeout", 300))
            api_key = config.get("api_key")
            max_tokens = int(config.get("max_tokens", 8000))
            return ReasoningAgentAdapter(
                agent_id=agent_id,
                model=model,
                provider=provider,
                timeout=timeout,
                api_key=api_key,
                max_tokens=max_tokens,
            )
        raise ValueError(f"Unsupported agent type '{agent_type}' for '{agent_id}'")

    def get_agent_descriptor(self, agent_id: str) -> Dict[str, Any]:
        descriptor = self._raw_config.get(agent_id)
        if descriptor is None:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return dict(descriptor)

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._raw_config)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self._config_path.exists():
            return {}
        data = yaml.safe_load(self._config_path.read_text())
        agents = data.get("agents") if isinstance(data, dict) else None
        if not isinstance(agents, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for agent_id, config in agents.items():
            if isinstance(config, dict):
                normalized[str(agent_id)] = config
        return normalized


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "agents.yaml"


def _resolve_command_part(part: Any) -> str:
    text = str(part)
    if text == "{python}":
        executable = sys.executable or "python3"
        return executable
    return text


def _ensure_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return None


def _string_mapping(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}
