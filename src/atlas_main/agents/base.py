"""Common adapter interfaces for external coding agents."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping, Protocol

from ..orchestrator.types import StepResult, StepSpec


class AgentAdapter(Protocol):
    """Contract for agent adapters."""

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        ...

    async def close(self) -> None:  # pragma: no cover - optional implementation
        ...


def step_to_payload(step: StepSpec) -> Dict[str, Any]:
    """Serialize StepSpec to a JSON friendly mapping."""
    payload = asdict(step)
    return payload
