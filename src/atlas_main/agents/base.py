"""Common adapter interfaces for external coding agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

from ..orchestrator.types import StepResult, StepSpec


@dataclass
class ConversationMessage:
    """Single message exchanged with an external agent during a loop."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingAgentSession(Protocol):
    """Protocol for multi-turn conversations with an external agent."""

    async def send(self, message: ConversationMessage) -> None:
        """Send a message to the agent."""

    async def receive(self, *, timeout: Optional[float] = None) -> Optional[ConversationMessage]:
        """Receive the next message from the agent. Returns None on clean EOF."""

    async def close(self) -> None:
        """Terminate the session."""


class AgentAdapter(Protocol):
    """Contract for agent adapters."""

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        ...

    async def open_session(
        self,
        *,
        task: str,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> StreamingAgentSession:
        """Open a streaming session for iterative collaboration."""
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional implementation
        ...


def step_to_payload(step: StepSpec) -> Dict[str, Any]:
    """Serialize StepSpec to a JSON friendly mapping."""
    payload = asdict(step)
    return payload
