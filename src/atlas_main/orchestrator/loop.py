"""Agent conversation loop scaffolding."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Mapping, Optional

from .types import StepSpec, TaskEvent
from ..agents.base import ConversationMessage, StreamingAgentSession


@dataclass
class LoopConfig:
    """Configuration for a conversation loop."""

    max_turns: int = 8
    receive_timeout: float = 90.0
    summary_after_completion: bool = True


@dataclass
class LoopTurn:
    """Single exchange between Atlas and the agent."""

    turn_index: int
    request: ConversationMessage
    response: Optional[ConversationMessage]


@dataclass
class LoopResult:
    """Outcome of a conversation loop."""

    turns: List[LoopTurn] = field(default_factory=list)
    completed: bool = False
    reason: str = ""


LoopEventCallback = Callable[[TaskEvent], None]


class AgentLoopController:
    """Controls multi-turn conversations with external agents."""

    def __init__(
        self,
        *,
        session: StreamingAgentSession,
        step: StepSpec,
        shared_context: Mapping[str, Any],
        config: Optional[LoopConfig] = None,
        event_callback: Optional[LoopEventCallback] = None,
    ) -> None:
        self._session = session
        self._step = step
        self._shared_context = dict(shared_context)
        self._config = config or LoopConfig()
        self._event_callback = event_callback

    async def run(
        self,
        *,
        prompts: List[ConversationMessage],
        generate_reply: Callable[[LoopTurn, LoopResult], Awaitable[Optional[ConversationMessage]]],
    ) -> LoopResult:
        """Run the loop using the provided prompt list and reply generator."""
        result = LoopResult()
        pending_prompts = list(prompts)
        max_turns = max(1, self._config.max_turns)

        for turn_index in range(1, max_turns + 1):
            if not pending_prompts:
                break
            request = pending_prompts.pop(0)
            await self._session.send(request)
            response = await self._receive_message()
            turn = LoopTurn(turn_index=turn_index, request=request, response=response)
            result.turns.append(turn)
            if response is None:
                result.completed = True
                result.reason = "agent_finished"
                break
            follow_up = await generate_reply(turn, result)
            if follow_up is None:
                result.completed = True
                result.reason = "controller_halted"
                break
            pending_prompts.append(follow_up)
        else:
            result.completed = False
            result.reason = "max_turns_reached"
        return result

    async def _receive_message(self) -> Optional[ConversationMessage]:
        timeout = self._config.receive_timeout
        try:
            return await asyncio.wait_for(self._session.receive(timeout=timeout), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        await self._session.close()
