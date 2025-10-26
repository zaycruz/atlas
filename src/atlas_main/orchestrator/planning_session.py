"""Bounded collaboration loop between Atlas and reasoning planner agents."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..agents.base import ConversationMessage
from ..agents.reasoning import ReasoningAgentAdapter
from .types import EnhancedPlan, PlanningContext, StepSpec


@dataclass
class PlanningTurn:
    role: str
    content: str
    metadata: Dict[str, Any]


class PlanningSessionError(RuntimeError):
    """Raised when the planning session cannot complete."""


class PlanningSession:
    def __init__(
        self,
        *,
        planner: ReasoningAgentAdapter,
        planning_context: PlanningContext,
        objective: str,
        max_rounds: int = 8,
        response_timeout: float = 60.0,
    ) -> None:
        self._planner = planner
        self._planning_context = planning_context
        self._objective = objective
        self._max_rounds = max(1, max_rounds)
        self._response_timeout = response_timeout
        self._turns: List[PlanningTurn] = []
        self._session = None

    @property
    def turns(self) -> List[PlanningTurn]:
        return list(self._turns)

    async def __aenter__(self) -> "PlanningSession":
        step = StepSpec(id="planning", description=self._objective, agent_id=self._planner.agent_id)
        shared_context: Mapping[str, Any] = {"planning_context": self._planning_context.__dict__}
        self._session = await self._planner.open_session(
            task=self._objective,
            step=step,
            shared_context=shared_context,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
        await self._planner.close()

    async def run(self) -> EnhancedPlan:
        if self._session is None:
            raise PlanningSessionError("session not started")
        for round_index in range(self._max_rounds):
            reply = await self._session.receive(timeout=self._response_timeout)
            if reply is None:
                raise PlanningSessionError("Planner did not respond in time")
            self._turns.append(
                PlanningTurn(role=reply.role, content=reply.content, metadata=dict(reply.metadata or {}))
            )
            candidate = self._extract_plan(reply.content)
            if candidate is not None:
                enriched = candidate
                enriched.reasoning_trace = reply.metadata.get("reasoning_trace", "")
                enriched.codebase_context = self._planning_context.codebase_structure
                enriched.task_id = reply.metadata.get("task_id", "")
                return enriched
            await self._session.send(
                ConversationMessage(
                    role="user",
                    content="Please provide the plan in the required JSON format.",
                    metadata={"round": round_index + 1},
                )
            )
        raise PlanningSessionError("Planner failed to produce a valid plan within round limit")

    def _extract_plan(self, payload: str) -> Optional[EnhancedPlan]:
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "steps" not in data:
            return None
        return EnhancedPlan(
            objective=data.get("objective", self._objective),
            steps=[self._to_step(step) for step in data.get("steps", []) if self._is_valid_step(step)],
            notes=data.get("reasoning", ""),
            parallel_groups=data.get("parallel_groups", []),
        )

    @staticmethod
    def _is_valid_step(data: Dict[str, Any]) -> bool:
        return isinstance(data, dict) and {"id", "description", "agent_id"}.issubset(data.keys())

    @staticmethod
    def _to_step(data: Dict[str, Any]) -> StepSpec:
        return StepSpec(
            id=str(data["id"]),
            description=str(data["description"]),
            agent_id=str(data["agent_id"]),
            inputs=dict(data.get("inputs", {})),
            tags=list(data.get("tags", [])),
            depends_on=list(data.get("depends_on", [])),
        )

