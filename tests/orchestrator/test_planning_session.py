from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from atlas_main.agents.base import ConversationMessage
from atlas_main.agents.reasoning import ReasoningAgentAdapter
from atlas_main.orchestrator.planning_session import PlanningSession, PlanningSessionError
from atlas_main.orchestrator.types import PlanningContext


class _FakeSession:
    def __init__(self) -> None:
        self.sent = []
        self.replies = asyncio.Queue()

    async def send(self, message: ConversationMessage) -> None:
        self.sent.append(message)

    async def receive(self, *, timeout: float | None = None):
        try:
            return await asyncio.wait_for(self.replies.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        return None


class _FakePlanner(ReasoningAgentAdapter):
    def __init__(self) -> None:
        self.agent_id = "planner"
        self.opened = False
        self.closed = False
        self.session = _FakeSession()

    async def open_session(self, *, task: str, step, shared_context):
        self.opened = True
        return self.session

    async def close(self) -> None:
        self.closed = True


def test_planning_session_success(monkeypatch):
    async def _run():
        planner = _FakePlanner()
        context = PlanningContext(objective="Add feature", repo_path="/repo")
        async with PlanningSession(planner=planner, planning_context=context, objective="Add feature", max_rounds=3) as session:
            plan_payload = {
                "objective": "Add feature",
                "steps": [
                    {"id": "plan", "description": "Plan", "agent_id": "planner"},
                ],
                "reasoning": "Because",
                "parallel_groups": [],
            }
            reply = ConversationMessage(role="assistant", content=json.dumps(plan_payload), metadata={})
            planner.session.replies.put_nowait(reply)
            plan = await session.run()
        assert planner.opened and planner.closed
        assert plan.objective == "Add feature"
        assert plan.steps[0].id == "plan"

    asyncio.run(_run())


def test_planning_session_timeout(monkeypatch):
    async def _run():
        planner = _FakePlanner()
        context = PlanningContext(objective="Add feature", repo_path="/repo")
        async with PlanningSession(
            planner=planner,
            planning_context=context,
            objective="Add feature",
            max_rounds=2,
            response_timeout=0.1,
        ) as session:
            with pytest.raises(PlanningSessionError):
                await session.run()

    asyncio.run(_run())


def test_planning_session_retries_invalid_payload(monkeypatch):
    async def _run():
        planner = _FakePlanner()
        context = PlanningContext(objective="Add feature", repo_path="/repo")
        async with PlanningSession(planner=planner, planning_context=context, objective="Add feature", max_rounds=3) as session:
            planner.session.replies.put_nowait(ConversationMessage(role="assistant", content="not-json", metadata={}))
            planner.session.replies.put_nowait(ConversationMessage(role="assistant", content=json.dumps({}), metadata={}))
            plan_payload = {
                "objective": "Add feature",
                "steps": [
                    {"id": "impl", "description": "Implement", "agent_id": "codex"},
                ],
                "reasoning": "Because",
                "parallel_groups": [],
            }
            planner.session.replies.put_nowait(ConversationMessage(role="assistant", content=json.dumps(plan_payload), metadata={}))
            plan = await session.run()
        assert plan.steps[0].id == "impl"
        assert len(planner.session.sent) >= 1

    asyncio.run(_run())
