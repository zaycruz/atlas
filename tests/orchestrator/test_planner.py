from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_main.orchestrator.planner import IntelligentPlanner, SimplePlanner
from atlas_main.orchestrator.types import EnhancedPlan, EnhancedStepSpec


class _NoopFactory:
    def __init__(self, agents: dict[str, object]) -> None:
        self._agents = agents

    async def create(self, agent_id: str) -> object:
        return self._agents[agent_id]

    def list_agents(self):
        return {key: {"type": "reasoning" if key.startswith("planner") else "other"} for key in self._agents}


class _ContextStub:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def analyze(self) -> dict[str, object]:
        return {"structure": {"root": []}, "languages": ["Python"]}


class _SessionStub:
    def __init__(self, *, plan: EnhancedPlan, **_: object) -> None:
        self._plan = plan

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self) -> EnhancedPlan:
        return self._plan


def test_simple_planner_returns_enhanced_plan(tmp_path: Path) -> None:
    planner = SimplePlanner()
    plan = planner.plan(objective="Add feature", repo_path=str(tmp_path))

    assert isinstance(plan, EnhancedPlan)
    assert len(plan.steps) == 3
    assert plan.steps[0].id == "plan"


def test_intelligent_planner_fallback_on_error(tmp_path: Path) -> None:
    factory = _NoopFactory({})
    planner = IntelligentPlanner(
        factory,
        planner_agent_id="planner-deepseek",
        context_cls=_ContextStub,
    )

    plan = asyncio.run(planner.plan(objective="Add login", repo_path=str(tmp_path)))

    assert isinstance(plan, EnhancedPlan)
    assert plan.steps[0].id == "plan"


def test_intelligent_planner_uses_session(tmp_path: Path) -> None:
    plan = EnhancedPlan(
        objective="Add feature",
        steps=[
            EnhancedStepSpec(
                id="impl",
                description="Implement",
                agent_id="codex",
            )
        ],
    )

    factory = _NoopFactory({"planner-deepseek": object()})
    planner = IntelligentPlanner(
        factory,
        context_cls=_ContextStub,
        session_factory=lambda **kwargs: _SessionStub(plan=plan, **kwargs),
    )

    result = asyncio.run(planner.plan(objective="Add feature", repo_path=str(tmp_path)))

    assert result.objective == "Add feature"
    assert result.steps[0].id == "impl"
    assert result.codebase_context


def test_should_plan_heuristics() -> None:
    planner = IntelligentPlanner(_NoopFactory({}), context_cls=_ContextStub)
    assert planner.should_plan("Add OAuth integration") is True
    assert planner.should_plan("Fix typo") is False
