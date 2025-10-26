from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atlas_main.orchestrator.types import EnhancedPlan, EnhancedStepSpec, StepResult, TaskResult
from atlas_main.tools import PlanAndExecuteTool, ToolError


class _Agent:
    def __init__(self, repo: Path) -> None:
        self._repo = repo

    def current_repo(self) -> str:
        return str(self._repo)


class _PlannerStub:
    def __init__(self, plan: EnhancedPlan) -> None:
        self._plan = plan

    async def plan(self, **_) -> EnhancedPlan:
        return self._plan


class _OrchestratorStub:
    def __init__(self, result: TaskResult) -> None:
        self._result = result

    async def run_task(self, task):
        return self._result


class _FactoryStub:
    def __init__(self) -> None:
        pass


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_plan_and_execute_success(repo: Path):
    plan = EnhancedPlan(
        objective="Example",
        steps=[
            EnhancedStepSpec(id="step", description="Do work", agent_id="codex"),
        ],
    )
    task_result = TaskResult(task_id="t1", status="succeeded", step_results=[
        StepResult(step_id="step", status="succeeded", summary="done")
    ])

    tool = PlanAndExecuteTool(
        _FactoryStub(),
        planner_factory=lambda planner_id: _PlannerStub(plan),
        orchestrator_factory=lambda branch_strategy, max_parallel: _OrchestratorStub(task_result),
    )

    message = tool.run(agent=_Agent(repo), objective="Example", repo_path=str(repo))

    assert "Plan executed successfully" in message


def test_plan_and_execute_failure(repo: Path):
    plan = EnhancedPlan(
        objective="Example",
        steps=[
            EnhancedStepSpec(id="step", description="Do work", agent_id="codex"),
        ],
    )
    task_result = TaskResult(task_id="t1", status="failed", step_results=[
        StepResult(step_id="step", status="failed", summary="oops")
    ])

    tool = PlanAndExecuteTool(
        _FactoryStub(),
        planner_factory=lambda planner_id: _PlannerStub(plan),
        orchestrator_factory=lambda branch_strategy, max_parallel: _OrchestratorStub(task_result),
    )

    with pytest.raises(ToolError):
        tool.run(agent=_Agent(repo), objective="Example", repo_path=str(repo))
