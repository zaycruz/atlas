from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

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
        self.result = result
        self.called_revision = False

    async def run_task_with_revision(self, task, *, max_revisions: int = 3):
        self.called_revision = True
        self.last_task = task
        self.last_max_revisions = max_revisions
        return self.result


class _FactoryStub:
    async def create(self, agent_id: str):
        raise NotImplementedError

    def list_agents(self):
        return {}


def test_plan_and_execute_invokes_revision_path(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = EnhancedPlan(
        objective="Example",
        steps=[EnhancedStepSpec(id="step", description="Work", agent_id="codex", inputs={"repo_path": str(repo)})],
    )
    task_result = TaskResult(
        task_id="t1",
        status="succeeded",
        step_results=[StepResult(step_id="step", status="succeeded", summary="done")],
    )

    orchestrator = _OrchestratorStub(task_result)
    tool = PlanAndExecuteTool(
        _FactoryStub(),
        planner_factory=lambda planner_id: _PlannerStub(plan),
        orchestrator_factory=lambda branch, max_parallel: orchestrator,
    )

    message = tool.run(agent=_Agent(repo), objective="Example", repo_path=str(repo))

    assert "Plan executed successfully" in message
    assert orchestrator.called_revision is True
    assert orchestrator.last_task.shared_context["planner_agent_id"] == "planner-deepseek"
    assert orchestrator.last_task.shared_context["plan"]["objective"] == "Example"


def test_plan_and_execute_propagates_failure(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = EnhancedPlan(
        objective="Example",
        steps=[EnhancedStepSpec(id="step", description="Work", agent_id="codex", inputs={"repo_path": str(repo)})],
    )
    failure = TaskResult(
        task_id="t1",
        status="failed",
        step_results=[StepResult(step_id="step", status="failed", summary="oops")],
    )
    orchestrator = _OrchestratorStub(failure)
    tool = PlanAndExecuteTool(
        _FactoryStub(),
        planner_factory=lambda planner_id: _PlannerStub(plan),
        orchestrator_factory=lambda branch, max_parallel: orchestrator,
    )

    try:
        tool.run(agent=_Agent(repo), objective="Example", repo_path=str(repo))
    except ToolError as exc:
        assert "Plan execution failed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ToolError was not raised")
