from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from atlas_main.orchestrator.parallel_engine import ParallelOrchestrator
from atlas_main.orchestrator.branches import BranchStrategy
from atlas_main.orchestrator.types import BranchConfig, EnhancedStepSpec, StepResult, StepSpec, TaskSpec


class _DelayedAgent:
    def __init__(self, delay: float = 0.1) -> None:
        self.delay = delay

    async def execute_step(self, *, step: StepSpec, shared_context: Any) -> StepResult:
        await asyncio.sleep(step.inputs.get("delay", self.delay))
        return StepResult(
            step_id=step.id,
            status="succeeded",
            summary="done",
        )


class _CommitAgent:
    def __init__(self, content: str) -> None:
        self.content = content

    async def execute_step(self, *, step: StepSpec, shared_context: Any) -> StepResult:
        repo = Path(step.inputs["repo_path"])
        file_path = repo / "app.txt"
        file_path.write_text(self.content, encoding="utf-8")
        subprocess.run(["git", "add", "app.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", f"Update {self.content}"], cwd=repo, check=True)
        return StepResult(
            step_id=step.id,
            status="succeeded",
            summary="committed",
        )


class _FakeFactory:
    def __init__(self, agents: dict[str, Any]) -> None:
        self._agents = agents

    async def create(self, agent_id: str) -> Any:
        return self._agents[agent_id]


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "app.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True)


def test_parallel_execution_speedup():
    steps = [
        EnhancedStepSpec(id="plan", description="plan", agent_id="delayed", inputs={"delay": 0.1}),
        EnhancedStepSpec(id="frontend", description="frontend", agent_id="delayed", inputs={"delay": 0.25}, depends_on=["plan"]),
        EnhancedStepSpec(id="backend", description="backend", agent_id="delayed", inputs={"delay": 0.25}, depends_on=["plan"]),
        EnhancedStepSpec(id="verify", description="verify", agent_id="delayed", inputs={"delay": 0.1}, depends_on=["frontend", "backend"]),
    ]
    task = TaskSpec(id="task", objective="Parallel test", steps=steps)

    factory = _FakeFactory({"delayed": _DelayedAgent()})
    orchestrator = ParallelOrchestrator(factory)

    start = time.perf_counter()
    result = asyncio.run(orchestrator.run_task(task))
    elapsed = time.perf_counter() - start

    assert result.succeeded
    assert len(result.step_results) == 4
    # Sequential execution would be roughly 0.7s; ensure we gained concurrency.
    assert elapsed < 0.55


def test_parallel_branch_merge(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    branch_strategy = BranchStrategy(repo)
    agents = {
        "committer": _CommitAgent("updated\n"),
    }
    factory = _FakeFactory(agents)
    orchestrator = ParallelOrchestrator(factory, branch_strategy=branch_strategy)

    step = EnhancedStepSpec(
        id="update",
        description="update file",
        agent_id="committer",
        inputs={"repo_path": str(repo)},
        branch_config=BranchConfig(base_branch="main"),
    )
    task = TaskSpec(id="task", objective="Branch test", steps=[step])

    result = asyncio.run(orchestrator.run_task(task))

    assert result.succeeded
    assert (repo / "app.txt").read_text(encoding="utf-8") == "updated\n"
    branches = subprocess.run(["git", "branch"], cwd=repo, check=True, capture_output=True, text=True).stdout
    assert "atlas/task/update" not in branches

