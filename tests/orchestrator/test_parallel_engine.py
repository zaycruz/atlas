from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from atlas_main.agents.base import ConversationMessage
from atlas_main.orchestrator.parallel_engine import ParallelOrchestrator
from atlas_main.orchestrator.branches import BranchStrategy
from atlas_main.orchestrator.types import BranchConfig, EnhancedPlan, EnhancedStepSpec, StepResult, StepSpec, TaskSpec


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


class _FailThenSucceedAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def execute_step(self, *, step: StepSpec, shared_context: Any) -> StepResult:
        self.calls += 1
        if step.id == "initial":
            return StepResult(
                step_id=step.id,
                status="failed",
                summary="Initial attempt failed",
                logs=["error: something went wrong"],
                metadata={"agent_id": "failing"},
            )
        return StepResult(step_id=step.id, status="succeeded", summary="fixed")


class _RevisionSession:
    def __init__(self, reply: ConversationMessage) -> None:
        self._reply = reply
        self._sent = False
        self.sent_messages: list[ConversationMessage] = []

    async def send(self, message: ConversationMessage) -> None:
        self.sent_messages.append(message)

    async def receive(self, *, timeout: float | None = None):
        if self.sent_messages and not self._sent:
            self._sent = True
            return self._reply
        return None

    async def close(self) -> None:
        return None


class _RevisionPlannerAdapter:
    def __init__(self, reply: ConversationMessage) -> None:
        self.agent_id = "planner-deepseek"
        self._reply = reply

    async def open_session(self, *, task: str, step, shared_context):
        return _RevisionSession(self._reply)

    async def close(self) -> None:
        return None


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


def test_run_task_with_revision(tmp_path: Path):
    agent = _FailThenSucceedAgent()
    revised_plan_payload = {
        "objective": "Fix issue",
        "steps": [
            {
                "id": "fix",
                "description": "Apply fix",
                "agent_id": "failing",
                "inputs": {"repo_path": str(tmp_path)},
            }
        ],
        "parallel_groups": [],
    }
    reply = ConversationMessage(role="assistant", content=json.dumps(revised_plan_payload), metadata={})

    factory = _FakeFactory(
        {
            "failing": agent,
            "planner-deepseek": _RevisionPlannerAdapter(reply),
        }
    )
    orchestrator = ParallelOrchestrator(factory)

    initial_plan = EnhancedPlan(
        objective="Fix issue",
        steps=[
            EnhancedStepSpec(
                id="initial",
                description="Initial attempt",
                agent_id="failing",
                inputs={"repo_path": str(tmp_path)},
            )
        ],
    )
    task = TaskSpec(
        id="task",
        objective="Fix issue",
        steps=list(initial_plan.steps),
        shared_context={
            "plan": asdict(initial_plan),
            "planner_agent_id": "planner-deepseek",
        },
    )

    result = asyncio.run(orchestrator.run_task_with_revision(task, max_revisions=2))

    assert result.succeeded
    assert any(sr.step_id == "fix" for sr in result.step_results)
    assert agent.calls >= 2
