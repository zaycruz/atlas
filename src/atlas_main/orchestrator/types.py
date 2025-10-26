"""Typed objects for the Atlas orchestrator vertical slice."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
TaskStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass
class Artifact:
    """Represents an artifact produced by an agent step."""

    kind: str
    content: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSpec:
    """Single unit of work that can be delegated to an agent."""

    id: str
    description: str
    agent_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Outcome of a delegated step."""

    step_id: str
    status: StepStatus
    summary: str
    artifacts: List[Artifact] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


@dataclass
class TaskSpec:
    """Orchestrator view of a multi-step task."""

    id: str
    objective: str
    steps: List[StepSpec]
    shared_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """High-level plan before execution (DAG)."""

    objective: str
    steps: List[StepSpec]
    notes: str = ""


@dataclass
class ParallelGroup:
    """Group of steps that can execute concurrently."""

    steps: List[StepSpec]
    merge_strategy: str = "smart"  # smart, sequential, octopus, manual


@dataclass
class BranchConfig:
    """Branch strategy for a step."""

    base_branch: str = "main"
    step_branch: str = ""
    auto_merge: bool = True


@dataclass
class EnhancedStepSpec(StepSpec):
    """Extended step with branch + parallelization metadata."""

    branch_config: Optional[BranchConfig] = None
    estimated_duration: Optional[int] = None  # seconds
    parallel_group_id: Optional[str] = None


@dataclass
class EnhancedPlan(Plan):
    """Plan with parallel execution groups and planner metadata."""

    parallel_groups: List[ParallelGroup] = field(default_factory=list)
    codebase_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: str = ""
    task_id: str = ""


@dataclass
class PlanningContext:
    """Context supplied to planner agents for plan generation."""

    objective: str
    repo_path: str
    codebase_structure: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    available_agents: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Final aggregation of step outcomes."""

    task_id: str
    status: TaskStatus
    step_results: List[StepResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


@dataclass
class TaskEvent:
    """Structured event emitted during orchestration."""

    type: str
    task_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
