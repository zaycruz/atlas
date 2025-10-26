from __future__ import annotations

from atlas_main.orchestrator.types import (
    BranchConfig,
    EnhancedPlan,
    EnhancedStepSpec,
    ParallelGroup,
    PlanningContext,
    StepSpec,
)


def test_parallel_group_defaults() -> None:
    step = StepSpec(id="s1", description="Do work", agent_id="codex")
    group = ParallelGroup(steps=[step])

    assert group.steps == [step]
    assert group.merge_strategy == "smart"


def test_branch_config_defaults_and_overrides() -> None:
    default_cfg = BranchConfig()
    assert default_cfg.base_branch == "main"
    assert default_cfg.step_branch == ""
    assert default_cfg.auto_merge is True

    override = BranchConfig(base_branch="develop", step_branch="atlas/task", auto_merge=False)
    assert override.base_branch == "develop"
    assert override.step_branch == "atlas/task"
    assert override.auto_merge is False


def test_enhanced_step_spec_extends_step_spec() -> None:
    branch_cfg = BranchConfig(step_branch="atlas/demo")
    enhanced = EnhancedStepSpec(
        id="plan",
        description="Create plan",
        agent_id="planner",
        branch_config=branch_cfg,
        estimated_duration=120,
        parallel_group_id="group-1",
    )

    assert isinstance(enhanced, StepSpec)
    assert enhanced.branch_config is branch_cfg
    assert enhanced.estimated_duration == 120
    assert enhanced.parallel_group_id == "group-1"
    assert enhanced.depends_on == []


def test_enhanced_plan_defaults_are_isolated() -> None:
    plan = EnhancedPlan(objective="Test", steps=[])

    assert plan.parallel_groups == []
    assert plan.codebase_context == {}
    assert plan.reasoning_trace == ""
    assert plan.task_id == ""

    group = ParallelGroup(steps=[StepSpec(id="s1", description="Work", agent_id="codex")])
    plan.parallel_groups.append(group)
    assert plan.parallel_groups == [group]

    second_plan = EnhancedPlan(objective="Other", steps=[])
    assert second_plan.parallel_groups == []


def test_planning_context_defaults() -> None:
    context = PlanningContext(objective="Improve docs", repo_path="/repo")

    assert context.codebase_structure == {}
    assert context.constraints == {}
    assert context.available_agents == []
