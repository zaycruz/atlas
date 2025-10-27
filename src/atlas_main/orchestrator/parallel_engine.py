"""Parallel orchestrator capable of executing independent step groups concurrently."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .branches import BranchStrategy
from .engine import EventCallback, Orchestrator, _call_agent, _maybe_close
from .feedback import FeedbackCollector
from .planning_session import PlanningSession, PlanningSessionError
from .types import (
    BranchConfig,
    EnhancedPlan,
    EnhancedStepSpec,
    ParallelGroup,
    PlanningContext,
    StepResult,
    StepSpec,
    TaskResult,
    TaskSpec,
)


class ParallelOrchestrator(Orchestrator):
    """Extend base orchestrator with parallel awareness and optional branch isolation."""

    def __init__(
        self,
        agent_factory: "AgentFactory",
        *,
        branch_strategy: Optional[BranchStrategy] = None,
        max_parallel: int = 4,
        event_callback: Optional[EventCallback] = None,
    ) -> None:
        super().__init__(agent_factory, event_callback=event_callback)
        self._branch_strategy = branch_strategy
        self._max_parallel = max(1, max_parallel)

    async def run_task(self, task: TaskSpec) -> TaskResult:
        if not task.steps:
            return TaskResult(task_id=task.id, status="succeeded", step_results=[])

        plan = self._upgrade_plan(task)
        ready_groups, missing = self._identify_parallel_groups(plan)

        shared_context: Dict[str, Any] = dict(task.shared_context)
        executed: Dict[str, StepResult] = {}
        failures = False

        if missing:
            failures = True
            for step in missing:
                result = StepResult(
                    step_id=step.id,
                    status="failed",
                    summary="Dependency cycle or missing dependency detected.",
                )
                executed[step.id] = result

        for group in ready_groups:
            if failures:
                for step in group:
                    if step.id in executed:
                        continue
                    skipped = StepResult(
                        step_id=step.id,
                        status="skipped",
                        summary="Skipped due to earlier failure.",
                    )
                    executed[step.id] = skipped
                    self._emit("step.skipped", task, {"step_id": step.id, "reason": skipped.summary})
                continue

            if len(group) > 1:
                results = await self._run_parallel_group(group, task, shared_context)
            else:
                results = [await self._run_single_step(group[0], task, shared_context)]

            for result in results:
                executed[result.step_id] = result
                if not result.succeeded:
                    failures = True

        status = "succeeded" if not failures else "failed"
        ordered_results = [executed.get(step.id) for step in plan.steps if step.id in executed]
        return TaskResult(task_id=task.id, status=status, step_results=ordered_results)

    async def run_task_with_revision(
        self,
        task: TaskSpec,
        *,
        max_revisions: int = 3,
    ) -> TaskResult:
        current_task = task
        result = await self.run_task(current_task)
        attempts = 0
        while not result.succeeded and attempts < max_revisions:
            failure_results = [sr for sr in result.step_results if sr and sr.status == "failed"]
            if not failure_results:
                break

            plan_dict = self._extract_plan_dict(current_task)
            if not plan_dict:
                break
            planner_agent_id = self._extract_planner_agent_id(current_task) or "planner-deepseek"
            try:
                planner_agent = await self._agent_factory.create(planner_agent_id)
            except Exception:
                break

            plan = self._plan_from_dict(plan_dict)
            collector = FeedbackCollector()
            contexts = [collector.capture_failure(sr) for sr in failure_results]

            planning_context = PlanningContext(
                objective=current_task.objective,
                repo_path=self._infer_repo_path(current_task),
                codebase_structure=plan.codebase_context,
            )

            try:
                session = PlanningSession(
                    planner=planner_agent,
                    planning_context=planning_context,
                    objective=current_task.objective,
                    max_rounds=8,
                )
                async with session as active:
                    revised_plan = await active.revise_plan(plan, contexts)
            except PlanningSessionError:
                break
            except Exception:
                break
            else:
                current_task = self._task_from_plan(current_task, revised_plan, planner_agent_id)
                result = await self.run_task(current_task)
                attempts += 1
                continue
            break
        return result

    # ------------------------------------------------------------------
    # Parallel execution helpers
    # ------------------------------------------------------------------
    async def _run_parallel_group(
        self,
        steps: Sequence[StepSpec],
        task: TaskSpec,
        shared_context: Dict[str, Any],
    ) -> List[StepResult]:
        sem = asyncio.Semaphore(self._max_parallel)
        tasks = [self._run_with_semaphore(sem, step, task, shared_context) for step in steps]
        return await asyncio.gather(*tasks)

    async def _run_with_semaphore(
        self,
        sem: asyncio.Semaphore,
        step: StepSpec,
        task: TaskSpec,
        shared_context: Dict[str, Any],
    ) -> StepResult:
        async with sem:
            return await self._run_single_step(step, task, shared_context)

    async def _run_single_step(
        self,
        step: StepSpec,
        task: TaskSpec,
        shared_context: Dict[str, Any],
    ) -> StepResult:
        step_inputs = dict(getattr(step, "inputs", {}))
        branch_name = None
        repo_path = step_inputs.get("repo_path")
        branch_cfg: Optional[BranchConfig] = getattr(step, "branch_config", None)

        if self._branch_strategy and repo_path and branch_cfg:
            branch_name = await self._create_branch(branch_cfg, task_id=task.id, step_id=step.id)
            step_inputs["checkout_branch"] = branch_name

        adapted_step = StepSpec(
            id=step.id,
            description=step.description,
            agent_id=step.agent_id,
            inputs=step_inputs,
            tags=list(step.tags),
            depends_on=list(step.depends_on),
        )

        self._emit("step.started", task, {"step_id": step.id, "agent_id": step.agent_id, "description": step.description})
        agent = await self._agent_factory.create(step.agent_id)
        try:
            result = await _call_agent(agent, adapted_step, shared_context)
        finally:
            await _maybe_close(agent)

        metadata: Dict[str, Any]
        if isinstance(result.metadata, dict):
            metadata = result.metadata
        else:
            metadata = {}
            result.metadata = metadata

        if branch_name:
            metadata.setdefault("branch", branch_name)

        if branch_name and branch_cfg and self._branch_strategy:
            if branch_cfg.auto_merge and result.succeeded:
                merge_result = await self._branch_strategy.merge_branches(
                    [branch_name],
                    target_branch=branch_cfg.base_branch,
                    strategy="sequential",
                )
                metadata["merge_result"] = asdict(merge_result)
                if merge_result.success:
                    await self._branch_strategy.cleanup_branches([branch_name], force=True)
                else:
                    result.status = "failed"
                    result.summary = (
                        result.summary.rstrip(".") + ". Merge failed: " + merge_result.message
                        if result.summary
                        else f"Merge failed: {merge_result.message}"
                    )
            elif branch_cfg and not branch_cfg.auto_merge:
                metadata["merge_pending"] = True

        self._emit(
            "step.completed",
            task,
            {
                "step_id": step.id,
                "status": result.status,
                "summary": result.summary,
                "artifacts": [artifact.kind for artifact in result.artifacts],
            },
        )
        shared_updates = metadata.get("shared_updates")
        if isinstance(shared_updates, dict):
            shared_context.update(shared_updates)
        return result

    async def _create_branch(self, cfg: BranchConfig, *, task_id: str, step_id: str) -> str:
        if cfg.step_branch:
            step_branch = cfg.step_branch
        else:
            step_branch = step_id
        return await self._branch_strategy.create_step_branch(
            task_id=task_id,
            step_id=step_branch,
            base_branch=cfg.base_branch,
        )

    # ------------------------------------------------------------------
    # Plan helpers
    # ------------------------------------------------------------------
    def _upgrade_plan(self, task: TaskSpec) -> EnhancedPlan:
        steps: List[EnhancedStepSpec] = []
        for step in task.steps:
            if isinstance(step, EnhancedStepSpec):
                steps.append(step)
            else:
                steps.append(self._to_enhanced(step))
        return EnhancedPlan(objective=task.objective, steps=steps)

    @staticmethod
    def _to_enhanced(step: StepSpec) -> EnhancedStepSpec:
        return EnhancedStepSpec(
            id=step.id,
            description=step.description,
            agent_id=step.agent_id,
            inputs=dict(step.inputs),
            tags=list(step.tags),
            depends_on=list(step.depends_on),
        )

    def _identify_parallel_groups(self, plan: EnhancedPlan) -> Tuple[List[List[EnhancedStepSpec]], List[EnhancedStepSpec]]:
        steps_by_id = {step.id: step for step in plan.steps}

        if plan.parallel_groups:
            groups: List[List[EnhancedStepSpec]] = []
            covered: set[str] = set()
            for group in plan.parallel_groups:
                group_steps = self._resolve_group_steps(group, steps_by_id)
                groups.append(group_steps)
                covered.update(step.id for step in group_steps)
            remaining = [step for step in plan.steps if step.id not in covered]
            if remaining:
                auto_groups, missing = self._topological_groups(remaining)
                groups.extend(auto_groups)
            else:
                auto_groups, missing = [], []
            return groups, missing

        return self._topological_groups(plan.steps)

    def _resolve_group_steps(
        self,
        group: ParallelGroup,
        steps_by_id: Dict[str, EnhancedStepSpec],
    ) -> List[EnhancedStepSpec]:
        resolved: List[EnhancedStepSpec] = []
        for step in group.steps:
            if isinstance(step, StepSpec):
                resolved.append(steps_by_id[step.id])
            elif isinstance(step, dict) and "id" in step:
                resolved.append(steps_by_id[str(step["id"])])
            elif isinstance(step, str):
                resolved.append(steps_by_id[step])
        return resolved

    def _topological_groups(
        self,
        steps: Iterable[EnhancedStepSpec],
    ) -> Tuple[List[List[EnhancedStepSpec]], List[EnhancedStepSpec]]:
        steps = list(steps)
        steps_by_id = {step.id: step for step in steps}
        dependents = defaultdict(set)
        in_degree = {step.id: len(step.depends_on) for step in steps}

        for step in steps:
            for dep in step.depends_on:
                dependents[dep].add(step.id)

        queue = deque([step_id for step_id, degree in in_degree.items() if degree == 0])
        groups: List[List[EnhancedStepSpec]] = []
        processed: set[str] = set()

        while queue:
            current_layer: List[EnhancedStepSpec] = []
            next_queue: deque[str] = deque()
            while queue:
                step_id = queue.popleft()
                processed.add(step_id)
                current_layer.append(steps_by_id[step_id])
                for child in dependents.get(step_id, set()):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            groups.append(current_layer)
            queue = next_queue

        missing = [steps_by_id[sid] for sid in steps_by_id.keys() - processed]
        return groups, missing

    def _extract_plan_dict(self, task: TaskSpec) -> Optional[Dict[str, Any]]:
        context = task.shared_context
        if isinstance(context, dict):
            plan_data = context.get("plan")
            if isinstance(plan_data, dict):
                return plan_data
        return None

    def _extract_planner_agent_id(self, task: TaskSpec) -> Optional[str]:
        context = task.shared_context
        if isinstance(context, dict):
            value = context.get("planner_agent_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _plan_from_dict(self, data: Dict[str, Any]) -> EnhancedPlan:
        steps: List[EnhancedStepSpec] = []
        for step_data in data.get("steps", []):
            branch_cfg_data = step_data.get("branch_config")
            branch_cfg = None
            if isinstance(branch_cfg_data, dict):
                branch_cfg = BranchConfig(
                    base_branch=str(branch_cfg_data.get("base_branch", "main")),
                    step_branch=str(branch_cfg_data.get("step_branch", "")),
                    auto_merge=bool(branch_cfg_data.get("auto_merge", True)),
                )
            steps.append(
                EnhancedStepSpec(
                    id=str(step_data.get("id", "")),
                    description=str(step_data.get("description", "")),
                    agent_id=str(step_data.get("agent_id", "")),
                    inputs=dict(step_data.get("inputs", {})),
                    tags=list(step_data.get("tags", [])),
                    depends_on=list(step_data.get("depends_on", [])),
                    branch_config=branch_cfg,
                    estimated_duration=step_data.get("estimated_duration"),
                    parallel_group_id=step_data.get("parallel_group_id"),
                )
            )
        plan = EnhancedPlan(
            objective=str(data.get("objective", "")),
            steps=steps,
            notes=str(data.get("notes", "")),
        )
        plan.codebase_context = data.get("codebase_context", {}) or {}
        plan.reasoning_trace = str(data.get("reasoning_trace", "")) if data.get("reasoning_trace") else ""
        plan.task_id = str(data.get("task_id", "")) if data.get("task_id") else ""
        return plan

    def _task_from_plan(
        self,
        original_task: TaskSpec,
        plan: EnhancedPlan,
        planner_agent_id: str,
    ) -> TaskSpec:
        shared_context: Dict[str, Any] = {}
        if isinstance(original_task.shared_context, dict):
            shared_context.update(original_task.shared_context)
        shared_context["plan"] = asdict(plan)
        shared_context["planner_agent_id"] = planner_agent_id
        return TaskSpec(
            id=original_task.id,
            objective=original_task.objective,
            steps=list(plan.steps),
            shared_context=shared_context,
            metadata=dict(getattr(original_task, "metadata", {}) or {}),
        )

    def _infer_repo_path(self, task: TaskSpec) -> str:
        for step in task.steps:
            inputs = getattr(step, "inputs", {})
            if isinstance(inputs, dict) and inputs.get("repo_path"):
                return str(inputs["repo_path"])
        return ""
