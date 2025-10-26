"""Simple planner that decomposes an objective into a DAG of steps.

This is a stub to enable sub-agent style delegation: a planner proposes
steps and suggested agents; the orchestrator then executes them.
"""
from __future__ import annotations

from typing import List

from .types import Plan, StepSpec


class SimplePlanner:
    """Rule-based planner producing a small DAG for common workflows."""

    def plan(self, *, objective: str, repo_path: str) -> Plan:
        # Always include a quick planning step with Claude when available
        plan_step = StepSpec(
            id="plan",
            description=f"Draft a concrete plan: {objective}",
            agent_id="claude-code",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["planning"],
        )
        # Implementation step via Codex
        impl_step = StepSpec(
            id="implement",
            description=f"Implement the agreed plan: {objective}",
            agent_id="codex",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["implementation"],
            depends_on=["plan"],
        )
        # Test and verify via Droid
        test_step = StepSpec(
            id="test",
            description="Run tests, summarize failures, and propose fixes if needed.",
            agent_id="droid",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["verification"],
            depends_on=["implement"],
        )

        steps: List[StepSpec] = [plan_step, impl_step, test_step]
        return Plan(objective=objective, steps=steps, notes="auto: simple 3-step plan")

