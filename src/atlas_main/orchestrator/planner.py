"""Planner implementations: simple fallback and reasoning-driven planning."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .context import CodebaseContext
from .planning_session import PlanningSession, PlanningSessionError
from .types import BranchConfig, EnhancedPlan, EnhancedStepSpec, PlanningContext, StepSpec


class SimplePlanner:
    """Heuristic planner that returns a fixed three-step workflow."""

    def plan(self, *, objective: str, repo_path: str) -> EnhancedPlan:
        plan_step = EnhancedStepSpec(
            id="plan",
            description=f"Draft a concrete plan: {objective}",
            agent_id="claude-code",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["planning"],
        )

        impl_step = EnhancedStepSpec(
            id="implement",
            description=f"Implement the agreed plan: {objective}",
            agent_id="codex",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["implementation"],
            depends_on=["plan"],
        )

        test_step = EnhancedStepSpec(
            id="test",
            description="Run tests, summarize failures, and propose fixes if needed.",
            agent_id="droid",
            inputs={"repo_path": repo_path, "allow_dirty": True},
            tags=["verification"],
            depends_on=["implement"],
        )

        return EnhancedPlan(
            objective=objective,
            steps=[plan_step, impl_step, test_step],
            notes="auto: simple 3-step plan",
        )


class IntelligentPlanner:
    """Planner that collaborates with a reasoning agent to build execution plans."""

    def __init__(
        self,
        agent_factory: "AgentFactory",
        *,
        planner_agent_id: str = "planner-deepseek",
        max_rounds: int = 8,
        context_cls: Callable[[Path], CodebaseContext] = CodebaseContext,
        session_factory: Optional[Callable[..., PlanningSession]] = None,
        fallback: Optional[SimplePlanner] = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._planner_agent_id = planner_agent_id
        self._max_rounds = max_rounds
        self._context_cls = context_cls
        self._session_factory = session_factory or self._default_session_factory
        self._fallback = fallback or SimplePlanner()

    async def plan(
        self,
        *,
        objective: str,
        repo_path: str,
        available_agents: Optional[List[str]] = None,
    ) -> EnhancedPlan:
        codebase_context = await self._analyze_codebase(repo_path)
        agents = available_agents or self._default_available_agents()
        planning_context = PlanningContext(
            objective=objective,
            repo_path=repo_path,
            codebase_structure=codebase_context,
            available_agents=agents,
        )

        planner_agent = None
        try:
            planner_agent = await self._agent_factory.create(self._planner_agent_id)
            session = self._session_factory(
                planner=planner_agent,
                planning_context=planning_context,
                objective=objective,
                max_rounds=self._max_rounds,
            )
            async with session as active:
                plan = await active.run()
        except (PlanningSessionError, Exception):
            if planner_agent and hasattr(planner_agent, "close"):
                await planner_agent.close()
            plan = self._fallback_plan(objective, repo_path, codebase_context)
        else:
            if not plan.codebase_context:
                plan.codebase_context = codebase_context
        finally:
            pass

        return plan

    def should_plan(self, objective: str) -> bool:
        normalized = objective.lower()
        if len(objective) < 50 and all(keyword not in normalized for keyword in ("add", "create", "refactor", "migrate")):
            return False
        if any(keyword in normalized for keyword in ("oauth", "authentication", "parallel", "refactor", "integration")):
            return True
        return len(normalized.split()) >= 8

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        try:
            analyzer = self._context_cls(Path(repo_path))
            return await analyzer.analyze()
        except Exception:
            return {}

    def _default_available_agents(self) -> List[str]:
        try:
            descriptors = self._agent_factory.list_agents()
            return [agent_id for agent_id, cfg in descriptors.items() if cfg.get("type") != "reasoning"] or [
                "codex",
                "claude-code",
                "droid",
            ]
        except Exception:
            return ["codex", "claude-code", "droid"]

    def _default_session_factory(self, **kwargs) -> PlanningSession:
        return PlanningSession(**kwargs)

    def _fallback_plan(self, objective: str, repo_path: str, codebase_context: Dict[str, Any]) -> EnhancedPlan:
        plan = self._fallback.plan(objective=objective, repo_path=repo_path)
        if not plan.codebase_context:
            plan.codebase_context = codebase_context
        return plan
