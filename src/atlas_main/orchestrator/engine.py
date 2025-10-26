"""Minimal orchestrator engine for delegating steps to coding agents."""
from __future__ import annotations

import asyncio
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional

from .types import Artifact, StepResult, StepSpec, TaskEvent, TaskResult, TaskSpec
from ..agents.common import git_status, git_diff

EventCallback = Callable[[TaskEvent], None]


class Orchestrator:
    """Sequential, event-driven task runner for the vertical slice."""

    def __init__(self, agent_factory: "AgentFactory", *, event_callback: Optional[EventCallback] = None) -> None:
        self._agent_factory = agent_factory
        self._event_callback = event_callback

    def _emit(self, event_type: str, task: TaskSpec, payload: Dict[str, Any]) -> None:
        if not self._event_callback:
            return
        self._event_callback(TaskEvent(type=event_type, task_id=task.id, payload=payload))

    async def run_task(self, task: TaskSpec) -> TaskResult:
        """Execute the provided task step-by-step."""
        self._emit("task.started", task, {"objective": task.objective})
        shared_context: Dict[str, Any] = dict(task.shared_context)
        step_results: list[StepResult] = []
        task_failed = False

        for step in task.steps:
            if task_failed:
                skipped = StepResult(
                    step_id=step.id,
                    status="skipped",
                    summary="Skipped due to previous failure",
                )
                step_results.append(skipped)
                self._emit("step.skipped", task, {"step_id": step.id, "reason": skipped.summary})
                continue

            repo_path = str(step.inputs.get("repo_path", "")) if isinstance(step.inputs, dict) else ""
            pre_git_status = ""
            allow_dirty = bool(step.inputs.get("allow_dirty")) if isinstance(step.inputs, dict) else False
            if repo_path:
                try:
                    pre_git_status = await git_status(repo_path)
                except Exception:
                    pre_git_status = ""
                if pre_git_status and "\n" in pre_git_status and not allow_dirty:
                    failed = StepResult(
                        step_id=step.id,
                        status="failed",
                        summary="Working tree is not clean. Commit or stash changes before delegating.",
                        logs=[pre_git_status],
                        metadata={"agent_id": step.agent_id, "pre_git_status": pre_git_status},
                    )
                    step_results.append(failed)
                    self._emit(
                        "step.completed",
                        task,
                        {
                            "step_id": step.id,
                            "status": failed.status,
                            "summary": failed.summary,
                            "artifacts": [],
                        },
                    )
                    task_failed = True
                    continue

            self._emit("step.started", task, {"step_id": step.id, "agent_id": step.agent_id, "description": step.description})
            agent = await self._agent_factory.create(step.agent_id)
            try:
                result = await _call_agent(agent, step, shared_context)
            except Exception as exc:  # pragma: no cover - defensive capture
                summary = f"Agent crashed: {exc}"
                result = StepResult(
                    step_id=step.id,
                    status="failed",
                    summary=summary,
                    logs=[traceback.format_exc()],
                )
            finally:
                try:
                    await _maybe_close(agent)
                except Exception:
                    pass

            step_results.append(result)
            if isinstance(result.metadata, dict):
                if pre_git_status:
                    result.metadata.setdefault("pre_git_status", pre_git_status)
            else:
                result.metadata = {"pre_git_status": pre_git_status} if pre_git_status else {}

            if repo_path:
                try:
                    post_status = await git_status(repo_path)
                except Exception:
                    post_status = ""
                if isinstance(result.metadata, dict):
                    if post_status:
                        result.metadata.setdefault("post_git_status", post_status)
                has_patch = any(artifact.kind == "patch" for artifact in result.artifacts)
                if result.succeeded and not has_patch:
                    try:
                        diff = await git_diff(repo_path)
                    except Exception:
                        diff = ""
                    if diff:
                        result.artifacts.append(Artifact(kind="patch", content=diff))
                        if isinstance(result.metadata, dict):
                            result.metadata["post_git_status"] = post_status or result.metadata.get("post_git_status", "")

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

            if result.succeeded:
                shared_updates = result.metadata.get("shared_updates")
                if isinstance(shared_updates, dict):
                    shared_context.update(shared_updates)
            else:
                task_failed = True

        status = "succeeded" if not task_failed else "failed"
        task_result = TaskResult(task_id=task.id, status=status, step_results=step_results)
        self._emit("task.completed", task, {"status": status})
        return task_result


async def _call_agent(agent: Any, step: StepSpec, shared_context: Dict[str, Any]) -> StepResult:
    """Invoke the agent and normalise responses."""
    execute = getattr(agent, "execute_step", None)
    if execute is None:
        raise RuntimeError(f"Agent {agent!r} does not implement execute_step")

    maybe_coro = execute(step=step, shared_context=shared_context)
    if asyncio.iscoroutine(maybe_coro) or isinstance(maybe_coro, Awaitable):
        result = await maybe_coro  # type: ignore[no-any-expr]
    else:
        result = maybe_coro

    if not isinstance(result, StepResult):
        raise RuntimeError(f"Agent returned invalid result: {result!r}")
    return result


async def _maybe_close(agent: Any) -> None:
    """Close agent if it exposes an async or sync close."""
    close = getattr(agent, "close", None)
    if close is None:
        return
    maybe_coro = close()
    if asyncio.iscoroutine(maybe_coro) or isinstance(maybe_coro, Awaitable):
        await maybe_coro  # type: ignore[no-any-expr]
