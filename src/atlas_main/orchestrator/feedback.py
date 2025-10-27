"""Failure feedback collection utilities for plan revision."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .types import StepResult


@dataclass
class FailureContext:
    step_id: str
    agent_id: str
    error_type: str
    error_message: str
    logs: List[str] = field(default_factory=list)
    git_diff: str = ""
    suggested_fixes: List[str] = field(default_factory=list)


class FeedbackCollector:
    """Extract actionable failure context from step results."""

    def capture_failure(self, result: StepResult) -> FailureContext:
        error_message = result.summary or "Unknown failure"
        error_type = self._infer_error_type(result)
        logs = list(result.logs or [])
        git_diff = self._extract_diff(result)
        agent_id = ""
        if isinstance(result.metadata, dict):
            agent_id = str(result.metadata.get("agent_id", ""))
        return FailureContext(
            step_id=result.step_id,
            agent_id=agent_id,
            error_type=error_type,
            error_message=error_message,
            logs=logs,
            git_diff=git_diff,
        )

    def format_for_planner(self, context: FailureContext) -> str:
        logs = "\n".join(context.logs[-10:]) if context.logs else "(no logs captured)"
        diff = context.git_diff[:2000] if context.git_diff else "(no diff available)"
        suggested = "\n".join(context.suggested_fixes) if context.suggested_fixes else "(none)"
        return (
            "STEP FAILURE REPORT:\n"
            f"Step: {context.step_id}\n"
            f"Agent: {context.agent_id or 'unknown'}\n"
            f"Error Type: {context.error_type}\n"
            f"Error Message: {context.error_message}\n\n"
            f"Logs (last 10 lines):\n{logs}\n\n"
            f"Git Diff (truncated):\n{diff}\n\n"
            f"Suggested Fixes:\n{suggested}\n\n"
            "Please revise the plan to address this failure."
        )

    @staticmethod
    def _extract_diff(result: StepResult) -> str:
        for artifact in result.artifacts:
            if artifact.kind == "patch" and artifact.content:
                return artifact.content
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        return str(metadata.get("post_git_status", ""))

    @staticmethod
    def _infer_error_type(result: StepResult) -> str:
        if isinstance(result.metadata, dict):
            error_type = result.metadata.get("error_type")
            if error_type:
                return str(error_type)
        if "timeout" in (result.summary or "").lower():
            return "timeout"
        if "merge" in (result.summary or "").lower():
            return "merge_failure"
        return "execution_error"

