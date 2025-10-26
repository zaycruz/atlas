"""Adapter for Factory Droid CLI."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from ..orchestrator.types import Artifact, StepResult, StepSpec
from .base import AgentAdapter
from .common import build_prompt, git_diff, git_status, resolve_repo_path, run_subprocess


class DroidAgentAdapter(AgentAdapter):
    """Runs `droid exec` and maps JSON output into a StepResult."""

    def __init__(
        self,
        *,
        agent_id: str,
        binary: str = "droid",
        timeout: float = 900.0,
        auto_level: str = "medium",
        env: Mapping[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.binary = binary
        self.timeout = timeout
        self.auto_level = auto_level
        self.env = dict(env or {})
        self.extra_args = extra_args or []

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        repo_path = resolve_repo_path(step)
        prompt = build_prompt(step, shared_context)

        command = [
            self.binary,
            "exec",
            "--output-format",
            "json",
            "--cwd",
            repo_path,
        ]
        if self.auto_level:
            command.extend(["--auto", self.auto_level])
        command.extend(self.extra_args)
        command.append(prompt)

        try:
            code, stdout, stderr = await run_subprocess(
                command,
                cwd=repo_path,
                env=self.env,
                timeout=self.timeout,
            )
        except Exception as exc:
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=f"Droid execution failed: {exc}",
                logs=[str(exc)],
                metadata={"agent_id": self.agent_id},
            )

        text_chunks: list[str] = []
        summary = "Droid completed."
        if stdout.strip():
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                text_chunks.append(stdout.strip())
            else:
                summary = data.get("summary") or summary
                output = data.get("output") or data.get("messages")
                if isinstance(output, str):
                    text_chunks.append(output)
                elif isinstance(output, list):
                    text_chunks.extend(str(item) for item in output)
                errors = data.get("errors")
                if errors:
                    if isinstance(errors, list):
                        text_chunks.extend(str(item) for item in errors)
                    else:
                        text_chunks.append(str(errors))

        if code != 0:
            if stderr:
                text_chunks.append(stderr)
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=summary,
                logs=text_chunks,
                metadata={"agent_id": self.agent_id, "returncode": code},
            )

        diff = await git_diff(repo_path)
        status = await git_status(repo_path)
        artifacts: list[Artifact] = []
        if diff:
            artifacts.append(Artifact(kind="patch", content=diff))

        metadata: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "git_status": status,
        }
        if stderr:
            text_chunks.append(stderr)

        return StepResult(
            step_id=step.id,
            status="succeeded",
            summary=summary,
            artifacts=artifacts,
            logs=text_chunks[-20:],
            metadata=metadata,
        )
