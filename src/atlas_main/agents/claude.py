"""Adapter for Claude Code CLI."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from ..orchestrator.types import Artifact, StepResult, StepSpec
from .base import AgentAdapter
from .common import build_prompt, git_diff, git_status, resolve_repo_path, run_subprocess


class ClaudeAgentAdapter(AgentAdapter):
    """Runs Claude Code in non-interactive streaming mode."""

    def __init__(
        self,
        *,
        agent_id: str,
        binary: str = "claude",
        timeout: float = 600.0,
        permission_mode: str = "acceptEdits",
        extra_args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.binary = binary
        self.timeout = timeout
        self.permission_mode = permission_mode
        self.extra_args = extra_args or []
        self.env = dict(env or {})

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        repo_path = resolve_repo_path(step)
        prompt = build_prompt(step, shared_context)

        command = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--permission-mode",
            self.permission_mode,
        ]
        command.extend(self.extra_args)
        command.append(prompt)

        env = dict(self.env)
        env.setdefault("HOME", repo_path)
        env.setdefault("CLAUDE_FIRST_PARTY_TELEMETRY_OPTOUT", "1")

        try:
            code, stdout, stderr = await run_subprocess(
                command, cwd=repo_path, env=env, timeout=self.timeout
            )
        except Exception as exc:
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=f"Claude execution failed: {exc}",
                logs=[str(exc)],
                metadata={"agent_id": self.agent_id},
            )

        events: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                text_chunks.append(line)
            else:
                events.append(data)
                content = data.get("text") or data.get("content")
                if isinstance(content, str):
                    text_chunks.append(content)

        summary = text_chunks[-1] if text_chunks else "Claude completed."
        if code != 0:
            logs = text_chunks
            if stderr:
                logs.append(stderr)
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=summary,
                logs=logs,
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
            "claude_event_count": len(events),
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
