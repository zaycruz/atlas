"""Process-based agent adapter."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Mapping, Optional

from ..orchestrator.types import Artifact, StepResult, StepSpec
from .base import AgentAdapter, step_to_payload


class ProcessAgentAdapter(AgentAdapter):
    """Runs an external CLI agent as a subprocess."""

    def __init__(self, *, agent_id: str, command: list[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, timeout: float = 120.0) -> None:
        self.agent_id = agent_id
        self.command = command
        self.cwd = cwd
        self.env = env or {}
        self.timeout = timeout

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        payload = {
            "step": step_to_payload(step),
            "shared_context": dict(shared_context),
        }
        proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env={**os.environ, **self.env},
        )

        stdin = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(stdin), timeout=self.timeout)

        if proc.returncode != 0:
            summary = f"Process exited with code {proc.returncode}"
            logs = _build_logs(stdout_data, stderr_data)
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=summary,
                logs=logs,
                metadata={"agent_id": self.agent_id},
            )

        try:
            decoded = stdout_data.decode("utf-8").strip()
            data = json.loads(decoded or "{}")
        except Exception as exc:  # pragma: no cover - defensive path
            logs = _build_logs(stdout_data, stderr_data)
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=f"Invalid JSON from agent: {exc}",
                logs=logs,
                metadata={"agent_id": self.agent_id},
            )

        status = data.get("status", "failed")
        summary = data.get("summary") or ""
        artifacts = [
            Artifact(
                kind=item.get("kind", "unknown"),
                content=item.get("content"),
                path=item.get("path"),
                metadata=item.get("metadata") or {},
            )
            for item in data.get("artifacts", [])
            if isinstance(item, dict)
        ]
        logs = data.get("logs") or []
        if stderr_data:
            logs.append(stderr_data.decode("utf-8", errors="replace"))

        metadata = data.get("metadata") or {}
        metadata["agent_id"] = self.agent_id
        return StepResult(
            step_id=step.id,
            status=status,
            summary=summary or "Agent completed",
            artifacts=artifacts,
            logs=logs,
            metadata=metadata,
        )

    async def close(self) -> None:  # noqa: D401
        return None


def _build_logs(stdout: bytes, stderr: bytes) -> list[str]:
    logs: list[str] = []
    if stdout:
        logs.append(stdout.decode("utf-8", errors="replace"))
    if stderr:
        logs.append(stderr.decode("utf-8", errors="replace"))
    return logs
