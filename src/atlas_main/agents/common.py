"""Shared helpers for CLI-based agent adapters."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ..orchestrator.types import StepSpec


async def run_subprocess(
    command: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 300.0,
) -> Tuple[int, str, str]:
    """Execute a subprocess and capture its output."""
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=dict(os.environ, **(env or {})),
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(None if input_text is None else input_text.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode, stdout_bytes.decode("utf-8", errors="replace"), stderr_bytes.decode(
        "utf-8", errors="replace"
    )


async def git_status(repo_path: str) -> str:
    """Return `git status -sb` for the repo, or empty string if not available."""
    command = ["git", "status", "-sb"]
    code, stdout, _ = await run_subprocess(command, cwd=repo_path, timeout=30.0)
    return stdout.strip() if code == 0 else ""


async def git_diff(repo_path: str) -> str:
    """Return git diff patch for repo, or empty string if none."""
    command = ["git", "diff", "--patch"]
    code, stdout, _ = await run_subprocess(command, cwd=repo_path, timeout=60.0)
    return stdout if code == 0 and stdout.strip() else ""


def build_prompt(step: StepSpec, shared_context: Mapping[str, Any]) -> str:
    """Construct a plain-text prompt combining step description and context."""
    sections: list[str] = [step.description.strip()]
    if shared_context:
        serialized = json.dumps(shared_context, indent=2, ensure_ascii=False)
        sections.append("Shared context:\n" + serialized)
    if step.inputs:
        serialized_inputs = json.dumps(step.inputs, indent=2, ensure_ascii=False)
        sections.append("Inputs:\n" + serialized_inputs)
    return "\n\n".join(section for section in sections if section.strip())


def resolve_repo_path(step: StepSpec) -> str:
    """Return repo path from step inputs or current working directory."""
    repo = step.inputs.get("repo_path") if isinstance(step.inputs, dict) else None
    if repo:
        return str(Path(repo).expanduser())
    return os.getcwd()
