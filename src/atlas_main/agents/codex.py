"""Adapter for GitHub Codex CLI."""
from __future__ import annotations

import asyncio
import json
from asyncio.subprocess import Process
from typing import Any, Dict, Mapping, Optional

from ..orchestrator.types import Artifact, StepResult, StepSpec
from .base import AgentAdapter, ConversationMessage, StreamingAgentSession
from .common import build_prompt, git_diff, git_status, resolve_repo_path, run_subprocess


class CodexAgentAdapter(AgentAdapter):
    """Run `codex exec` in JSON mode and convert artifacts to StepResult."""

    def __init__(
        self,
        *,
        agent_id: str,
        binary: str = "codex",
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.binary = binary
        self.timeout = timeout
        self.extra_args = extra_args or ["--full-auto"]
        self.env = dict(env or {})

    async def execute_step(self, *, step: StepSpec, shared_context: Mapping[str, Any]) -> StepResult:
        repo_path = resolve_repo_path(step)
        prompt = build_prompt(step, shared_context)
        command = [self.binary, "exec", "--json", "--cd", repo_path]
        command.extend(self.extra_args)
        command.append("-")

        try:
            code, stdout, stderr = await run_subprocess(
                command,
                cwd=repo_path,
                env=self.env,
                input_text=prompt,
                timeout=self.timeout,
            )
        except Exception as exc:
            return StepResult(
                step_id=step.id,
                status="failed",
                summary=f"Codex execution failed: {exc}",
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
                content = data.get("content")
                if isinstance(content, str):
                    text_chunks.append(content)
                elif isinstance(content, list):
                    text_chunks.extend(str(item) for item in content)

        summary = text_chunks[-1] if text_chunks else "Codex completed."
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
            "codex_event_count": len(events),
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

    async def open_session(
        self,
        *,
        task: str,
        step: StepSpec,
        shared_context: Mapping[str, Any],
    ) -> StreamingAgentSession:
        repo_path = resolve_repo_path(step)
        command = [
            self.binary,
            "exec",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--cd",
            repo_path,
        ]
        command.extend(self.extra_args)
        env = dict(self.env)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path,
            env=env,
        )
        greeting = ConversationMessage(
            role="system",
            content=task,
            metadata={"shared_context": shared_context},
        )
        session = _CodexStreamingSession(
            process=process,
            greeting=greeting,
            agent_id=self.agent_id,
        )
        await session.start()
        return session

    async def close(self) -> None:
        return None


class _CodexStreamingSession(StreamingAgentSession):
    """Streaming session for Codex CLI using JSONL protocol."""

    def __init__(
        self,
        *,
        process: Process,
        greeting: ConversationMessage,
        agent_id: str,
    ) -> None:
        self._process = process
        self._agent_id = agent_id
        self._greeting = greeting
        self._stdout = process.stdout
        self._stdin = process.stdin
        self._stderr = process.stderr
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.send(self._greeting)
        self._started = True

    async def send(self, message: ConversationMessage) -> None:
        payload = {
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata,
        }
        data = json.dumps(payload, ensure_ascii=False) + "\n"
        if not self._stdin:
            raise RuntimeError("Codex session stdin closed")
        self._stdin.write(data.encode("utf-8"))
        await self._stdin.drain()

    async def receive(self, *, timeout: Optional[float] = None) -> Optional[ConversationMessage]:
        if not self._stdout:
            return None
        try:
            line = await asyncio.wait_for(self._stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if not line:
            return None
        raw = line.decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ConversationMessage(
                role="assistant",
                content=raw,
                metadata={"agent_id": self._agent_id, "raw": True},
            )
        role = str(data.get("role") or "assistant")
        content = str(data.get("content") or "")
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("agent_id", self._agent_id)
        return ConversationMessage(role=role, content=content, metadata=metadata)

    async def close(self) -> None:
        if self._stdin:
            try:
                self._stdin.write_eof()
            except Exception:
                pass
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
