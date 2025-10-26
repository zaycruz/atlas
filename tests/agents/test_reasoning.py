from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from atlas_main.agents.reasoning import ReasoningAgentAdapter
from atlas_main.orchestrator.types import StepSpec


class _FakeOllamaClient:
    def __init__(self, plan: Dict[str, Any], reasoning: str = "Reasoning about the plan.") -> None:
        self.plan = plan
        self.reasoning = reasoning
        self.calls: List[Dict[str, Any]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = f"<think>{self.reasoning}</think>\n{json.dumps(self.plan)}"
        return {
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": f"<think>{self.reasoning}</think>",
            },
            "context": {"dummy": True},
        }

    def close(self) -> None:  # pragma: no cover - not triggered in tests
        return None


class _FakeResponse:
    def __init__(self, body: Dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> Dict[str, Any]:
        return self._body


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.posts: List[Dict[str, Any]] = []
        self.closed = False

    async def post(self, url: str, **kwargs) -> _FakeResponse:
        payload = {"url": url, **kwargs}
        self.posts.append(payload)
        return self._response

    async def aclose(self) -> None:
        self.closed = True


def _build_planning_context() -> Dict[str, Any]:
    return {
        "objective": "Add OAuth integration",
        "codebase_context": {
            "structure": {"src": ["app.py", "auth.py"], "tests": ["test_auth.py"]},
            "languages": ["python"],
            "frameworks": ["fastapi"],
            "entry_points": ["src/app.py"],
            "test_framework": "pytest",
            "git_status": "clean",
            "recent_commits": ["abcd123 Fix bug", "efgh456 Improve tests"],
        },
        "constraints": {"max_rounds": 8},
        "available_agents": ["codex", "claude-code", "droid"],
    }


def _build_step() -> StepSpec:
    planning_context = _build_planning_context()
    return StepSpec(
        id="plan",
        description="Draft an execution plan for OAuth integration.",
        agent_id="planner-deepseek",
        inputs={"planning_context": planning_context},
    )


def test_execute_step_with_deepseek_via_ollama() -> None:
    plan = {
        "objective": "Add OAuth integration",
        "steps": [
            {"id": "frontend", "description": "Update UI", "agent_id": "codex", "depends_on": []},
            {"id": "backend", "description": "Implement OAuth backend", "agent_id": "droid", "depends_on": []},
        ],
        "parallel_groups": [],
        "reasoning": "Planned parallel work for frontend and backend.",
    }
    fake_client = _FakeOllamaClient(plan)
    step = _build_step()
    shared_context = {"planning_context": _build_planning_context()}

    async def _run() -> Any:
        adapter = ReasoningAgentAdapter(
            agent_id="planner-deepseek",
            model="deepseek-r1",
            provider="ollama",
            ollama_client=fake_client,
        )
        try:
            return await adapter.execute_step(step=step, shared_context=shared_context)
        finally:
            await adapter.close()

    result = asyncio.run(_run())

    assert result.succeeded
    assert result.metadata["provider"] == "ollama"
    assert "reasoning_trace" in result.metadata
    assert result.artifacts and result.artifacts[0].kind == "plan"

    plan_artifact = json.loads(result.artifacts[0].content or "{}")
    assert plan_artifact["objective"] == "Add OAuth integration"
    assert len(fake_client.calls) == 1


def test_execute_step_with_anthropic_client() -> None:
    plan = {
        "objective": "Add OAuth integration",
        "steps": [
            {"id": "plan", "description": "Design workflow", "agent_id": "codex", "depends_on": []}
        ],
        "parallel_groups": [],
        "reasoning": "Single step plan.",
    }
    response = _FakeResponse(
        {
            "content": [{"type": "text", "text": json.dumps(plan)}],
            "usage": {"input_tokens": 123, "output_tokens": 456},
        }
    )
    fake_client = _FakeAsyncClient(response)
    step = _build_step()
    shared_context = {"planning_context": _build_planning_context()}

    async def _run() -> Any:
        adapter = ReasoningAgentAdapter(
            agent_id="planner-sonnet",
            model="claude-sonnet-4.5",
            provider="anthropic",
            api_key="test-key",
            http_client=fake_client,
        )
        try:
            return await adapter.execute_step(step=step, shared_context=shared_context)
        finally:
            await adapter.close()

    result = asyncio.run(_run())

    assert result.succeeded
    assert result.metadata["provider"] == "anthropic"
    assert result.metadata["usage"] == {"input_tokens": 123, "output_tokens": 456}
    artifact_plan = json.loads(result.artifacts[0].content or "{}")
    assert artifact_plan["steps"][0]["id"] == "plan"
    assert fake_client.posts, "expected anthropic client to be invoked"


def test_open_session_produces_initial_reply() -> None:
    plan = {
        "objective": "Add OAuth integration",
        "steps": [],
        "parallel_groups": [],
        "reasoning": "Initial plan stub.",
    }
    fake_client = _FakeOllamaClient(plan)
    step = _build_step()
    shared_context = {"planning_context": _build_planning_context()}

    async def _run() -> Any:
        adapter = ReasoningAgentAdapter(
            agent_id="planner-deepseek",
            model="deepseek-r1",
            provider="ollama",
            ollama_client=fake_client,
        )
        try:
            session = await adapter.open_session(
                task="Collaborate on planning",
                step=step,
                shared_context=shared_context,
            )
            try:
                return await asyncio.wait_for(session.receive(), timeout=1.0)
            finally:
                await session.close()
        finally:
            await adapter.close()

    reply = asyncio.run(_run())

    assert reply is not None
    assert reply.role == "assistant"
    assert "reasoning_trace" in (reply.metadata or {})
