from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from atlas_main.agent import AtlasAgent
from atlas_main.tools import Tool


class _FakeClient:
    def __init__(self):
        self._round = 0

    def chat(
        self,
        *,
        model,
        messages,
        stream=False,
        options=None,
        tools=None,
        context=None,
        keep_alive=None,
        request_timeout=None,
    ):
        return {"message": {"content": "• summary"}}

    def chat_stream(self, *, model, messages, tools=None, options=None, keep_alive=None, request_timeout=None):  # noqa: D401
        """Yield fake streaming chunks; advance round each call."""
        self._round += 1
        if self._round == 1:
            # Native function call to web_search
            yield {
                "content": "(thinking)",
                "tool_calls": [
                    {"function": {"name": "web_search", "arguments": {"query": "pytest"}}}
                ],
            }
        elif self._round == 2:
            # Inline markup tool request for read_file
            yield {
                "content": "<<tool:read_file|%s>>" % json.dumps({"path": str(Path(__file__).resolve())}),
                "tool_calls": [],
            }
        else:
            # Final answer without tool calls
            yield {"content": "All done.", "tool_calls": []}

    def close(self):
        pass

    def embed(self, model, text):  # pragma: no cover - simple stub
        return [0.0, 0.0, 0.0]


def test_agent_handles_multiple_tool_calls(tmp_path: Path):
    client = _FakeClient()
    os.environ["ATLAS_MEMORY_DIR"] = str(tmp_path)
    agent = AtlasAgent(client)

    # Monkeypatch web_search tool to avoid network
    web = agent.tools._tools.get("web_search")
    assert web is not None
    web.run = lambda *, agent=None, query, max_results=3: "Search results: OK"  # type: ignore

    out = agent.respond("Test multi-tool")

    # Final answer should be returned
    assert "All done" in out


class _FakeConcurrentClient:
    def __init__(self) -> None:
        self._round = 0

    def chat(
        self,
        *,
        model,
        messages,
        stream=False,
        options=None,
        tools=None,
        context=None,
        keep_alive=None,
        request_timeout=None,
    ):
        return {"message": {"content": "• summary"}}

    def chat_stream(self, *, model, messages, tools=None, options=None, keep_alive=None, request_timeout=None):  # noqa: D401
        self._round += 1
        if self._round == 1:
            yield {
                "content": "(thinking)",
                "tool_calls": [
                    {"function": {"name": "barrier_tool_a", "arguments": {}}},
                    {"function": {"name": "barrier_tool_b", "arguments": {}}},
                ],
            }
        else:
            yield {"content": "All done.", "tool_calls": []}

    def close(self) -> None:
        pass

    def embed(self, model, text):  # pragma: no cover - simple stub
        return [0.0, 0.0, 0.0]


class _BarrierToolA(Tool):
    name = "barrier_tool_a"
    description = ""
    args_hint = ""

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def run(self, *, agent=None) -> str:  # type: ignore[override]
        self._barrier.wait()
        return "A"


class _BarrierToolB(Tool):
    name = "barrier_tool_b"
    description = ""
    args_hint = ""

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier

    def run(self, *, agent=None) -> str:  # type: ignore[override]
        self._barrier.wait()
        return "B"


def test_agent_runs_tool_calls_concurrently(tmp_path: Path):
    barrier = threading.Barrier(2, timeout=2)
    client = _FakeConcurrentClient()
    os.environ["ATLAS_MEMORY_DIR"] = str(tmp_path)
    agent = AtlasAgent(client)

    agent.tools.register(_BarrierToolA(barrier))
    agent.tools.register(_BarrierToolB(barrier))

    out = agent.respond("Trigger concurrent tools")

    assert "All done" in out
