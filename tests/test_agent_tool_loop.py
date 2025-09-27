from __future__ import annotations

import json
from pathlib import Path

from atlas_main.agent import AtlasAgent


class _FakeClient:
    def __init__(self):
        self._round = 0

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


def test_agent_handles_multiple_tool_calls(tmp_path: Path):
    client = _FakeClient()
    agent = AtlasAgent(client)  # registers web_search, read_file, list_dir, write_file

    # Monkeypatch web_search tool to avoid network
    web = agent.tools._tools.get("web_search")
    assert web is not None
    web.run = lambda *, agent=None, query, max_results=3: "Search results: OK"  # type: ignore

    out = agent.respond("Test multi-tool")

    # Final answer should be returned
    assert "All done" in out
