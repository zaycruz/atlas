"""Tests for tool registry and web search integration."""
from __future__ import annotations

from datetime import datetime
from typing import List
import re

import pytest

import atlas_main.tools as tools_module
from atlas_main.agent import AtlasAgent
from atlas_main.memory import WorkingMemoryConfig
from atlas_main.tools import (
    ToolError,
    WebSearchTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    ShellCommandTool,
    CurrentTimeTool,
    AlpacaAccountTool,
    ToolRegistry,
)
from atlas_main.tools_memory import FetchFactTool
from atlas_main.memory_layers import SemanticMemory


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200, json_data=None) -> None:
        self.text = text
        self.status_code = status_code
        self._json = json_data

    def json(self):  # pragma: no cover - trivial accessor
        if self._json is None:
            raise ValueError("No JSON payload available")
        return self._json


class _FakeSession:
    """Simple FIFO response session for deterministic testing."""

    def __init__(self, responses: List[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.kwargs: list[dict] = []

    def get(self, url: str, *_, **kwargs) -> _FakeResponse:  # pragma: no cover - simple passthrough
        self.calls.append(url)
        self.kwargs.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeSession received more calls than prepared responses")
        return self._responses.pop(0)


def _build_search_response() -> str:
    return (
        "Title: Example search\n\n"
        "URL Source: https://duckduckgo.com/?q=MemGPT\n\n"
        "Markdown Content:\n"
        "1.   example.com [![Image](https://external-content.duckduckgo.com/ip3/example.com.ico)]"
        "(https://duckduckgo.com/?q=MemGPT+site:example.com \"Search domain example.com\") "
        "Example snippet explaining the latest MemGPT research.\n"
        "2.   ### Videos\n"
    )


def _build_content_response() -> str:
    return (
        "Title: Example Domain\n\n"
        "Markdown Content:\n"
        "MemGPT is the successor to Letta with persistent memory and advanced tools."
    )


def test_web_search_tool_run_aggregates_results_and_content() -> None:
    session = _FakeSession(
        [
            _FakeResponse(_build_search_response()),
            _FakeResponse(_build_content_response()),
        ]
    )
    tool = WebSearchTool(session=session)

    output = tool.run(query="MemGPT", max_results=1)

    assert "Search summary for 'MemGPT':" in output
    assert "Example snippet explaining the latest MemGPT research." in output
    assert "MemGPT is the successor to Letta" in output
    assert "example.com (https://example.com)" in output
    assert session.calls[0].startswith("https://r.jina.ai/https://duckduckgo.com/?q=MemGPT")


def test_web_search_tool_titles_only_with_meta(monkeypatch) -> None:
    session = _FakeSession(
        [
            _FakeResponse(_build_search_response()),
            _FakeResponse(_build_content_response()),
            _FakeResponse("<html><meta name=\"description\" content=\"Example meta\"></html>"),
        ]
    )
    tool = WebSearchTool(session=session)
    output = tool.run(query="MemGPT", max_results=1, titles_only=True, include_meta=True)

    assert "1. example.com" in output
    assert "Example meta" in output


def test_web_search_tool_domain_query_encoding() -> None:
    session = _FakeSession([
        _FakeResponse(_build_search_response()),
        _FakeResponse(_build_content_response()),
    ])
    tool = WebSearchTool(session=session)

    _ = tool.run(query="MemGPT", domain="example.com", max_results=1)

    assert "site%3Aexample.com" in session.calls[0]


def test_tool_registry_emits_function_specs() -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WebSearchTool(session=_FakeSession([_FakeResponse("", 500)])))

    specs = registry.render_function_specs()

    assert any(item.get("function", {}).get("name") == "read_file" for item in specs)
    assert all(item.get("type") == "function" for item in specs)
    assert any("capabilities" in item.get("function", {}) for item in specs)


def test_current_time_tool_outputs_isoformatted_values() -> None:
    tool = CurrentTimeTool()

    output = tool.run()

    lines = output.splitlines()
    assert lines[0] == "Current time:"

    parsed = {}
    for line in lines[1:]:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        parsed[key] = value

    assert "Local" in parsed
    assert "UTC" in parsed
    # fromisoformat validates the timestamps include timezone information
    datetime.fromisoformat(parsed["Local"])
    datetime.fromisoformat(parsed["UTC"])


def test_alpaca_account_tool_fetches_account_snapshot(monkeypatch) -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                "",
                json_data={
                    "status": "ACTIVE",
                    "account_number": "123456789",
                    "currency": "USD",
                    "cash": "10000",
                    "buying_power": "20000",
                    "portfolio_value": "12000",
                    "last_equity": "11800",
                    "trading_blocked": False,
                },
            ),
            _FakeResponse(
                "",
                json_data=[
                    {
                        "symbol": "AAPL",
                        "qty": "10",
                        "side": "long",
                        "avg_entry_price": "120.50",
                        "market_value": "1300",
                        "unrealized_pl": "95",
                        "unrealized_plpc": "0.073",
                        "current_price": "130",
                    }
                ],
            ),
            _FakeResponse(
                "",
                json_data=[
                    {
                        "id": "order-1",
                        "symbol": "AAPL",
                        "qty": "5",
                        "side": "buy",
                        "type": "limit",
                        "status": "accepted",
                        "limit_price": "125",
                        "filled_qty": "0",
                        "created_at": "2024-05-01T12:30:00Z",
                    },
                    {
                        "id": "order-2",
                        "symbol": "TSLA",
                        "qty": "2",
                        "side": "sell",
                        "type": "market",
                        "status": "filled",
                        "filled_qty": "2",
                        "created_at": "2024-05-01T10:00:00Z",
                    },
                ],
            ),
        ]
    )
    tool = AlpacaAccountTool(session=session, base_url="https://example.com")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

    output = tool.run(order_status="all", order_limit=2, order_direction="asc")

    assert "Alpaca Account Overview:" in output
    assert "Status: ACTIVE" in output
    assert "- AAPL: 10 long" in output
    assert "Recent Orders [status=all, direction=asc, limit=2] (2)" in output
    assert session.calls == [
        "https://example.com/v2/account",
        "https://example.com/v2/positions",
        "https://example.com/v2/orders",
    ]
    assert session.kwargs[2]["params"]["status"] == "all"


def test_alpaca_account_tool_loads_credentials_from_dotenv(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("APCA_API_KEY_ID=from_env\nAPCA_API_SECRET_KEY=from_secret\n")

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tools_module, "_ALPACA_DOTENV_LOADED", False, raising=False)

    session = _FakeSession(
        [
            _FakeResponse("", json_data={}),
            _FakeResponse("", json_data=[]),
            _FakeResponse("", json_data=[]),
        ]
    )
    tool = AlpacaAccountTool(session=session, base_url="https://example.com")

    tool.run()

    headers = session.kwargs[0]["headers"]
    assert headers["APCA-API-KEY-ID"] == "from_env"
    assert headers["APCA-API-SECRET-KEY"] == "from_secret"


def test_alpaca_account_tool_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    tool = AlpacaAccountTool(session=_FakeSession([]))

    with pytest.raises(ToolError):
        tool.run()


def test_alpaca_account_tool_reports_http_error(monkeypatch) -> None:
    session = _FakeSession([
        _FakeResponse("", status_code=401, json_data={"message": "Unauthorized"})
    ])
    tool = AlpacaAccountTool(session=session, base_url="https://example.com")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")

    with pytest.raises(ToolError) as exc:
        tool.run()

    assert "401" in str(exc.value)


def test_parse_markdown_results_skips_media_blocks() -> None:
    tool = WebSearchTool(session=_FakeSession([]))
    markdown = (
        "1.   example.com [link](https://duckduckgo.com/?q=MemGPT+site:example.com) Snippet here.\n"
        "2.   ### Videos\n"
    )

    results = tool._parse_markdown_results(markdown)

    assert results == [
        {
            "title": "example.com",
            "snippet": "Snippet here.",
            "url": "https://example.com",
        }
    ]


def test_fetch_content_without_url_raises() -> None:
    session = _FakeSession([])
    tool = WebSearchTool(session=session)

    with pytest.raises(ToolError):
        tool._fetch_content_simple("")


def test_read_file_tool_reads_content(tmp_path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("Alpha\nBeta\nGamma")
    tool = ReadFileTool()

    output = tool.run(path=str(target))

    assert "File:" in output
    assert "Alpha" in output
    assert "Beta" in output


def test_read_file_tool_respects_max_lines_and_pattern(tmp_path) -> None:
    target = tmp_path / "log.txt"
    target.write_text("error one\nwarning two\nerror three\ninfo four")
    tool = ReadFileTool()

    output = tool.run(
        path=str(target),
        max_lines=2,
        pattern="error",
        case_sensitive=False,
    )

    assert "<<error>> one" in output
    assert "error three" not in output
    assert "(truncated output)" in output


def test_write_file_tool_respects_overwrite(tmp_path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("old")
    tool = WriteFileTool()

    with pytest.raises(ToolError):
        tool.run(path=str(target), content="new")

    tool.run(path=str(target), content="new", overwrite=True)
    assert target.read_text() == "new"


def test_write_file_tool_atomic_preserve_and_diff(tmp_path) -> None:
    target = tmp_path / "config.ini"
    target.write_text("[section]\nvalue=1\n")
    tool = WriteFileTool()

    before = target.stat()

    output = tool.run(
        path=str(target),
        content="[section]\nvalue=2\n",
        overwrite=True,
        atomic=True,
        preserve_times=True,
        show_diff=True,
    )

    after = target.stat()
    assert target.read_text() == "[section]\nvalue=2\n"
    assert "Diff" in output
    assert "-value=1" in output
    assert "+value=2" in output
    assert after.st_mtime == pytest.approx(before.st_mtime, rel=0, abs=1e-6)


def test_list_dir_tool_hides_hidden_by_default(tmp_path) -> None:
    (tmp_path / "visible.txt").write_text("ok")
    (tmp_path / ".secret").write_text("hidden")
    tool = ListDirectoryTool()

    summary = tool.run(path=str(tmp_path))

    assert "visible.txt" in summary
    assert ".secret" not in summary

    summary_hidden = tool.run(path=str(tmp_path), show_hidden=True)
    assert ".secret" in summary_hidden


def test_list_dir_tool_recursive_human(tmp_path) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "main.py").write_text("print('hi')\n")
    deeper = nested / "pkg"
    deeper.mkdir()
    (deeper / "__init__.py").write_text("")

    tool = ListDirectoryTool()

    summary = tool.run(
        path=str(tmp_path),
        recursive=True,
        depth=2,
        human=True,
        max_entries=10,
    )

    assert "src/" in summary
    assert "  main.py" in summary
    assert "pkg/" in summary


def test_shell_command_tool_executes_echo(tmp_path) -> None:
    tool = ShellCommandTool()

    result = tool.run(command="echo atlas-shell", cwd=str(tmp_path))

    assert "atlas-shell" in result
    assert "exit_code: 0" in result


def test_shell_command_tool_interactive(tmp_path) -> None:
    tool = ShellCommandTool()

    result = tool.run(command="printf 'hello\\n'", cwd=str(tmp_path), interactive=True)

    assert "interactive" in result
    assert "hello" in result


def test_shell_command_tool_retries(tmp_path) -> None:
    tool = ShellCommandTool()

    result = tool.run(command="false", cwd=str(tmp_path), retries=2)

    assert "attempts: 3" in result
    assert "exit_code: 1" in result


def test_tool_policy_deny(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ATLAS_TOOL_POLICY", "deny")
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())
    with pytest.raises(ToolError):
        registry.run("list_dir", arguments={"path": str(tmp_path)})
    monkeypatch.delenv("ATLAS_TOOL_POLICY", raising=False)


def test_fetch_fact_tool_returns_full_fact(tmp_path) -> None:
    semantic_path = tmp_path / "semantic.json"
    semantic = SemanticMemory(semantic_path, embed_fn=None)
    fact = semantic.add_fact("Atlas remembers shipped features", tags=["release"])

    class _LM:
        def __init__(self, semantic_memory):
            self.semantic = semantic_memory

    class _Agent:
        def __init__(self, layered_memory):
            self.layered_memory = layered_memory

    agent = _Agent(_LM(semantic))
    tool = FetchFactTool()

    output = tool.run(agent=agent, id=fact["id"][:8])

    assert "Atlas remembers shipped features" in output
    assert fact["id"][:8] in output


def test_tool_policy_allow_noninteractive(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ATLAS_TOOL_POLICY", "ask")
    monkeypatch.setenv("ATLAS_TOOL_POLICY_NONINTERACTIVE", "allow")
    registry = ToolRegistry()
    registry.register(ListDirectoryTool())
    # Should not prompt in tests; defaults to allow
    output = registry.run("list_dir", arguments={"path": str(tmp_path)})
    assert "Directory" in output
    monkeypatch.delenv("ATLAS_TOOL_POLICY", raising=False)
    monkeypatch.delenv("ATLAS_TOOL_POLICY_NONINTERACTIVE", raising=False)


class _StubClient:
    def chat(self, **_):
        return {"message": {"content": "ok"}}

    def chat_stream(self, **_):
        return iter([
            {"content": "done", "tool_calls": []}
        ])

    def embed(self, model, text):
        # simple deterministic vector based on hash
        return [float(len(text) % 7), 0.0, 1.0]

    def close(self):
        pass


def test_memory_tools_return_relevant_results(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path / "memory"))
    agent = AtlasAgent(_StubClient())
    try:
        lm = agent.layered_memory
        lm.episodic.log("tailscale question", "tailscale answer", metadata={"tags": ["tailscale"]})
        fact = lm.semantic.add_fact("Tailscale is a mesh VPN service", tags=["tailscale"])
        other = lm.semantic.add_fact("WireGuard forms encrypted tunnels", tags=["vpn"])
        # graph functionality removed; no rebuild step
        output_ep = agent.tools.run("memory.search_episodes", agent=agent, arguments={"query": "tailscale", "limit": 3})
        assert "tailscale" in output_ep.lower()
        output_fact = agent.tools.run("memory.search_facts", agent=agent, arguments={"query": "tailscale", "limit": 3})
        assert "mesh vpn" in output_fact.lower()
        # graph exploration removed; skip explore_graph tool
    finally:
        agent.close()
    monkeypatch.delenv("ATLAS_MEMORY_DIR", raising=False)


def test_memory_save_fact_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path / "memory"))
    agent = AtlasAgent(_StubClient())
    try:
        arguments = {
            "text": "Tailscale status must be checked before reporting IPs",
            "tags": ["tailscale", "workflow"],
            "confidence": 0.9,
        }
        result = agent.tools.run("memory.save_fact", agent=agent, arguments=arguments)
        assert "Fact" in result
        follow_up = agent.tools.run(
            "memory.search_facts",
            agent=agent,
            arguments={"query": "tailscale status", "limit": 3},
        )
        assert "tailscale status" in follow_up.lower()
    finally:
        agent.close()
    monkeypatch.delenv("ATLAS_MEMORY_DIR", raising=False)


def test_memory_fact_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path / "memory"))
    fact_text = "Mac Mini 2 IP addresses come from `tailscale status`."
    tags = ["mac-mini-2", "tailscale", "ip"]

    agent = AtlasAgent(_StubClient())
    try:
        result = agent.tools.run(
            "memory.save_fact",
            agent=agent,
            arguments={"text": fact_text, "tags": tags, "confidence": 0.95},
        )
        assert "fact" in result.lower()
        match = re.search(r"\(([0-9a-fA-F-]{8,})\)", result)
        assert match, "expected fact id in save_fact output"
        fact_id = match.group(1)
    finally:
        agent.close()

    agent_reloaded = AtlasAgent(_StubClient())
    try:
        fetched = agent_reloaded.tools.run(
            "memory.fetch_fact",
            agent=agent_reloaded,
            arguments={"id": fact_id[:8]},
        )
        assert fact_text in fetched
    finally:
        agent_reloaded.close()

    monkeypatch.delenv("ATLAS_MEMORY_DIR", raising=False)


def test_working_memory_compaction_triggers_near_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("ATLAS_KV_CACHE", "0")
    monkeypatch.setenv("ATLAS_COMPACT_THRESHOLD", "80")
    monkeypatch.setenv("ATLAS_COMPACT_TARGET", "55")
    monkeypatch.setenv("ATLAS_COMPACT_MIN_PREFIX", "3")
    monkeypatch.setenv("ATLAS_CONTEXT_WINDOW", "1200")
    monkeypatch.setenv("ATLAS_CONTEXT_SAFETY", "0")

    config = WorkingMemoryConfig(max_turns=20, token_budget=120, enable_token_awareness=True)
    agent = AtlasAgent(_StubClient(), working_memory_config=config)
    try:
        payload = "chat turn " + ("ABCD" * 30)  # ~120 chars ≈ 30 tokens
        for _ in range(6):
            agent.working_memory.add_user(payload)

        stats = agent.working_memory.get_stats()
        assert stats["token_pct"] > 80

        compacted = agent._maybe_compact_working_memory(stats, None)
        assert compacted

        messages = agent.working_memory.to_messages()
        assert messages
        first = messages[0]
        assert first.get("summary") is True
        assert first.get("pinned") is True
    finally:
        agent.close()

    for name in [
        "ATLAS_MEMORY_DIR",
        "ATLAS_KV_CACHE",
        "ATLAS_COMPACT_THRESHOLD",
        "ATLAS_COMPACT_TARGET",
        "ATLAS_COMPACT_MIN_PREFIX",
        "ATLAS_CONTEXT_WINDOW",
        "ATLAS_CONTEXT_SAFETY",
    ]:
        monkeypatch.delenv(name, raising=False)
