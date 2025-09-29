"""Tests for tool registry and web search integration."""
from __future__ import annotations

from typing import List

import pytest

from atlas_main.tools import (
    ToolError,
    WebSearchTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    ShellCommandTool,
    ToolRegistry,
)


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _FakeSession:
    """Simple FIFO response session for deterministic testing."""

    def __init__(self, responses: List[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *_, **__) -> _FakeResponse:  # pragma: no cover - simple passthrough
        self.calls.append(url)
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
