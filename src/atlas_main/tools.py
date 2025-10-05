"""Tool registry and web search implementation for Atlas Lite."""
from __future__ import annotations

import asyncio
import difflib
import os
import pty
import re
import select
import subprocess
import tempfile
import threading
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .tools_browser import BrowserSession, DEFAULT_BUDGET, DEFAULT_TOPN


USER_AGENT = "AtlasLite/1.0 (+https://github.com)"
DEFAULT_FILE_CHUNK = 6000
MAX_SHELL_OUTPUT = 4000


class ToolError(RuntimeError):
    """Raised when a tool cannot complete its task."""


def _resolve_user_path(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise ToolError("path is required")
    candidate = Path(raw_path).expanduser()
    try:
        return candidate.resolve(strict=False)
    except Exception as exc:  # pragma: no cover - platform specific errors
        raise ToolError(f"unable to resolve path: {exc}") from exc


@dataclass
class ToolDescription:
    name: str
    description: str
    args_hint: str


class Tool:
    """Base class for tools usable by the agent."""

    name: str = ""
    description: str = ""
    args_hint: str = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def describe(self) -> ToolDescription:
        return ToolDescription(self.name, self.description, self.args_hint)

    def run(self, *, agent=None, **kwargs: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def function_parameters(self) -> Dict[str, Any]:
        return self.parameters_schema


class ToolRegistry:
    """Maintains available tools and executes them safely."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def render_instructions(self) -> str:
        lines: List[str] = []
        for tool in self._tools.values():
            desc = tool.describe()
            lines.append(f"- {desc.name}: {desc.description}")
            if desc.args_hint:
                lines.append(f"  args: {desc.args_hint}")
        return "\n".join(lines) if lines else "(no tools available)"

    def run(self, name: str, *, agent=None, arguments: Optional[Dict[str, Any]] = None) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise ToolError(f"Unknown tool: {name}")
        args = arguments or {}
        try:
            return tool.run(agent=agent, **args)
        except ToolError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ToolError(f"Tool '{name}' failed: {exc}") from exc

    def list_names(self) -> Iterable[str]:
        return list(self._tools.keys())

    def render_function_specs(self) -> List[Dict[str, Any]]:
        """Render tool metadata for function-calling capable models."""
        specs: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.function_parameters(),
                    },
                }
            )
        return specs


class WebSearchTool(Tool):
    """Web search with Crawl4AI-powered content extraction."""

    name = "web_search"
    description = "Search the web and extract clean content from results."
    args_hint = (
        "query (str, required); max_results (int, optional, default 3); domain (str, optional); "
        "titles_only (bool, optional); include_meta (bool, optional)"
    )

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()
        self._sync_crawler = None
        self._async_crawler_cls = None
        self._async_cache_mode = None
        self._init_crawler()

    def _init_crawler(self) -> None:
        """Detect available Crawl4AI backends."""
        try:
            from crawl4ai import WebCrawler  # type: ignore
        except Exception:
            WebCrawler = None  # type: ignore

        if WebCrawler:
            try:
                self._sync_crawler = WebCrawler(verbose=False)
            except Exception:
                self._sync_crawler = None

        if self._sync_crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler, CacheMode  # type: ignore
            except Exception:
                self._async_crawler_cls = None
                self._async_cache_mode = None
            else:
                self._async_crawler_cls = AsyncWebCrawler
                self._async_cache_mode = CacheMode

    def run(
        self,
        *,
        agent=None,
        query: str,
        max_results: int = 3,
        domain: Optional[str] = None,
        titles_only: bool = False,
        include_meta: bool = False,
    ) -> str:  # type: ignore[override]
        query = query.strip()
        if not query:
            raise ToolError("web_search requires a non-empty 'query'")

        max_results = max(1, min(int(max_results or 3), 5))

        if domain:
            domain = domain.strip()
            if domain and f"site:{domain}" not in query:
                query = f"{query} site:{domain}"

        # Get search results from DuckDuckGo
        search_results = self._search_duckduckgo(query)
        if not search_results:
            return f"No results found for '{query}'."

        sources: List[Dict[str, str]] = []
        for result in search_results[:max_results]:
            snippet = (result.get("snippet") or "").strip()
            title = (result.get("title") or result.get("text") or "Result").strip()
            url = (result.get("url") or result.get("first_url") or "").strip()

            content = ""
            meta_description = ""
            if url:
                try:
                    content = self._crawl_content(url)
                except (ToolError, Exception) as exc:
                    try:
                        content = self._fetch_content_simple(url)
                    except Exception:
                        content = f"(content unavailable: {type(exc).__name__})"

            content = (content or "").strip()
            if content.startswith("(content unavailable") and snippet:
                content = ""

            if include_meta and url:
                try:
                    meta_description = self._fetch_meta_description(url)
                except Exception:
                    meta_description = ""

            sources.append(
                {
                    "title": title or "Result",
                    "url": url,
                    "snippet": snippet,
                    "content": content,
                    "meta": meta_description.strip(),
                }
            )

        header = f"Search summary for '{query}':"
        if titles_only:
            lines = [header]
            for idx, source in enumerate(sources[:max_results]):
                title = source.get("title") or "Result"
                url = source.get("url") or ""
                meta = source.get("meta") or ""
                line = f"{idx + 1}. {title}"
                if url:
                    line += f" ({url})"
                if include_meta and meta:
                    line += f" — {meta}"
                lines.append(line)
            return "\n".join(lines)

        summary = self._compose_takeaways(sources)
        details = self._compose_details(sources)

        parts = [header]
        if summary:
            parts.append(summary)
        if details:
            parts.append("")
            parts.append(details)
        return "\n".join([part for part in parts if part]).strip()

    def _crawl_content(self, url: str) -> str:
        """Extract content using Crawl4AI."""
        if not (self._sync_crawler or self._async_crawler_cls):
            raise ToolError("Crawl4AI not available")
        
        if self._sync_crawler is not None:
            return self._crawl_sync(url)
        return self._crawl_async(url)

    def _crawl_sync(self, url: str) -> str:
        try:
            result = self._sync_crawler.run(url=url, verbose=False, user_agent=USER_AGENT)
        except Exception as exc:  # pragma: no cover - library errors
            raise ToolError(f"Crawl4AI sync crawler failed: {exc}") from exc

        return self._render_crawl_result(result)

    def _crawl_async(self, url: str) -> str:
        if self._async_crawler_cls is None or self._async_cache_mode is None:
            raise ToolError("Crawl4AI async crawler unavailable")

        async def crawl_once() -> Any:
            async with self._async_crawler_cls(verbose=False, warning=False) as crawler:
                return await crawler.arun(
                    url=url,
                    cache_mode=self._async_cache_mode.BYPASS,
                    verbose=False,
                    user_agent=USER_AGENT,
                )

        result = self._run_coroutine(crawl_once())
        return self._render_crawl_result(result)

    @staticmethod
    def _run_coroutine(coro: Any) -> Any:
        if not asyncio.iscoroutine(coro):  # pragma: no cover - defensive
            return coro

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: list[Any] = []
        error: list[BaseException] = []

        def runner() -> None:
            try:
                result.append(asyncio.run(coro))
            except BaseException as exc:  # pragma: no cover - thread propagation
                error.append(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if error:
            raise error[0]
        return result[0] if result else None

    def _render_crawl_result(self, result: Any) -> str:
        if result is None:
            raise ToolError("Crawl4AI returned no data")

        success = getattr(result, "success", False)
        if not success:
            message = getattr(result, "error_message", "unknown error")
            raise ToolError(f"Crawl4AI failed: {message}")

        candidates: List[str] = []

        markdown_v2 = getattr(result, "markdown_v2", None)
        if markdown_v2:
            for attr in ("fit_markdown", "markdown_with_citations", "raw_markdown"):
                value = getattr(markdown_v2, attr, None)
                if value:
                    candidates.append(str(value))

        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str):
            candidates.append(markdown)
        elif markdown:
            for attr in ("fit_markdown", "markdown_with_citations", "raw_markdown"):
                value = getattr(markdown, attr, None)
                if value:
                    candidates.append(str(value))

        for attr in ("extracted_content", "cleaned_html", "fit_html", "html"):
            value = getattr(result, attr, None)
            if value:
                candidates.append(str(value))

        for candidate in candidates:
            text = self._normalize_content(candidate)
            if text:
                return text[:1500]

        raise ToolError("Crawl4AI returned empty content")

    @staticmethod
    def _normalize_content(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if "<" in stripped and ">" in stripped:
            stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped)
        return stripped

    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search using DuckDuckGo via jina.ai."""
        encoded = requests.utils.quote(query, safe="")
        url = f"https://r.jina.ai/https://duckduckgo.com/?q={encoded}&ia=web"
        try:
            response = self._session.get(url, timeout=8, headers={"User-Agent": USER_AGENT})
            if response.status_code >= 400:
                raise ToolError(f"DuckDuckGo returned status {response.status_code}")
            text = response.text.strip()
            if not text:
                raise ToolError("DuckDuckGo returned an empty response")
            results = self._parse_markdown_results(text)
            if not results:
                raise ToolError("No parsable results returned")
            return results
        except Exception as e:
            raise ToolError(f"DuckDuckGo search failed: {e}")

    def _parse_markdown_results(self, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        pattern = re.compile(r"^(\d+)\.\s+(.*)")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = pattern.match(stripped)
            if not match:
                continue
            body = match.group(2).strip()
            if not body:
                continue
            if "###" in body:
                # Skip aggregated media sections
                continue
            clean_body = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body)
            url = self._extract_url(body)
            if not url:
                continue
            clean_body = re.sub(r"\[[^\]]*\]\([^)]+\)", "", clean_body)
            clean_body = re.sub(r"\s+", " ", clean_body).strip()
            clean_body = clean_body.replace("[]", "").strip()
            if not clean_body:
                continue
            if " " in clean_body:
                title, snippet = clean_body.split(" ", 1)
            else:
                title, snippet = clean_body, ""
            if not snippet or title.lower().startswith("more"):
                continue
            snippet = re.sub(r"\(https://duckduckgo\.com[^)]*\)\s*", "", snippet)
            snippet = snippet.strip()
            if not snippet:
                continue
            results.append({
                "title": title.strip(),
                "snippet": snippet.strip(),
                "url": url,
            })
        return results

    def _extract_url(self, body: str) -> str:
        for candidate in re.findall(r"https?://[^)\s]+", body):
            if "duckduckgo.com" in candidate or "external-content.duckduckgo.com" in candidate:
                continue
            return candidate
        site_match = re.search(r"site:([A-Za-z0-9.-]+)", body)
        if site_match:
            domain = site_match.group(1)
            if not domain.startswith("http"):
                return f"https://{domain}"
            return domain
        domain_match = re.match(r"([A-Za-z0-9.-]+)", body)
        if domain_match:
            domain = domain_match.group(1)
            if "." in domain:
                return f"https://{domain}"
        return ""

    def _fetch_content_simple(self, url: str) -> str:
        """Simple fallback content fetching."""
        if not url:
            raise ToolError("missing URL for content fetch")
        
        normalized = url if url.startswith(("http://", "https://")) else f"https://{url.lstrip('/')}"
        
        try:
            # Try jina.ai first
            go_url = f"https://r.jina.ai/{normalized}"
            response = self._session.get(go_url, timeout=5, headers={"User-Agent": USER_AGENT})
            if response.status_code < 400:
                text = response.text.strip()
                if text:
                    cleaned = self._normalize_content(text)
                    return cleaned[:1200]
        except Exception:
            pass
            
        try:
            # Fallback to direct fetch with simple HTML cleaning
            response = self._session.get(normalized, timeout=3, headers={"User-Agent": USER_AGENT})
            if response.status_code >= 400:
                raise ToolError(f"HTTP {response.status_code}")
            
            text = self._normalize_content(response.text)
            return text[:800]
        except Exception as e:
            raise ToolError(f"Content fetch failed: {e}")

    def _compose_takeaways(self, sources: List[Dict[str, str]]) -> str:
        sentences: List[str] = []
        for source in sources:
            snippet = source.get("snippet") or ""
            content = source.get("content") or ""
            for piece in (snippet, content):
                if not piece:
                    continue
                for sentence in self._extract_sentences(piece, limit=2):
                    sentences.append(sentence)
                    if len(sentences) >= 4:
                        break
                if len(sentences) >= 4:
                    break
            if len(sentences) >= 4:
                break
        if not sentences:
            return "No substantive information was retrieved."
        combined = " ".join(sentences)
        return combined[:600]

    def _compose_details(self, sources: List[Dict[str, str]]) -> str:
        detail_lines: List[str] = []
        for source in sources:
            url = source.get("url") or ""
            title = source.get("title") or "Result"
            snippet = source.get("snippet") or ""
            content = source.get("content") or ""
            meta = source.get("meta") or ""
            sentences: List[str] = []
            for piece in (snippet, content):
                if not piece:
                    continue
                remaining = max(0, 2 - len(sentences))
                if remaining == 0:
                    break
                sentences.extend(self._extract_sentences(piece, limit=remaining))
                if len(sentences) >= 2:
                    break
            if not sentences:
                continue
            detail = " ".join(sentences)
            detail = self._ensure_sentence(detail)
            if url:
                line = f"From {title} ({url}), {detail}"
            else:
                line = f"From {title}, {detail}"
            if meta:
                line += f" (meta: {meta})"
            detail_lines.append(line)
        return "\n\n".join(detail_lines)

    def _fetch_meta_description(self, url: str) -> str:
        try:
            response = self._session.get(url, timeout=4, headers={"User-Agent": USER_AGENT})
            if response.status_code >= 400:
                return ""
            html = response.text
        except Exception:
            return ""
        match = re.search(r'<meta[^>]+name=["\"]description["\"][^>]+content=["\"]([^"\"]+)["\"]', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r'<meta[^>]+content=["\"]([^"\"]+)["\"][^>]+name=["\"]description["\"]', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_sentences(text: str, *, limit: int = 2) -> List[str]:
        if limit <= 0:
            return []
        if not text:
            return []
        cleaned = text.strip()
        if not cleaned:
            return []
        segments = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences: List[str] = []
        for segment in segments:
            candidate = segment.strip()
            if not candidate:
                continue
            sentences.append(candidate)
            if len(sentences) >= limit:
                break
        return sentences

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        trimmed = text.strip()
        if not trimmed:
            return ""
        if trimmed[-1] not in ".!?":
            return trimmed + "."
        return trimmed


class ReadFileTool(Tool):
    """Read text from files on disk."""

    name = "read_file"
    description = "Read a UTF-8 text file with optional offset/length and filtering."  # noqa: RUF015
    args_hint = (
        "path (str, required); offset (int, optional); length (int, optional); "
        "max_lines (int, optional); pattern (str, optional); case_sensitive (bool, optional)"
    )

    def __init__(self, *, max_chars: int = DEFAULT_FILE_CHUNK) -> None:
        self._max_chars = max_chars

    def run(
        self,
        *,
        agent=None,
        path: str,
        offset: int = 0,
        length: Optional[int] = None,
        max_lines: Optional[int] = None,
        pattern: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> str:  # type: ignore[override]
        resolved = _resolve_user_path(path)
        if not resolved.is_file():
            raise ToolError(f"not a file: {resolved}")

        safe_offset = max(int(offset or 0), 0)
        chunk_limit = None if length is None else max(int(length), 0)
        max_read = self._max_chars if chunk_limit is None else chunk_limit

        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(safe_offset)
                data = handle.read(max_read + 1 if chunk_limit is None else max_read)
        except Exception as exc:
            raise ToolError(f"failed to read {resolved}: {exc}") from exc

        truncated = False
        if chunk_limit is None and len(data) > self._max_chars:
            data = data[: self._max_chars]
            truncated = True

        filtered = data

        if max_lines is not None:
            safe_lines = max(int(max_lines), 0)
            if safe_lines and safe_lines < len(filtered.splitlines()):
                filtered = "\n".join(filtered.splitlines()[:safe_lines])
                truncated = True

        highlight_note = ""
        if pattern:
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pattern, flags)
            except re.error as exc:
                raise ToolError(f"invalid regex pattern: {exc}") from exc

            def _repl(match: re.Match[str]) -> str:
                return f"<<{match.group(0)}>>"

            highlighted = regex.sub(_repl, filtered)
            if highlighted != filtered:
                highlight_note = f"(pattern highlighted with <<>> markers: {pattern})"
            filtered = highlighted

        lines = [f"File: {resolved}"]
        lines.append("")
        lines.append(filtered)
        if highlight_note:
            lines.append(f"\n{highlight_note}")
        if truncated:
            lines.append(f"\n(truncated output)")
        return "\n".join(lines)


class WriteFileTool(Tool):
    """Write text content to disk."""

    name = "write_file"
    description = "Create or update a text file; requires explicit overwrite/append."  # noqa: RUF015
    args_hint = (
        "path (str, required); content (str, required); overwrite (bool, optional); "
        "append (bool, optional); create_dirs (bool, optional); atomic (bool, optional); "
        "preserve_times (bool, optional); show_diff (bool, optional)"
    )

    def run(
        self,
        *,
        agent=None,
        path: str,
        content: str,
        overwrite: bool = False,
        append: bool = False,
        create_dirs: bool = False,
        atomic: bool = False,
        preserve_times: bool = False,
        show_diff: bool = False,
    ) -> str:  # type: ignore[override]
        if append and overwrite:
            raise ToolError("set either append or overwrite, not both")
        if atomic and append:
            raise ToolError("atomic writes are incompatible with append mode")

        resolved = _resolve_user_path(path)
        parent = resolved.parent
        if not parent.exists():
            if create_dirs:
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    raise ToolError(f"failed to create parent directories: {exc}") from exc
            else:
                raise ToolError(f"parent directory does not exist: {parent}")

        if resolved.exists() and not resolved.is_file():
            raise ToolError(f"path exists and is not a file: {resolved}")

        if resolved.exists() and not (overwrite or append):
            raise ToolError("file exists; set overwrite=True or append=True")

        old_stat = None
        old_content = ""
        if resolved.exists():
            try:
                old_stat = resolved.stat()
                old_content = resolved.read_text(encoding="utf-8")
            except Exception:
                old_content = ""

        final_content = content
        if append:
            final_content = (old_content or "") + content
            try:
                with resolved.open("a", encoding="utf-8") as handle:
                    handle.write(content)
            except Exception as exc:
                raise ToolError(f"failed to append {resolved}: {exc}") from exc
        else:
            if atomic:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        delete=False,
                        dir=str(parent),
                    ) as tmp:
                        tmp.write(content)
                        tmp_path = Path(tmp.name)
                    if tmp_path is not None:
                        if old_stat is not None:
                            try:
                                tmp_path.chmod(old_stat.st_mode)
                            except Exception:
                                pass
                        else:
                            try:
                                tmp_path.chmod(0o644)
                            except Exception:
                                pass
                        os.replace(tmp_path, resolved)
                except Exception as exc:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    raise ToolError(f"failed to write {resolved} atomically: {exc}") from exc
            else:
                try:
                    with resolved.open("w", encoding="utf-8") as handle:
                        handle.write(content)
                except Exception as exc:
                    raise ToolError(f"failed to write {resolved}: {exc}") from exc

        if preserve_times and old_stat and resolved.exists():
            try:
                os.utime(resolved, (old_stat.st_atime, old_stat.st_mtime))
            except Exception:
                pass

        diff_text = ""
        if show_diff:
            before = old_content.splitlines()
            after = final_content.splitlines()
            diff_lines = list(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile="before",
                    tofile="after",
                    lineterm="",
                )
            )
            if diff_lines:
                diff_text = "\n".join(diff_lines)

        action = "Appended" if append else "Wrote"
        result_lines = [f"{action} {len(content)} characters to {resolved}"]
        if preserve_times and old_stat:
            result_lines.append("Preserved original timestamps")
        if diff_text:
            result_lines.append("\nDiff:\n" + diff_text)
        return "\n".join(result_lines)


class ShellCommandTool(Tool):
    """Execute shell commands on the local system."""

    name = "shell_command"
    description = "Run a shell command and return stdout/stderr."  # noqa: RUF015
    args_hint = (
        "command (str, required); timeout (int, optional); cwd (str, optional); "
        "sudo (bool, optional); interactive (bool, optional); retries (int, optional)"
    )

    def run(
        self,
        *,
        agent=None,
        command: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
        sudo: bool = False,
        interactive: bool = False,
        retries: int = 0,
    ) -> str:  # type: ignore[override]
        cmd = (command or "").strip()
        if not cmd:
            raise ToolError("shell_command requires a non-empty 'command'")

        resolved_cwd: Optional[str] = None
        if cwd:
            path = Path(cwd).expanduser()
            if not path.exists():
                raise ToolError(f"cwd does not exist: {path}")
            if not path.is_dir():
                raise ToolError(f"cwd is not a directory: {path}")
            resolved_cwd = str(path)

        safe_timeout = max(1, int(timeout or 0))

        base_cmd: List[str] = ["/bin/bash", "-lc", cmd]
        if sudo:
            sudo_path = shutil.which("sudo")
            if not sudo_path:
                raise ToolError("sudo requested but not available on PATH")
            base_cmd = [sudo_path, "-n"] + base_cmd

        attempts = max(1, int(retries) + 1)
        stdout = ""
        stderr = ""
        exit_code = 0

        for attempt in range(attempts):
            try:
                if interactive:
                    exit_code, stdout = self._run_interactive(base_cmd, resolved_cwd, safe_timeout)
                    stderr = ""
                else:
                    completed = subprocess.run(
                        base_cmd,
                        capture_output=True,
                        text=True,
                        timeout=safe_timeout,
                        cwd=resolved_cwd,
                    )
                    exit_code = completed.returncode
                    stdout = (completed.stdout or "").strip()
                    stderr = (completed.stderr or "").strip()
            except subprocess.TimeoutExpired as exc:
                if attempt < attempts - 1:
                    continue
                raise ToolError(f"command timed out after {safe_timeout}s") from exc
            except FileNotFoundError as exc:
                raise ToolError("/bin/bash not found on system") from exc
            except Exception as exc:  # pragma: no cover - defensive
                if attempt < attempts - 1:
                    time.sleep(0.1)
                    continue
                raise ToolError(f"command failed: {exc}") from exc

            if exit_code == 0 or attempt == attempts - 1:
                break

        lines = [f"$ {cmd}"]
        if resolved_cwd:
            lines.append(f"(cwd: {resolved_cwd})")
        lines.append(f"attempts: {attempt + 1}")
        if sudo:
            lines.append("sudo: enabled")
        if interactive:
            lines.append("interactive: captured pseudo-tty session")
        if stdout:
            lines.append("\nstdout:\n" + stdout)
        if stderr:
            lines.append("\nstderr:\n" + stderr)
        lines.append(f"\nexit_code: {exit_code}")

        output = "\n".join(lines).strip()
        if len(output) > MAX_SHELL_OUTPUT:
            output = output[: MAX_SHELL_OUTPUT - 20] + "\n(truncated)"
        return output

    def _run_interactive(
        self,
        cmd: List[str],
        cwd: Optional[str],
        timeout: int,
    ) -> tuple[int, str]:
        master, slave = pty.openpty()
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=cwd,
            text=False,
            close_fds=True,
        )
        os.close(slave)
        chunks: List[str] = []
        try:
            while True:
                if timeout and (time.monotonic() - start) > timeout:
                    proc.kill()
                    raise subprocess.TimeoutExpired(cmd, timeout)
                r, _, _ = select.select([master], [], [], 0.1)
                if master in r:
                    try:
                        data = os.read(master, 1024)
                    except OSError:
                        break
                    if not data:
                        break
                    chunks.append(data.decode(errors="replace"))
                if proc.poll() is not None and not r:
                    break
            while True:
                try:
                    data = os.read(master, 1024)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data.decode(errors="replace"))
        finally:
            os.close(master)
        proc.wait()
        return proc.returncode, "".join(chunks).strip()


class ListDirectoryTool(Tool):
    """List entries in a directory."""

    name = "list_dir"
    description = "List directory contents (non-recursive)."
    args_hint = (
        "path (str, optional, default '.'); show_hidden (bool, optional); max_entries (int, optional); "
        "recursive (bool, optional); depth (int, optional); human (bool, optional)"
    )

    def run(
        self,
        *,
        agent=None,
        path: str = ".",
        show_hidden: bool = False,
        max_entries: int = 100,
        recursive: bool = False,
        depth: Optional[int] = None,
        human: bool = False,
    ) -> str:  # type: ignore[override]
        resolved = _resolve_user_path(path)
        if resolved.is_file():
            resolved = resolved.parent
        if not resolved.exists():
            raise ToolError(f"directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ToolError(f"not a directory: {resolved}")

        limit = max(int(max_entries or 0), 0) or 100

        max_depth = None if depth is None else max(0, int(depth))

        lines = [f"Directory: {resolved}"]
        collected: List[str] = []

        def _human_size(value: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            size = float(value)
            for unit in units:
                if size < 1024 or unit == units[-1]:
                    return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
                size /= 1024
            return f"{value}B"

        def _format_entry(entry: Path) -> str:
            suffix = "/" if entry.is_dir() else ""
            if not human:
                return entry.name + suffix
            try:
                size = entry.stat().st_size if entry.is_file() else sum(
                    child.stat().st_size for child in entry.iterdir() if child.is_file()
                ) if entry.is_dir() else 0
            except Exception:
                size = 0
            size_text = _human_size(size)
            return f"{entry.name + suffix} ({size_text})"

        def _walk(current: Path, current_depth: int = 0, prefix: str = "") -> None:
            nonlocal collected
            try:
                entries = sorted(current.iterdir(), key=lambda item: item.name.lower())
            except Exception as exc:
                collected.append(prefix + f"<error: {exc}>")
                return
            for entry in entries:
                if not show_hidden and entry.name.startswith('.'):
                    continue
                collected.append(prefix + _format_entry(entry))
                if len(collected) >= limit:
                    return
                if recursive and entry.is_dir() and not entry.is_symlink():
                    if max_depth is None or current_depth < max_depth:
                        _walk(entry, current_depth + 1, prefix + "  ")
                        if len(collected) >= limit:
                            return

        _walk(resolved, 0, "" if not recursive else "")

        if not collected:
            lines.append("(empty)")
            return "\n".join(lines)

        lines.extend(collected)
        if len(collected) >= limit:
            lines.append("... limit reached")
        return "\n".join(lines)


class _BrowserToolBase(Tool):
    def __init__(self, session_resolver):
        self._session_resolver = session_resolver

    def _session(self, agent) -> BrowserSession:
        if agent is None:
            raise ToolError("browser tool requires agent context")
        session = self._session_resolver(agent)
        if session is None:
            raise ToolError("browser session unavailable")
        return session


class BrowserSearchTool(_BrowserToolBase):
    name = "browser.search"
    description = "Federated web search with persistent browsing state."
    args_hint = "query (str); topn (int); sources (list[str]); time_range (str); site (str); budget_tokens (int)"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "topn": {"type": "integer", "minimum": 1},
            "sources": {"type": "array", "items": {"type": "string"}},
            "time_range": {"type": "string"},
            "site": {"type": "string"},
            "budget_tokens": {"type": "integer", "minimum": 256},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, *, agent=None, **kwargs: Any) -> str:  # type: ignore[override]
        session = self._session(agent)
        query = (kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("browser.search requires 'query'")
        topn = int(kwargs.get("topn") or os.getenv("ATLAS_SEARCH_TOPN", DEFAULT_TOPN))
        sources = kwargs.get("sources")
        time_range = kwargs.get("time_range")
        site = kwargs.get("site")
        budget = int(kwargs.get("budget_tokens") or DEFAULT_BUDGET)
        result = session.search(
            query=query,
            topn=topn,
            sources=sources,
            time_range=time_range,
            site=site,
            budget_tokens=budget,
        )
        return result["pageText"]


class BrowserOpenTool(_BrowserToolBase):
    name = "browser.open"
    description = "Open a search result or scroll the current page."
    args_hint = "id (int|str); cursor (int); loc (int); num_lines (int)"
    parameters_schema = {
        "type": "object",
        "properties": {
            "id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            "cursor": {"type": "integer"},
            "loc": {"type": "integer"},
            "num_lines": {"type": "integer", "minimum": -1},
        },
        "additionalProperties": False,
    }

    def run(self, *, agent=None, **kwargs: Any) -> str:  # type: ignore[override]
        session = self._session(agent)
        result = session.open(
            id=kwargs.get("id"),
            cursor=int(kwargs.get("cursor", -1)),
            loc=int(kwargs.get("loc", -1)),
            num_lines=int(kwargs.get("num_lines", -1)),
        )
        return result["pageText"]


class BrowserFindTool(_BrowserToolBase):
    name = "browser.find"
    description = "Find text within the current browser page."
    args_hint = "pattern (str); cursor (int)"
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "cursor": {"type": "integer"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def run(self, *, agent=None, **kwargs: Any) -> str:  # type: ignore[override]
        session = self._session(agent)
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("browser.find requires 'pattern'")
        result = session.find(pattern=pattern, cursor=int(kwargs.get("cursor", -1)))
        return result["pageText"]

if "__all__" not in globals():
    __all__: List[str] = []
__all__ += ["BrowserSearchTool", "BrowserOpenTool", "BrowserFindTool"]
