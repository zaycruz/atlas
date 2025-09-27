"""Tool registry and web search implementation for Atlas Lite."""
from __future__ import annotations

import asyncio
import re
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


USER_AGENT = "AtlasLite/1.0 (+https://github.com)"
DEFAULT_FILE_CHUNK = 6000


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

    def describe(self) -> ToolDescription:
        return ToolDescription(self.name, self.description, self.args_hint)

    def run(self, *, agent=None, **kwargs: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError


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
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        },
                    },
                }
            )
        return specs


class WebSearchTool(Tool):
    """Web search with Crawl4AI-powered content extraction."""

    name = "web_search"
    description = "Search the web and extract clean content from results."
    args_hint = "query (str, required); max_results (int, optional, default 3)"

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

    def run(self, *, agent=None, query: str, max_results: int = 3) -> str:  # type: ignore[override]
        query = query.strip()
        if not query:
            raise ToolError("web_search requires a non-empty 'query'")
        
        max_results = max(1, min(int(max_results or 3), 5))
        
        # Get search results from DuckDuckGo
        search_results = self._search_duckduckgo(query)
        if not search_results:
            return f"No results found for '{query}'."

        summaries = []
        for idx, result in enumerate(search_results[:max_results]):
            snippet = result.get("snippet") or ""
            title = result.get("title") or result.get("text") or "Result"
            url = result.get("url") or result.get("first_url") or ""
            
            # Use Crawl4AI to extract clean content
            content = ""
            if url:
                try:
                    content = self._crawl_content(url)
                except (ToolError, Exception) as exc:
                    # Fallback to simple fetch if Crawl4AI fails
                    try:
                        content = self._fetch_content_simple(url)
                    except Exception:
                        content = f"(content unavailable: {type(exc).__name__})"

            section = textwrap.dedent(
                f"""Result {idx + 1}: {title}
URL: {url or 'n/a'}
Snippet: {snippet or 'n/a'}
Content: {content or 'n/a'}"""
            ).strip()
            summaries.append(section)

        header = f"Web search results for: {query}"
        return header + "\n\n" + "\n\n".join(summaries)

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


class ReadFileTool(Tool):
    """Read text from files on disk."""

    name = "read_file"
    description = "Read a UTF-8 text file, optionally with offset/length."  # noqa: RUF015
    args_hint = "path (str, required); offset (int, optional); length (int, optional)"

    def __init__(self, *, max_chars: int = DEFAULT_FILE_CHUNK) -> None:
        self._max_chars = max_chars

    def run(
        self,
        *,
        agent=None,
        path: str,
        offset: int = 0,
        length: Optional[int] = None,
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

        lines = [f"File: {resolved}", ""]
        lines.append(data)
        if truncated:
            lines.append(f"\n(truncated to {self._max_chars} characters)")
        return "\n".join(lines)


class WriteFileTool(Tool):
    """Write text content to disk."""

    name = "write_file"
    description = "Create or update a text file; requires explicit overwrite/append."  # noqa: RUF015
    args_hint = (
        "path (str, required); content (str, required); overwrite (bool, optional); "
        "append (bool, optional); create_dirs (bool, optional)"
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
    ) -> str:  # type: ignore[override]
        if append and overwrite:
            raise ToolError("set either append or overwrite, not both")

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

        mode = "a" if append else "w"
        try:
            with resolved.open(mode, encoding="utf-8") as handle:
                handle.write(content)
        except Exception as exc:
            raise ToolError(f"failed to write {resolved}: {exc}") from exc

        action = "Appended" if append else "Wrote"
        return f"{action} {len(content)} characters to {resolved}"


class ListDirectoryTool(Tool):
    """List entries in a directory."""

    name = "list_dir"
    description = "List directory contents (non-recursive)."
    args_hint = "path (str, optional, default '.'); show_hidden (bool, optional); max_entries (int, optional)"

    def run(
        self,
        *,
        agent=None,
        path: str = ".",
        show_hidden: bool = False,
        max_entries: int = 100,
    ) -> str:  # type: ignore[override]
        resolved = _resolve_user_path(path)
        if resolved.is_file():
            resolved = resolved.parent
        if not resolved.exists():
            raise ToolError(f"directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ToolError(f"not a directory: {resolved}")

        limit = max(int(max_entries or 0), 0) or 100

        try:
            entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
        except Exception as exc:
            raise ToolError(f"failed to list {resolved}: {exc}") from exc

        lines = [f"Directory: {resolved}"]
        visible: List[Path] = []
        for entry in entries:
            if not show_hidden and entry.name.startswith('.'):
                continue
            visible.append(entry)
            if len(visible) >= limit:
                break

        if not visible:
            lines.append("(empty)")
            return "\n".join(lines)

        for entry in visible:
            suffix = "/" if entry.is_dir() else ""
            lines.append(entry.name + suffix)

        remaining = len([e for e in entries if show_hidden or not e.name.startswith('.')]) - len(visible)
        if remaining > 0:
            lines.append(f"... {remaining} more entries hidden")
        return "\n".join(lines)
