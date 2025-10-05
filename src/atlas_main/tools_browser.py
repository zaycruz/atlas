"""Browser-style tool set powered by the search package."""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

from .search import RankedDoc
from .search import engines, rank
from .search.crawler import CrawlError, fetch

STATE_VERSION = 1
DEFAULT_VIEW_TOKENS = 1024
DEFAULT_SOURCES = [src.strip() for src in os.getenv("ATLAS_SEARCH_SOURCES", "ddg").split(",") if src.strip()]
DEFAULT_TOPN = int(os.getenv("ATLAS_SEARCH_TOPN", "8"))
DEFAULT_OPEN_TOPK = int(os.getenv("ATLAS_SEARCH_OPEN_TOPK", "3"))
DEFAULT_BUDGET = int(os.getenv("ATLAS_SEARCH_MAX_TOKENS", "8000"))

LogFn = Callable[[str, Dict[str, Union[str, int, float]]], None]


@dataclass
class PageView:
    url: str
    title: str
    text: str
    lines: List[str]
    links: Dict[int, str]
    metadata: Dict[str, str]
    created_at: datetime


@dataclass
class BrowserState:
    page_stack: List[str] = field(default_factory=list)
    url_to_page: Dict[str, PageView] = field(default_factory=dict)
    view_tokens: int = DEFAULT_VIEW_TOKENS
    version: int = STATE_VERSION


def _estimate_tokens(text: str) -> int:
    return int(len(text) / 4)


def _domain(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.netloc or url


def _wrap(text: str, width: int = 90) -> List[str]:
    wrapped: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width, replace_whitespace=True) or [""])
    return wrapped


class BrowserSession:
    """Stateful helper for federated search and navigation."""

    def __init__(
        self,
        *,
        state: Optional[BrowserState] = None,
        embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None,
        logger: Optional[LogFn] = None,
    ) -> None:
        self.state = state or BrowserState()
        self.embed_fn = embed_fn
        self._logger = logger

    # ------------------------------------------------------------------
    def search(
        self,
        *,
        query: str,
        topn: int = DEFAULT_TOPN,
        sources: Optional[List[str]] = None,
        time_range: Optional[str] = None,
        site: Optional[str] = None,
        budget_tokens: int = DEFAULT_BUDGET,
    ) -> Dict[str, object]:
        sources = sources or DEFAULT_SOURCES
        ranked_docs = self._run_search(query, sources, topn, time_range, site)
        budget_remaining = budget_tokens
        summary_lines, links = self._build_summary(query, ranked_docs)
        summary_text = "\n".join(summary_lines) if summary_lines else "No results found."
        page = PageView(
            url=f"search://{datetime.utcnow().isoformat()}?q={query}",
            title=f"Search results for '{query}'",
            text=summary_text,
            lines=_wrap(summary_text),
            links=links,
            metadata={"query": query, "sources": ",".join(sources)},
            created_at=datetime.utcnow(),
        )
        self._push_page(page)
        self._log("search.query", {"query": query, "sources": ",".join(sources), "topn": topn})

        open_topk = min(DEFAULT_OPEN_TOPK, len(ranked_docs))
        opened_urls: List[str] = []
        for doc in ranked_docs[:open_topk]:
            if budget_remaining <= 0:
                break
            try:
                page_view = self._open_url(doc.url)
                tokens = _estimate_tokens(page_view.text)
                budget_remaining -= tokens
                opened_urls.append(doc.url)
            except CrawlError:
                continue
        self._log(
            "search.budget",
            {"requested": budget_tokens, "remaining": max(budget_remaining, 0), "opened": len(opened_urls)},
        )
        return {"state": self._serialize_state(), "pageText": self._render_page(page)}

    def open(
        self,
        *,
        id: Optional[Union[int, str]] = None,
        cursor: int = -1,
        loc: int = -1,
        num_lines: int = -1,
    ) -> Dict[str, object]:
        page = self._resolve_page_for_open(id, cursor)
        if isinstance(id, str) and id:
            page = self._open_url(id)
        elif isinstance(id, int):
            link_url = page.links.get(id)
            if not link_url:
                raise CrawlError(f"No link with id {id}")
            page = self._open_url(link_url)

        rendered = self._render_page(page, loc=loc, num_lines=num_lines)
        self._log("browser.open", {"url": page.url, "loc": loc, "num_lines": num_lines})
        return {"state": self._serialize_state(), "pageText": rendered}

    def find(self, *, pattern: str, cursor: int = -1) -> Dict[str, object]:
        page = self._resolve_page_for_open(None, cursor)
        lowered = page.text.lower()
        matches: List[str] = []
        start = 0
        while True:
            idx = lowered.find(pattern.lower(), start)
            if idx == -1:
                break
            window = page.text[max(0, idx - 100) : idx + 200]
            matches.append(window.replace("\n", " "))
            start = idx + len(pattern)
        if not matches:
            result_text = f"No matches for '{pattern}'"
        else:
            result_text = "\n\n".join(f"[{i}] …{snippet}…" for i, snippet in enumerate(matches))
        find_page = PageView(
            url=f"find://{page.url}?pattern={pattern}",
            title=f"Find '{pattern}'",
            text=result_text,
            lines=_wrap(result_text),
            links={},
            metadata={"pattern": pattern, "source": page.url},
            created_at=datetime.utcnow(),
        )
        self._push_page(find_page)
        self._log("browser.find", {"pattern": pattern, "matches": len(matches)})
        return {"state": self._serialize_state(), "pageText": self._render_page(find_page)}

    # ------------------------------------------------------------------
    def _run_search(
        self,
        query: str,
        sources: List[str],
        topn: int,
        time_range: Optional[str],
        site: Optional[str],
    ) -> List[RankedDoc]:
        raw_results = engines.federated_search(
            query,
            sources=sources,
            topn=topn,
            time_range=time_range,
            site=site,
        )
        if len(raw_results) < 2 and site is None:
            alternate_query = f"{query} overview"
            self._log("search.reformulate", {"query": query, "alt": alternate_query})
            raw_results.extend(
                engines.federated_search(
                    alternate_query,
                    sources=sources,
                    topn=topn,
                    time_range=time_range,
                    site=site,
                )
            )
        ranked_docs = rank.rank_results(query, raw_results, embed_fn=self.embed_fn)
        return ranked_docs

    def _build_summary(self, query: str, docs: List[RankedDoc]) -> Tuple[List[str], Dict[int, str]]:
        lines: List[str] = []
        links: Dict[int, str] = {}
        if not docs:
            return lines, links
        max_score = max(doc.score for doc in docs) or 1.0
        token_counts: Dict[str, int] = {}
        for doc in docs[:5]:
            for token in doc.snippet.split():
                token = token.lower().strip(",.;:!?")
                if len(token) >= 5 and token not in query.lower():
                    token_counts[token] = token_counts.get(token, 0) + 1
        consensus = [word for word, count in sorted(token_counts.items(), key=lambda item: item[1], reverse=True) if count >= 2][:5]
        if consensus:
            lines.append("Consensus keywords: " + ", ".join(consensus))
            lines.append("")
        for idx, doc in enumerate(docs):
            confidence = max(0.1, min(0.99, doc.score / max_score))
            lines.append(f"[{idx}] {doc.title} ({_domain(doc.url)}) — confidence {confidence:.2f}")
            snippet = doc.snippet[:200].strip() if doc.snippet else "(no snippet)"
            lines.append(f"     {snippet}")
            lines.append(f"     reason={doc.reason}")
            links[idx] = doc.url
        lines.append("")
        lines.append("Citations:")
        for idx, url in links.items():
            lines.append(f"  [{idx}] {url}")
        return lines, links

    def _open_url(self, url: str) -> PageView:
        if url in self.state.url_to_page:
            page = self.state.url_to_page[url]
            self._push_page(page)
            return page
        page_content = fetch(url)
        text = page_content.text or ""
        title = page_content.meta.get("title") or url
        lines = _wrap(text)
        links = _extract_links(page_content.html or "", base_url=url)
        page = PageView(
            url=url,
            title=title,
            text=text,
            lines=lines,
            links=links,
            metadata=page_content.meta,
            created_at=page_content.fetched_at,
        )
        self.state.url_to_page[url] = page
        self.state.page_stack.append(url)
        self._log("crawl.fetch", {"url": url, "words": len(text.split()), "links": len(links)})
        return page

    def _push_page(self, page: PageView) -> None:
        if page.url not in self.state.url_to_page:
            self.state.url_to_page[page.url] = page
        if not self.state.page_stack or self.state.page_stack[-1] != page.url:
            self.state.page_stack.append(page.url)
        if len(self.state.page_stack) > 50:
            self.state.page_stack = self.state.page_stack[-50:]

    def _resolve_page_for_open(self, id: Optional[Union[int, str]], cursor: int) -> PageView:
        if cursor >= 0 and cursor < len(self.state.page_stack):
            url = self.state.page_stack[cursor]
            return self.state.url_to_page[url]
        if self.state.page_stack:
            return self.state.url_to_page[self.state.page_stack[-1]]
        raise CrawlError("No pages available")

    def _render_page(self, page: PageView, *, loc: int = -1, num_lines: int = -1) -> str:
        lines = page.lines
        total = len(lines)
        if loc >= 0:
            start = max(0, loc)
        elif num_lines > 0:
            start = max(0, total - num_lines)
        else:
            start = 0
        if num_lines > 0:
            end = min(total, start + num_lines)
        else:
            end = total
        slice_lines = lines[start:end]
        numbered = [f"L{start + idx}: {line}" for idx, line in enumerate(slice_lines)]
        footer = [
            "",
            f"Links ({len(page.links)}):"
        ]
        for idx, url in page.links.items():
            footer.append(f"[{idx}] {url}")
        footer.append("")
        footer.append(f"Showing lines {start}-{end} of {total}")
        return "\n".join(numbered + footer)

    def _serialize_state(self) -> Dict[str, object]:
        return {
            "page_stack": list(self.state.page_stack),
            "view_tokens": self.state.view_tokens,
            "version": self.state.version,
        }

    def _log(self, event: str, data: Dict[str, Union[str, int, float]]) -> None:
        if self._logger:
            self._logger(event, data)


def _extract_links(html: str, *, base_url: str) -> Dict[int, str]:
    links: Dict[int, str] = {}
    if not html:
        return links
    import re
    from urllib.parse import urljoin

    pattern = re.compile(r'<a[^>]+href=["\'](?P<href>[^"\']+)["\'][^>]*>', re.IGNORECASE)
    for idx, match in enumerate(pattern.finditer(html)):
        href = match.group("href")
        if href.startswith("javascript:"):
            continue
        links[idx] = urljoin(base_url, href)
        if len(links) >= 50:
            break
    return links


__all__ = ["BrowserSession", "BrowserState", "PageView"]
