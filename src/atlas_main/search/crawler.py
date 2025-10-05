"""Simple HTTP crawler with caching, PDF, and optional headless fallbacks."""
from __future__ import annotations

import time
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

from . import PageContent
from .cache import page_get, page_put
from .extract import clean_text

USER_AGENT = "AtlasCrawler/1.0 (+https://github.com)"
DEFAULT_TIMEOUT = 15
DEFAULT_CACHE_TTL = 7 * 24 * 3600

# crude per-host cooldown storage
_LAST_FETCH: Dict[str, float] = {}
_COOLDOWN_SECONDS = 1.5


class CrawlError(RuntimeError):
    """Raised when a fetch attempt fails."""


def _respect_cooldown(host: str) -> None:
    now = time.time()
    last = _LAST_FETCH.get(host, 0)
    if now - last < _COOLDOWN_SECONDS:
        time.sleep(_COOLDOWN_SECONDS - (now - last))
    _LAST_FETCH[host] = time.time()


def fetch(
    url: str,
    *,
    allow_cache: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    use_headless: bool = False,
) -> PageContent:
    canonical_url = url.strip()
    cache_key = canonical_url
    if allow_cache:
        cached = page_get(cache_key)
        if cached:
            return cached

    parsed = urlparse(canonical_url)
    if parsed.scheme not in {"http", "https"}:
        raise CrawlError(f"Unsupported URL scheme for {canonical_url}")

    if use_headless:
        page = _fetch_headless(canonical_url)
    else:
        _respect_cooldown(parsed.netloc)
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}

        try:
            response = requests.get(canonical_url, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            raise CrawlError(f"Failed to fetch {canonical_url}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        final_url = str(response.url)
        if "pdf" in content_type or canonical_url.lower().endswith(".pdf"):
            text = _extract_pdf(response.content)
            html = None
            extra_meta = {}
        else:
            html = response.text if "html" in content_type else None
            text, extra_meta = clean_text(response.text, canonical_url)
        meta = {
            "status": str(response.status_code),
            "content_type": content_type,
            "final_url": final_url,
        }
        meta.update(extra_meta)
        page = PageContent(
            url=canonical_url,
            text=text,
            html=html,
            meta=meta,
            fetched_at=datetime.utcnow(),
        )
    if allow_cache:
        page_put(page, ttl_s=DEFAULT_CACHE_TTL)
    return page


def _fetch_headless(url: str) -> PageContent:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise CrawlError("Headless browsing requires playwright") from exc

    with sync_playwright() as p:  # pragma: no cover - network dependent
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT * 1000)
        page.goto(url, wait_until="networkidle")
        html = page.content()
        text, meta = clean_text(html, url)
        browser.close()
    meta.update({"status": "200", "content_type": "text/html", "final_url": url})
    return PageContent(url=url, text=text, html=html, meta=meta, fetched_at=datetime.utcnow())


def _extract_pdf(content: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return ""
    buffer = BytesIO(content)
    try:
        return extract_text(buffer) or ""
    except Exception:
        return ""


__all__ = ["fetch", "CrawlError"]
