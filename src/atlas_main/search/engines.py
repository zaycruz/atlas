"""Search engine adapters and federation helpers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import requests

from . import SearchResult
from .cache import serp_get, serp_put

USER_AGENT = os.getenv("ATLAS_SEARCH_USER_AGENT", "AtlasSearch/1.0 (+https://github.com)")
DEFAULT_TIMEOUT = int(os.getenv("ATLAS_SEARCH_TIMEOUT", "12"))


def _http_get(url: str, *, params: Optional[Dict[str, str]] = None) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}
    response = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    # Drop tracking parameters commonly seen
    whitelist = {"id", "p", "q", "s", "v", "t"}
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False) if k.lower() in whitelist]
    canonical = parsed._replace(query=urlencode(query_pairs), fragment="")
    return urlunparse(canonical)


def _result_key(result: SearchResult) -> str:
    return _canonicalize_url(result.url)


def _normalize_result(payload: Dict[str, str], *, source: str) -> Optional[SearchResult]:
    title = (payload.get("title") or payload.get("name") or "").strip()
    url = (payload.get("url") or payload.get("link") or payload.get("href") or "").strip()
    snippet = (payload.get("snippet") or payload.get("description") or payload.get("text") or "").strip()
    if not title or not url:
        return None
    extras = {k: v for k, v in payload.items() if isinstance(v, str) and k not in {"title", "url", "snippet", "description", "text"}}
    return SearchResult(title=title, url=url, snippet=snippet, source=source, published_at=payload.get("date"), extras=extras)


def search_ddg(query: str, *, topn: int = 8, time_range: Optional[str] = None, site: Optional[str] = None) -> List[SearchResult]:
    if site:
        query = f"{query} site:{site}"
    params = {
        "q": query,
        "kl": "us-en",
        "kp": "-2",
        "kc": "1",
        "kz": "-1",
        "k1": "-1",
        "t": "atlas",
        "ia": "web",
        "format": "json",
        "no_redirect": "1",
    }
    if time_range in {"d", "w", "m", "y"}:
        params["df"] = time_range
    try:
        response = _http_get("https://duckduckgo.com/", params=params)
        data = response.json()
    except Exception:
        return []

    results: List[SearchResult] = []
    for raw in data.get("results", [])[:topn * 2]:
        normalized = _normalize_result(raw, source="ddg")
        if normalized:
            results.append(normalized)
    # fallback to "RelatedTopics" if direct results empty
    if not results:
        for raw in data.get("RelatedTopics", [])[:topn * 2]:
            if isinstance(raw, dict):
                normalized = _normalize_result(raw, source="ddg")
                if normalized:
                    results.append(normalized)
    return results[:topn]


def _bing_headers() -> Dict[str, str]:
    key = os.getenv("BING_API_KEY")
    return {"Ocp-Apim-Subscription-Key": key} if key else {}


def search_bing(query: str, *, topn: int = 8, site: Optional[str] = None) -> List[SearchResult]:
    if not os.getenv("BING_API_KEY"):
        return []
    if site:
        query = f"{query} site:{site}"
    params = {"q": query, "count": min(topn, 50)}
    try:
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"User-Agent": USER_AGENT, **_bing_headers()},
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    web_pages = data.get("webPages", {}).get("value", [])
    results: List[SearchResult] = []
    for raw in web_pages[:topn * 2]:
        normalized = _normalize_result(raw, source="bing")
        if normalized:
            results.append(normalized)
    return results[:topn]


def search_google_serper(query: str, *, topn: int = 8, site: Optional[str] = None, time_range: Optional[str] = None) -> List[SearchResult]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return []
    if site:
        query = f"{query} site:{site}"
    payload = {"q": query, "num": min(topn, 10)}
    if time_range:
        payload["tbs"] = f"qdr:{time_range}"
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    organic = data.get("organic", [])
    results: List[SearchResult] = []
    for raw in organic[:topn * 2]:
        normalized = _normalize_result(raw, source="google")
        if normalized:
            results.append(normalized)
    return results[:topn]


def normalize_and_dedupe(results: Sequence[SearchResult]) -> List[SearchResult]:
    seen: Dict[str, SearchResult] = {}
    for item in results:
        key = _result_key(item)
        if key not in seen:
            seen[key] = item
    return list(seen.values())


def _serp_cache_key(query: str, sources: Sequence[str], topn: int, site: Optional[str], time_range: Optional[str]) -> str:
    raw = json.dumps({"q": query, "sources": list(sources), "topn": topn, "site": site, "time_range": time_range}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def federated_search(
    query: str,
    *,
    sources: Sequence[str],
    topn: int,
    time_range: Optional[str] = None,
    site: Optional[str] = None,
    cache_ttl_s: int = 86_400,
) -> List[SearchResult]:
    sources = [src.lower().strip() for src in sources if src]
    cache_key = _serp_cache_key(query, sources, topn, site, time_range)
    cached = serp_get(cache_key)
    if cached is not None:
        return cached

    aggregator: List[SearchResult] = []
    for source in sources:
        if source in {"ddg", "duckduckgo"}:
            aggregator.extend(search_ddg(query, topn=topn, time_range=time_range, site=site))
        elif source == "bing":
            aggregator.extend(search_bing(query, topn=topn, site=site))
        elif source in {"google", "serper"}:
            aggregator.extend(search_google_serper(query, topn=topn, site=site, time_range=time_range))

    normalized = normalize_and_dedupe(aggregator)
    serp_put(cache_key, normalized, ttl_s=cache_ttl_s)
    return normalized


__all__ = [
    "SearchResult",
    "federated_search",
    "normalize_and_dedupe",
    "search_ddg",
    "search_bing",
    "search_google_serper",
]
