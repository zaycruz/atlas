"""Filesystem-backed cache for SERP and page results."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from . import PageContent, SearchResult

CACHE_ROOT = Path(os.getenv("ATLAS_SEARCH_CACHE_DIR", "~/.atlas/search_cache")).expanduser()
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
SERP_DIR = CACHE_ROOT / "serp"
PAGE_DIR = CACHE_ROOT / "pages"
SERP_DIR.mkdir(parents=True, exist_ok=True)
PAGE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def serp_get(key: str) -> Optional[List[SearchResult]]:
    path = SERP_DIR / f"{key}.json"
    data = _read_json(path)
    if not data:
        return None
    expiry = data.get("expiry", 0)
    if expiry and expiry < time.time():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    results = [SearchResult(**item) for item in data.get("results", [])]
    return results


def serp_put(key: str, results: List[SearchResult], *, ttl_s: int) -> None:
    payload = {
        "results": [asdict(item) for item in results],
        "expiry": time.time() + max(ttl_s, 0),
    }
    _write_json(SERP_DIR / f"{key}.json", payload)


def page_get(key: str) -> Optional[PageContent]:
    path = PAGE_DIR / f"{_sanitize_filename(key)}.json"
    data = _read_json(path)
    if not data:
        return None
    expiry = data.get("expiry", 0)
    if expiry and expiry < time.time():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None
    try:
        fetched_at_raw = data.get("fetched_at")
        fetched_at = datetime.fromisoformat(fetched_at_raw) if fetched_at_raw else datetime.utcnow()
        page = PageContent(
            url=data["url"],
            text=data.get("text", ""),
            html=data.get("html"),
            meta=data.get("meta", {}),
            fetched_at=fetched_at,
        )
    except Exception:
        return None
    return page


def page_put(page: PageContent, *, ttl_s: int) -> None:
    timestamp = page.fetched_at.isoformat(timespec="seconds")
    payload = {
        "url": page.url,
        "text": page.text,
        "html": page.html,
        "meta": page.meta,
        "fetched_at": timestamp,
        "expiry": time.time() + max(ttl_s, 0),
    }
    filename = _sanitize_filename(page.url)
    _write_json(PAGE_DIR / f"{filename}.json", payload)


def _sanitize_filename(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return digest


__all__ = ["serp_get", "serp_put", "page_get", "page_put"]
