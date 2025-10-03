"""Search package providing multi-engine web retrieval and ranking utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class SearchResult:
    """Normalized search result metadata prior to ranking."""

    title: str
    url: str
    snippet: str
    source: str
    published_at: Optional[str] = None
    extras: Dict[str, str] = field(default_factory=dict)


@dataclass
class RankedDoc:
    """Ranked search result with score diagnostics."""

    url: str
    title: str
    score: float
    reason: str
    source: str
    snippet: str
    published_at: Optional[str] = None


@dataclass
class PageContent:
    """Fetched and cleaned page content."""

    url: str
    text: str
    html: Optional[str]
    meta: Dict[str, str]
    fetched_at: datetime


__all__ = ["SearchResult", "RankedDoc", "PageContent"]
