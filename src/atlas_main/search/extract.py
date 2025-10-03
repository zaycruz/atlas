"""Basic HTML to text extraction."""
from __future__ import annotations

import html
import re
from typing import Dict, Tuple

META_RE = re.compile(r'<meta\s+[^>]*?(name|property)=["\'](?P<name>[^"\']+)["\'][^>]*?content=["\'](?P<content>[^"\']+)["\']', re.IGNORECASE)
TITLE_RE = re.compile(r"<title>(?P<title>[\s\S]*?)</title>", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
STRUCTURAL_RE = re.compile(r"<(header|footer|nav)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(raw_html: str, url: str) -> Tuple[str, Dict[str, str]]:
    """Convert HTML into readable text and capture metadata."""
    if not raw_html:
        return "", {"title": url}
    cleaned = SCRIPT_STYLE_RE.sub(" ", raw_html)
    cleaned = STRUCTURAL_RE.sub(" ", cleaned)
    title_match = TITLE_RE.search(cleaned)
    title = html.unescape(title_match.group("title").strip()) if title_match else url
    metas: Dict[str, str] = {"title": title}
    for match in META_RE.finditer(cleaned):
        key = match.group("name").lower()
        metas[key] = match.group("content").strip()
    text = TAG_RE.sub(" ", cleaned)
    text = html.unescape(text)
    # Collapse elongated whitespace while preserving paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip(), metas


__all__ = ["clean_text"]
