"""Utility helpers for redacting sensitive content."""
from __future__ import annotations

import os
import re
from typing import Any, Dict

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_API_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9]{16,}|[A-Za-z0-9]{32,})\b")


def _enabled() -> bool:
    value = os.getenv("ATLAS_REDACT", "1").strip().lower()
    return value not in {"0", "false", "off"}


def redact_text(text: str) -> str:
    if not text or not _enabled():
        return text
    redacted = _EMAIL_RE.sub("<redacted_email>", text)
    redacted = _API_KEY_RE.sub("<redacted_token>", redacted)
    return redacted


def redact_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    if not _enabled():
        return {str(k): str(v) for k, v in mapping.items()}
    return {str(k): redact_text(str(v)) for k, v in mapping.items()}
