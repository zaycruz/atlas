"""Memory-aware tools for retrieving episodes and facts."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .tools import Tool, ToolError


def _require_layered_memory(agent) -> Any:
    layered = getattr(agent, "layered_memory", None)
    if layered is None:
        raise ToolError("Layered memory is disabled")
    return layered


class SearchEpisodesTool(Tool):
    """Retrieve episodic memories using vector search."""

    name = "memory.search_episodes"
    description = "Search episodic memory for relevant conversation snippets."
    args_hint = "query (str, required); limit (int, optional)"
    capabilities = frozenset({"memory:read"})
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, *, agent=None, query: str, limit: int = 5) -> str:  # type: ignore[override]
        if not query or not query.strip():
            raise ToolError("query parameter is required")
        layered = _require_layered_memory(agent)
        limit = max(1, min(int(limit or 0), 10))
        results = layered.episodic.recall(query, top_k=limit)
        if not results:
            return "No episodic memories matched that query."
        lines: List[str] = [f"Top episodic matches for '{query.strip()}':"]
        for score, payload in results:
            ts = payload.get("ts")
            stamp = ""
            if ts:
                stamp = f"{ts:.0f}"
            user = (payload.get("user") or "").strip()
            assistant = (payload.get("assistant") or "").strip()
            summary = assistant or user
            if not summary:
                summary = "(empty turn)"
            lines.append(f"- [{score:.3f}]{' ' + stamp if stamp else ''} {summary[:220]}")
        return "\n".join(lines)


class SearchFactsTool(Tool):
    """Retrieve semantic facts from long-term memory."""

    name = "memory.search_facts"
    description = "Search semantic facts from long-term memory."
    args_hint = "query (str, required); limit (int, optional)"
    capabilities = frozenset({"memory:read"})
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, *, agent=None, query: str, limit: int = 5) -> str:  # type: ignore[override]
        if not query or not query.strip():
            raise ToolError("query parameter is required")
        layered = _require_layered_memory(agent)
        limit = max(1, min(int(limit or 0), 10))
        results = layered.semantic.recall(query, top_k=limit)
        if not results:
            return "No semantic facts matched that query."
        lines: List[str] = [f"Top semantic facts for '{query.strip()}':"]
        for score, fact in results:
            text = str(fact.get("text", "")).strip()
            fact_id = str(fact.get("id") or "")[:8]
            tags = fact.get("tags") or []
            tag_part = f" tags: {', '.join(tags[:3])}" if tags else ""
            lines.append(f"- [{score:.3f}] ({fact_id}) {text[:220]}{tag_part}")
        return "\n".join(lines)


class FetchFactTool(Tool):
    """Retrieve the full contents of a semantic fact by identifier."""

    name = "memory.fetch_fact"
    description = "Fetch a stored semantic fact by ID (accepts UUID prefix)."
    args_hint = "id (str, required)"
    capabilities = frozenset({"memory:read"})
    parameters_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    def run(self, *, agent=None, id: str) -> str:  # type: ignore[override]
        identifier = (id or "").strip()
        if not identifier:
            raise ToolError("id parameter is required")
        layered = _require_layered_memory(agent)
        semantic = getattr(layered, "semantic", None)
        if semantic is None:
            raise ToolError("Semantic memory is not available")
        try:
            fact = semantic.resolve_fact(identifier)
        except ValueError:
            return (
                "Multiple facts share that prefix. Please provide the full UUID."
            )
        if not fact:
            return f"No fact found matching '{identifier}'."
        fact_id = str(fact.get("id") or "")
        text = str(fact.get("text", "")).strip()
        lines: List[str] = [f"Fact {fact_id}: {text}"]
        tags = fact.get("tags") or []
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        source = fact.get("source")
        if source:
            lines.append(f"Source: {source}")
        confidence = fact.get("confidence")
        quality = fact.get("quality")
        uses = fact.get("uses")
        stats_parts: List[str] = []
        if quality is not None:
            stats_parts.append(f"quality={float(quality):.2f}")
        if confidence is not None:
            stats_parts.append(f"confidence={float(confidence):.2f}")
        if uses is not None:
            stats_parts.append(f"uses={int(uses)}")
        if stats_parts:
            lines.append("Stats: " + ", ".join(stats_parts))
        last_access = fact.get("last_access_ts")
        if isinstance(last_access, (int, float)) and last_access > 0:
            lines.append(f"Last accessed: {int(last_access)}")
        return "\n".join(lines)


class SaveFactTool(Tool):
    """Persist a semantic fact into long-term memory."""

    name = "memory.save_fact"
    description = "Write a semantic fact with optional tags, quality, confidence, and source."
    args_hint = "text (str, required); tags (list[str], optional); confidence (float, optional); quality (float, optional); source (str, optional)"
    capabilities = frozenset({"memory:write"})
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "confidence": {"type": "number"},
            "quality": {"type": "number"},
            "source": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(
        self,
        *,
        agent=None,
        text: str,
        tags: Optional[Sequence[str]] = None,
        confidence: Optional[float] = None,
        quality: Optional[float] = None,
        source: Optional[str] = None,
    ) -> str:  # type: ignore[override]
        normalized = (text or "").strip()
        if not normalized:
            raise ToolError("text parameter is required")
        layered = _require_layered_memory(agent)
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        fact = layered.semantic.add_fact(
            normalized,
            source=source.strip() if source and source.strip() else None,
            confidence=confidence,
            quality=quality,
            tags=clean_tags,
        )
        if not fact:
            return "No fact stored (input was empty or rejected)."
        fact_id = str(fact.get("id") or "")[:8]
        applied_tags = fact.get("tags") or []
        tag_part = f" tags: {', '.join(applied_tags[:3])}" if applied_tags else ""
        status = "updated" if fact.get("uses", 0) else "stored"
        return f"Fact {status}: ({fact_id}) {fact.get('text', '')[:220]}{tag_part}"


__all__ = [
    "SearchEpisodesTool",
    "SearchFactsTool",
    "FetchFactTool",
    "SaveFactTool",
]
