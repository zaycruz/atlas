"""Memory-aware tools for retrieving episodes, facts, and knowledge graph links."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

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
    description = "Search semantic facts and show linked knowledge graph connections."
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
            links = fact.get("links") or []
            tag_part = f" tags: {', '.join(tags[:3])}" if tags else ""
            link_part = ""
            if links:
                linked = ", ".join(str(link.get("target", ""))[:8] for link in links[:3])
                if linked:
                    link_part = f" links: {linked}"
            lines.append(f"- [{score:.3f}] ({fact_id}) {text[:220]}{tag_part}{link_part}")
        return "\n".join(lines)


class ExploreKnowledgeTool(Tool):
    """Explore the knowledge graph around specific fact IDs."""

    name = "memory.explore_graph"
    description = "Show neighbouring facts in the knowledge graph."
    args_hint = "ids (list[str], required); limit (int, optional); relation (str, optional)"
    capabilities = frozenset({"memory:read"})
    parameters_schema = {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "relation": {"type": "string"},
        },
        "required": ["ids"],
        "additionalProperties": False,
    }

    def run(
        self,
        *,
        agent=None,
        ids: Iterable[str],
        limit: int = 10,
        relation: Optional[str] = None,
    ) -> str:  # type: ignore[override]
        layered = _require_layered_memory(agent)
        ids = [str(item).strip() for item in ids if str(item).strip()]
        if not ids:
            raise ToolError("ids must contain at least one fact identifier")
        relation_filter = (relation or "").strip().lower()
        limit = max(1, min(int(limit or 0), 20))
        lines: List[str] = []
        count = 0
        for fact_id in ids:
            fact = layered.semantic.get_fact(fact_id)
            if not fact:
                lines.append(f"- Fact {fact_id} not found.")
                continue
            title = str(fact.get("text", ""))[:120]
            lines.append(f"Fact {fact_id}: {title}")
            neighbors = layered.semantic.graph.links_for(fact_id)
            if relation_filter:
                neighbors = [link for link in neighbors if str(link.get("type", "")).lower() == relation_filter]
            if not neighbors:
                lines.append("  (no linked facts)")
                continue
            for link in neighbors:
                if count >= limit:
                    break
                target = link.get("target")
                relation_name = link.get("type") or "related"
                target_fact = layered.semantic.get_fact(target) if target else None
                summary = (target_fact or {}).get("text", "")
                lines.append(f"  - {relation_name} → {str(target)[:8]}: {str(summary)[:160]}")
                count += 1
            if count >= limit:
                break
        return "\n".join(lines) if lines else "No graph links available."


__all__ = ["SearchEpisodesTool", "SearchFactsTool", "ExploreKnowledgeTool"]
