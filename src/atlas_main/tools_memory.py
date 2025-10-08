"""Memory retrieval tools for Atlas agent."""
from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING

from .tools import Tool, ToolError

if TYPE_CHECKING:
    from .agent import AtlasAgent
    from .websocket_server import AtlasWebSocketServer


class SearchMemoriesTool(Tool):
    """Search across all memory layers (episodic, semantic facts, reflections) for relevant information."""

    @property
    def name(self) -> str:
        return "search_memories"

    @property
    def description(self) -> str:
        return "Search your memory for relevant past conversations, learned facts, and reflections. Use this when you need to recall specific information the user mentioned previously or lessons you've learned."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what you're looking for in memory",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def run(self, *, agent: Any = None, **kwargs: Any) -> str:
        query = kwargs.get("query")
        limit = kwargs.get("limit", 5)

        if not query:
            raise ToolError("query parameter is required")

        if not isinstance(limit, int) or limit < 1:
            limit = 5
        limit = min(limit, 20)  # Cap at 20 results

        # Access layered memory from agent
        if not hasattr(agent, "layered_memory") or agent.layered_memory is None:
            return "Memory system is not available."

        memory = agent.layered_memory

        # Search each layer
        results = []

        # Episodic memories (conversations)
        try:
            episodic_results = memory.episodic.recall(query, top_k=limit)
            for score, record in episodic_results:
                user_text = record.get("user", "")
                assistant_text = record.get("assistant", "")
                if user_text or assistant_text:
                    snippet = f"User: {user_text}\nAssistant: {assistant_text}" if user_text else assistant_text
                    results.append({
                        "type": "conversation",
                        "score": round(score, 3),
                        "content": snippet[:300],
                    })
        except Exception:
            pass

        # Semantic facts
        try:
            fact_results = memory.semantic.recall(query, top_k=limit)
            for score, fact in fact_results:
                text = fact.get("text", "")
                tags = fact.get("tags", [])
                if text:
                    results.append({
                        "type": "fact",
                        "score": round(score, 3),
                        "content": text,
                        "tags": tags[:3] if tags else [],
                    })
        except Exception:
            pass

        # Reflections
        try:
            reflection_results = memory.reflections.recall(query, top_k=limit)
            for score, reflection in reflection_results:
                text = reflection.get("text", "")
                tags = reflection.get("tags", [])
                if text:
                    results.append({
                        "type": "reflection",
                        "score": round(score, 3),
                        "content": text,
                        "tags": tags[:3] if tags else [],
                    })
        except Exception:
            pass

        if not results:
            return f"No relevant memories found for query: '{query}'"

        # Sort by score and take top results
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        # Format output
        output_lines = [f"Found {len(results)} relevant memories for '{query}':\n"]
        for i, result in enumerate(results, 1):
            type_label = result["type"].upper()
            score = result["score"]
            content = result["content"]
            tags = result.get("tags", [])

            output_lines.append(f"{i}. [{type_label}] (relevance: {score})")
            output_lines.append(f"   {content}")
            if tags:
                output_lines.append(f"   Tags: {', '.join(tags)}")
            output_lines.append("")

        return "\n".join(output_lines)


class RecallReflectionsTool(Tool):
    """Retrieve learned lessons and reflections that match a specific context."""

    @property
    def name(self) -> str:
        return "recall_reflections"

    @property
    def description(self) -> str:
        return "Search specifically for learned lessons, preferences, and reflections. Use this when you need to check 'how does the user prefer X done?' or 'what have I learned about Y?'"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What kind of lesson or reflection are you looking for?",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of reflections to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    def run(self, *, agent: Any = None, **kwargs: Any) -> str:
        query = kwargs.get("query")
        limit = kwargs.get("limit", 3)

        if not query:
            raise ToolError("query parameter is required")

        if not isinstance(limit, int) or limit < 1:
            limit = 3
        limit = min(limit, 10)

        if not hasattr(agent, "layered_memory") or agent.layered_memory is None:
            return "Memory system is not available."

        memory = agent.layered_memory

        try:
            results = memory.reflections.recall(query, top_k=limit)
        except Exception as e:
            return f"Error searching reflections: {e}"

        if not results:
            return f"No reflections found matching: '{query}'"

        output_lines = [f"Found {len(results)} relevant reflections for '{query}':\n"]
        for i, (score, reflection) in enumerate(results, 1):
            text = reflection.get("text", "")
            tags = reflection.get("tags", [])
            confidence = reflection.get("confidence")
            uses = reflection.get("uses", 0)

            output_lines.append(f"{i}. (relevance: {round(score, 3)}, used {uses} times)")
            output_lines.append(f"   {text}")
            if tags:
                output_lines.append(f"   Tags: {', '.join(tags)}")
            if confidence:
                output_lines.append(f"   Confidence: {round(confidence, 2)}")
            output_lines.append("")

        return "\n".join(output_lines)


class SearchKGTool(Tool):
    """Search the knowledge graph for entities, topics, and relationships."""

    @property
    def name(self) -> str:
        return "search_kg"

    @property
    def description(self) -> str:
        return "Search the knowledge graph for entities, topics, projects, decisions, and tasks. Use this to find structured information and discover relationships between concepts."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding entities in the knowledge graph",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    def run(self, *, agent: Any = None, **kwargs: Any) -> str:
        query = kwargs.get("query")
        limit = kwargs.get("limit", 10)

        if not query:
            raise ToolError("query parameter is required")

        if not isinstance(limit, int) or limit < 1:
            limit = 10
        limit = min(limit, 50)

        # Get graph store from agent's websocket server context
        # This is a bit indirect, but necessary since tools don't have direct access
        graph_store = getattr(agent, "_graph_store", None)
        if graph_store is None:
            return "Knowledge graph is not available."

        try:
            nodes = graph_store.search_nodes(query, limit=limit)
        except Exception as e:
            return f"Error searching knowledge graph: {e}"

        if not nodes:
            return f"No entities found in knowledge graph matching: '{query}'"

        output_lines = [f"Found {len(nodes)} entities in knowledge graph for '{query}':\n"]
        for i, node in enumerate(nodes, 1):
            node_type = node.type
            label = node.label
            props = node.props

            output_lines.append(f"{i}. {label} ({node_type})")
            output_lines.append(f"   ID: {node.id}")
            if props:
                # Show first few properties
                prop_strs = [f"{k}={v}" for k, v in list(props.items())[:3]]
                if prop_strs:
                    output_lines.append(f"   Properties: {', '.join(prop_strs)}")
            output_lines.append("")

        return "\n".join(output_lines)


class ExploreKGTool(Tool):
    """Explore relationships in the knowledge graph by traversing from specific nodes."""

    @property
    def name(self) -> str:
        return "explore_kg"

    @property
    def description(self) -> str:
        return "Explore the knowledge graph by finding neighbors and relationships of specific nodes. Use this after search_kg to discover how entities are connected."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of node IDs to explore from (obtained from search_kg)",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to traverse (1-3, default: 1)",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum total nodes to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["node_ids"],
        }

    def run(self, *, agent: Any = None, **kwargs: Any) -> str:
        node_ids = kwargs.get("node_ids")
        depth = kwargs.get("depth", 1)
        limit = kwargs.get("limit", 20)

        if not node_ids or not isinstance(node_ids, list):
            raise ToolError("node_ids must be a non-empty list")

        if not isinstance(depth, int) or depth < 1:
            depth = 1
        depth = min(depth, 3)

        if not isinstance(limit, int) or limit < 1:
            limit = 20
        limit = min(limit, 100)

        graph_store = getattr(agent, "_graph_store", None)
        if graph_store is None:
            return "Knowledge graph is not available."

        try:
            nodes, edges = graph_store.subgraph(node_ids, depth=depth, limit=limit)
        except Exception as e:
            return f"Error exploring knowledge graph: {e}"

        if not nodes:
            return f"No nodes found when exploring from IDs: {node_ids}"

        # Build output
        output_lines = [
            f"Explored knowledge graph from {len(node_ids)} starting nodes (depth={depth}):",
            f"Found {len(nodes)} nodes and {len(edges)} relationships\n"
        ]

        # Group nodes by type
        nodes_by_type: dict[str, list] = {}
        for node in nodes:
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            nodes_by_type[node.type].append(node)

        # Show nodes by type
        output_lines.append("NODES:")
        for node_type, type_nodes in sorted(nodes_by_type.items()):
            output_lines.append(f"\n{node_type.upper()}S:")
            for node in type_nodes[:10]:  # Limit per type
                output_lines.append(f"  - {node.label} (ID: {node.id})")

        # Show relationships
        if edges:
            output_lines.append(f"\n\nRELATIONSHIPS ({len(edges)} total):")
            # Create node ID to label mapping
            node_map = {node.id: node.label for node in nodes}

            for i, edge in enumerate(edges[:15], 1):  # Show first 15 relationships
                src_label = node_map.get(edge.src, f"ID:{edge.src}")
                dst_label = node_map.get(edge.dst, f"ID:{edge.dst}")
                output_lines.append(f"{i}. {src_label} --[{edge.type}]--> {dst_label}")

            if len(edges) > 15:
                output_lines.append(f"... and {len(edges) - 15} more relationships")

        return "\n".join(output_lines)


__all__ = [
    "SearchMemoriesTool",
    "RecallReflectionsTool",
    "SearchKGTool",
    "ExploreKGTool",
]
