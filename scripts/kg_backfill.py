#!/usr/bin/env python3
"""Populate the knowledge graph from existing semantic and reflection memory files."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

from atlas_main.kg_config import KnowledgeGraphConfig, ensure_parent
from atlas_main.knowledge_graph import SQLiteGraphStore
from atlas_main.kg_extractor import MemoryGraphExtractor, MemoryPayload
from atlas_main.memory_layers import LayeredMemoryConfig


def _iter_semantic(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        facts = payload.get("facts", []) if isinstance(payload, dict) else []
    except Exception:
        facts = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        yield {
            "id": str(fact.get("id") or fact.get("uuid") or time.time_ns()),
            "text": text,
            "tags": fact.get("tags", []),
            "ts": float(fact.get("ts", time.time())),
            "source": fact.get("source"),
        }


def _iter_reflections(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        lessons = payload.get("lessons", []) if isinstance(payload, dict) else []
    except Exception:
        lessons = []
    for item in lessons:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        yield {
            "id": str(item.get("id") or time.time_ns()),
            "text": text,
            "tags": item.get("tags", []),
            "ts": float(item.get("ts", time.time())),
        }


def main() -> int:
    config = KnowledgeGraphConfig.from_env()
    if not config.enabled:
        print("Knowledge graph is disabled (set ATLAS_KG_ENABLED=1).", file=sys.stderr)
        return 1

    ensure_parent(config.db_path)
    store = SQLiteGraphStore(config)
    extractor = MemoryGraphExtractor()

    memory_config = LayeredMemoryConfig()
    semantic_items = list(_iter_semantic(memory_config.semantic_path))
    reflection_items = list(_iter_reflections(memory_config.reflections_path))

    total_nodes = 0
    total_edges = 0

    for item in semantic_items:
        payload = MemoryPayload(
            id=item["id"],
            kind="semantic",
            text=item["text"],
            tags=item.get("tags", []),
            ts=item.get("ts", time.time()),
            source=item.get("source"),
        )
        result = extractor.extract(payload)
        _apply_extraction(store, result)
        total_nodes += len(result.nodes)
        total_edges += len(result.edges)

    for item in reflection_items:
        payload = MemoryPayload(
            id=item["id"],
            kind="reflection",
            text=item["text"],
            tags=item.get("tags", []),
            ts=item.get("ts", time.time()),
            source=None,
        )
        result = extractor.extract(payload)
        _apply_extraction(store, result)
        total_nodes += len(result.nodes)
        total_edges += len(result.edges)

    store.close()
    print(f"Backfill complete. Nodes ingested: {total_nodes}, edges: {total_edges}")
    return 0


def _apply_extraction(store: SQLiteGraphStore, result) -> None:
    key_to_id: dict[str, int] = {}
    for node in result.nodes:
        node_id = store.upsert_node(node.type, node.label, node.props)
        key_to_id[node.key] = node_id
    for edge in result.edges:
        src_id = key_to_id.get(edge.source)
        dst_id = key_to_id.get(edge.target)
        if src_id is None or dst_id is None:
            continue
        store.upsert_edge(src_id, dst_id, edge.type, edge.props)


if __name__ == "__main__":
    raise SystemExit(main())
