"""SQLite-backed property graph store for Atlas."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .kg_config import KnowledgeGraphConfig, ensure_parent


@dataclass
class NodeRecord:
    id: int
    type: str
    label: str
    props: Dict[str, str]


@dataclass
class EdgeRecord:
    id: int
    src: int
    dst: int
    type: str


class GraphStore:
    """Abstract graph storage interface."""

    def upsert_node(self, node_type: str, label: str, props: Optional[Dict[str, str]] = None) -> int:
        raise NotImplementedError

    def upsert_edge(
        self,
        src_id: int,
        dst_id: int,
        edge_type: str,
        props: Optional[Dict[str, str]] = None,
    ) -> int:
        raise NotImplementedError

    def search_nodes(self, query: str, *, limit: int = 20) -> List[NodeRecord]:
        raise NotImplementedError

    def neighbors(
        self,
        node_id: int,
        *,
        types: Optional[Sequence[str]] = None,
        limit: int = 25,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        raise NotImplementedError

    def subgraph(
        self,
        seed_ids: Iterable[int],
        *,
        depth: int = 1,
        limit: int = 64,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        raise NotImplementedError

    def recent_nodes(self, *, limit: int = 20) -> List[NodeRecord]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SQLiteGraphStore(GraphStore):
    """SQLite implementation of the property graph store."""

    def __init__(self, config: KnowledgeGraphConfig) -> None:
        self.config = config
        ensure_parent(config.db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(config.db_path))
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def upsert_node(self, node_type: str, label: str, props: Optional[Dict[str, str]] = None) -> int:
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("label cannot be empty")
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO nodes(type, label, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(type, label) DO UPDATE SET label=excluded.label
                RETURNING id
                """,
                (node_type, normalized_label, int(time.time())),
            )
            row = cur.fetchone()
            node_id = int(row[0])
            if props:
                self._upsert_node_props(node_id, props)
            if self.config.fts_enabled:
                self._update_fts(node_id, normalized_label, props or {})
            return node_id

    def upsert_edge(
        self,
        src_id: int,
        dst_id: int,
        edge_type: str,
        props: Optional[Dict[str, str]] = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO edges(src, dst, type, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(src, dst, type) DO UPDATE SET type=excluded.type
                RETURNING id
                """,
                (src_id, dst_id, edge_type, int(time.time())),
            )
            row = cur.fetchone()
            edge_id = int(row[0])
            if props:
                self._upsert_edge_props(edge_id, props)
            return edge_id

    def search_nodes(self, query: str, *, limit: int = 20) -> List[NodeRecord]:
        limit = max(1, min(limit, 100))
        with self._lock:
            if self.config.fts_enabled:
                cursor = self._conn.execute(
                    """
                    SELECT n.id, n.type, n.label
                    FROM node_fts f
                    JOIN nodes n ON n.id = f.rowid
                    WHERE node_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
            else:
                like_query = f"%{query.lower()}%"
                cursor = self._conn.execute(
                    """
                    SELECT id, type, label
                    FROM nodes
                    WHERE lower(label) LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (like_query, limit),
                )
            rows = cursor.fetchall()
        return [self._row_to_node(row) for row in rows]

    def neighbors(
        self,
        node_id: int,
        *,
        types: Optional[Sequence[str]] = None,
        limit: int = 25,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        limit = max(1, min(limit, 200))
        params: List[object] = [node_id, node_id]
        type_filter = ""
        if types:
            placeholders = ",".join("?" for _ in types)
            type_filter = f" AND e.type IN ({placeholders})"
            params.extend(types)
        query = f"""
            SELECT e.id AS edge_id, e.src, e.dst, e.type,
                   n.id AS node_id, n.type AS node_type, n.label AS node_label
            FROM edges e
            JOIN nodes n ON (n.id = CASE WHEN e.src = ? THEN e.dst ELSE e.src END)
            WHERE (e.src = ? OR e.dst = ?){type_filter}
            ORDER BY e.created_at DESC
            LIMIT ?
        """
        params.insert(2, node_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        seen_nodes: Dict[int, NodeRecord] = {}
        edges: List[EdgeRecord] = []
        for row in rows:
            edge_id = int(row["edge_id"])
            src = int(row["src"])
            dst = int(row["dst"])
            edges.append(EdgeRecord(id=edge_id, src=src, dst=dst, type=row["type"]))
            node_id = int(row["node_id"])
            if node_id not in seen_nodes:
                seen_nodes[node_id] = self._fetch_node_with_props(node_id)
        return list(seen_nodes.values()), edges

    def subgraph(
        self,
        seed_ids: Iterable[int],
        *,
        depth: int = 1,
        limit: int = 64,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        depth = max(0, min(depth, 3))
        limit = max(1, min(limit, 512))
        visited: Dict[int, NodeRecord] = {}
        collected_edges: Dict[int, EdgeRecord] = {}
        frontier = list({int(s) for s in seed_ids})
        current_depth = 0
        while frontier and current_depth <= depth and len(visited) < limit:
            next_frontier: List[int] = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                node = self._fetch_node_with_props(node_id)
                if not node:
                    continue
                visited[node_id] = node
                neighbors, edges = self.neighbors(node_id, limit=limit)
                for neighbor in neighbors:
                    if neighbor.id not in visited and neighbor.id not in next_frontier:
                        next_frontier.append(neighbor.id)
                for edge in edges:
                    if edge.id not in collected_edges:
                        collected_edges[edge.id] = edge
                if len(visited) >= limit:
                    break
            frontier = next_frontier
            current_depth += 1
        return list(visited.values()), list(collected_edges.values())

    def recent_nodes(self, *, limit: int = 20) -> List[NodeRecord]:
        limit = max(1, min(limit, 100))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, type, label FROM nodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initialize_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(type, label)
                );

                CREATE TABLE IF NOT EXISTS node_props (
                    node_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    PRIMARY KEY (node_id, key),
                    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src INTEGER NOT NULL,
                    dst INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(src, dst, type),
                    FOREIGN KEY(src) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(dst) REFERENCES nodes(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS edge_props (
                    edge_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    PRIMARY KEY (edge_id, key),
                    FOREIGN KEY(edge_id) REFERENCES edges(id) ON DELETE CASCADE
                );
                """
            )
            if self.config.fts_enabled:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS node_fts
                    USING fts5(label, text, content='')
                    """
                )

    def _upsert_node_props(self, node_id: int, props: Dict[str, str]) -> None:
        for key, value in props.items():
            self._conn.execute(
                """
                INSERT INTO node_props(node_id, key, value)
                VALUES(?, ?, ?)
                ON CONFLICT(node_id, key) DO UPDATE SET value=excluded.value
                """,
                (node_id, key, str(value)),
            )

    def _upsert_edge_props(self, edge_id: int, props: Dict[str, str]) -> None:
        for key, value in props.items():
            self._conn.execute(
                """
                INSERT INTO edge_props(edge_id, key, value)
                VALUES(?, ?, ?)
                ON CONFLICT(edge_id, key) DO UPDATE SET value=excluded.value
                """,
                (edge_id, key, str(value)),
            )

    def _update_fts(self, node_id: int, label: str, props: Dict[str, str]) -> None:
        text = " ".join(str(v) for v in props.values())
        self._conn.execute("INSERT INTO node_fts(node_fts, rowid, label, text) VALUES('delete', ?, '', '')", (node_id,))
        self._conn.execute(
            "INSERT INTO node_fts(rowid, label, text) VALUES(?, ?, ?)",
            (node_id, label, text),
        )

    def _row_to_node(self, row: sqlite3.Row) -> NodeRecord:
        props = self._fetch_node_props(int(row["id"]))
        return NodeRecord(id=int(row["id"]), type=str(row["type"]), label=str(row["label"]), props=props)

    def _fetch_node_with_props(self, node_id: int) -> Optional[NodeRecord]:
        with self._lock:
            row = self._conn.execute("SELECT id, type, label FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def _fetch_node_props(self, node_id: int) -> Dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM node_props WHERE node_id = ?",
                (node_id,),
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}


__all__ = [
    "NodeRecord",
    "EdgeRecord",
    "GraphStore",
    "SQLiteGraphStore",
]
