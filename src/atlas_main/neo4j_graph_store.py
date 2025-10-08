"""Neo4j-backed property graph store for Atlas."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from .knowledge_graph import GraphStore, NodeRecord, EdgeRecord
from .kg_config import KnowledgeGraphConfig


class Neo4jGraphStore(GraphStore):
    """Neo4j implementation of the property graph store."""

    def __init__(self, config: KnowledgeGraphConfig) -> None:
        self.config = config

        # Get Neo4j connection settings from config or use defaults
        uri = getattr(config, 'neo4j_uri', 'bolt://localhost:7687')
        username = getattr(config, 'neo4j_username', 'neo4j')
        password = getattr(config, 'neo4j_password', 'neo4j')

        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

        # Initialize schema (indexes and constraints)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create indexes and constraints for efficient querying."""
        with self.driver.session() as session:
            # Create constraint for unique (type, label) combination
            # This ensures we can upsert nodes by type+label
            try:
                session.run("""
                    CREATE CONSTRAINT node_type_label_unique IF NOT EXISTS
                    FOR (n:Node)
                    REQUIRE (n.type, n.label) IS UNIQUE
                """)
            except Neo4jError:
                pass  # Constraint might already exist

            # Create index on node labels for search
            try:
                session.run("""
                    CREATE INDEX node_label_index IF NOT EXISTS
                    FOR (n:Node)
                    ON (n.label)
                """)
            except Neo4jError:
                pass

            # Create full-text search index if enabled
            if self.config.fts_enabled:
                try:
                    session.run("""
                        CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS
                        FOR (n:Node)
                        ON EACH [n.label, n.text]
                    """)
                except Neo4jError:
                    pass

    def upsert_node(self, node_type: str, label: str, props: Optional[Dict[str, str]] = None) -> int:
        """Create or update a node, returning its internal Neo4j ID."""
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError("label cannot be empty")

        props = props or {}
        created_at = int(time.time())

        # Build properties dict for Cypher
        node_props = {
            'type': node_type,
            'label': normalized_label,
            'created_at': created_at,
            **props
        }

        # Create text field for full-text search
        if self.config.fts_enabled:
            text = " ".join(str(v) for v in props.values())
            node_props['text'] = text

        with self.driver.session() as session:
            result = session.run("""
                MERGE (n:Node {type: $type, label: $label})
                SET n += $props
                RETURN id(n) AS node_id
            """, type=node_type, label=normalized_label, props=node_props)

            record = result.single()
            return int(record['node_id'])

    def upsert_edge(
        self,
        src_id: int,
        dst_id: int,
        edge_type: str,
        props: Optional[Dict[str, str]] = None,
    ) -> int:
        """Create or update an edge between two nodes."""
        props = props or {}
        created_at = int(time.time())

        edge_props = {
            'type': edge_type,
            'created_at': created_at,
            **props
        }

        with self.driver.session() as session:
            # Use MERGE to ensure only one edge of this type between these nodes
            result = session.run("""
                MATCH (src:Node), (dst:Node)
                WHERE id(src) = $src_id AND id(dst) = $dst_id
                MERGE (src)-[r:RELATED {type: $edge_type}]->(dst)
                SET r += $props
                RETURN id(r) AS edge_id
            """, src_id=src_id, dst_id=dst_id, edge_type=edge_type, props=edge_props)

            record = result.single()
            if record is None:
                raise ValueError(f"Could not create edge: nodes {src_id} or {dst_id} not found")
            return int(record['edge_id'])

    def search_nodes(self, query: str, *, limit: int = 20) -> List[NodeRecord]:
        """Search nodes using full-text search or label matching."""
        limit = max(1, min(limit, 100))

        with self.driver.session() as session:
            if self.config.fts_enabled:
                # Use Neo4j full-text search
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('node_fulltext', $query)
                    YIELD node, score
                    RETURN id(node) AS node_id,
                           node.type AS type,
                           node.label AS label,
                           properties(node) AS props
                    ORDER BY score DESC
                    LIMIT $limit
                """, query=query, limit=limit)
            else:
                # Fallback to CONTAINS search
                result = session.run("""
                    MATCH (n:Node)
                    WHERE toLower(n.label) CONTAINS toLower($query)
                    RETURN id(n) AS node_id,
                           n.type AS type,
                           n.label AS label,
                           properties(n) AS props
                    ORDER BY n.created_at DESC
                    LIMIT $limit
                """, query=query, limit=limit)

            return [self._record_to_node(record) for record in result]

    def neighbors(
        self,
        node_id: int,
        *,
        types: Optional[Sequence[str]] = None,
        limit: int = 25,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        """Get neighboring nodes and their connecting edges."""
        limit = max(1, min(limit, 200))

        # Build type filter
        type_filter = ""
        if types:
            type_list = ", ".join(f"'{t}'" for t in types)
            type_filter = f"AND r.type IN [{type_list}]"

        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (center:Node)-[r:RELATED]-(neighbor:Node)
                WHERE id(center) = $node_id {type_filter}
                RETURN id(r) AS edge_id,
                       id(startNode(r)) AS src_id,
                       id(endNode(r)) AS dst_id,
                       r.type AS edge_type,
                       id(neighbor) AS neighbor_id,
                       neighbor.type AS neighbor_type,
                       neighbor.label AS neighbor_label,
                       properties(neighbor) AS neighbor_props
                ORDER BY r.created_at DESC
                LIMIT $limit
            """, node_id=node_id, limit=limit)

            nodes: Dict[int, NodeRecord] = {}
            edges: List[EdgeRecord] = []

            for record in result:
                # Add edge
                edge = EdgeRecord(
                    id=int(record['edge_id']),
                    src=int(record['src_id']),
                    dst=int(record['dst_id']),
                    type=record['edge_type']
                )
                edges.append(edge)

                # Add neighbor node if not already seen
                neighbor_id = int(record['neighbor_id'])
                if neighbor_id not in nodes:
                    nodes[neighbor_id] = NodeRecord(
                        id=neighbor_id,
                        type=record['neighbor_type'],
                        label=record['neighbor_label'],
                        props=self._clean_props(record['neighbor_props'])
                    )

            return list(nodes.values()), edges

    def subgraph(
        self,
        seed_ids: Iterable[int],
        *,
        depth: int = 1,
        limit: int = 64,
    ) -> Tuple[List[NodeRecord], List[EdgeRecord]]:
        """Extract a subgraph starting from seed nodes using variable-length path matching."""
        depth = max(0, min(depth, 3))
        limit = max(1, min(limit, 512))

        seed_list = list(set(int(s) for s in seed_ids))

        # Build variable-length pattern based on depth
        if depth == 0:
            # Just return seed nodes
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:Node)
                    WHERE id(n) IN $seed_ids
                    RETURN id(n) AS node_id,
                           n.type AS node_type,
                           n.label AS node_label,
                           properties(n) AS node_props
                    LIMIT $limit
                """, seed_ids=seed_list, limit=limit)

                nodes = {}
                for record in result:
                    node_id = int(record['node_id'])
                    nodes[node_id] = NodeRecord(
                        id=node_id,
                        type=record['node_type'],
                        label=record['node_label'],
                        props=self._clean_props(record['node_props'])
                    )
                return list(nodes.values()), []

        with self.driver.session() as session:
            # Use variable-length path matching
            min_depth = 1
            max_depth = depth
            result = session.run(f"""
                MATCH path = (seed:Node)-[*{min_depth}..{max_depth}]-(n:Node)
                WHERE id(seed) IN $seed_ids
                WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
                UNWIND path_nodes AS node
                WITH collect(DISTINCT node) AS all_nodes, path_rels
                UNWIND path_rels AS rel
                WITH all_nodes, collect(DISTINCT rel) AS all_rels
                UNWIND all_nodes AS n
                UNWIND all_rels AS r
                RETURN DISTINCT
                    id(n) AS node_id,
                    n.type AS node_type,
                    n.label AS node_label,
                    properties(n) AS node_props,
                    id(r) AS edge_id,
                    id(startNode(r)) AS src_id,
                    id(endNode(r)) AS dst_id,
                    r.type AS edge_type
                LIMIT $limit
            """, seed_ids=seed_list, limit=limit)

            nodes: Dict[int, NodeRecord] = {}
            edges: Dict[int, EdgeRecord] = {}

            for record in result:
                # Add node
                node_id = int(record['node_id'])
                if node_id not in nodes:
                    nodes[node_id] = NodeRecord(
                        id=node_id,
                        type=record['node_type'],
                        label=record['node_label'],
                        props=self._clean_props(record['node_props'])
                    )

                # Add edge
                edge_id = record['edge_id']
                if edge_id is not None and edge_id not in edges:
                    edges[edge_id] = EdgeRecord(
                        id=int(edge_id),
                        src=int(record['src_id']),
                        dst=int(record['dst_id']),
                        type=record['edge_type']
                    )

            return list(nodes.values()), list(edges.values())

    def recent_nodes(self, *, limit: int = 20) -> List[NodeRecord]:
        """Get most recently created nodes."""
        limit = max(1, min(limit, 100))

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Node)
                RETURN id(n) AS node_id,
                       n.type AS type,
                       n.label AS label,
                       properties(n) AS props
                ORDER BY n.created_at DESC
                LIMIT $limit
            """, limit=limit)

            return [self._record_to_node(record) for record in result]

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_to_node(self, record) -> NodeRecord:
        """Convert a Neo4j record to a NodeRecord."""
        props = self._clean_props(record['props'])
        return NodeRecord(
            id=int(record['node_id']),
            type=record['type'],
            label=record['label'],
            props=props
        )

    def _clean_props(self, props: Dict) -> Dict[str, str]:
        """Remove internal properties and convert values to strings."""
        internal_keys = {'type', 'label', 'created_at', 'text'}
        return {
            k: str(v)
            for k, v in props.items()
            if k not in internal_keys
        }


__all__ = [
    "Neo4jGraphStore",
]
