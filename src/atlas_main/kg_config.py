"""Configuration helpers for the Atlas knowledge graph."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional


_DEFAULT_GRAPH_PATH = Path(os.getenv("ATLAS_DATA_DIR", "~/.atlas")).expanduser() / "graph.db"
_DEFAULT_QUEUE = 200
_DEFAULT_BATCH = 64


@dataclass
class KnowledgeGraphConfig:
    """Runtime configuration for the graph pipeline."""

    enabled: bool = False
    db_path: Path = _DEFAULT_GRAPH_PATH
    max_queue: int = 200
    batch_size: int = 64
    fts_enabled: bool = True

    # Graph backend selection
    backend: str = "neo4j"  # "sqlite" or "neo4j"

    # Neo4j connection settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "atlas123"

    @classmethod
    def from_env(cls) -> "KnowledgeGraphConfig":
        enabled = os.getenv("ATLAS_KG_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        db_override = os.getenv("ATLAS_KG_DB")
        queue_override = os.getenv("ATLAS_KG_QUEUE")
        batch_override = os.getenv("ATLAS_KG_BATCH")
        fts_override = os.getenv("ATLAS_KG_FTS")
        backend_override = os.getenv("ATLAS_KG_BACKEND", "neo4j")

        # Neo4j connection settings from environment
        neo4j_uri = os.getenv("ATLAS_NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("ATLAS_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("ATLAS_NEO4J_PASSWORD", "atlas123")

        db_path: Path = _DEFAULT_GRAPH_PATH
        if db_override:
            db_path = Path(db_override).expanduser()

        max_queue = _DEFAULT_QUEUE
        if queue_override:
            try:
                parsed = int(queue_override)
                max_queue = max(10, parsed)
            except ValueError:
                pass

        batch_size = _DEFAULT_BATCH
        if batch_override:
            try:
                parsed = int(batch_override)
                batch_size = max(8, parsed)
            except ValueError:
                pass

        fts_enabled = True
        if fts_override:
            fts_enabled = fts_override.strip().lower() not in {"0", "false", "off"}

        return cls(
            enabled=enabled,
            db_path=db_path,
            max_queue=max_queue,
            batch_size=batch_size,
            fts_enabled=fts_enabled,
            backend=backend_override,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
        )


def ensure_parent(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["KnowledgeGraphConfig", "ensure_parent"]
