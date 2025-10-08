"""Async ingestion pipeline for the Atlas knowledge graph."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from .kg_config import KnowledgeGraphConfig
from .knowledge_graph import GraphStore
from .kg_extractor import ExtractionResult, MemoryGraphExtractor, MemoryPayload

LOGGER = logging.getLogger(__name__)


_pipeline: Optional["KnowledgeGraphPipeline"] = None


class KnowledgeGraphPipeline:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        config: KnowledgeGraphConfig,
        store: GraphStore,
        extractor: MemoryGraphExtractor,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._loop = loop
        self._config = config
        self._store = store
        self._extractor = extractor
        self._event_callback = event_callback
        self._queue: asyncio.Queue[MemoryPayload] = asyncio.Queue(maxsize=config.max_queue)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = self._loop.create_task(self._worker())

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None

    def submit_memory(self, payload: MemoryPayload) -> None:
        if not self._running:
            return
        def _enqueue() -> None:
            try:
                self._queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(payload)
                except asyncio.QueueFull:
                    LOGGER.debug("Dropped KG payload due to persistent full queue")
        self._loop.call_soon_threadsafe(_enqueue)

    async def _worker(self) -> None:
        LOGGER.info("Knowledge graph pipeline started")
        try:
            while True:
                payload = await self._queue.get()
                try:
                    result = self._extractor.extract(payload)
                    self._apply_extraction(result)
                    if self._event_callback:
                        self._event_callback(
                            {
                                "type": "kg_update",
                                "payload": {
                                    "memory_id": payload.id,
                                    "kind": payload.kind,
                                    "nodes": len(result.nodes),
                                    "edges": len(result.edges),
                                },
                            }
                        )
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Failed to ingest memory into knowledge graph")
        except asyncio.CancelledError:
            LOGGER.info("Knowledge graph pipeline stopping")
            raise

    def _apply_extraction(self, result: ExtractionResult) -> None:
        key_to_id: dict[str, int] = {}
        for node in result.nodes:
            node_id = self._store.upsert_node(node.type, node.label, node.props)
            key_to_id[node.key] = node_id
        for edge in result.edges:
            src_id = key_to_id.get(edge.source)
            dst_id = key_to_id.get(edge.target)
            if src_id is None or dst_id is None:
                continue
            self._store.upsert_edge(src_id, dst_id, edge.type, edge.props)


def set_pipeline(pipeline: Optional[KnowledgeGraphPipeline]) -> None:
    global _pipeline
    _pipeline = pipeline
    if _pipeline:
        _pipeline.start()


def get_pipeline() -> Optional[KnowledgeGraphPipeline]:
    return _pipeline


def clear_pipeline() -> None:
    global _pipeline
    _pipeline = None


__all__ = [
    "KnowledgeGraphPipeline",
    "set_pipeline",
    "get_pipeline",
    "clear_pipeline",
    "MemoryPayload",
]
