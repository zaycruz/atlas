"""Layered memory for Jarvis-like agent: episodic, semantic, reflections, assembly.

All components are local and optional. Designed to run without external services.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EmbedFn = Callable[[str], Optional[Sequence[float]]]


LOGGER = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EpisodicSQLiteMemory:
    """Lightweight episodic memory using SQLite and local embeddings."""

    def __init__(self, db_path: Path, embed_fn: Optional[EmbedFn] = None, max_records: int = 2000) -> None:
        self.db_path = Path(db_path).expanduser()
        self.embed_fn = embed_fn
        self.max_records = max_records
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                user TEXT,
                assistant TEXT,
                embedding TEXT
            );
            """
        )
        self._conn.commit()

    def log(self, user: str, assistant: str) -> None:
        ts = time.time()
        combined = f"User: {user}\nAssistant: {assistant}".strip()
        emb: Optional[List[float]] = None
        if self.embed_fn is not None:
            try:
                vec = self.embed_fn(combined)
                if vec:
                    emb = list(vec)
            except Exception:
                emb = None
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO episodes (id, ts, user, assistant, embedding) VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                ts,
                user,
                assistant,
                json.dumps(emb) if emb is not None else None,
            ),
        )
        # Trim table to max_records by deleting oldest
        cur.execute("SELECT COUNT(*) FROM episodes")
        (count,) = cur.fetchone()
        if count and count > self.max_records:
            to_delete = count - self.max_records
            cur.execute("DELETE FROM episodes WHERE id IN (SELECT id FROM episodes ORDER BY ts ASC LIMIT ?)", (to_delete,))
        self._conn.commit()

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        if top_k <= 0:
            return []
        if not self.embed_fn:
            return []
        qvec = self.embed_fn(query)
        if not qvec:
            return []
        q = np.asarray(qvec, dtype=float)
        cur = self._conn.cursor()
        cur.execute("SELECT id, ts, user, assistant, embedding FROM episodes")
        rows = cur.fetchall()
        scored: List[Tuple[float, dict]] = []
        for _id, ts, user, assistant, emb_json in rows:
            if not emb_json:
                continue
            try:
                emb = json.loads(emb_json)
                v = np.asarray(emb, dtype=float)
            except Exception:
                continue
            if v.shape != q.shape:
                continue
            s = _cosine(q, v)
            scored.append((s, {"ts": ts, "user": user or "", "assistant": assistant or ""}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def recent(self, top_k: int = 3) -> List[dict]:
        if top_k <= 0:
            return []
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, user, assistant FROM episodes ORDER BY ts DESC LIMIT ?",
            (int(top_k),),
        )
        rows = cur.fetchall()
        recent: List[dict] = []
        for ts, user, assistant in rows:
            recent.append({"ts": ts, "user": user or "", "assistant": assistant or ""})
        return recent


class SemanticMemory:
    """Durable facts loaded from a JSON file; optional embeddings for recall."""

    def __init__(self, json_path: Path, embed_fn: Optional[EmbedFn] = None) -> None:
        self.json_path = Path(json_path).expanduser()
        self.embed_fn = embed_fn
        self._facts: List[dict] = []
        self._embeddings: List[Optional[np.ndarray]] = []
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            self._facts = []
            self._embeddings = []
            return
        try:
            raw = json.loads(self.json_path.read_text())
            facts = raw if isinstance(raw, list) else raw.get("facts", [])
        except Exception:
            facts = []
        clean_facts: List[dict] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            entry = {
                "text": text,
                "ts": float(fact.get("ts", time.time())),
            }
            source = fact.get("source")
            if source:
                entry["source"] = str(source)
            confidence = fact.get("confidence")
            if confidence is not None:
                try:
                    entry["confidence"] = float(confidence)
                except (TypeError, ValueError):
                    pass
            clean_facts.append(entry)
        self._facts = clean_facts
        self._embeddings = [None] * len(self._facts)

    def _persist(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"facts": self._facts}
        self.json_path.write_text(json.dumps(payload, indent=2))

    def add_fact(
        self,
        text: str,
        *,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[dict]:
        normalized = (text or "").strip()
        if not normalized:
            return None
        key = normalized.lower()
        for fact in self._facts:
            if str(fact.get("text", "")).strip().lower() == key:
                return fact
        entry: dict[str, Any] = {"text": normalized, "ts": time.time()}
        if source:
            entry["source"] = source
        if confidence is not None:
            try:
                entry["confidence"] = float(confidence)
            except (TypeError, ValueError):
                pass
        self._facts.append(entry)
        self._embeddings.append(None)
        try:
            if self.embed_fn is not None:
                vec = self.embed_fn(normalized)
                if vec:
                    self._embeddings[-1] = np.asarray(vec, dtype=float)
        except Exception:
            self._embeddings[-1] = None
        self._persist()
        return entry

    def extend_facts(
        self,
        facts: Iterable[dict[str, Any] | str],
        *,
        source: Optional[str] = None,
    ) -> int:
        added = 0
        for item in facts:
            if isinstance(item, dict):
                text = str(item.get("text", ""))
                confidence = item.get("confidence")
                inserted = self.add_fact(text, source=item.get("source") or source, confidence=confidence)
            else:
                inserted = self.add_fact(str(item), source=source)
            if inserted:
                added += 1
        return added

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        if not self._facts or not self.embed_fn or top_k <= 0:
            return []
        qvec = self.embed_fn(query)
        if not qvec:
            return []
        q = np.asarray(qvec, dtype=float)
        scored: List[Tuple[float, dict]] = []
        for i, fact in enumerate(self._facts):
            if self._embeddings[i] is None:
                try:
                    v = self.embed_fn(str(fact.get("text", "")))
                    self._embeddings[i] = None if not v else np.asarray(v, dtype=float)
                except Exception:
                    self._embeddings[i] = None
            v = self._embeddings[i]
            if v is None or v.shape != q.shape:
                continue
            scored.append((_cosine(q, v), fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def head(self, top_k: int = 3) -> List[dict]:
        if top_k <= 0:
            return []
        return self._facts[:top_k]

    def prune(self, max_items: int, *, keep_indices: Optional[set[int]] = None) -> int:
        """Trim stored facts to `max_items`, prioritizing confidence then recency.

        `keep_indices` allows callers to protect specific entries (e.g. after review).
        """
        if max_items <= 0:
            removed = len(self._facts)
            if removed:
                self._facts = []
                self._embeddings = []
                self._persist()
            return removed
        if len(self._facts) <= max_items:
            return 0

        if keep_indices is None:
            def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    ts = float(item.get("ts", 0.0))
                except (TypeError, ValueError):
                    ts = 0.0
                return (confidence, ts)

            indexed = list(enumerate(self._facts))
            indexed.sort(key=lambda pair: _sort_key(pair[1]), reverse=True)
            keep_indices = {idx for idx, _ in indexed[:max_items]}
        else:
            keep_indices = set(keep_indices)

        new_facts: List[dict] = []
        new_embeddings: List[Optional[np.ndarray]] = []
        removed = 0
        for idx, fact in enumerate(self._facts):
            if idx in keep_indices:
                new_facts.append(fact)
                new_embeddings.append(self._embeddings[idx] if idx < len(self._embeddings) else None)
            else:
                removed += 1
        if removed:
            self._facts = new_facts
            self._embeddings = new_embeddings
            self._persist()
        return removed


class ReflectionMemory:
    """Lessons learned stored in a JSON lines or list file."""

    def __init__(self, skills_path: Path) -> None:
        self.skills_path = Path(skills_path).expanduser()
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.skills_path.exists():
            self.skills_path.write_text(json.dumps({"lessons": []}, indent=2))

    def add(self, text: str, *, confidence: Optional[float] = None) -> None:
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        normalized = (text or "").strip()
        if not normalized:
            return
        key = normalized.lower()
        for item in lessons:
            existing = str(item.get("text", "")).strip().lower()
            if existing == key:
                return
        lesson: dict[str, Any] = {"ts": time.time(), "text": normalized}
        if confidence is not None:
            try:
                lesson["confidence"] = float(confidence)
            except (TypeError, ValueError):
                pass
        lessons.append(lesson)
        self.skills_path.write_text(json.dumps({"lessons": lessons}, indent=2))

    def recent(self, n: int = 5) -> List[dict]:
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        return lessons[-n:]

    def all(self) -> List[dict]:
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        return lessons

    def prune(self, max_items: int, *, keep_tail: Optional[int] = None) -> int:
        if max_items <= 0:
            total = 0
            try:
                data = json.loads(self.skills_path.read_text())
                lessons = data.get("lessons", [])
                total = len(lessons)
            except Exception:
                lessons = []
            self.skills_path.write_text(json.dumps({"lessons": []}, indent=2))
            return total
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        if keep_tail is not None:
            max_items = max(max_items, keep_tail)
        if len(lessons) <= max_items:
            return 0
        lessons.sort(key=lambda item: float(item.get("ts", 0.0)))
        removed = len(lessons) - max_items
        retained = lessons[-max_items:]
        self.skills_path.write_text(json.dumps({"lessons": retained}, indent=2))
        return removed


@dataclass
class LayeredMemoryConfig:
    base_dir: Path = Path(os.getenv("ATLAS_MEMORY_DIR", "~/.atlas/memory")).expanduser()
    embed_model: str = os.getenv("ATLAS_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
    episodic_filename: str = "episodes.sqlite3"
    semantic_filename: str = "semantic.json"
    reflections_filename: str = "reflections.json"
    summary_model: Optional[str] = None
    memory_model: Optional[str] = None
    min_fact_confidence: Optional[float] = None
    min_reflection_confidence: Optional[float] = None
    prune_semantic_max_items: int = 400
    prune_reflections_max_items: int = 200
    max_episodic_records: int = 2000
    k_ep: int = 3
    k_facts: int = 3
    k_reflections: int = 3
    summary_style: str = "bullets"

    episodic_path: Path = field(init=False)
    semantic_path: Path = field(init=False)
    reflections_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_path = self.base_dir / self.episodic_filename
        self.semantic_path = self.base_dir / self.semantic_filename
        self.reflections_path = self.base_dir / self.reflections_filename
        if not self.summary_model:
            configured = os.getenv("ATLAS_SUMMARY_MODEL")
            fallback = "phi3:latest"
            self.summary_model = (configured or fallback).strip() or fallback
        if not self.memory_model:
            configured = (
                os.getenv("ATLAS_MEMORY_MODEL")
                or os.getenv("ATLAS_FACT_MODEL")
            )
            fallback = self.summary_model or "phi3:latest"
            self.memory_model = (configured or fallback).strip() or fallback
        if self.min_fact_confidence is None:
            self.min_fact_confidence = self._float_env("ATLAS_MEMORY_MIN_FACT_CONF", default=0.6)
        if self.min_reflection_confidence is None:
            self.min_reflection_confidence = self._float_env("ATLAS_MEMORY_MIN_REFL_CONF", default=0.5)

    @staticmethod
    def _float_env(name: str, *, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return default


@dataclass
class LayeredMemorySnapshot:
    summary: str
    rendered: str
    assembled: AssembledContext


class LayeredMemoryManager:
    """Coordinates layered memory recall, rendering, and logging."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        *,
        config: Optional[LayeredMemoryConfig] = None,
    ) -> None:
        self.config = config or LayeredMemoryConfig()
        self.embed_fn = embed_fn
        self._debug_enabled = self._env_truth("ATLAS_MEMORY_DEBUG")
        self.episodic = EpisodicSQLiteMemory(
            self.config.episodic_path,
            embed_fn=embed_fn,
            max_records=self.config.max_episodic_records,
        )
        self.semantic = SemanticMemory(self.config.semantic_path, embed_fn=embed_fn)
        self.reflections = ReflectionMemory(self.config.reflections_path)
        self.assembler = ContextAssembler(
            episodic=self.episodic,
            semantic=self.semantic,
            reflections=self.reflections,
        )
        self._stats: dict[str, dict[str, float]] = {
            "harvest": {
                "attempts": 0,
                "parse_failures": 0,
                "accepted_facts": 0,
                "accepted_reflections": 0,
                "rejected_low_confidence": 0,
            },
            "prune": {
                "runs": 0,
                "semantic_removed": 0,
                "reflections_removed": 0,
                "auto_invocations": 0,
                "reviews": 0,
                "review_rescued": 0,
            },
        }
        self._turns_since_prune = 0
        self._debug(
            "LayeredMemoryManager initialized (base_dir=%s, embed_model=%s)",
            self.config.base_dir,
            self.config.embed_model,
        )

    def log_interaction(self, user: str, assistant: str) -> None:
        try:
            self.episodic.log(user, assistant)
            self._debug(
                "Logged interaction to episodic memory (user='%s...', assistant='%s...')",
                (user or "").strip()[:40],
                (assistant or "").strip()[:40],
            )
        except Exception:
            pass

    def process_turn(self, user: str, assistant: str, *, client: Any | None = None) -> None:
        """Persist the exchange and optionally harvest long-term insights."""
        self.log_interaction(user, assistant)
        if client is not None:
            self._harvest_long_term_layers(user, assistant, client=client)
            self._maybe_auto_prune()

    def assemble(self, query: str) -> AssembledContext:
        self._debug("Assembling memory context for query='%s'", query.strip()[:60])
        assembled = self.assembler.assemble(
            query,
            k_ep=self.config.k_ep,
            k_facts=self.config.k_facts,
            k_lessons=self.config.k_reflections,
        )
        meta = self.assembler.last_metadata
        episodic_meta = meta.get("episodic", [])
        semantic_meta = meta.get("semantic", [])
        reflection_meta = meta.get("reflections", [])
        if episodic_meta:
            preview = "; ".join(
                f"{item['score']:.3f}: {item['text']}" for item in episodic_meta
            )
            self._debug("Episodic recall hits (%d): %s", len(episodic_meta), preview)
        else:
            self._debug("Episodic recall returned no vector matches")
        if semantic_meta:
            preview = "; ".join(
                f"{item['score']:.3f}: {item['text']}" for item in semantic_meta
            )
            self._debug("Semantic recall hits (%d): %s", len(semantic_meta), preview)
        else:
            self._debug("Semantic recall returned no matches")
        if reflection_meta:
            preview = "; ".join(item["text"] for item in reflection_meta)
            self._debug("Reflections surfaced (%d): %s", len(reflection_meta), preview)
        else:
            self._debug("No reflections available")

        if not assembled.episodic:
            fallback = []
            for item in self.episodic.recent(self.config.k_ep):
                text = (item.get("assistant") or item.get("user") or "").strip().replace("\n", " ")
                if text:
                    fallback.append(f"- {text[:200]}")
            if fallback:
                self._debug(
                    "Using recency fallback for episodic layer (count=%d)",
                    len(fallback),
                )
                assembled = AssembledContext(episodic=fallback, facts=assembled.facts, reflections=assembled.reflections)
        if not assembled.facts:
            facts = []
            for fact in self.semantic.head(self.config.k_facts):
                text = str(fact.get("text", "")).strip().replace("\n", " ")
                if text:
                    facts.append(f"- {text[:200]}")
            if facts:
                self._debug(
                    "Using head fallback for semantic layer (count=%d)",
                    len(facts),
                )
                assembled = AssembledContext(episodic=assembled.episodic, facts=facts, reflections=assembled.reflections)
        return assembled

    def render(self, assembled: AssembledContext) -> str:
        return self.assembler.render(assembled)

    def summarize(self, assembled: AssembledContext, *, client: Any) -> str:
        if not hasattr(client, "chat"):
            return ""
        try:
            summary = summarize_assembled_context_abstractive(
                assembled,
                client,
                style=self.config.summary_style,
                model=self.config.summary_model,
            )
            self._debug("Generated memory summary (%d chars)", len(summary))
            return summary
        except Exception:
            self._debug("Summary generation failed", exc_info=True)
            return ""

    def build_snapshot(self, query: str, *, client: Any) -> LayeredMemorySnapshot:
        assembled = self.assemble(query)
        rendered = self.render(assembled)
        summary = ""
        if assembled.episodic or assembled.facts or assembled.reflections:
            summary = self.summarize(assembled, client=client)
        else:
            self._debug("No memory content available for snapshot")
        return LayeredMemorySnapshot(summary=summary.strip(), rendered=rendered.strip(), assembled=assembled)

    def _debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        if self._debug_enabled:
            LOGGER.info(message, *args, **kwargs)

    @staticmethod
    def _env_truth(name: str) -> bool:
        value = os.getenv(name)
        if value is None:
            return False
        normalized = value.strip().lower()
        return normalized not in {"", "0", "false", "off"}

    def _harvest_long_term_layers(self, user: str, assistant: str, *, client: Any) -> None:
        self._stats["harvest"]["attempts"] += 1
        if not hasattr(client, "chat"):
            return
        model = (self.config.memory_model or "phi3:latest").strip()
        messages = self._build_insight_messages(user, assistant)
        try:
            response = client.chat(model=model, messages=messages, stream=False)
        except Exception:
            self._debug("Harvesting long-term memory failed", exc_info=True)
            self._stats["harvest"]["parse_failures"] += 1
            return

        content = self._extract_message_content(response)
        payload = self._parse_insight_payload(content)
        if not isinstance(payload, dict):
            self._stats["harvest"]["parse_failures"] += 1
            return

        facts_raw = payload.get("facts", [])
        lessons_raw = payload.get("reflections", payload.get("lessons", []))
        fact_items = self._normalize_harvest_items(
            facts_raw,
            min_confidence=self.config.min_fact_confidence or 0.0,
            kind="fact",
        )
        lesson_items = self._normalize_harvest_items(
            lessons_raw,
            min_confidence=self.config.min_reflection_confidence or 0.0,
            kind="reflection",
        )

        max_facts = self._safe_int_env("ATLAS_MEMORY_MAX_FACTS", 5)
        max_reflections = self._safe_int_env("ATLAS_MEMORY_MAX_REFLECTIONS", 3)
        fact_items = fact_items[:max(0, max_facts)]
        lesson_items = lesson_items[:max(0, max_reflections)]

        added_facts = 0
        if fact_items:
            added_facts = self.semantic.extend_facts(fact_items, source="conversation")
            self._stats["harvest"]["accepted_facts"] += added_facts
        added_reflections = 0
        if lesson_items:
            before = len(lesson_items)
            for lesson in lesson_items:
                self.reflections.add(lesson["text"], confidence=lesson.get("confidence"))
            added_reflections = before
            self._stats["harvest"]["accepted_reflections"] += added_reflections

        if added_facts or added_reflections:
            self._debug(
                "Harvested long-term memory (facts=%d, reflections=%d)",
                added_facts,
                added_reflections,
            )

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return a deep copy of harvest/prune metrics for CLI display."""
        return deepcopy(self._stats)

    def prune_long_term(
        self,
        *,
        semantic_limit: Optional[int] = None,
        reflections_limit: Optional[int] = None,
        review_client: Any | None = None,
    ) -> dict[str, int]:
        """Prune long-term stores, optionally consulting an LLM before removals."""
        result = {"semantic_removed": 0, "reflections_removed": 0}
        semantic_limit = semantic_limit if semantic_limit is not None else self.config.prune_semantic_max_items
        reflections_limit = (
            reflections_limit
            if reflections_limit is not None
            else self.config.prune_reflections_max_items
        )

        if semantic_limit is not None and semantic_limit >= 0:
            keep_indices = None
            if review_client is not None:
                keep_indices = self._review_semantic_keep_indices(semantic_limit, review_client)
            removed_semantic = self.semantic.prune(semantic_limit, keep_indices=keep_indices)
            result["semantic_removed"] = int(removed_semantic)

        if reflections_limit is not None and reflections_limit >= 0:
            keep_tail = None
            if review_client is not None:
                keep_tail = self._review_reflection_tail(reflections_limit, review_client)
            removed_reflections = self.reflections.prune(reflections_limit, keep_tail=keep_tail)
            result["reflections_removed"] = int(removed_reflections)

        if result["semantic_removed"] or result["reflections_removed"]:
            self._stats["prune"]["runs"] += 1
            self._stats["prune"]["semantic_removed"] += result["semantic_removed"]
            self._stats["prune"]["reflections_removed"] += result["reflections_removed"]
            self._debug(
                "Manual prune removed semantic=%d reflections=%d",
                result["semantic_removed"],
                result["reflections_removed"],
            )
        return result

    def _maybe_auto_prune(self) -> None:
        """Run lightweight deterministic pruning when stores exceed configured caps."""
        semantic_limit = self.config.prune_semantic_max_items
        reflections_limit = self.config.prune_reflections_max_items
        removed_semantic = 0
        removed_reflections = 0
        if semantic_limit and semantic_limit > 0 and len(self.semantic._facts) > semantic_limit:
            removed_semantic = self.semantic.prune(semantic_limit)
        if reflections_limit and reflections_limit > 0:
            total_reflections = len(self.reflections.all())
            if total_reflections > reflections_limit:
                removed_reflections = self.reflections.prune(reflections_limit)
        if removed_semantic or removed_reflections:
            self._stats["prune"]["runs"] += 1
            self._stats["prune"]["auto_invocations"] += 1
            self._stats["prune"]["semantic_removed"] += removed_semantic
            self._stats["prune"]["reflections_removed"] += removed_reflections
            self._debug(
                "Auto-pruned memories (semantic=%d, reflections=%d)",
                removed_semantic,
                removed_reflections,
            )

    def _review_semantic_keep_indices(self, limit: int, client: Any) -> Optional[set[int]]:
        facts = self.semantic._facts
        if not hasattr(client, "chat") or len(facts) <= limit:
            return None
        indexed = list(enumerate(facts))
        indexed.sort(key=lambda pair: self._semantic_sort_key(pair[1]), reverse=True)
        keep_indices = {idx for idx, _ in indexed[:limit]}
        drop_candidates = [(idx, facts[idx]) for idx, _ in indexed[limit:]]
        rescued = self._review_candidates(client=client, kind="semantic", candidates=drop_candidates)
        if rescued:
            keep_indices.update(rescued)
            ordered = sorted(keep_indices, key=lambda i: self._semantic_sort_key(facts[i]), reverse=True)
            keep_indices = set(ordered[:limit])
            self._stats["prune"]["review_rescued"] += len(rescued)
        if rescued is not None:
            self._stats["prune"]["reviews"] += 1
        return keep_indices

    def _review_reflection_tail(self, limit: int, client: Any) -> Optional[int]:
        lessons = self.reflections.all()
        if not hasattr(client, "chat") or len(lessons) <= limit:
            return None
        indexed = list(enumerate(lessons))
        indexed.sort(key=lambda pair: float(pair[1].get("ts", 0.0)))
        drop_candidates = indexed[:-limit]
        rescued = self._review_candidates(client=client, kind="reflections", candidates=drop_candidates)
        if rescued:
            # Ensure rescued tail entries survive by extending tail length by rescued count
            self._stats["prune"]["review_rescued"] += len(rescued)
            self._stats["prune"]["reviews"] += 1
            return min(limit + len(rescued), len(lessons))
        if rescued is not None:
            self._stats["prune"]["reviews"] += 1
        return None

    def _review_candidates(
        self,
        *,
        client: Any,
        kind: str,
        candidates: list[tuple[int, dict]],
    ) -> Optional[set[int]]:
        if not candidates or not hasattr(client, "chat"):
            return None
        try:
            lines = []
            for idx, payload in candidates:
                text = str(payload.get("text", "")).strip()
                conf = payload.get("confidence")
                source = payload.get("source")
                fragment = text or json.dumps(payload)
                meta_bits = []
                if conf is not None:
                    meta_bits.append(f"confidence={conf}")
                if source:
                    meta_bits.append(f"source={source}")
                suffix = f" ({', '.join(meta_bits)})" if meta_bits else ""
                lines.append(f"{idx}: {fragment}{suffix}")
            instruction = (
                "You review candidate long-term memory entries marked for deletion. "
                "Reply with JSON specifying which indices to KEEP, using {\"keep\": [..]}."
            )
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Candidates:\n" + "\n".join(lines)},
            ]
            response = client.chat(model=self.config.memory_model, messages=messages, stream=False)
            content = self._extract_message_content(response)
            data = self._parse_insight_payload(content)
            raw_keep = data.get("keep") or data.get("rescue") or []
            keep_indices: set[int] = set()
            for value in raw_keep:
                try:
                    keep_indices.add(int(value))
                except (TypeError, ValueError):
                    continue
            return keep_indices if keep_indices else None
        except Exception:
            self._debug("Review request for %s pruning failed", kind, exc_info=True)
            return None

    def _semantic_sort_key(self, fact: dict[str, Any]) -> tuple[float, float]:
        try:
            confidence = float(fact.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            ts = float(fact.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        return (confidence, ts)

    def _normalize_harvest_items(
        self,
        raw_items: Iterable[Any],
        *,
        min_confidence: float,
        kind: str,
    ) -> List[dict[str, Any]]:
        normalized: List[dict[str, Any]] = []
        minimum = max(0.0, float(min_confidence))
        for item in raw_items:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                confidence = item.get("confidence")
                source = item.get("source")
            else:
                text = str(item).strip()
                confidence = None
                source = None
            if not text:
                continue
            conf_val = None
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                except (TypeError, ValueError):
                    conf_val = None
            if conf_val is None:
                conf_val = 1.0  # Assume high confidence when model omits the field
            if conf_val < minimum:
                self._stats["harvest"]["rejected_low_confidence"] += 1
                continue
            entry: dict[str, Any] = {"text": text, "confidence": conf_val}
            if source:
                entry["source"] = str(source)
            normalized.append(entry)
        return normalized

    @staticmethod
    def _safe_int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    @staticmethod
    def _build_insight_messages(user: str, assistant: str) -> List[dict[str, str]]:
        instruction = (
            "You analyze the most recent exchange between a user and an assistant. "
            "Extract durable semantic facts about the user, their projects, goals, or environment, "
            "and capture actionable reflections or lessons for the assistant. "
            "Respond in compact JSON with two arrays: \"facts\" and \"reflections\"."
        )
        turn = (
            f"User said:\n{user.strip()}\n\n"
            f"Assistant replied:\n{assistant.strip()}"
        )
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": turn},
        ]

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message") or {}
            content = message.get("content") or response.get("response")
            if content:
                return str(content)
            choices = response.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    msg = choice.get("message") if isinstance(choice, dict) else None
                    if msg and msg.get("content"):
                        return str(msg["content"])
        elif isinstance(response, str):
            return response
        return ""

    @staticmethod
    def _parse_insight_payload(text: str) -> dict[str, Any]:
        snippet = text.strip()
        if not snippet:
            return {}
        fence = re.search(r"```json\s*([\s\S]*?)```", snippet, re.IGNORECASE)
        if fence:
            snippet = fence.group(1).strip()
        else:
            fence = re.search(r"```\s*([\s\S]*?)```", snippet)
            if fence:
                snippet = fence.group(1).strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", snippet)
        if match:
            fragment = match.group(0)
            try:
                parsed = json.loads(fragment)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}


@dataclass
class AssembledContext:
    episodic: List[str]
    facts: List[str]
    reflections: List[str]


class ContextAssembler:
    """Collect relevant snippets from each memory layer and render text."""

    def __init__(
        self,
        episodic: Optional[EpisodicSQLiteMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        reflections: Optional[ReflectionMemory] = None,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.reflections = reflections
        self._last_metadata: dict[str, Any] = {}

    def assemble(self, query: str, *, k_ep: int = 3, k_facts: int = 3, k_lessons: int = 3) -> AssembledContext:
        episodic_snips: List[str] = []
        episodic_meta: List[dict[str, Any]] = []
        if self.episodic:
            for score, rec in self.episodic.recall(query, top_k=k_ep):
                text = (rec.get("assistant") or rec.get("user") or "").strip().replace("\n", " ")
                if text:
                    episodic_snips.append(f"- {text[:200]}")
                    episodic_meta.append({
                        "score": float(score),
                        "text": text[:120],
                    })

        fact_snips: List[str] = []
        fact_meta: List[dict[str, Any]] = []
        if self.semantic:
            for score, fact in self.semantic.recall(query, top_k=k_facts):
                text = str(fact.get("text", "")).strip().replace("\n", " ")
                if text:
                    fact_snips.append(f"- {text[:200]}")
                    fact_meta.append({
                        "score": float(score),
                        "text": text[:120],
                    })

        lesson_snips: List[str] = []
        lesson_meta: List[dict[str, Any]] = []
        if self.reflections:
            for item in self.reflections.recent(k_lessons):
                text = str(item.get("text", "")).strip().replace("\n", " ")
                if text:
                    lesson_snips.append(f"- {text[:200]}")
                    lesson_meta.append({
                        "text": text[:120],
                    })

        self._last_metadata = {
            "episodic": episodic_meta,
            "semantic": fact_meta,
            "reflections": lesson_meta,
        }
        return AssembledContext(episodic=episodic_snips, facts=fact_snips, reflections=lesson_snips)

    def render(self, assembled: AssembledContext) -> str:
        parts: List[str] = []
        if assembled.episodic:
            parts.append("Relevant episodes:\n" + "\n".join(assembled.episodic))
        if assembled.facts:
            parts.append("Relevant facts:\n" + "\n".join(assembled.facts))
        if assembled.reflections:
            parts.append("Relevant reflections:\n" + "\n".join(assembled.reflections))
        return "\n\n".join(parts)

    @property
    def last_metadata(self) -> dict[str, Any]:
        return self._last_metadata


def summarize_assembled_context_abstractive(
    assembled: AssembledContext,
    client: Any,
    *,
    model: Optional[str] = None,
    style: str = "bullets",
) -> str:
    """Summarize a combined context (episodes, facts, reflections) via local LLM.

    - model defaults to env ATLAS_SUMMARY_MODEL or 'phi3:latest'
    - style: 'bullets' or 'paragraph'
    """
    chosen_model = (model or os.getenv("ATLAS_SUMMARY_MODEL") or "phi3:latest").strip()

    parts: List[str] = []
    if assembled.episodic:
        parts.append("Episodes:\n" + "\n".join(assembled.episodic))
    if assembled.facts:
        parts.append("Facts:\n" + "\n".join(assembled.facts))
    if assembled.reflections:
        parts.append("Reflections:\n" + "\n".join(assembled.reflections))

    if not parts:
        return "(no context to summarize)"
    doc = "\n\n".join(parts)

    instruction = (
        "You are a concise summarizer. Given episodes, facts, and reflections, produce a short, "
        "high-signal summary that captures goals, tasks, decisions, lessons, and outcomes."
    )
    if style == "bullets":
        instruction += " Respond with 3-6 bullet points."
    else:
        instruction += " Respond with a 2-4 sentence paragraph."

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": doc},
    ]

    try:
        resp = client.chat(model=chosen_model, messages=messages, stream=False)
    except Exception as exc:  # pragma: no cover
        return f"(summary failed: {exc})"

    content = ""
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        content = msg.get("content") or resp.get("response") or ""
    return content.strip() or "(empty summary)"
