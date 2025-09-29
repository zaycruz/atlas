"""Layered memory for Jarvis-like agent: episodic, semantic, reflections, assembly.

All components are local and optional. Designed to run without external services.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import logging
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
            "INSERT OR REPLACE INTO episodes (id, ts, user, assistant, embedding) VALUES (?, ?, ?, ?, ?)",
            (str(ts), ts, user, assistant, json.dumps(emb) if emb is not None else None),
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
            data = json.loads(self.json_path.read_text())
            facts = data if isinstance(data, list) else data.get("facts", [])
        except Exception:
            facts = []
        self._facts = [f for f in facts if isinstance(f, dict) and f.get("text")]
        self._embeddings = [None] * len(self._facts)

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


class ReflectionMemory:
    """Lessons learned stored in a JSON lines or list file."""

    def __init__(self, skills_path: Path) -> None:
        self.skills_path = Path(skills_path).expanduser()
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.skills_path.exists():
            self.skills_path.write_text(json.dumps({"lessons": []}, indent=2))

    def add(self, text: str) -> None:
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        lessons.append({"ts": time.time(), "text": text})
        self.skills_path.write_text(json.dumps({"lessons": lessons}, indent=2))

    def recent(self, n: int = 5) -> List[dict]:
        try:
            data = json.loads(self.skills_path.read_text())
            lessons = data.get("lessons", [])
        except Exception:
            lessons = []
        return lessons[-n:]


@dataclass
class LayeredMemoryConfig:
    base_dir: Path = Path(os.getenv("ATLAS_MEMORY_DIR", "~/.atlas/memory")).expanduser()
    embed_model: str = os.getenv("ATLAS_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
    episodic_filename: str = "episodes.sqlite3"
    semantic_filename: str = "semantic.json"
    reflections_filename: str = "reflections.json"
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
