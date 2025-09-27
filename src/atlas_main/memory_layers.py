"""Layered memory for Jarvis-like agent: episodic, semantic, reflections, assembly.

All components are local and optional. Designed to run without external services.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EmbedFn = Callable[[str], Optional[Sequence[float]]]


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

    def assemble(self, query: str, *, k_ep: int = 3, k_facts: int = 3, k_lessons: int = 3) -> AssembledContext:
        episodic_snips: List[str] = []
        if self.episodic:
            for score, rec in self.episodic.recall(query, top_k=k_ep):
                text = (rec.get("assistant") or rec.get("user") or "").strip().replace("\n", " ")
                if text:
                    episodic_snips.append(f"- {text[:200]}")

        fact_snips: List[str] = []
        if self.semantic:
            for score, fact in self.semantic.recall(query, top_k=k_facts):
                text = str(fact.get("text", "")).strip().replace("\n", " ")
                if text:
                    fact_snips.append(f"- {text[:200]}")

        lesson_snips: List[str] = []
        if self.reflections:
            for item in self.reflections.recent(k_lessons):
                text = str(item.get("text", "")).strip().replace("\n", " ")
                if text:
                    lesson_snips.append(f"- {text[:200]}")

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
