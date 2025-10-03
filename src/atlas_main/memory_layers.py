"""Layered memory for Jarvis-like agent: episodic, semantic, reflections, assembly.

All components are local and optional. Designed to run without external services.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
import logging
from collections import deque
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


def _quality_features(text: str) -> tuple[float, dict[str, float]]:
    """Heuristic quality score inspired by MemGPT/Letta retention gating."""

    normalized = " ".join(text.strip().split())
    if not normalized:
        return 0.0, {
            "length": 0.0,
            "tokens": 0.0,
            "diversity": 0.0,
            "structure": 0.0,
            "actionability": 0.0,
        }

    tokens = re.findall(r"[\w'-]+", normalized.lower())
    token_count = len(tokens)
    unique_tokens = len(set(tokens))
    char_count = len(normalized)
    avg_token_len = (char_count / token_count) if token_count else 0.0

    diversity = (unique_tokens / max(1, token_count))
    structure = 0.0
    if re.search(r"[.;:?!]", normalized):
        structure += 0.6
    if re.search(r"\b(and|but|because|so|then|while)\b", normalized):
        structure += 0.4
    structure = min(structure, 1.0)

    verbs = {
        "am",
        "are",
        "is",
        "was",
        "were",
        "be",
        "being",
        "been",
        "plan",
        "plans",
        "planning",
        "should",
        "need",
        "needs",
        "prefer",
        "prefers",
        "remember",
        "remind",
        "track",
        "tracks",
        "monitor",
        "monitors",
        "offer",
        "offers",
        "recommend",
        "recommends",
        "suggest",
        "suggests",
        "ensure",
        "ensures",
        "consider",
        "considers",
        "prioritize",
        "prioritizes",
        "maintain",
        "maintains",
        "focus",
        "focusing",
        "improve",
        "improves",
        "learn",
        "learns",
        "learned",
        "avoid",
        "avoids",
        "want",
        "wants",
        "use",
        "uses",
        "update",
        "updates",
    }
    has_verb = any(token in verbs for token in tokens)
    has_subject = bool(re.search(r"\b(i|we|user|assistant|project|system|task|goal|plan)\b", normalized.lower()))
    has_numbers = bool(re.search(r"\d", normalized))
    imperative = bool(
        re.match(
            r"\b(remember|ensure|offer|recommend|suggest|consider|prioritize|avoid|focus|track|monitor|document|confirm|ask|review|summarize|plan|schedule|double-check)\b",
            normalized.lower(),
        )
    )

    length_score = min(1.0, char_count / 120.0)
    token_score = min(1.0, token_count / 18.0)
    diversity_score = min(1.0, diversity * 1.3)
    structure_score = structure
    actionability = 0.0
    if has_verb:
        actionability += 0.6
    if has_subject:
        actionability += 0.25
    if has_numbers:
        actionability += 0.15
    if imperative:
        actionability = max(actionability, 0.55)
    actionability = min(1.0, actionability)

    score = (
        0.25 * length_score
        + 0.2 * token_score
        + 0.15 * diversity_score
        + 0.15 * structure_score
        + 0.25 * actionability
    )
    score = max(0.0, min(1.0, score + max(0.0, min(avg_token_len / 8.0, 0.1))))

    features = {
        "length": length_score,
        "tokens": token_score,
        "diversity": diversity_score,
        "structure": structure_score,
        "actionability": actionability,
    }
    return score, features


class KnowledgeGraph:
    """Lightweight adjacency map relating stored facts."""

    def __init__(self, graph_path: Path) -> None:
        self.graph_path = Path(graph_path).expanduser()
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._edges: dict[str, list[dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.graph_path.exists():
            self._edges = {}
            return
        try:
            data = json.loads(self.graph_path.read_text())
            edges = data.get("edges", {}) if isinstance(data, dict) else {}
            normalized: dict[str, list[dict[str, str]]] = {}
            for node, links in edges.items():
                if not isinstance(links, list):
                    continue
                cleaned: list[dict[str, str]] = []
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    target = str(link.get("target", "")).strip()
                    rel = str(link.get("type", "")).strip() or "related"
                    if not target:
                        continue
                    cleaned.append({"target": target, "type": rel})
                if cleaned:
                    normalized[str(node)] = cleaned
            self._edges = normalized
        except Exception:
            self._edges = {}

    def _persist(self) -> None:
        payload = {"edges": self._edges}
        try:
            self.graph_path.write_text(json.dumps(payload, indent=2))
        except Exception:
            LOGGER.debug("Failed to persist knowledge graph", exc_info=True)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = {token for token in re.findall(r"[A-Za-z0-9']+", text.lower()) if len(token) > 2}
        return tokens

    @staticmethod
    def _detect_negation(text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in {" not ", " no ", " never ", " without "})

    @staticmethod
    def _detect_dependency(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in {"requires", "depends", "needs", "prerequisite"})

    def _infer_relation(self, text_a: str, text_b: str) -> Optional[str]:
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return None
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union == 0:
            return None
        jaccard = overlap / union
        if jaccard < 0.3:
            return None
        neg_a = self._detect_negation(text_a)
        neg_b = self._detect_negation(text_b)
        if neg_a != neg_b and jaccard >= 0.35:
            return "contradicts"
        if self._detect_dependency(text_a) or self._detect_dependency(text_b):
            return "depends"
        return "related"

    @staticmethod
    def _dedupe(edges: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        result: list[dict[str, str]] = []
        for edge in edges:
            target = edge.get("target")
            rel = edge.get("type") or "related"
            if not target:
                continue
            key = (target, rel)
            if key in seen:
                continue
            seen.add(key)
            result.append({"target": target, "type": rel})
        return result

    def rebuild(self, facts: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        fact_list = [fact for fact in facts if fact.get("id")]
        adjacency: dict[str, list[dict[str, str]]] = {
            str(fact["id"]): [] for fact in fact_list
        }
        for idx, fact in enumerate(fact_list):
            text_a = str(fact.get("text", ""))
            id_a = str(fact.get("id"))
            for other in fact_list[idx + 1 :]:
                text_b = str(other.get("text", ""))
                id_b = str(other.get("id"))
                relation = self._infer_relation(text_a, text_b)
                if not relation:
                    continue
                adjacency[id_a].append({"target": id_b, "type": relation})
                adjacency[id_b].append({"target": id_a, "type": relation})
        for node, edges in adjacency.items():
            adjacency[node] = self._dedupe(edges)
        self._edges = adjacency
        self._persist()
        return {node: list(edges) for node, edges in adjacency.items()}

    def remove(self, fact_ids: Iterable[str]) -> None:
        removal = {str(fid) for fid in fact_ids}
        if not removal:
            return
        new_edges: dict[str, list[dict[str, str]]] = {}
        for node, edges in self._edges.items():
            if node in removal:
                continue
            filtered = [edge for edge in edges if edge.get("target") not in removal]
            new_edges[node] = filtered
        self._edges = new_edges
        self._persist()

    def links_for(self, fact_id: str) -> list[dict[str, str]]:
        return [dict(edge) for edge in self._edges.get(str(fact_id), [])]

class EpisodicSQLiteMemory:
    """Lightweight episodic memory using SQLite and local embeddings."""

    def __init__(self, db_path: Path, embed_fn: Optional[EmbedFn] = None, max_records: int = 2000) -> None:
        self.db_path = Path(db_path).expanduser()
        self.embed_fn = embed_fn
        self.max_records = max_records
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            LOGGER.debug("Failed to close episodic SQLite connection", exc_info=True)
        finally:
            self._conn = None

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                user TEXT,
                assistant TEXT,
                embedding TEXT,
                metadata TEXT
            );
            """
        )
        try:
            cur.execute("PRAGMA table_info(episodes)")
            columns = {row[1] for row in cur.fetchall()}
            if "metadata" not in columns:
                cur.execute("ALTER TABLE episodes ADD COLUMN metadata TEXT")
        except Exception:
            LOGGER.debug("Failed to ensure episodic metadata column", exc_info=True)
        self._conn.commit()

    def log(self, user: str, assistant: str, *, metadata: Optional[dict[str, Any]] = None) -> None:
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
            "INSERT INTO episodes (id, ts, user, assistant, embedding, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                ts,
                user,
                assistant,
                json.dumps(emb) if emb is not None else None,
                json.dumps(metadata, ensure_ascii=False) if metadata else None,
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
        cur.execute("SELECT id, ts, user, assistant, embedding, metadata FROM episodes")
        rows = cur.fetchall()
        scored: List[Tuple[float, dict]] = []
        for _id, ts, user, assistant, emb_json, metadata_json in rows:
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
            meta: dict[str, Any] = {}
            if metadata_json:
                try:
                    parsed_meta = json.loads(metadata_json)
                    if isinstance(parsed_meta, dict):
                        meta = parsed_meta
                except Exception:
                    meta = {}
            scored.append(
                (
                    s,
                    {
                        "ts": ts,
                        "user": user or "",
                        "assistant": assistant or "",
                        "metadata": meta,
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def recent(self, top_k: int = 3) -> List[dict]:
        if top_k <= 0:
            return []
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, user, assistant, metadata FROM episodes ORDER BY ts DESC LIMIT ?",
            (int(top_k),),
        )
        rows = cur.fetchall()
        recent: List[dict] = []
        for ts, user, assistant, metadata in rows:
            meta = None
            if metadata:
                try:
                    meta = json.loads(metadata)
                except Exception:
                    meta = None
            recent.append(
                {
                    "ts": ts,
                    "user": user or "",
                    "assistant": assistant or "",
                    "metadata": meta or {},
                }
            )
        return recent


class SemanticMemory:
    """Durable facts loaded from a JSON file; optional embeddings for recall."""

    def __init__(
        self,
        json_path: Path,
        embed_fn: Optional[EmbedFn] = None,
        *,
        graph_path: Optional[Path] = None,
    ) -> None:
        self.json_path = Path(json_path).expanduser()
        self.embed_fn = embed_fn
        self.graph = KnowledgeGraph(graph_path or (self.json_path.with_name("knowledge_graph.json")))
        self._facts: List[dict] = []
        self._embeddings: List[Optional[np.ndarray]] = []
        self._load()
        self._last_recalled: List[dict] = []

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
            entry: dict[str, Any] = {
                "id": str(fact.get("id") or uuid.uuid4()),
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
            uses = fact.get("uses")
            try:
                entry["uses"] = max(0, int(uses))
            except (TypeError, ValueError):
                entry["uses"] = 0
            try:
                entry["last_access_ts"] = float(fact.get("last_access_ts", entry["ts"]))
            except (TypeError, ValueError):
                entry["last_access_ts"] = entry["ts"]
            quality = fact.get("quality")
            if quality is not None:
                try:
                    entry["quality"] = float(quality)
                except (TypeError, ValueError):
                    entry["quality"] = _quality_features(text)[0]
            else:
                entry["quality"] = _quality_features(text)[0]
            entry["tags"] = self._normalize_tags(fact.get("tags"))
            entry["links"] = []
            clean_facts.append(entry)
        self._facts = clean_facts
        self._embeddings = [None] * len(self._facts)
        if self._facts:
            self._apply_graph_links()

    def _persist(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"facts": self._facts}
        self.json_path.write_text(json.dumps(payload, indent=2))

    def _apply_graph_links(self) -> None:
        if not self._facts:
            self.graph.remove([])
            self._persist()
            return
        link_map = self.graph.rebuild(self._facts)
        for fact in self._facts:
            fact["links"] = link_map.get(fact["id"], [])
        self._persist()

    def _normalize_tags(self, tags: Any) -> list[str]:
        if not tags:
            return []
        normalized: set[str] = set()
        if isinstance(tags, (list, tuple, set)):
            iterable = tags
        else:
            iterable = [tags]
        for tag in iterable:
            text = str(tag or "").strip().lower()
            if not text:
                continue
            normalized.add(text)
        return sorted(normalized)

    def _find_similar_fact(self, vector: np.ndarray, *, threshold: float = 0.9) -> Optional[int]:
        if vector.size == 0:
            return None
        for idx, existing in enumerate(self._embeddings):
            if existing is None:
                continue
            if existing.shape != vector.shape:
                continue
            score = _cosine(existing, vector)
            if score >= threshold:
                return idx
        return None

    def add_fact(
        self,
        text: str,
        *,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
        quality: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
        update_graph: bool = True,
    ) -> Optional[dict]:
        normalized = (text or "").strip()
        if not normalized:
            return None
        new_tags = self._normalize_tags(tags)
        key = normalized.lower()
        graph_dirty = False
        for fact in self._facts:
            if str(fact.get("text", "")).strip().lower() == key:
                now = time.time()
                fact["ts"] = now
                if new_tags:
                    merged = set(fact.get("tags", [])) | set(new_tags)
                    fact["tags"] = sorted(merged)
                    graph_dirty = True
                if confidence is not None:
                    try:
                        fact["confidence"] = float(confidence)
                    except (TypeError, ValueError):
                        pass
                if quality is not None:
                    try:
                        fact["quality"] = max(float(quality), fact.get("quality", 0.0))
                    except (TypeError, ValueError):
                        pass
                fact["uses"] = int(fact.get("uses", 0))
                fact["last_access_ts"] = now
                self._persist()
                if update_graph and graph_dirty:
                    self._apply_graph_links()
                return fact

        entry: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "text": normalized,
            "ts": time.time(),
            "uses": 0,
            "tags": new_tags,
            "links": [],
        }
        if source:
            entry["source"] = source
        if confidence is not None:
            try:
                entry["confidence"] = float(confidence)
            except (TypeError, ValueError):
                pass
        quality_score = None
        if quality is not None:
            try:
                quality_score = float(quality)
            except (TypeError, ValueError):
                quality_score = None
        if quality_score is None:
            quality_score = _quality_features(normalized)[0]
        entry["quality"] = quality_score
        entry["last_access_ts"] = entry["ts"]
        self._facts.append(entry)
        self._embeddings.append(None)
        graph_dirty = True
        try:
            if self.embed_fn is not None:
                vec = self.embed_fn(normalized)
                if vec:
                    new_vec = np.asarray(vec, dtype=float)
                    duplicate_idx = self._find_similar_fact(new_vec, threshold=0.92)
                    if duplicate_idx is not None:
                        existing = self._facts[duplicate_idx]
                        existing["ts"] = time.time()
                        existing["quality"] = max(existing.get("quality", 0.0), quality_score)
                        if new_tags:
                            merged = set(existing.get("tags", [])) | set(new_tags)
                            existing["tags"] = sorted(merged)
                        if "confidence" in entry and entry["confidence"] is not None:
                            try:
                                existing_conf = float(existing.get("confidence", 0.0))
                            except (TypeError, ValueError):
                                existing_conf = 0.0
                            try:
                                new_conf = float(entry["confidence"])
                            except (TypeError, ValueError):
                                new_conf = existing_conf
                            existing["confidence"] = max(existing_conf, new_conf)
                        existing["uses"] = int(existing.get("uses", 0))
                        existing["last_access_ts"] = time.time()
                        self._facts.pop()
                        self._embeddings.pop()
                        self._persist()
                        if update_graph and graph_dirty:
                            self._apply_graph_links()
                        return existing
                    self._embeddings[-1] = new_vec
        except Exception:
            self._embeddings[-1] = None
        self._persist()
        if update_graph and graph_dirty:
            self._apply_graph_links()
        return entry

    def extend_facts(
        self,
        facts: Iterable[dict[str, Any] | str],
        *,
        source: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        inserted: List[dict] = []
        batch_tags = self._normalize_tags(tags)
        for item in facts:
            if isinstance(item, dict):
                text = str(item.get("text", ""))
                confidence = item.get("confidence")
                quality = item.get("quality")
                item_tags = self._normalize_tags(item.get("tags"))
                merged_tags = sorted(set(batch_tags) | set(item_tags))
                inserted_fact = self.add_fact(
                    text,
                    source=item.get("source") or source,
                    confidence=confidence,
                    quality=quality,
                    tags=merged_tags,
                    update_graph=False,
                )
            else:
                inserted_fact = self.add_fact(
                    str(item),
                    source=source,
                    tags=batch_tags,
                    update_graph=False,
                )
            if inserted_fact:
                inserted.append(inserted_fact)
        if inserted:
            self._apply_graph_links()
        else:
            self._persist()
        return inserted

    def update_fact(
        self,
        fact_id: str,
        *,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        quality: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[dict]:
        for fact in self._facts:
            if fact.get("id") == fact_id:
                if text and text.strip():
                    fact["text"] = text.strip()
                if confidence is not None:
                    try:
                        fact["confidence"] = float(confidence)
                    except (TypeError, ValueError):
                        pass
                if quality is not None:
                    try:
                        fact["quality"] = float(quality)
                    except (TypeError, ValueError):
                        pass
                if tags is not None:
                    fact["tags"] = self._normalize_tags(tags)
                fact["ts"] = time.time()
                fact["last_access_ts"] = fact["ts"]
                self._embeddings = [None] * len(self._facts)
                self._persist()
                self._apply_graph_links()
                return fact
        return None

    def remove_fact(self, fact_id: str) -> bool:
        to_remove = None
        for idx, fact in enumerate(self._facts):
            if fact.get("id") == fact_id:
                to_remove = idx
                break
        if to_remove is None:
            return False
        self._facts.pop(to_remove)
        self._embeddings = [None] * len(self._facts)
        self._persist()
        self._apply_graph_links()
        return True

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[float, dict]]:
        if not self._facts or not self.embed_fn or top_k <= 0:
            return []
        qvec = self.embed_fn(query)
        if not qvec:
            return []
        q = np.asarray(qvec, dtype=float)
        query_tokens = self._tokenize_query(query)
        scored: List[Tuple[float, dict, int]] = []
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
            base = _cosine(q, v)
            bonus = self._tag_bonus(query_tokens, fact.get("tags", []))
            scored.append((base + bonus, fact, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[:top_k]
        if not chosen:
            return []
        now = time.time()
        dirty = False
        for _, _fact, idx in chosen:
            try:
                self._facts[idx]["uses"] = int(self._facts[idx].get("uses", 0)) + 1
            except (TypeError, ValueError):
                self._facts[idx]["uses"] = 1
            self._facts[idx]["last_access_ts"] = now
            dirty = True
        if dirty:
            self._persist()
        result = [(score, dict(fact)) for score, fact, _ in chosen]
        self._last_recalled = [dict(fact) for _, fact, _ in chosen]
        return result

    def head(self, top_k: int = 3) -> List[dict]:
        if top_k <= 0:
            return []
        if not self._facts:
            return []
        now = time.time()
        indices = list(range(len(self._facts)))
        indices.sort(key=lambda idx: self._fact_priority(self._facts[idx], now), reverse=True)
        selected: List[dict] = []
        dirty = False
        for idx in indices[:top_k]:
            fact = self._facts[idx]
            try:
                self._facts[idx]["uses"] = int(self._facts[idx].get("uses", 0)) + 1
            except (TypeError, ValueError):
                self._facts[idx]["uses"] = 1
            self._facts[idx]["last_access_ts"] = now
            dirty = True
            selected.append(dict(fact))
        if dirty:
            self._persist()
        return selected

    def prune(self, max_items: int, *, keep_indices: Optional[set[int]] = None) -> int:
        if max_items <= 0:
            removed_ids = [fact.get("id") for fact in self._facts]
            removed = len(self._facts)
            self._facts = []
            self._embeddings = []
            self._persist()
            if removed_ids:
                self.graph.remove(removed_ids)
            return removed
        if not self._facts or len(self._facts) <= max_items:
            return 0
        indexed = list(enumerate(self._facts))
        indexed.sort(key=lambda pair: self._fact_priority(pair[1]), reverse=True)
        keep = indexed[:max_items]
        keep_set = {idx for idx, _ in keep}
        if keep_indices:
            keep_set.update(idx for idx in keep_indices if 0 <= idx < len(self._facts))
        ordered_keep = sorted(keep_set, key=lambda idx: self._fact_priority(self._facts[idx]), reverse=True)
        retained_indices = ordered_keep[:max_items]
        retained = [self._facts[idx] for idx in sorted(retained_indices)]
        removed_ids = [fact.get("id") for fact in self._facts if fact not in retained]
        removed = len(self._facts) - len(retained)
        self._facts = retained
        self._embeddings = [None] * len(self._facts)
        self._persist()
        if removed_ids:
            self.graph.remove(removed_ids)
        if self._facts:
            self._apply_graph_links()
        return removed

    def get_fact(self, fact_id: str) -> Optional[dict]:
        for fact in self._facts:
            if fact.get("id") == fact_id:
                return dict(fact)
        return None

    @property
    def last_recalled(self) -> List[dict]:
        return [dict(item) for item in self._last_recalled]

    def _fact_priority(self, fact: dict[str, Any], now: Optional[float] = None) -> float:
        now = now or time.time()
        try:
            quality = float(fact.get("quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0
        try:
            confidence = float(fact.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            created_ts = float(fact.get("ts", 0.0))
        except (TypeError, ValueError):
            created_ts = 0.0
        try:
            last_access = float(fact.get("last_access_ts", created_ts))
        except (TypeError, ValueError):
            last_access = created_ts
        try:
            uses = float(fact.get("uses", 0.0))
        except (TypeError, ValueError):
            uses = 0.0

        age_hours = max(0.0, (now - created_ts) / 3600.0)
        recency_hours = max(0.0, (now - last_access) / 3600.0)
        age_decay = math.exp(-age_hours / 96.0)
        recency_decay = math.exp(-recency_hours / 48.0)
        usage_bonus = math.log1p(uses) / 4.0
        tag_bonus = min(0.1, len(fact.get("tags", [])) * 0.02)
        return (
            0.45 * quality
            + 0.1 * confidence
            + 0.25 * age_decay
            + 0.15 * recency_decay
            + usage_bonus
            + tag_bonus
        )

    @staticmethod
    def _tokenize_query(query: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9']+", query.lower()) if len(token) > 2}

    @staticmethod
    def _tag_bonus(query_tokens: set[str], tags: Iterable[str]) -> float:
        if not tags or not query_tokens:
            return 0.0
        matches = 0
        for tag in tags:
            pieces = tag.split(":", 1)
            raw = pieces[-1]
            if any(part in query_tokens for part in raw.split()):
                matches += 1
        return min(0.2, matches * 0.08)


class ReflectionMemory:
    """Lessons learned stored in a JSON lines or list file."""

    def __init__(self, skills_path: Path) -> None:
        self.skills_path = Path(skills_path).expanduser()
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.skills_path.exists():
            self.skills_path.write_text(json.dumps({"lessons": []}, indent=2))
        self._last_recent: List[dict[str, Any]] = []

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if not tags:
            return []
        if isinstance(tags, (list, tuple, set)):
            iterable = tags
        else:
            iterable = [tags]
        normalized: set[str] = set()
        for tag in iterable:
            text = str(tag or "").strip().lower()
            if text:
                normalized.add(text)
        return sorted(normalized)

    def _read_lessons(self) -> List[dict[str, Any]]:
        try:
            data = json.loads(self.skills_path.read_text())
            raw = data.get("lessons", [])
        except Exception:
            raw = []
        lessons: List[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            entry: dict[str, Any] = {"id": str(item.get("id") or uuid.uuid4()), "text": text}
            try:
                entry["ts"] = float(item.get("ts", time.time()))
            except (TypeError, ValueError):
                entry["ts"] = time.time()
            confidence = item.get("confidence")
            if confidence is not None:
                try:
                    entry["confidence"] = float(confidence)
                except (TypeError, ValueError):
                    pass
            uses = item.get("uses")
            try:
                entry["uses"] = max(0, int(uses))
            except (TypeError, ValueError):
                entry["uses"] = 0
            try:
                entry["last_access_ts"] = float(item.get("last_access_ts", entry["ts"]))
            except (TypeError, ValueError):
                entry["last_access_ts"] = entry["ts"]
            quality = item.get("quality")
            if quality is not None:
                try:
                    entry["quality"] = float(quality)
                except (TypeError, ValueError):
                    entry["quality"] = _quality_features(text)[0]
            else:
                entry["quality"] = _quality_features(text)[0]
            entry["tags"] = self._normalize_tags(item.get("tags"))
            lessons.append(entry)
        return lessons

    def _write_lessons(self, lessons: Iterable[dict[str, Any]]) -> None:
        payload = {"lessons": list(lessons)}
        self.skills_path.write_text(json.dumps(payload, indent=2))

    def add(
        self,
        text: str,
        *,
        confidence: Optional[float] = None,
        quality: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        lessons = self._read_lessons()
        key = normalized.lower()
        for lesson in lessons:
            existing = str(lesson.get("text", "")).strip().lower()
            if existing == key:
                now = time.time()
                lesson["ts"] = now
                lesson["last_access_ts"] = now
                lesson["uses"] = int(lesson.get("uses", 0))
                new_tags = self._normalize_tags(tags)
                if new_tags:
                    merged = set(lesson.get("tags", [])) | set(new_tags)
                    lesson["tags"] = sorted(merged)
                if confidence is not None:
                    try:
                        lesson["confidence"] = float(confidence)
                    except (TypeError, ValueError):
                        pass
                if quality is not None:
                    try:
                        lesson["quality"] = max(float(quality), lesson.get("quality", 0.0))
                    except (TypeError, ValueError):
                        pass
                self._write_lessons(lessons)
                return False
        entry: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "text": normalized,
            "uses": 0,
        }
        if confidence is not None:
            try:
                entry["confidence"] = float(confidence)
            except (TypeError, ValueError):
                pass
        if quality is not None:
            try:
                entry["quality"] = float(quality)
            except (TypeError, ValueError):
                entry["quality"] = _quality_features(normalized)[0]
        else:
            entry["quality"] = _quality_features(normalized)[0]
        entry["last_access_ts"] = entry["ts"]
        entry["tags"] = self._normalize_tags(tags)
        lessons.append(entry)
        self._write_lessons(lessons)
        return True

    def update(
        self,
        lesson_id: str,
        *,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        quality: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional[dict[str, Any]]:
        lessons = self._read_lessons()
        updated = None
        for lesson in lessons:
            if lesson.get("id") == lesson_id:
                if text and text.strip():
                    lesson["text"] = text.strip()
                if confidence is not None:
                    try:
                        lesson["confidence"] = float(confidence)
                    except (TypeError, ValueError):
                        pass
                if quality is not None:
                    try:
                        lesson["quality"] = float(quality)
                    except (TypeError, ValueError):
                        pass
                if tags is not None:
                    lesson["tags"] = self._normalize_tags(tags)
                lesson["ts"] = time.time()
                lesson["last_access_ts"] = lesson["ts"]
                lesson["uses"] = int(lesson.get("uses", 0))
                updated = dict(lesson)
                break
        if updated:
            self._write_lessons(lessons)
        return updated

    def remove(self, lesson_id: str) -> bool:
        lessons = self._read_lessons()
        filtered = [lesson for lesson in lessons if lesson.get("id") != lesson_id]
        if len(filtered) == len(lessons):
            return False
        self._write_lessons(filtered)
        return True

    def recent(self, n: int = 5) -> List[dict]:
        if n <= 0:
            return []
        lessons = self._read_lessons()
        if not lessons:
            return []
        now = time.time()
        indexed = list(enumerate(lessons))
        indexed.sort(key=lambda pair: self._lesson_priority(pair[1], now), reverse=True)
        chosen = indexed[:n]
        dirty = False
        for idx, lesson in chosen:
            try:
                lessons[idx]["uses"] = int(lessons[idx].get("uses", 0)) + 1
            except (TypeError, ValueError):
                lessons[idx]["uses"] = 1
            lessons[idx]["last_access_ts"] = now
            dirty = True
        if dirty:
            self._write_lessons(lessons)
        self._last_recent = [dict(lesson) for _, lesson in chosen]
        return [dict(lesson) for _, lesson in chosen]

    def all(self) -> List[dict]:
        return [dict(lesson) for lesson in self._read_lessons()]

    @property
    def last_recent(self) -> List[dict[str, Any]]:
        return [dict(item) for item in self._last_recent]

    def prune(self, max_items: int, *, keep_tail: Optional[int] = None) -> int:
        lessons = self._read_lessons()
        if max_items <= 0:
            removed = len(lessons)
            self._write_lessons([])
            return removed
        if not lessons or len(lessons) <= max_items:
            return 0
        tail_indices: set[int] = set()
        if keep_tail is not None and keep_tail > 0:
            ordered_by_ts = sorted(
                list(enumerate(lessons)),
                key=lambda pair: float(pair[1].get("ts", 0.0)),
            )
            tail_indices = {idx for idx, _ in ordered_by_ts[-keep_tail:]}
        indexed = list(enumerate(lessons))
        indexed.sort(key=lambda pair: self._lesson_priority(pair[1]), reverse=True)
        keep_indices = {idx for idx, _ in indexed[:max_items]}
        keep_indices.update(tail_indices)
        if len(keep_indices) > max_items:
            prioritized = sorted(
                list(keep_indices),
                key=lambda idx: self._lesson_priority(lessons[idx]),
                reverse=True,
            )
            keep_indices = set(prioritized[:max_items])
        retained = [lessons[idx] for idx in sorted(keep_indices, key=lambda i: float(lessons[i].get("ts", 0.0)))]
        removed = len(lessons) - len(retained)
        self._write_lessons(retained)
        return removed

    def _lesson_priority(self, lesson: dict[str, Any], now: Optional[float] = None) -> float:
        now = now or time.time()
        try:
            quality = float(lesson.get("quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0
        try:
            confidence = float(lesson.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            created_ts = float(lesson.get("ts", 0.0))
        except (TypeError, ValueError):
            created_ts = 0.0
        try:
            last_access = float(lesson.get("last_access_ts", created_ts))
        except (TypeError, ValueError):
            last_access = created_ts
        try:
            uses = float(lesson.get("uses", 0.0))
        except (TypeError, ValueError):
            uses = 0.0

        age_hours = max(0.0, (now - created_ts) / 3600.0)
        recency_hours = max(0.0, (now - last_access) / 3600.0)
        age_decay = math.exp(-age_hours / 96.0)  # ~4 day scale
        recency_decay = math.exp(-recency_hours / 48.0)
        usage_bonus = math.log1p(uses) / 4.0
        return (
            0.5 * quality
            + 0.1 * confidence
            + 0.25 * age_decay
            + 0.15 * recency_decay
            + usage_bonus
        )


@dataclass
class LayeredMemoryConfig:
    base_dir: Path = Path(os.getenv("ATLAS_MEMORY_DIR", "~/.atlas/memory")).expanduser()
    embed_model: str = os.getenv("ATLAS_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
    episodic_filename: str = "episodes.sqlite3"
    semantic_filename: str = "semantic.json"
    reflections_filename: str = "reflections.json"
    graph_filename: str = "knowledge_graph.json"
    adaptive_filename: str = "adaptive_thresholds.json"
    summary_model: Optional[str] = None
    memory_model: Optional[str] = None
    critic_model: Optional[str] = None
    critic_enabled: bool = True
    min_fact_confidence: Optional[float] = None
    min_reflection_confidence: Optional[float] = None
    min_fact_quality: Optional[float] = None
    min_reflection_quality: Optional[float] = None
    prune_semantic_max_items: int = 400
    prune_reflections_max_items: int = 200
    max_episodic_records: int = 2000
    k_ep: int = 3
    k_facts: int = 3
    k_reflections: int = 3
    summary_style: str = "bullets"
    audit_interval_turns: int = 12
    audit_window: int = 6
    audit_sample_size: int = 5

    episodic_path: Path = field(init=False)
    semantic_path: Path = field(init=False)
    reflections_path: Path = field(init=False)
    graph_path: Path = field(init=False)
    adaptive_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_path = self.base_dir / self.episodic_filename
        self.semantic_path = self.base_dir / self.semantic_filename
        self.reflections_path = self.base_dir / self.reflections_filename
        self.graph_path = self.base_dir / self.graph_filename
        self.adaptive_path = self.base_dir / self.adaptive_filename
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
        if not self.critic_model:
            configured = os.getenv("ATLAS_MEMORY_CRITIC_MODEL")
            self.critic_model = (configured or self.memory_model or "phi3:latest").strip()
        self.critic_enabled = self._bool_env("ATLAS_MEMORY_CRITIC", default=self.critic_enabled)
        if self.min_fact_confidence is None:
            self.min_fact_confidence = self._float_env("ATLAS_MEMORY_MIN_FACT_CONF", default=0.6)
        if self.min_reflection_confidence is None:
            self.min_reflection_confidence = self._float_env("ATLAS_MEMORY_MIN_REFL_CONF", default=0.5)
        if self.min_fact_quality is None:
            self.min_fact_quality = self._float_env("ATLAS_MEMORY_MIN_FACT_QUALITY", default=0.3)
        if self.min_reflection_quality is None:
            self.min_reflection_quality = self._float_env("ATLAS_MEMORY_MIN_REFL_QUALITY", default=0.35)
        self.audit_interval_turns = max(0, int(os.getenv("ATLAS_MEMORY_AUDIT_INTERVAL", self.audit_interval_turns)))
        self.audit_window = max(1, int(os.getenv("ATLAS_MEMORY_AUDIT_WINDOW", self.audit_window)))
        self.audit_sample_size = max(1, int(os.getenv("ATLAS_MEMORY_AUDIT_SAMPLE", self.audit_sample_size)))

    @staticmethod
    def _float_env(name: str, *, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _bool_env(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
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
        self.semantic = SemanticMemory(
            self.config.semantic_path,
            embed_fn=embed_fn,
            graph_path=self.config.graph_path,
        )
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
                "rejected_low_quality": 0,
                "critic_failures": 0,
                "critic_rejected": 0,
            },
            "prune": {
                "runs": 0,
                "semantic_removed": 0,
                "reflections_removed": 0,
                "auto_invocations": 0,
                "reviews": 0,
                "review_rescued": 0,
            },
            "audit": {
                "runs": 0,
                "facts_reviewed": 0,
                "facts_dropped": 0,
                "facts_updated": 0,
                "reflections_updated": 0,
                "reflections_dropped": 0,
            },
        }
        self._turns_since_prune = 0
        self._turns_since_audit = 0
        self._recent_turns: deque[dict[str, Any]] = deque(maxlen=max(2, self.config.audit_window))
        self._adaptive_thresholds = self._load_adaptive_thresholds()
        self._adaptive_thresholds_dirty = False
        self._debug(
            "LayeredMemoryManager initialized (base_dir=%s, embed_model=%s)",
            self.config.base_dir,
            self.config.embed_model,
        )

    def close(self) -> None:
        self._save_adaptive_thresholds()
        episodic = getattr(self, "episodic", None)
        if episodic is None:
            return
        try:
            episodic.close()
        except Exception:
            pass

    def log_interaction(self, user: str, assistant: str, *, metadata: Optional[dict[str, Any]] = None) -> None:
        try:
            self.episodic.log(user, assistant, metadata=metadata)
            self._debug(
                "Logged interaction to episodic memory (user='%s...', assistant='%s...')",
                (user or "").strip()[:40],
                (assistant or "").strip()[:40],
            )
        except Exception:
            pass

    def process_turn(
        self,
        user: str,
        assistant: str,
        *,
        client: Any | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist the exchange and optionally harvest long-term insights."""
        tags = self._derive_tags(user, assistant, metadata)
        payload = dict(metadata or {})
        if tags:
            payload["tags"] = sorted(tags)
        self.log_interaction(user, assistant, metadata=payload or None)
        self._recent_turns.append(
            {
                "user": (user or "").strip(),
                "assistant": (assistant or "").strip(),
                "tags": sorted(tags),
            }
        )
        if client is not None:
            self._harvest_long_term_layers(user, assistant, client=client, tags=tags)
            self._maybe_auto_prune()
            self._maybe_run_audit(client=client)
        if self._adaptive_thresholds_dirty:
            self._save_adaptive_thresholds()

    def assemble(self, query: str) -> AssembledContext:
        self._debug("Assembling memory context for query='%s'", query.strip()[:60])
        assembled = self.assembler.assemble(
            query,
            k_ep=self.config.k_ep,
            k_facts=self.config.k_facts,
            k_lessons=self.config.k_reflections,
        )
        self._register_retrieval_feedback("fact", self.semantic.last_recalled)
        self._register_retrieval_feedback("reflection", self.reflections.last_recent)
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

    def _harvest_long_term_layers(
        self,
        user: str,
        assistant: str,
        *,
        client: Any,
        tags: set[str],
    ) -> None:
        self._stats["harvest"]["attempts"] += 1
        if not hasattr(client, "chat"):
            return
        model = (self.config.memory_model or "phi3:latest").strip()
        messages = self._build_insight_messages(user, assistant)
        if tags and messages:
            try:
                messages[-1]["content"] += "\n\nContext tags: " + ", ".join(sorted(tags))
            except Exception:
                pass
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
            context_tags=tags,
        )
        lesson_items = self._normalize_harvest_items(
            lessons_raw,
            min_confidence=self.config.min_reflection_confidence or 0.0,
            kind="reflection",
            context_tags=tags,
        )

        fact_items = self._apply_critic(
            fact_items,
            kind="fact",
            client=client,
            tags=tags,
            user=user,
            assistant=assistant,
        )
        lesson_items = self._apply_critic(
            lesson_items,
            kind="reflection",
            client=client,
            tags=tags,
            user=user,
            assistant=assistant,
        )

        max_facts = self._safe_int_env("ATLAS_MEMORY_MAX_FACTS", 5)
        max_reflections = self._safe_int_env("ATLAS_MEMORY_MAX_REFLECTIONS", 3)
        fact_items = fact_items[:max(0, max_facts)]
        lesson_items = lesson_items[:max(0, max_reflections)]

        added_facts = 0
        if fact_items:
            added_records = self.semantic.extend_facts(
                fact_items,
                source="conversation",
                tags=tags,
            )
            added_facts = len(added_records)
            self._stats["harvest"]["accepted_facts"] += added_facts
        added_reflections = 0
        if lesson_items:
            for lesson in lesson_items:
                inserted = self.reflections.add(
                    lesson["text"],
                    confidence=lesson.get("confidence"),
                    quality=lesson.get("quality"),
                    tags=lesson.get("tags") or tags,
                )
                if inserted:
                    added_reflections += 1
            if added_reflections:
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

    @staticmethod
    def _normalize_tags(tags: Optional[Iterable[str] | str]) -> list[str]:
        if not tags:
            return []
        if isinstance(tags, (list, tuple, set)):
            iterable = tags
        else:
            iterable = [tags]
        normalized: set[str] = set()
        for tag in iterable:
            text = str(tag or "").strip().lower()
            if text:
                normalized.add(text)
        return sorted(normalized)

    def _derive_tags(
        self,
        user: str,
        assistant: str,
        metadata: Optional[dict[str, Any]],
    ) -> set[str]:
        tags = set(self._normalize_tags((metadata or {}).get("tags")))
        objective = (metadata or {}).get("objective")
        if objective:
            slug = self._slug_tag(str(objective))
            if slug:
                tags.add(f"objective:{slug}")
        for tool_name in self._normalize_tags((metadata or {}).get("tools")):
            tags.add(f"tool:{tool_name}")
        if not tags:
            tags.update(self._extract_keywords(user))
        if assistant:
            tags.update(self._extract_keywords(assistant, prefix="response"))
        return {tag for tag in tags if tag}

    @staticmethod
    def _slug_tag(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return "general"
        return "-".join(tokens[:5])[:40]

    @staticmethod
    def _extract_keywords(text: str, prefix: str = "topic", limit: int = 4) -> set[str]:
        words = [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text or "")]
        stopwords = {
            "the",
            "and",
            "that",
            "with",
            "have",
            "this",
            "from",
            "your",
            "about",
            "there",
            "what",
            "would",
            "could",
            "should",
            "please",
            "thanks",
        }
        keywords: list[str] = []
        for word in words:
            if len(word) < 4 or word in stopwords:
                continue
            keywords.append(word)
            if len(keywords) >= limit:
                break
        return {f"{prefix}:{kw}" for kw in keywords}

    def _normalize_harvest_items(
        self,
        raw_items: Iterable[Any],
        *,
        min_confidence: float,
        kind: str,
        context_tags: Optional[Iterable[str]] = None,
    ) -> List[dict[str, Any]]:
        normalized: List[dict[str, Any]] = []
        minimum = max(0.0, float(min_confidence))
        default_quality = (
            self.config.min_reflection_quality
            if kind == "reflection"
            else self.config.min_fact_quality
        )
        base_threshold = self._adaptive_min_quality(
            kind,
            float(default_quality or 0.0),
            tags=self._normalize_tags(context_tags),
        )
        base_tags = set(self._normalize_tags(context_tags))
        for item in raw_items:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                confidence = item.get("confidence")
                source = item.get("source")
                quality_override = item.get("quality")
                item_tags = self._normalize_tags(item.get("tags"))
            else:
                text = str(item).strip()
                confidence = None
                source = None
                quality_override = None
                item_tags = []
            if not text:
                continue
            candidate_tags = sorted(base_tags | set(item_tags))
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
            quality_score = None
            if isinstance(quality_override, (int, float)):
                try:
                    quality_score = float(quality_override)
                except (TypeError, ValueError):
                    quality_score = None
            if quality_score is None:
                quality_score, _ = _quality_features(text)
            threshold = self._adaptive_min_quality(
                kind,
                base_threshold,
                tags=candidate_tags,
            )
            if quality_score < threshold:
                self._stats["harvest"]["rejected_low_quality"] += 1
                continue
            entry: dict[str, Any] = {
                "text": text,
                "confidence": conf_val,
                "quality": quality_score,
            }
            if source:
                entry["source"] = str(source)
            if candidate_tags:
                entry["tags"] = candidate_tags
            normalized.append(entry)
        return normalized

    def _adaptive_min_quality(
        self,
        kind: str,
        default: float,
        *,
        tags: Optional[Iterable[str]] = None,
    ) -> float:
        state = self._adaptive_thresholds.get(kind, {})
        value = state.get("threshold", default)
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            threshold = default
        if not tags:
            return threshold
        tag_bias = state.get("tag_bias") if isinstance(state, dict) else None
        if not isinstance(tag_bias, dict) or not tag_bias:
            return threshold
        score = 0.0
        for tag in tags:
            try:
                score += float(tag_bias.get(tag, 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
        if score <= 0:
            return threshold
        adjustment = min(0.1, math.log1p(score) / 12.0)
        return max(0.05, threshold - adjustment)

    def _load_adaptive_thresholds(self) -> dict[str, dict[str, Any]]:
        baseline = {
            "fact": {
                "threshold": float(self.config.min_fact_quality or 0.3),
                "success": 0,
                "attempts": 0,
                "avg_quality": float(self.config.min_fact_quality or 0.3),
                "tag_bias": {},
            },
            "reflection": {
                "threshold": float(self.config.min_reflection_quality or 0.35),
                "success": 0,
                "attempts": 0,
                "avg_quality": float(self.config.min_reflection_quality or 0.35),
                "tag_bias": {},
            },
        }
        path = self.config.adaptive_path
        if not path.exists():
            return baseline
        try:
            data = json.loads(path.read_text())
        except Exception:
            return baseline
        if not isinstance(data, dict):
            return baseline
        for kind_key in ("fact", "reflection"):
            state = data.get(kind_key, {})
            if not isinstance(state, dict):
                continue
            for key in ("threshold", "success", "attempts", "avg_quality"):
                if key in state:
                    try:
                        baseline[kind_key][key] = float(state[key])
                    except (TypeError, ValueError):
                        continue
            tag_bias = state.get("tag_bias")
            if isinstance(tag_bias, dict):
                clean_bias: dict[str, float] = {}
                for tag, value in tag_bias.items():
                    try:
                        clean_bias[str(tag)] = float(value)
                    except (TypeError, ValueError):
                        continue
                baseline[kind_key]["tag_bias"] = clean_bias
        self.config.min_fact_quality = baseline["fact"]["threshold"]
        self.config.min_reflection_quality = baseline["reflection"]["threshold"]
        return baseline

    def _save_adaptive_thresholds(self) -> None:
        if not getattr(self, "_adaptive_thresholds_dirty", False):
            return
        payload = self._adaptive_thresholds
        try:
            self.config.adaptive_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.adaptive_path.write_text(json.dumps(payload, indent=2))
            self._adaptive_thresholds_dirty = False
        except Exception:
            LOGGER.debug("Failed to persist adaptive thresholds", exc_info=True)

    def _register_retrieval_feedback(self, kind: str, items: Iterable[dict[str, Any]]) -> None:
        state = self._adaptive_thresholds.setdefault(
            kind,
            {
                "threshold": self.config.min_fact_quality if kind == "fact" else self.config.min_reflection_quality,
                "success": 0,
                "attempts": 0,
                "avg_quality": 0.0,
                "tag_bias": {},
            },
        )
        state["attempts"] = state.get("attempts", 0) + 1
        items_list = [item for item in items if isinstance(item, dict)]
        if items_list:
            state["success"] = state.get("success", 0) + 1
            qualities: list[float] = []
            for item in items_list:
                try:
                    qualities.append(float(item.get("quality", 0.0)))
                except (TypeError, ValueError):
                    continue
                for tag in item.get("tags", []):
                    bias = state.setdefault("tag_bias", {})
                    try:
                        previous = float(bias.get(tag, 0.0))
                    except (TypeError, ValueError):
                        previous = 0.0
                    bias[tag] = round(0.85 * previous + 1.0, 4)
            if qualities:
                avg = sum(qualities) / len(qualities)
                current = state.get("avg_quality", avg)
                state["avg_quality"] = 0.7 * current + 0.3 * avg
        else:
            state["dry_spells"] = state.get("dry_spells", 0) + 1
        self._adjust_quality_threshold(kind)
        self._adaptive_thresholds_dirty = True

    def _adjust_quality_threshold(self, kind: str) -> None:
        state = self._adaptive_thresholds.get(kind, {})
        attempts = max(1.0, float(state.get("attempts", 0)))
        success_rate = float(state.get("success", 0)) / attempts
        threshold = float(state.get("threshold", 0.3))
        avg_quality = float(state.get("avg_quality", threshold))
        if success_rate < 0.3:
            threshold = max(0.15, threshold - 0.05)
        elif success_rate > 0.7 and avg_quality > threshold + 0.05:
            threshold = min(0.9, threshold + 0.03)
        state["threshold"] = round(threshold, 4)
        if kind == "fact":
            self.config.min_fact_quality = threshold
        else:
            self.config.min_reflection_quality = threshold

    def _apply_critic(
        self,
        items: List[dict[str, Any]],
        *,
        kind: str,
        client: Any,
        tags: set[str],
        user: str,
        assistant: str,
    ) -> List[dict[str, Any]]:
        if not items or not self.config.critic_enabled or not hasattr(client, "chat"):
            return items
        kind_label = "facts" if kind == "fact" else "reflections"
        preview_lines = []
        for idx, item in enumerate(items, start=1):
            preview_lines.append(f"{idx}. {item.get('text', '')}")
        instruction = (
            "You are a rigorous memory critic. Approve only high-quality, durable "
            f"{kind_label} worth storing long term."
            " You may rewrite entries for clarity, but keep intent."
            " Respond with JSON like {\"accept\": [...], \"reject\": [indices], \"notes\": \"\"}."
        )
        tags_line = ", ".join(sorted(tags)) or "(none)"
        content = (
            f"Context tags: {tags_line}\n"
            f"User said: {user.strip()}\n"
            f"Assistant replied: {assistant.strip()}\n\n"
            f"Candidates:\n" + "\n".join(preview_lines)
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
        try:
            response = client.chat(
                model=self.config.critic_model or self.config.memory_model,
                messages=messages,
                stream=False,
            )
        except Exception:
            self._stats["harvest"]["critic_failures"] += 1
            return items
        data = self._parse_insight_payload(self._extract_message_content(response))
        if not isinstance(data, dict):
            self._stats["harvest"]["critic_failures"] += 1
            return items
        accepted_raw = (
            data.get("accept")
            or data.get("approved")
            or data.get("keep")
            or []
        )
        rejected = data.get("reject") or []
        try:
            rejected_count = len([r for r in rejected if isinstance(r, (int, float, str))])
        except TypeError:
            rejected_count = 0
        revised: List[dict[str, Any]] = []
        for entry in accepted_raw:
            if isinstance(entry, str):
                text = entry.strip()
                if not text:
                    continue
                revised.append({"text": text})
            elif isinstance(entry, dict):
                text = str(entry.get("text", "")).strip()
                if not text:
                    continue
                candidate: dict[str, Any] = {"text": text}
                if "confidence" in entry:
                    candidate["confidence"] = entry.get("confidence")
                if "quality" in entry:
                    candidate["quality"] = entry.get("quality")
                if entry.get("tags"):
                    candidate["tags"] = entry["tags"]
                revised.append(candidate)
        if not revised:
            self._stats["harvest"]["critic_failures"] += 1
            return items
        if rejected_count:
            self._stats["harvest"]["critic_rejected"] += rejected_count
        final: List[dict[str, Any]] = []
        for original, candidate in zip(items, revised):
            merged = dict(original)
            merged.update({k: v for k, v in candidate.items() if v is not None})
            merged_tags = set(merged.get("tags", [])) | set(candidate.get("tags", [])) | tags
            if merged_tags:
                merged["tags"] = sorted(merged_tags)
            final.append(merged)
        if len(revised) < len(items):
            self._stats["harvest"]["critic_rejected"] += len(items) - len(revised)
        return final

    def _maybe_run_audit(self, *, client: Any) -> None:
        if not self.config.audit_interval_turns or not hasattr(client, "chat"):
            return
        if len(self._recent_turns) < 2:
            return
        self._turns_since_audit += 1
        if self._turns_since_audit < self.config.audit_interval_turns:
            return
        self._turns_since_audit = 0
        audit_tags: set[str] = set()
        for turn in self._recent_turns:
            audit_tags.update(turn.get("tags", []))
        self._run_memory_audit(client, tags=audit_tags)

    def _run_memory_audit(self, client: Any, *, tags: set[str]) -> None:
        if not hasattr(client, "chat"):
            return
        recent = list(self._recent_turns)[-self.config.audit_window :]
        facts = self.semantic.head(self.config.audit_sample_size)
        reflections = self.reflections.recent(self.config.audit_sample_size)
        if not facts and not reflections:
            return
        self._stats["audit"]["runs"] += 1
        self._stats["audit"]["facts_reviewed"] += len(facts)
        summary_lines = []
        for idx, turn in enumerate(recent, start=1):
            summary_lines.append(
                f"{idx}. User: {turn.get('user', '')}\n   Assistant: {turn.get('assistant', '')}"
            )
        payload = {
            "recent_turns": summary_lines,
            "facts": [
                {
                    "id": fact.get("id"),
                    "text": fact.get("text"),
                    "confidence": fact.get("confidence"),
                    "quality": fact.get("quality"),
                    "tags": fact.get("tags", []),
                    "links": fact.get("links", []),
                }
                for fact in facts
            ],
            "reflections": [
                {
                    "id": lesson.get("id"),
                    "text": lesson.get("text"),
                    "confidence": lesson.get("confidence"),
                    "quality": lesson.get("quality"),
                    "tags": lesson.get("tags", []),
                }
                for lesson in reflections
            ],
        }
        instruction = (
            "You audit long-term memory for consistency with recent conversations. "
            "Return JSON with optional sections: facts.drop, facts.update, facts.add, "
            "reflections.drop, reflections.update, reflections.add."
        )
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]
        try:
            response = client.chat(
                model=self.config.memory_model,
                messages=messages,
                stream=False,
            )
        except Exception:
            LOGGER.debug("Memory audit failed", exc_info=True)
            return
        data = self._parse_insight_payload(self._extract_message_content(response))
        if not isinstance(data, dict):
            return
        recent_tags = sorted(tags)
        # Handle fact removals
        for fact_id in data.get("facts", {}).get("drop", []) if isinstance(data.get("facts"), dict) else []:
            if isinstance(fact_id, str) and self.semantic.remove_fact(fact_id):
                self._stats["audit"]["facts_dropped"] += 1
        # Handle fact updates
        for update in data.get("facts", {}).get("update", []) if isinstance(data.get("facts"), dict) else []:
            if isinstance(update, dict) and update.get("id"):
                updated = self.semantic.update_fact(
                    update["id"],
                    text=update.get("text"),
                    confidence=update.get("confidence"),
                    quality=update.get("quality"),
                    tags=update.get("tags") or recent_tags,
                )
                if updated:
                    self._stats["audit"]["facts_updated"] += 1
        # Handle fact additions
        for addition in data.get("facts", {}).get("add", []) if isinstance(data.get("facts"), dict) else []:
            if isinstance(addition, dict) or isinstance(addition, str):
                payload = addition if isinstance(addition, dict) else {"text": addition}
                payload.setdefault("tags", recent_tags)
                self.semantic.extend_facts([payload], source="audit", tags=recent_tags)
        reflections_block = data.get("reflections")
        if isinstance(reflections_block, dict):
            for ref_id in reflections_block.get("drop", []) or []:
                if isinstance(ref_id, str) and self.reflections.remove(ref_id):
                    self._stats["audit"]["reflections_dropped"] += 1
            for update in reflections_block.get("update", []) or []:
                if isinstance(update, dict) and update.get("id"):
                    updated = self.reflections.update(
                        update["id"],
                        text=update.get("text"),
                        confidence=update.get("confidence"),
                        quality=update.get("quality"),
                        tags=update.get("tags") or recent_tags,
                    )
                    if updated:
                        self._stats["audit"]["reflections_updated"] += 1
            for addition in reflections_block.get("add", []) or []:
                if isinstance(addition, dict):
                    self.reflections.add(
                        addition.get("text", ""),
                        confidence=addition.get("confidence"),
                        quality=addition.get("quality"),
                        tags=addition.get("tags") or recent_tags,
                    )
                elif isinstance(addition, str):
                    self.reflections.add(addition, tags=recent_tags)

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
                    descriptor = text[:200]
                    extras: List[str] = []
                    tags = fact.get("tags", [])
                    if tags:
                        extras.append("tags: " + ", ".join(tags[:3]))
                    links = fact.get("links", [])
                    if links and self.semantic:
                        link_labels: List[str] = []
                        for link in links[:2]:
                            target_id = link.get("target")
                            if not target_id:
                                continue
                            target_fact = self.semantic.get_fact(target_id)
                            if not target_fact:
                                continue
                            link_labels.append(
                                f"{link.get('type', 'related')}→{str(target_fact.get('text', ''))[:40].strip()}"
                            )
                        if link_labels:
                            extras.append("links: " + "; ".join(link_labels))
                    if extras:
                        descriptor += " (" + "; ".join(extras) + ")"
                    fact_snips.append(f"- {descriptor}")
                    fact_meta.append({
                        "score": float(score),
                        "text": text[:120],
                        "tags": tags,
                        "links": links,
                    })

        lesson_snips: List[str] = []
        lesson_meta: List[dict[str, Any]] = []
        if self.reflections:
            for item in self.reflections.recent(k_lessons):
                text = str(item.get("text", "")).strip().replace("\n", " ")
                if text:
                    descriptor = text[:200]
                    tags = item.get("tags", [])
                    if tags:
                        descriptor += " (tags: " + ", ".join(tags[:3]) + ")"
                    lesson_snips.append(f"- {descriptor}")
                    lesson_meta.append({
                        "text": text[:120],
                        "tags": tags,
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
