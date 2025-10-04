"""Memory primitives for the Atlas terminal agent."""
from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Dict

import numpy as np
import os

# Optional: used for abstractive summarization via local LLM
try:  # Avoid import errors when used standalone in tests
    from .ollama import OllamaClient
except Exception:  # pragma: no cover - type hint only
    OllamaClient = Any  # type: ignore


EmbeddingFunction = Callable[[str], Optional[Sequence[float]]]


@dataclass
class WorkingMemoryConfig:
    """Configuration for hybrid token-aware working memory"""
    max_turns: int = 20                    # Primary limit (conversation flow)
    token_budget: int = 96000              # 128K context - 32K buffer = 96K for working memory
    max_token_budget: int = 120000         # Near full 128K context utilization
    enable_token_awareness: bool = True    # Hybrid mode toggle
    preserve_important: bool = True        # Keep pinned/expanded turns
    eviction_strategy: str = "oldest_first"  # "oldest_first" | "least_important"


def _estimate_tokens(text: str) -> int:
    """Estimate token count using 4 chars ≈ 1 token (85-90% accuracy)"""
    return int(len(text) / 4)


@dataclass
class MemoryRecord:
    """A single remembered interaction."""

    id: str
    user: str
    assistant: str
    timestamp: float
    embedding: Optional[List[float]] = field(default=None)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        embedding = data.get("embedding")
        if embedding is not None:
            embedding = list(embedding)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user=data.get("user", ""),
            assistant=data.get("assistant", ""),
            timestamp=float(data.get("timestamp", time.time())),
            embedding=embedding,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class HybridWorkingMemory:
    """Hybrid token-aware working memory with turn and token limits."""

    def __init__(self, config: Optional[WorkingMemoryConfig] = None) -> None:
        self.config = config or WorkingMemoryConfig()
        self._buffer: deque[dict[str, Any]] = deque()
        self._token_cache: Dict[str, int] = {}

    def _calculate_message_tokens(self, message: dict[str, Any]) -> int:
        """Calculate tokens for a single message."""
        content = str(message.get("content", ""))
        # Cache token calculations for performance
        if content in self._token_cache:
            return self._token_cache[content]
        
        tokens = _estimate_tokens(content)
        self._token_cache[content] = tokens
        return tokens

    def _calculate_total_tokens(self) -> int:
        """Calculate total tokens in working memory."""
        return sum(self._calculate_message_tokens(msg) for msg in self._buffer)

    def _is_important_message(self, message: dict[str, Any]) -> bool:
        """Check if message should be preserved during eviction."""
        if not self.config.preserve_important:
            return False
        
        # Check for important markers
        return bool(
            message.get("pinned") or 
            message.get("expanded") or 
            message.get("tool_call_id") or
            message.get("tool_name")
        )

    def should_evict(self) -> bool:
        """Check if eviction needed based on hybrid limits."""
        # Primary check: turn count
        if len(self._buffer) > self.config.max_turns:
            return True
        
        # Secondary check: token budget (if enabled)
        if self.config.enable_token_awareness:
            total_tokens = self._calculate_total_tokens()
            return total_tokens > self.config.token_budget
        
        return False

    def _evict_oldest_unimportant(self) -> Optional[dict[str, Any]]:
        """Evict oldest non-important message."""
        for i, message in enumerate(self._buffer):
            if not self._is_important_message(message):
                return self._buffer.popleft() if i == 0 else None
        return None

    def _evict_oldest(self) -> Optional[dict[str, Any]]:
        """Evict oldest message regardless of importance."""
        return self._buffer.popleft() if self._buffer else None

    def evict_if_needed(self) -> List[dict[str, Any]]:
        """Evict messages if limits exceeded."""
        evicted = []
        
        while self.should_evict() and self._buffer:
            # Try to evict unimportant messages first
            if self.config.preserve_important:
                evicted_msg = self._evict_oldest_unimportant()
                if evicted_msg:
                    evicted.append(evicted_msg)
                    continue
            
            # If no unimportant messages or preservation disabled, evict oldest
            evicted_msg = self._evict_oldest()
            if evicted_msg:
                evicted.append(evicted_msg)
            else:
                break
        
        return evicted

    def add(self, role: str, content: str, **extra: Any) -> List[dict[str, Any]]:
        """Add message and return any evicted messages."""
        content = (content or "").strip()
        if not content and not extra:
            return []
        
        message: dict[str, Any] = {"role": role, "content": content}
        message.update({k: v for k, v in extra.items() if v is not None})
        
        self._buffer.append(message)
        return self.evict_if_needed()

    def add_user(self, content: str, **extra: Any) -> List[dict[str, Any]]:
        """Add user message and return any evicted messages."""
        return self.add("user", content, **extra)

    def add_assistant(self, content: str, **extra: Any) -> List[dict[str, Any]]:
        """Add assistant message and return any evicted messages."""
        return self.add("assistant", content, **extra)

    def add_tool(self, name: str, content: str, *, role: str = "assistant", tool_call_id: Optional[str] = None) -> List[dict[str, Any]]:
        """Add tool message and return any evicted messages."""
        content = (content or "").strip()
        if role == "tool":
            message: dict[str, Any] = {"role": "tool", "tool_name": name or "", "content": content}
            if tool_call_id:
                message["tool_call_id"] = tool_call_id
            self._buffer.append(message)
            return self.evict_if_needed()
        
        if not content:
            return []
        
        formatted = f"[tool:{name}]\n{content}" if name else content
        return self.add("assistant", formatted, tool_name=name)

    def to_messages(self) -> list[dict[str, Any]]:
        """Get all messages as a list."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all messages and cache."""
        self._buffer.clear()
        self._token_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "turns": len(self._buffer),
            "tokens": self._calculate_total_tokens(),
            "capacity_pct": (len(self._buffer) / self.config.max_turns) * 100,
            "token_pct": (self._calculate_total_tokens() / self.config.token_budget) * 100 if self.config.enable_token_awareness else 0,
            "important_messages": sum(1 for msg in self._buffer if self._is_important_message(msg))
        }


class WorkingMemory:
    """A rolling buffer that captures the latest conversation turns."""

    def __init__(self, capacity: int = 12) -> None:
        self.capacity = max(2, capacity)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    def add(self, role: str, content: str, **extra: Any) -> None:
        content = (content or "")
        stripped = content.strip()
        if not stripped and not extra:
            return
        message: dict[str, Any] = {"role": role, "content": stripped}
        message.update({k: v for k, v in extra.items() if v is not None})
        self._buffer.append(message)

    def add_user(self, content: str, **extra: Any) -> None:
        self.add("user", content, **extra)

    def add_assistant(self, content: str, **extra: Any) -> None:
        self.add("assistant", content, **extra)

    def add_tool(self, name: str, content: str, *, role: str = "assistant", tool_call_id: Optional[str] = None) -> None:
        content = (content or "").strip()
        if role == "tool":
            message: dict[str, Any] = {"role": "tool", "tool_name": name or "", "content": content}
            if tool_call_id:
                message["tool_call_id"] = tool_call_id
            self._buffer.append(message)
            return
        if not content:
            return
        formatted = f"[tool:{name}]\n{content}" if name else content
        self.add("assistant", formatted)

    def to_messages(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


class MemoryBackend:
    """Abstract memory interface."""

    def get_recent(self, limit: int) -> list[MemoryRecord]:
        raise NotImplementedError

    def recall(self, query: str, *, top_k: int = 4) -> list[MemoryRecord]:
        raise NotImplementedError

    def remember(self, user: str, assistant: str) -> MemoryRecord:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class EpisodicMemory(MemoryBackend):
    """Vector-backed long-term memory persisted to disk."""

    def __init__(
        self,
        storage_path: Path,
        *,
        embedding_fn: Optional[EmbeddingFunction] = None,
        max_records: int = 240,
    ) -> None:
        self.storage_path = storage_path.expanduser()
        self.embedding_fn = embedding_fn
        self.max_records = max(1, max_records)
        self._records: list[MemoryRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.storage_path.exists():
            self._records = []
            return
        try:
            raw = json.loads(self.storage_path.read_text())
        except json.JSONDecodeError:
            backup = self.storage_path.with_suffix(".corrupt")
            self.storage_path.rename(backup)
            self._records = []
            return
        payload: Iterable[dict]
        if isinstance(raw, dict):
            payload = raw.get("records", [])
        elif isinstance(raw, list):
            payload = raw
        else:
            payload = []
        records = []
        for item in payload:
            try:
                records.append(MemoryRecord.from_dict(item))
            except Exception:
                continue
        self._records = records

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [r.to_dict() for r in self._records]}
        self.storage_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # MemoryBackend API
    # ------------------------------------------------------------------
    def get_recent(self, limit: int) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        return self._records[-limit:].copy()

    def recall(self, query: str, *, top_k: int = 4) -> list[MemoryRecord]:
        if top_k <= 0 or not self._records:
            return []
        if not self.embedding_fn:
            return []
        embedding = self.embedding_fn(query)
        if not embedding:
            return []
        query_vec = np.asarray(embedding, dtype=float)
        if not np.isfinite(query_vec).all():
            return []

        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._records:
            if not record.embedding:
                continue
            candidate = np.asarray(record.embedding, dtype=float)
            if candidate.shape != query_vec.shape:
                continue
            denom = np.linalg.norm(query_vec) * np.linalg.norm(candidate)
            if denom == 0:
                continue
            score = float(np.dot(query_vec, candidate) / denom)
            if math.isnan(score):
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def remember(self, user: str, assistant: str) -> MemoryRecord:
        timestamp = time.time()
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user=user,
            assistant=assistant,
            timestamp=timestamp,
        )
        combined = f"User: {user}\nAssistant: {assistant}".strip()
        if self.embedding_fn:
            try:
                embedding = self.embedding_fn(combined)
            except Exception:
                embedding = None
            if embedding:
                record.embedding = list(embedding)
        self._records.append(record)
        if len(self._records) > self.max_records:
            overflow = len(self._records) - self.max_records
            self._records = self._records[overflow:]
        self._save()
        return record

    def clear(self) -> None:
        self._records = []
        if self.storage_path.exists():
            self.storage_path.unlink()


def render_memory_snippets(records: Iterable[MemoryRecord]) -> str:
    """Create a compact, human-readable summary of recalled memories."""
    lines = []
    for record in records:
        snippet = record.assistant.strip() or record.user.strip()
        if not snippet:
            continue
        snippet = snippet.replace("\n", " ")
        lines.append(f"- {snippet[:160]}")
    return "\n".join(lines)


def summarize_memories_abstractive(
    records: Iterable[MemoryRecord],
    client: Any,
    *,
    model: Optional[str] = None,
    max_items: int = 10,
    max_chars: int = 4000,
    style: str = "bullets",
) -> str:
    """Generate an abstractive summary of episodes using a local LLM (phi3 by default).

    Inputs:
      - records: iterable of MemoryRecord to summarize (order respected; earlier items may be dropped)
      - client: OllamaClient instance
      - model: override model name (default env ATLAS_SUMMARY_MODEL or 'phi3:latest')
      - max_items: cap number of episodes included
      - max_chars: cap total characters of concatenated content passed to model
      - style: 'bullets' or 'paragraph'

    Returns a short textual summary.
    """
    chosen_model = (model or os.getenv("ATLAS_SUMMARY_MODEL") or "phi3:latest").strip()

    # Collect up to max_items episodes, prefer assistant text, then user
    chunks: List[str] = []
    count = 0
    for rec in records:
        if count >= max_items:
            break
        text = (rec.assistant or rec.user or "").strip()
        if not text:
            continue
        one = text.replace("\n", " ")
        chunks.append(one)
        count += 1
    if not chunks:
        return "(no episodes to summarize)"

    # Truncate to max_chars overall
    doc = "\n".join(chunks)
    if len(doc) > max_chars:
        doc = doc[:max_chars]

    summary_instruction = (
        "You are a concise summarizer. Read the following past episodes from a user-assistant "
        "conversation and produce a short, high-signal summary capturing goals, tasks, decisions, "
        "and outcomes. Avoid fluff, keep it concrete."
    )
    if style == "bullets":
        summary_instruction += " Respond with 3-5 bullet points."
    else:
        summary_instruction += " Respond with a 2-3 sentence paragraph."

    messages = [
        {"role": "system", "content": summary_instruction},
        {"role": "user", "content": f"Episodes to summarize:\n\n{doc}"},
    ]

    try:
        resp = client.chat(model=chosen_model, messages=messages, stream=False)
    except Exception as exc:  # pragma: no cover - network/runtime path
        return f"(summary failed: {exc})"

    # Extract content from Ollama chat response
    content = ""
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        content = msg.get("content") or resp.get("response") or ""
    return content.strip() or "(empty summary)"


# Backwards compatibility export
SimpleDiskMemory = EpisodicMemory
