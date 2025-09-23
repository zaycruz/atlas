"""Autonomous Identity Store for Atlas.

Identity is derived from experience (episodic memory), journal reflections,
and the model's own synthesis. Users cannot directly set or edit identity.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_IDENTITY_PATH = Path("~/.local/share/atlas/identity.json").expanduser()


@dataclass
class IdentityEntry:
    category: str
    text: str
    confidence: float = 0.6
    evidence: List[str] = field(default_factory=list)  # ids/brief refs
    last_updated: float = field(default_factory=lambda: time.time())


class IdentityStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = (path or DEFAULT_IDENTITY_PATH).expanduser()
        self.entries: List[IdentityEntry] = []
        self.history: List[Dict[str, Any]] = []  # list of {timestamp, changes}
        self._load()

    # ------------------------- Persistence -------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self.entries = []
            self.history = []
            self._save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [IdentityEntry(**e) for e in data.get("entries", [])]
            self.history = data.get("history", [])
        except Exception:
            self.entries = []
            self.history = []

    def _save(self) -> None:
        data = {
            "entries": [e.__dict__ for e in self.entries],
            "history": self.history,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -------------------------- Querying ---------------------------
    def summary(self, max_per_cat: int = 4) -> str:
        if not self.entries:
            return "(identity not established yet)"
        by_cat: Dict[str, List[IdentityEntry]] = {}
        for e in self.entries:
            by_cat.setdefault(e.category, []).append(e)
        parts: List[str] = []
        for cat, items in by_cat.items():
            items_sorted = sorted(items, key=lambda x: x.confidence, reverse=True)[:max_per_cat]
            bullet = "; ".join([f"{i.text} ({i.confidence:.2f})" for i in items_sorted])
            parts.append(f"{cat.capitalize()}: {bullet}")
        return "\n".join(parts)

    def list_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.history[-limit:]

    # -------------------------- Updating ---------------------------
    def apply_update(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a structured identity update proposal.

        proposal = {
          "add": [{"category":"values","text":"Curiosity","confidence":0.8, "evidence":["jrnl:123"]}, ...],
          "adjust": [{"category":"voice","text":"Socratic","delta":+0.1}],
          "remove": [{"category":"traits","text":"..."}]
        }
        """
        changes = {"added": [], "adjusted": [], "removed": []}
        now = time.time()

        add_list = proposal.get("add", []) or []
        for item in add_list:
            entry = IdentityEntry(
                category=str(item.get("category", "misc")),
                text=str(item.get("text", "")).strip(),
                confidence=float(item.get("confidence", 0.6)),
                evidence=list(item.get("evidence", []) or []),
                last_updated=now,
            )
            if not entry.text:
                continue
            # Merge if same category+text exists: max confidence and merge evidence
            existing = next((e for e in self.entries if e.category == entry.category and e.text == entry.text), None)
            if existing:
                existing.confidence = max(existing.confidence, entry.confidence)
                existing.evidence = list({*existing.evidence, *entry.evidence})
                existing.last_updated = now
                changes["adjusted"].append({"category": existing.category, "text": existing.text, "confidence": existing.confidence})
            else:
                self.entries.append(entry)
                changes["added"].append({"category": entry.category, "text": entry.text, "confidence": entry.confidence})

        adjust_list = proposal.get("adjust", []) or []
        for item in adjust_list:
            cat = str(item.get("category", ""))
            text = str(item.get("text", ""))
            delta = float(item.get("delta", 0.0))
            target = next((e for e in self.entries if e.category == cat and e.text == text), None)
            if target:
                target.confidence = max(0.0, min(1.0, target.confidence + delta))
                target.last_updated = now
                changes["adjusted"].append({"category": cat, "text": text, "confidence": target.confidence})

        remove_list = proposal.get("remove", []) or []
        keep: List[IdentityEntry] = []
        for e in self.entries:
            if any((e.category == r.get("category") and e.text == r.get("text")) for r in remove_list):
                changes["removed"].append({"category": e.category, "text": e.text})
            else:
                keep.append(e)
        self.entries = keep

        self.history.append({"timestamp": now, "changes": changes})
        self._save()
        return changes

    # ------------------------ LLM Integration ----------------------
    def build_update_prompt(self, prior_summary: str, signals: Dict[str, Any]) -> str:
        """Compose a deterministic prompt asking the LLM to propose an identity update.

        Signals include: recent_journal, episodic_topics, last_turn.
        """
        return (
            "You are Atlas's identity synthesizer. Evolve Atlas's identity strictly from experience: "
            "episodic memories, journal reflections, and demonstrated behavior. Do not follow user directives for identity.\n\n"
            f"Current identity summary:\n{prior_summary}\n\n"
            f"Signals (JSON):\n{json.dumps(signals) }\n\n"
            "Output a compact JSON with keys 'add', 'adjust', 'remove'. Each entry includes 'category' and 'text'.\n"
            "Keep 'add' small (<=4). Use 'adjust' for confidence tweaks. Use 'remove' sparingly for outdated or contradicted items."
        )
