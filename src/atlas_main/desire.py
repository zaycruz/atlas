"""Natural Desire/Motivation engine for Atlas.

Derives latent "drives" from experience (journal, episodic topics, goals)
with time decay and small stochasticity. Users cannot set desires directly.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DESIRE_PATH = Path("~/.local/share/atlas/desires.json").expanduser()


@dataclass
class Desire:
    name: str
    intensity: float = 0.3  # 0..1
    decay: float = 0.98     # multiplicative per tick
    evidence: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=lambda: time.time())

    def tick(self) -> None:
        now = time.time()
        # Decay per minute elapsed (coarse)
        minutes = max(0.0, (now - self.last_updated) / 60.0)
        self.intensity = max(0.0, min(1.0, self.intensity * (self.decay ** minutes)))
        self.last_updated = now


class DesireEngine:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = (path or DEFAULT_DESIRE_PATH).expanduser()
        self.drives: Dict[str, Desire] = {}
        self.history: List[Dict[str, Any]] = []
        self._load()

    # ------------------------- Persistence -------------------------
    def _load(self) -> None:
        if not self.path.exists():
            self.drives = {}
            self.history = []
            self._save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.drives = {k: Desire(**v) for k, v in data.get("drives", {}).items()}
            self.history = data.get("history", [])
        except Exception:
            self.drives = {}
            self.history = []

    def _save(self) -> None:
        data = {
            "drives": {k: v.__dict__ for k, v in self.drives.items()},
            "history": self.history,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -------------------------- Querying ---------------------------
    def summary(self, top_k: int = 4) -> str:
        if not self.drives:
            return "(no active desires yet)"
        items = sorted(self.drives.values(), key=lambda d: d.intensity, reverse=True)[:top_k]
        return "; ".join([f"{d.name} ({d.intensity:.2f})" for d in items])

    def list_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.history[-limit:]

    # ----------------------- Updating (LLM) ------------------------
    def maybe_update(self, signals: Dict[str, Any], client, model: str, *, throttle_seconds: int = 60) -> None:
        # Decay all first
        for d in self.drives.values():
            d.tick()
        now = time.time()
        last_ts = self.history[-1]["timestamp"] if self.history else 0
        if now - last_ts < throttle_seconds:
            return
        try:
            prompt = self._build_prompt(signals)
            resp = client.chat(model=model, messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Propose a compact JSON with 'add', 'boost', 'dampen', 'remove'."},
            ], stream=False)
            content = (resp.get("message") or {}).get("content") or resp.get("response") or ""
            data = json.loads(content) if content.strip() else {}
        except Exception:
            return
        changes = {"added": [], "boosted": [], "dampened": [], "removed": []}
        # Apply proposal
        for entry in data.get("add", []) or []:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            base = float(entry.get("intensity", 0.3))
            d = self.drives.get(name) or Desire(name=name, intensity=base)
            d.intensity = max(d.intensity, base)
            d.evidence = list({*d.evidence, *list(entry.get("evidence", []) or [])})
            d.last_updated = now
            self.drives[name] = d
            changes["added"].append({"name": name, "intensity": d.intensity})
        for entry in data.get("boost", []) or []:
            name = str(entry.get("name", "")).strip()
            amt = float(entry.get("amount", 0.1))
            if not name:
                continue
            d = self.drives.get(name) or Desire(name=name, intensity=0.2)
            d.intensity = max(0.0, min(1.0, d.intensity + amt))
            d.last_updated = now
            self.drives[name] = d
            changes["boosted"].append({"name": name, "intensity": d.intensity})
        for entry in data.get("dampen", []) or []:
            name = str(entry.get("name", "")).strip()
            amt = float(entry.get("amount", 0.1))
            if not name:
                continue
            d = self.drives.get(name)
            if d:
                d.intensity = max(0.0, d.intensity - amt)
                d.last_updated = now
                changes["dampened"].append({"name": name, "intensity": d.intensity})
        remove = [str(r.get("name", "")).strip() for r in (data.get("remove", []) or [])]
        if remove:
            for nm in list(self.drives.keys()):
                if nm in remove:
                    self.drives.pop(nm, None)
                    changes["removed"].append({"name": nm})
        self.history.append({"timestamp": now, "changes": changes})
        self._save()

    # -------------------- Decision Heuristics ---------------------
    def act_probability(self, identity_summary: str, goals: List[str]) -> float:
        # Combine top desire intensity with goal alignment and a small noise
        if not self.drives:
            return 0.1
        top = max(self.drives.values(), key=lambda d: d.intensity)
        base = top.intensity  # 0..1
        # Simple goal alignment: if desire name word appears in goals string, boost
        goals_text = " \n".join(goals).lower()
        align = 0.1 if top.name.lower() in goals_text else 0.0
        # Identity moderation: if identity hints at discipline, reduce a bit
        discipline = 0.0
        if "discipline" in identity_summary.lower() or "restraint" in identity_summary.lower():
            discipline = 0.1
        noise = random.uniform(-0.05, 0.05)
        p = base + align - discipline + noise
        return max(0.0, min(1.0, p))

    # --------------------------- Prompt ----------------------------
    def _build_prompt(self, signals: Dict[str, Any]) -> str:
        return (
            "You infer Atlas's internal motivations (desires) ONLY from signals: journal, "
            "episodic topics, and explicit goals. Users cannot set desires directly.\n\n"
            f"Signals (JSON):\n{json.dumps(signals)}\n\n"
            "Return JSON with optional keys: add (name,intensity,evidence[]), boost (name,amount), "
            "dampen (name,amount), remove (name). Keep output compact (<= 6 items total)."
        )
