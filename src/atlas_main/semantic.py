"""Semantic memory management for Atlas."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .ollama import OllamaClient, OllamaError


DEFAULT_STATE = {
    "profile": {},
    "preferences": [],
    "goals": [],
}


@dataclass
class SemanticMemory:
    storage_path: Path
    client: OllamaClient
    model: str
    state: Dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_STATE)))

    def __post_init__(self) -> None:
        self.storage_path = self.storage_path.expanduser()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.storage_path.exists():
            self.state = json.loads(json.dumps(DEFAULT_STATE))
            return
        try:
            self.state = json.loads(self.storage_path.read_text())
        except json.JSONDecodeError:
            backup = self.storage_path.with_suffix(".corrupt")
            self.storage_path.rename(backup)
            self.state = json.loads(json.dumps(DEFAULT_STATE))

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    def summary(self) -> str:
        profile_lines = [f"{key}: {value}" for key, value in self.state.get("profile", {}).items()]
        preference_lines = [f"- {item}" for item in self.state.get("preferences", [])]
        goal_lines = [f"- {item}" for item in self.state.get("goals", [])]

        sections = []
        if profile_lines:
            sections.append("Profile\n" + "\n".join(profile_lines))
        if preference_lines:
            sections.append("Preferences\n" + "\n".join(preference_lines))
        if goal_lines:
            sections.append("Goals\n" + "\n".join(goal_lines))
        return "\n\n".join(sections)

    def update(self, user_text: str, assistant_text: str) -> None:
        """Ask the LLM to extract durable facts from the latest exchange."""
        prompt = (
            "You are Atlas's long-term memory manager. Given the latest conversation, update the "
            "user's profile, preferences (things they like/dislike), and goals (tasks they plan to do). "
            "Return ONLY JSON with optional keys: profile (object), preferences (array of strings), goals (array of strings). "
            "Omit keys that have no updates."
        )
        conversation = (
            f"Current stored memory: {json.dumps(self.state)}\n"
            f"Latest user message: {user_text}\nLatest assistant reply: {assistant_text}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation},
        ]
        try:
            response = self.client.chat(model=self.model, messages=messages, stream=False)
        except OllamaError:
            return

        payload = response.get("message") or {}
        content = payload.get("content") or response.get("response") or ""
        content = content.strip()
        if not content:
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        if "profile" in data and isinstance(data["profile"], dict):
            self.state.setdefault("profile", {})
            for key, value in data["profile"].items():
                if isinstance(value, str) and value.strip():
                    self.state["profile"][key] = value.strip()

        if "preferences" in data and isinstance(data["preferences"], list):
            current = set(self.state.get("preferences", []))
            for item in data["preferences"]:
                if isinstance(item, str) and item.strip():
                    current.add(item.strip())
            self.state["preferences"] = sorted(current)

        if "goals" in data and isinstance(data["goals"], list):
            existing = self.state.get("goals", [])
            new_goals: List[str] = []
            for item in data["goals"]:
                if isinstance(item, str) and item.strip():
                    new_goals.append(item.strip())
            merged = {goal: True for goal in existing}
            for goal in new_goals:
                merged[goal] = True
            self.state["goals"] = list(merged.keys())

        self._save()

    def reset(self) -> None:
        self.state = json.loads(json.dumps(DEFAULT_STATE))
        self._save()
