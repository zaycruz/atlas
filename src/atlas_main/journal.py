"""Reflective journal support for Atlas."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class JournalEntry:
    id: str
    created_at: float
    title: str
    content: str

    @classmethod
    def create(cls, title: str, content: str) -> "JournalEntry":
        return cls(id=str(uuid.uuid4()), created_at=time.time(), title=title.strip(), content=content.strip())


class Journal:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path.expanduser()
        self.entries: List[JournalEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self.entries = []
            return
        try:
            data = json.loads(self.storage_path.read_text())
        except json.JSONDecodeError:
            backup = self.storage_path.with_suffix(".corrupt")
            self.storage_path.rename(backup)
            self.entries = []
            return
        entries: List[JournalEntry] = []
        for item in data.get("entries", []):
            try:
                entry = JournalEntry(
                    id=str(item.get("id", uuid.uuid4())),
                    created_at=float(item.get("created_at", time.time())),
                    title=str(item.get("title", "")).strip(),
                    content=str(item.get("content", "")).strip(),
                )
                entries.append(entry)
            except Exception:
                continue
        self.entries = entries

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(entry) for entry in self.entries]}
        self.storage_path.write_text(json.dumps(payload, indent=2))

    def add_entry(self, title: str, content: str) -> JournalEntry:
        entry = JournalEntry.create(title, content)
        self.entries.append(entry)
        self._save()
        return entry

    def all_entries(self) -> List[JournalEntry]:
        return list(self.entries)

    def recent(self, limit: int = 5) -> List[JournalEntry]:
        if limit <= 0:
            return []
        return self.entries[-limit:]

    def find_by_keyword(self, keyword: str) -> List[JournalEntry]:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            return []
        matches = []
        for entry in self.entries:
            haystack = f"{entry.title}\n{entry.content}".lower()
            if keyword_lower in haystack:
                matches.append(entry)
        return matches
