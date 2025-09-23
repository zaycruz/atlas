"""Reflective journal support for Atlas."""
from __future__ import annotations

import json
import sqlite3
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
    tags: List[str] = None
    links: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.links is None:
            self.links = []

    @classmethod
    def create(cls, title: str, content: str, tags: List[str] = None, links: List[str] = None) -> "JournalEntry":
        return cls(
            id=str(uuid.uuid4()), 
            created_at=time.time(), 
            title=title.strip(), 
            content=content.strip(),
            tags=tags or [],
            links=links or []
        )


class Journal:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path.expanduser()
        self.conn = sqlite3.connect(str(self.storage_path))
        self._create_table()
        self.entries: List[JournalEntry] = []
        self._load()

    def _create_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY,
                created_at REAL,
                title TEXT,
                content TEXT,
                tags TEXT,  -- JSON list
                links TEXT  -- JSON list
            )
        """)
        self.conn.commit()

    def _load(self) -> None:
        cursor = self.conn.execute("SELECT COUNT(*) FROM journal")
        count = cursor.fetchone()[0]
        if count == 0 and self.storage_path.with_suffix('.json').exists():
            # Migrate from JSON
            json_path = self.storage_path.with_suffix('.json')
            try:
                data = json.loads(json_path.read_text())
                for item in data.get("entries", []):
                    try:
                        entry = JournalEntry(
                            id=str(item.get("id", uuid.uuid4())),
                            created_at=float(item.get("created_at", time.time())),
                            title=str(item.get("title", "")).strip(),
                            content=str(item.get("content", "")).strip(),
                            tags=[],  # No tags in old format
                            links=[]
                        )
                        self.conn.execute(
                            "INSERT INTO journal (id, created_at, title, content, tags, links) VALUES (?, ?, ?, ?, ?, ?)",
                            (entry.id, entry.created_at, entry.title, entry.content, json.dumps(entry.tags), json.dumps(entry.links))
                        )
                    except Exception:
                        continue
                self.conn.commit()
                print(f"Migrated journal from {json_path} to SQLite.")
            except Exception as e:
                print(f"Warning: Journal migration failed: {e}")
        
        # Load from SQLite
        cursor = self.conn.execute("SELECT id, created_at, title, content, tags, links FROM journal ORDER BY created_at")
        entries = []
        for row in cursor:
            id_, created_at, title, content, tags_json, links_json = row
            tags = json.loads(tags_json) if tags_json else []
            links = json.loads(links_json) if links_json else []
            entry = JournalEntry(
                id=id_,
                created_at=created_at,
                title=title,
                content=content,
                tags=tags,
                links=links
            )
            entries.append(entry)
        self.entries = entries

    def _save(self) -> None:
        # Since we insert on add_entry, _save is not needed, but keep for compatibility
        pass

    def add_entry(self, title: str, content: str, tags: List[str] = None, links: List[str] = None) -> JournalEntry:
        entry = JournalEntry.create(title, content, tags, links)
        self.conn.execute(
            "INSERT INTO journal (id, created_at, title, content, tags, links) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.id, entry.created_at, entry.title, entry.content, json.dumps(entry.tags), json.dumps(entry.links))
        )
        self.conn.commit()
        self.entries.append(entry)
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
