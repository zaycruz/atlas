from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MemoryEvent:
    time: str
    type: str
    detail: str


@dataclass
class AtlasMemory:
    episodes: int = 124
    facts: int = 86
    insights: int = 32
    _events: List[MemoryEvent] = field(default_factory=list)
    _tool_usage: Dict[str, int] = field(default_factory=lambda: {
        'web_search': 14,
        'shell': 9,
        'file_read': 12,
        'memory_write': 7,
    })

    def __post_init__(self) -> None:
        if not self._events:
            self._events = [
                MemoryEvent('09:41', 'EPISODE', 'Reviewed system boot diagnostics.'),
                MemoryEvent('09:38', 'FACT', 'Atlas connected to Ollama: qwen3:latest.'),
                MemoryEvent('09:25', 'INSIGHT', 'User prefers concise action plans.'),
                MemoryEvent('09:12', 'EPISODE', 'Indexed project workspace for quick search.'),
                MemoryEvent('08:55', 'FACT', 'Daily summary exported to /logs/atlas.'),
            ]

    def get_stats(self) -> Dict[str, int]:
        return {
            'episodes': self.episodes,
            'facts': self.facts,
            'insights': self.insights,
        }

    def context_usage(self) -> Dict[str, int]:
        current = random.randint(12, 22)
        max_tokens = 32
        percentage = int((current / max_tokens) * 100)
        return {
            'current': current,
            'max': max_tokens,
            'percentage': percentage,
        }

    def layers(self) -> Dict[str, int]:
        return self.get_stats()

    def events(self) -> List[Dict[str, str]]:
        return [event.__dict__ for event in self._events[-10:]]

    def add_event(self, event_type: str, detail: str) -> None:
        timestamp = time.strftime('%H:%M')
        self._events.append(MemoryEvent(timestamp, event_type, detail))

    def topic_distribution(self) -> List[Dict[str, int]]:
        return [
            {'topic': 'System Ops', 'percentage': 36},
            {'topic': 'Research', 'percentage': 28},
            {'topic': 'Planning', 'percentage': 22},
            {'topic': 'Support', 'percentage': 14},
        ]

    def processes(self) -> List[Dict[str, int]]:
        return [
            {'name': 'atlas-agent', 'cpu': 24, 'mem': 512},
            {'name': 'ollama-server', 'cpu': 36, 'mem': 2048},
            {'name': 'memory-harvester', 'cpu': 12, 'mem': 256},
            {'name': 'context-assembler', 'cpu': 18, 'mem': 384},
        ]

    def file_access(self) -> List[Dict[str, str]]:
        return [
            {'path': '~/Atlas/logs/session.log', 'action': 'WRITE', 'time': '09:41:22'},
            {'path': '~/Projects/atlas/notes.md', 'action': 'READ', 'time': '09:33:08'},
            {'path': '~/Atlas/memory/semantic.json', 'action': 'WRITE', 'time': '09:21:45'},
            {'path': '~/Atlas/memory/reflections.json', 'action': 'READ', 'time': '09:18:11'},
        ]
