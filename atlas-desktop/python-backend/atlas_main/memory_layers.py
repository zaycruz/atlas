from dataclasses import dataclass
from typing import Dict


@dataclass
class MemoryLayers:
    episodes: int
    facts: int
    insights: int

    def as_dict(self) -> Dict[str, int]:
        return {
            'episodes': self.episodes,
            'facts': self.facts,
            'insights': self.insights,
        }
