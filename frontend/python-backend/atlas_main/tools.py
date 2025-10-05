from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ToolRun:
    id: str
    name: str
    summary: str
    time: str


@dataclass
class ToolRegistry:
    _tools: Dict[str, str] = field(default_factory=lambda: {
        'web_search': 'Searches the web using DuckDuckGo and Crawl4AI',
        'shell': 'Executes shell commands in a sandboxed environment',
        'file_read': 'Reads files from the project workspace',
        'memory_write': 'Persists information to the long-term stores',
    })
    _usage: Dict[str, int] = field(default_factory=lambda: {
        'web_search': 14,
        'shell': 9,
        'file_read': 12,
        'memory_write': 7,
    })
    _recent_runs: List[ToolRun] = field(default_factory=lambda: [
        ToolRun('tool-4981', 'Web Search', 'Gathered current market intel for AI assistants.', '09:32'),
        ToolRun('tool-4975', 'File Read', 'Parsed roadmap.md for outstanding items.', '09:18'),
        ToolRun('tool-4960', 'Shell Command', 'Monitored GPU utilization via nvidia-smi.', '08:50'),
    ])

    def list_tools(self) -> Dict[str, str]:
        return self._tools

    def usage_stats(self) -> List[Dict[str, int]]:
        return [{'tool': name, 'count': count} for name, count in self._usage.items()]

    def recent_runs(self) -> List[Dict[str, str]]:
        return [run.__dict__ for run in self._recent_runs]

    def record_run(self, tool_name: str, summary: str) -> None:
        self._usage[tool_name] = self._usage.get(tool_name, 0) + 1
        run_id = f"tool-{random.randint(5000, 9999)}"
        timestamp = time.strftime('%H:%M')
        self._recent_runs.insert(0, ToolRun(run_id, tool_name.title(), summary, timestamp))
        self._recent_runs = self._recent_runs[:10]
