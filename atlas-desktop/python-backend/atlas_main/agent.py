import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Dict

from .memory import AtlasMemory
from .tools import ToolRegistry


@dataclass
class AtlasMetrics:
    tokens: int = 240_000
    operations: int = 120
    inference: int = 140


@dataclass
class AtlasAgent:
    memory: AtlasMemory = field(default_factory=AtlasMemory)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    metrics: AtlasMetrics = field(default_factory=AtlasMetrics)

    async def process_command(self, command: str) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        if command.lower() in {'help', '?'}:
            return {
                'type': 'help',
                'message': 'Available commands: help, memory stats, tool list, ping'
            }

        if command.lower() == 'ping':
            return {'type': 'pong', 'message': 'pong'}

        if command.lower().startswith('memory'):
            return {'type': 'memory', 'payload': self.memory.get_stats()}

        if command.lower().startswith('tool'):
            return {'type': 'tools', 'payload': self.tools.list_tools()}

        self.metrics.tokens += random.randint(32, 180)
        self.metrics.operations += 1
        self.metrics.inference = max(90, min(220, self.metrics.inference + random.randint(-12, 12)))

        self.memory.add_event('COMMAND', f'Executed command: {command}')

        return {
            'type': 'echo',
            'message': f'Command received: {command}',
            'metrics': self.get_memory_metrics()
        }

    def get_memory_metrics(self) -> Dict[str, Any]:
        usage = self.memory.context_usage()
        layers = self.memory.layers()
        events = self.memory.events()
        tools = self.tools.recent_runs()

        return {
            'atlas': {
                'tokens': self.metrics.tokens,
                'operations': self.metrics.operations,
                'inference': self.metrics.inference
            },
            'memoryLayers': layers,
            'contextUsage': usage,
            'memoryEvents': events,
            'toolRuns': tools,
            'topicDistribution': self.memory.topic_distribution(),
            'toolUsage': self.tools.usage_stats(),
            'processes': self.memory.processes(),
            'fileAccess': self.memory.file_access()
        }
