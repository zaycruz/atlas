import React, { useEffect, useState, useRef } from 'react';
import { Header } from './components/Header';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { TerminalTab } from './components/TerminalTab';
import { AnalyticsTab } from './components/AnalyticsTab';
import { NetworkTab } from './components/NetworkTab';
import { SystemTab } from './components/SystemTab';
import { useWebSocket } from './hooks/useWebSocket';
import { useSystemMetrics } from './hooks/useSystemMetrics';
import type {
  AtlasMetrics,
  ContextUsage,
  FileAccess,
  GraphEdge,
  GraphNode,
  MemoryEvent,
  MemoryLayers,
  Process,
  TerminalEntry,
  ToolRun,
  TopicDistribution,
  ToolUsageStats
} from './types';

const WS_URL = 'ws://localhost:8765';

const DEFAULT_TERMINAL: TerminalEntry[] = [
  {
    type: 'system',
    text: 'Initializing ATLAS command terminal...'
  },
  {
    type: 'system',
    text: 'Connecting to local agent at ws://localhost:8765'
  }
];

const DEFAULT_MEMORY_EVENTS: MemoryEvent[] = [
  { time: '09:41', type: 'EPISODE', detail: 'Reviewed system boot diagnostics.' },
  { time: '09:38', type: 'FACT', detail: 'Atlas connected to Ollama: qwen3:latest.' },
  { time: '09:25', type: 'INSIGHT', detail: 'User prefers concise action plans.' },
  { time: '09:12', type: 'EPISODE', detail: 'Indexed project workspace for quick search.' },
  { time: '08:55', type: 'FACT', detail: 'Daily summary exported to /logs/atlas.' }
];

const DEFAULT_TOOL_RUNS: ToolRun[] = [
  {
    id: 'tool-4981',
    name: 'Web Search',
    summary: 'Gathered current market intel for AI assistants.',
    time: '09:32'
  },
  {
    id: 'tool-4975',
    name: 'File Read',
    summary: 'Parsed roadmap.md for outstanding items.',
    time: '09:18'
  },
  {
    id: 'tool-4960',
    name: 'Shell Command',
    summary: 'Monitored GPU utilization via nvidia-smi.',
    time: '08:50'
  }
];

const DEFAULT_TOPICS: TopicDistribution[] = [
  { topic: 'System Ops', percentage: 36 },
  { topic: 'Research', percentage: 28 },
  { topic: 'Planning', percentage: 22 },
  { topic: 'Support', percentage: 14 }
];

const DEFAULT_TOOL_USAGE: ToolUsageStats[] = [
  { tool: 'web_search', count: 14 },
  { tool: 'shell', count: 9 },
  { tool: 'file_read', count: 12 },
  { tool: 'memory_write', count: 7 }
];

const DEFAULT_PROCESSES: Process[] = [
  { name: 'atlas-agent', cpu: 24, mem: 512 },
  { name: 'ollama-server', cpu: 36, mem: 2048 },
  { name: 'memory-harvester', cpu: 12, mem: 256 },
  { name: 'context-assembler', cpu: 18, mem: 384 }
];

const DEFAULT_FILE_ACCESS: FileAccess[] = [
  { path: '~/Atlas/logs/session.log', action: 'WRITE', time: '09:41:22' },
  { path: '~/Projects/atlas/notes.md', action: 'READ', time: '09:33:08' },
  { path: '~/Atlas/memory/semantic.json', action: 'WRITE', time: '09:21:45' },
  { path: '~/Atlas/memory/reflections.json', action: 'READ', time: '09:18:11' }
];

const DEFAULT_GRAPH_NODES: GraphNode[] = [
  { id: 1, label: 'ATLAS', x: 50, y: 30, size: 'large' },
  { id: 2, label: 'User Goals', x: 25, y: 55, size: 'medium' },
  { id: 3, label: 'System Health', x: 70, y: 55, size: 'medium' },
  { id: 4, label: 'Research', x: 20, y: 80, size: 'small' },
  { id: 5, label: 'Memory Ops', x: 80, y: 80, size: 'small' },
  { id: 6, label: 'Tools', x: 50, y: 75, size: 'small' }
];

const DEFAULT_GRAPH_EDGES: GraphEdge[] = [
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 2, to: 4 },
  { from: 3, to: 5 },
  { from: 1, to: 6 },
  { from: 6, to: 4 }
];

const DEFAULT_MEMORY_LAYERS: MemoryLayers = {
  episodes: 124,
  facts: 86,
  insights: 32
};

const DEFAULT_CONTEXT_USAGE: ContextUsage = {
  current: 18,
  max: 32,
  percentage: Math.round((18 / 32) * 100)
};

const DEFAULT_ATLAS_METRICS: AtlasMetrics = {
  tokens: 241_238,
  operations: 128,
  inference: 142
};

const App: React.FC = () => {
  const [time, setTime] = useState(() => new Date());
  const [activeModule, setActiveModule] = useState('terminal');
  const systemMetrics = useSystemMetrics();
  const { isConnected, messages, clearMessages, sendMessage } = useWebSocket(WS_URL);

  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState<TerminalEntry[]>(DEFAULT_TERMINAL);
  const [streamingResponse, setStreamingResponse] = useState<string>('');
  const streamingResponseRef = useRef<string>('');
  const lastProcessedSequence = useRef<number>(0);
  const lastConnectionId = useRef<number>(0);
  const [atlasMetrics, setAtlasMetrics] = useState<AtlasMetrics>(DEFAULT_ATLAS_METRICS);
  const [memoryLayers, setMemoryLayers] = useState<MemoryLayers>(DEFAULT_MEMORY_LAYERS);
  const [contextUsage, setContextUsage] = useState<ContextUsage>(DEFAULT_CONTEXT_USAGE);
  const [memoryEvents, setMemoryEvents] = useState<MemoryEvent[]>(DEFAULT_MEMORY_EVENTS);
  const [toolRuns, setToolRuns] = useState<ToolRun[]>(DEFAULT_TOOL_RUNS);
  const [topicDistribution, setTopicDistribution] = useState<TopicDistribution[]>(DEFAULT_TOPICS);
  const [toolUsage, setToolUsage] = useState<ToolUsageStats[]>(DEFAULT_TOOL_USAGE);
  const [processes, setProcesses] = useState<Process[]>(DEFAULT_PROCESSES);
  const [fileAccess, setFileAccess] = useState<FileAccess[]>(DEFAULT_FILE_ACCESS);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const incrementer = setInterval(() => {
      setAtlasMetrics((prev) => ({
        tokens: prev.tokens + Math.floor(Math.random() * 120),
        operations: prev.operations + 1,
        inference: Math.max(80, Math.min(220, prev.inference + Math.floor(Math.random() * 12 - 6)))
      }));
      setContextUsage((prev) => {
        const current = Math.max(8, Math.min(prev.max, prev.current + (Math.random() > 0.5 ? 1 : -1)));
        const percentage = Math.round((current / prev.max) * 100);
        return { ...prev, current, percentage };
      });
    }, 5000);

    return () => clearInterval(incrementer);
  }, []);

  useEffect(() => {
    if (!messages.length) {
      return;
    }

    for (const message of messages) {
      console.log('[App] Processing message:', message.type, message);

      const connectionId =
        typeof message._connectionId === 'number' && message._connectionId > 0
          ? message._connectionId
          : lastConnectionId.current;

      if (connectionId !== lastConnectionId.current) {
        console.log('[App] Detected new connection, resetting stream state');
        lastConnectionId.current = connectionId;
        lastProcessedSequence.current = 0;
        streamingResponseRef.current = '';
        setStreamingResponse('');
      }

      const sequence =
        typeof message._sequence === 'number' && message._sequence > 0
          ? message._sequence
          : lastProcessedSequence.current + 1;

      if (sequence <= lastProcessedSequence.current) {
        console.log('[App] Message already processed, skipping');
        continue;
      }

      lastProcessedSequence.current = sequence;

      if (message.type === 'response_chunk') {
        const chunk = typeof message.payload === 'string' ? message.payload : '';
        const isFinal = message.is_final === true;

        if (isFinal) {
          if (streamingResponseRef.current.trim()) {
            setTerminalHistory((prev) => [
              ...prev,
              { type: 'success', text: streamingResponseRef.current }
            ]);
          }
          setStreamingResponse('');
          streamingResponseRef.current = '';
        } else {
          setStreamingResponse((prev) => {
            const next = prev + chunk;
            streamingResponseRef.current = next;
            return next;
          });
        }
      }

      if (message.type === 'response') {
        const text =
          typeof message.payload === 'string'
            ? message.payload
            : JSON.stringify(message.payload, null, 2);
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'success', text }
        ]);
      }

      if (message.type === 'error') {
        const text =
          typeof message.payload === 'string'
            ? message.payload
            : JSON.stringify(message.payload, null, 2);
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'error', text }
        ]);
      }

      if (message.type === 'metrics') {
        const payload = message.payload as Partial<{
          system: AtlasMetrics;
          atlas: AtlasMetrics;
          memoryLayers: MemoryLayers;
          contextUsage: ContextUsage;
          memoryEvents: MemoryEvent[];
          toolRuns: ToolRun[];
          topicDistribution: TopicDistribution[];
          toolUsage: ToolUsageStats[];
          processes: Process[];
          fileAccess: FileAccess[];
        }>;

        if (payload.atlas) {
          setAtlasMetrics(payload.atlas);
        }
        if (payload.memoryLayers) {
          setMemoryLayers(payload.memoryLayers);
        }
        if (payload.contextUsage) {
          setContextUsage(payload.contextUsage);
        }
        if (payload.memoryEvents) {
          setMemoryEvents(payload.memoryEvents);
        }
        if (payload.toolRuns) {
          setToolRuns(payload.toolRuns);
        }
        if (payload.topicDistribution) {
          setTopicDistribution(payload.topicDistribution);
        }
        if (payload.toolUsage) {
          setToolUsage(payload.toolUsage);
        }
        if (payload.processes) {
          setProcesses(payload.processes);
        }
        if (payload.fileAccess) {
          setFileAccess(payload.fileAccess);
        }
      }
    }

    clearMessages(lastConnectionId.current, lastProcessedSequence.current);
  }, [messages, clearMessages]);

  useEffect(() => {
    if (isConnected) {
      console.log('[App] WebSocket connected, requesting initial metrics');
      // Small delay to ensure connection is fully ready
      setTimeout(() => {
        sendMessage({ type: 'get_metrics' });
      }, 100);
    }
  }, [isConnected, sendMessage]);

  const handleCommand = () => {
    console.log('[App] handleCommand called, input:', terminalInput);
    if (!terminalInput.trim()) return;

    const commandText = `$ ${terminalInput.trim()}`;
    setTerminalHistory((prev) => [
      ...prev,
      { type: 'command', text: commandText }
    ]);

    console.log('[App] isConnected:', isConnected, 'sending message');
    if (isConnected) {
      const message = { type: 'command', payload: terminalInput.trim() };
      console.log('[App] Sending WebSocket message:', message);
      sendMessage(message);
    } else {
      console.log('[App] WebSocket not connected, showing error');
      setTerminalHistory((prev) => [
        ...prev,
        { type: 'error', text: 'WebSocket disconnected. Unable to send command.' }
      ]);
    }

    setTerminalInput('');
  };

  return (
    <div className="min-h-screen bg-atlas-black text-atlas-green-500 flex flex-col">
      <Header time={time} />
      <div className="flex-1 grid grid-cols-12">
        <LeftSidebar
          activeModule={activeModule}
          setActiveModule={setActiveModule}
          systemMetrics={systemMetrics}
          atlasMetrics={atlasMetrics}
        />
        <div className="col-span-7 border-r border-atlas-green-900 flex flex-col">
          {activeModule === 'analytics' && (
            <AnalyticsTab
              topicDistribution={topicDistribution}
              toolUsage={toolUsage}
              memoryLayers={memoryLayers}
              contextUsage={contextUsage}
            />
          )}
          {activeModule === 'network' && (
            <NetworkTab nodes={DEFAULT_GRAPH_NODES} edges={DEFAULT_GRAPH_EDGES} />
          )}
          {activeModule === 'system' && (
            <SystemTab processes={processes} fileAccess={fileAccess} />
          )}
          {activeModule === 'terminal' && (
            <TerminalTab
              history={terminalHistory}
              input={terminalInput}
              setInput={setTerminalInput}
              onCommand={handleCommand}
              streamingText={streamingResponse}
            />
          )}
        </div>
        <RightSidebar
          contextUsage={contextUsage}
          memoryLayers={memoryLayers}
          memoryEvents={memoryEvents}
          toolRuns={toolRuns}
        />
      </div>
      <footer className="px-3 py-1 border-t border-atlas-green-900 text-[9px] text-atlas-green-700 flex justify-between">
        <span>WS: {isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
        <span>© ATLAS Systems</span>
      </footer>
    </div>
  );
};

export default App;
