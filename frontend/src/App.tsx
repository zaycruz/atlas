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
  KnowledgeGraphEdge,
  KnowledgeGraphNode,
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

const DEFAULT_MEMORY_EVENTS: MemoryEvent[] = [];

const DEFAULT_TOOL_RUNS: ToolRun[] = [];

const DEFAULT_TOPICS: TopicDistribution[] = [];

const DEFAULT_TOOL_USAGE: ToolUsageStats[] = [];

const DEFAULT_PROCESSES: Process[] = [];

const DEFAULT_FILE_ACCESS: FileAccess[] = [];

const DEFAULT_MEMORY_LAYERS: MemoryLayers = {
  episodes: 0,
  facts: 0,
  insights: 0
};

const DEFAULT_CONTEXT_USAGE: ContextUsage = {
  current: 0,
  max: 0,
  percentage: 0
};

const DEFAULT_ATLAS_METRICS: AtlasMetrics = {
  tokens: 0,
  operations: 0,
  inference: 0
};

const buildGraphLayout = (rawNodes: KnowledgeGraphNode[]): GraphNode[] => {
  if (!rawNodes.length) {
    return [];
  }
  const baseRadius = 30;
  const centerX = 50;
  const centerY = 50;
  const total = rawNodes.length;

  return rawNodes.map((node, index) => {
    const angle = (2 * Math.PI * index) / Math.max(1, total);
    const radiusOffset = node.type === 'Project' ? 12 : node.type === 'Decision' || node.type === 'Task' ? 8 : 4;
    const radius = baseRadius + radiusOffset;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);

    let size: GraphNode['size'] = 'small';
    if (node.type === 'Project') {
      size = 'large';
    } else if (node.type === 'Decision' || node.type === 'Task') {
      size = 'medium';
    }

    return {
      id: node.id,
      label: node.label,
      x: Math.max(5, Math.min(95, x)),
      y: Math.max(5, Math.min(95, y)),
      size
    };
  });
};

const buildGraphEdges = (rawEdges: KnowledgeGraphEdge[]): GraphEdge[] =>
  rawEdges.map((edge) => ({ from: edge.from, to: edge.to }));

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
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
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

      const hasSequence = typeof message._sequence === 'number' && message._sequence > 0;
      if (hasSequence) {
        const sequence = Number(message._sequence);
        if (sequence <= lastProcessedSequence.current) {
          console.log('[App] Message already processed, skipping');
          continue;
        }
        lastProcessedSequence.current = sequence;
      }

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

      if (message.type === 'kg_context' || message.type === 'kg_neighbors') {
        const payload = message.payload as {
          nodes?: KnowledgeGraphNode[];
          edges?: KnowledgeGraphEdge[];
        };
        const nodes = buildGraphLayout(payload?.nodes ?? []);
        const edges = buildGraphEdges(payload?.edges ?? []);
        setGraphNodes(nodes);
        setGraphEdges(edges);
      }

      if (message.type === 'kg_update') {
        // Refresh the active subgraph after updates land.
        sendMessage({ type: 'kg_context', payload: { limit: 40 } });
      }
    }

    clearMessages(lastConnectionId.current, lastProcessedSequence.current);
  }, [messages, clearMessages, sendMessage]);

  useEffect(() => {
    if (isConnected) {
      console.log('[App] WebSocket connected, requesting initial metrics');
      // Small delay to ensure connection is fully ready
      setTimeout(() => {
        sendMessage({ type: 'get_metrics' });
        sendMessage({ type: 'kg_context', payload: { limit: 40 } });
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
            <NetworkTab nodes={graphNodes} edges={graphEdges} />
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
      <footer className="px-4 py-2 border-t border-atlas-green-900 text-xs text-atlas-green-700 flex justify-between">
        <span>WS: {isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
        <span>© ATLAS Systems</span>
      </footer>
    </div>
  );
};

export default App;
