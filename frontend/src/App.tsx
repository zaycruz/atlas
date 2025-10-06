import React, { useEffect, useState, useRef } from 'react';
import { Header } from './components/Header';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { ChatTab, type ChatTabRef } from './components/ChatTab';
import { TerminalFooter } from './components/TerminalFooter';
import { AnalyticsTab } from './components/AnalyticsTab';
import { NetworkTab } from './components/NetworkTab';
import { SystemTab } from './components/SystemTab';
import type { AIModel } from './components/ModelToggler';
import { UserProfile, type UserProfileData } from './components/UserProfile';
import type { AgentState } from './components/AgentStatus';
import { useWebSocket } from './hooks/useWebSocket';
import { useSystemMetrics } from './hooks/useSystemMetrics';
import { useCommandHistory } from './hooks/useCommandHistory';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
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

const DEFAULT_CHAT: TerminalEntry[] = [
  {
    type: 'system',
    text: 'ATLAS Interface Initialized'
  }
];

const DEFAULT_TERMINAL: TerminalEntry[] = [];

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

type ModelPullState = {
  model: string | null;
  status: 'idle' | 'started' | 'progress' | 'completed' | 'error';
  message?: string;
};

const App: React.FC = () => {
  const [time, setTime] = useState(() => new Date());
  const [activeModule, setActiveModule] = useState('terminal');
  const [currentModel, setCurrentModel] = useState<AIModel>('qwen3:latest');
  const [installedModels, setInstalledModels] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelPullStatus, setModelPullStatus] = useState<ModelPullState>({ model: null, status: 'idle' });
  const [agentStatus, setAgentStatus] = useState<AgentState>('idle');
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isTerminalExpanded, setIsTerminalExpanded] = useState(false);
  const [userProfile, setUserProfile] = useState<UserProfileData>({
    name: '',
    role: '',
    expertise: [],
    workingStyle: 'balanced',
    preferences: {
      codeComments: true,
      stepByStep: false,
      askBeforeAction: true
    }
  });

  const systemMetrics = useSystemMetrics();
  const { isConnected, messages, clearMessages, sendMessage } = useWebSocket(WS_URL);
  const commandHistory = useCommandHistory();
  const terminalCommandHistory = useCommandHistory();
  const chatTabRef = useRef<ChatTabRef>(null);

  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<TerminalEntry[]>(DEFAULT_CHAT);
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
  const modelClearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'k',
      meta: true,
      ctrl: true,
      handler: () => {
        setActiveModule('terminal');
        setTimeout(() => chatTabRef.current?.focusInput(), 100);
      },
      description: 'Focus chat input'
    },
    {
      key: 't',
      meta: true,
      ctrl: true,
      handler: () => {
        setIsTerminalExpanded((prev) => !prev);
      },
      description: 'Toggle terminal'
    },
    {
      key: 'Escape',
      handler: () => {
        if (isProfileOpen) {
          setIsProfileOpen(false);
        }
      },
      description: 'Close modals'
    }
  ]);

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
            const timestamp = Date.now();
            setChatHistory((prev) => [
              ...prev,
              { type: 'success', text: streamingResponseRef.current, timestamp }
            ]);
            setTerminalHistory((prev) => [
              ...prev,
              { type: 'success', text: streamingResponseRef.current, timestamp }
            ]);
          }
          setStreamingResponse('');
          streamingResponseRef.current = '';
          setAgentStatus('idle');
        } else {
          setAgentStatus('responding');
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
        const timestamp = Date.now();
        setChatHistory((prev) => [
          ...prev,
          { type: 'success', text, timestamp }
        ]);
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'success', text, timestamp }
        ]);
      }

      if (message.type === 'error') {
        const text =
          typeof message.payload === 'string'
            ? message.payload
            : JSON.stringify(message.payload, null, 2);
        const timestamp = Date.now();
        setChatHistory((prev) => [
          ...prev,
          { type: 'error', text, timestamp }
        ]);
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'error', text, timestamp }
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

      if (message.type === 'models_list') {
        const payload = message.payload as Partial<{
          installed: string[];
          available: string[];
          current: string;
        }>;
        if (Array.isArray(payload.installed)) {
          setInstalledModels(payload.installed);
        }
        if (Array.isArray(payload.available)) {
          setAvailableModels(payload.available);
        }
        if (payload.current) {
          setCurrentModel(payload.current as AIModel);
        }
      }

      if (message.type === 'model_pull') {
        const payload = message.payload as {
          model?: string;
          status?: string;
          message?: string;
          completed?: number;
          total?: number;
        };
        const model = payload.model || null;
        let status: ModelPullState['status'] = 'progress';
        const rawStatus = (payload.status || '').toLowerCase();
        if (rawStatus === 'started') {
          status = 'started';
        } else if (['completed', 'complete', 'done'].includes(rawStatus)) {
          status = 'completed';
        } else if (rawStatus === 'error' || rawStatus === 'failed') {
          status = 'error';
        } else {
          status = 'progress';
        }

        let messageText = payload.message || payload.status;
        if (!messageText && typeof payload.completed === 'number' && typeof payload.total === 'number') {
          const percent = payload.total > 0 ? Math.round((payload.completed / payload.total) * 100) : undefined;
          if (percent !== undefined && !Number.isNaN(percent)) {
            messageText = `${percent}%`;
          }
        }

        if (status === 'started' && !messageText) {
          messageText = 'Starting pull...';
        }

        if (status === 'completed' && !messageText) {
          messageText = 'Pull complete';
        }

        if (status === 'error' && !messageText) {
          messageText = 'Pull failed';
        }

        setModelPullStatus({ model, status, message: messageText });
      }

      if (message.type === 'model_updated') {
        const payload = message.payload as { model?: string };
        if (payload.model) {
          setCurrentModel(payload.model as AIModel);
        }
      }

      if (message.type === 'shell_start') {
        const payload = message.payload as { command?: string; cwd?: string | null };
        const info = payload.command ? `Running: ${payload.command}` : 'Command started';
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'system', text: info, timestamp: Date.now() }
        ]);
      }

      if (message.type === 'shell_output') {
        const payload = message.payload as { data?: string; stream?: string };
        if (payload.data) {
          const stream = (payload.stream || 'stdout').toLowerCase();
          const entryType: TerminalEntry['type'] = stream === 'stderr' ? 'error' : 'success';
          const prefix = stream === 'stderr' ? '[stderr] ' : '';
          setTerminalHistory((prev) => [
            ...prev,
            { type: entryType, text: `${prefix}${payload.data}`, timestamp: Date.now() }
          ]);
        }
      }

      if (message.type === 'shell_error') {
        const payload = message.payload as { message?: string };
        const text = payload.message || 'Shell command error';
        setTerminalHistory((prev) => [
          ...prev,
          { type: 'error', text, timestamp: Date.now() }
        ]);
      }

      if (message.type === 'shell_complete') {
        const payload = message.payload as { exit_code?: number; timed_out?: boolean };
        const exitCode = typeof payload.exit_code === 'number' ? payload.exit_code : 'unknown';
        const text = payload.timed_out
          ? `Command terminated after timeout`
          : `Command exited with code ${exitCode}`;
        setTerminalHistory((prev) => [
          ...prev,
          { type: payload.timed_out ? 'warn' : 'system', text, timestamp: Date.now() }
        ]);
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
        sendMessage({ type: 'list_models' });
      }, 100);
    }
  }, [isConnected, sendMessage]);

  useEffect(() => {
    if (modelPullStatus.status === 'completed' || modelPullStatus.status === 'error') {
      if (modelClearTimer.current) {
        clearTimeout(modelClearTimer.current);
      }
      modelClearTimer.current = setTimeout(() => {
        setModelPullStatus({ model: null, status: 'idle' });
      }, 2500);
    }
    return () => {
      if (modelClearTimer.current) {
        clearTimeout(modelClearTimer.current);
        modelClearTimer.current = null;
      }
    };
  }, [modelPullStatus.status]);

  const handleChatCommand = () => {
    console.log('[App] handleChatCommand called, input:', chatInput);
    if (!chatInput.trim()) return;

    const trimmedInput = chatInput.trim();
    const timestamp = Date.now();

    // Add to command history
    commandHistory.addCommand(trimmedInput);

    // Add to chat history (without $ prefix)
    setChatHistory((prev) => [
      ...prev,
      { type: 'command', text: trimmedInput, timestamp }
    ]);

    // Add to terminal history (with $ prefix)
    const commandText = `$ ${trimmedInput}`;
    setTerminalHistory((prev) => [
      ...prev,
      { type: 'command', text: commandText, timestamp }
    ]);

    console.log('[App] isConnected:', isConnected, 'sending message');
    if (isConnected) {
      const message = { type: 'command', payload: trimmedInput };
      console.log('[App] Sending WebSocket message:', message);
      setAgentStatus('processing');
      sendMessage(message);
    } else {
      console.log('[App] WebSocket not connected, showing error');
      const errorMsg = 'WebSocket disconnected. Unable to send command.';
      const timestamp = Date.now();
      setChatHistory((prev) => [
        ...prev,
        { type: 'error', text: errorMsg, timestamp }
      ]);
      setTerminalHistory((prev) => [
        ...prev,
        { type: 'error', text: errorMsg, timestamp }
      ]);
      setAgentStatus('idle');
    }

    setChatInput('');
    commandHistory.resetPosition();
  };

  const handleTerminalCommand = () => {
    console.log('[App] handleTerminalCommand called, input:', terminalInput);
    if (!terminalInput.trim()) return;

    const trimmedInput = terminalInput.trim();
    const commandText = `$ ${trimmedInput}`;
    const timestamp = Date.now();
    const commandId = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);

    terminalCommandHistory.addCommand(trimmedInput);
    terminalCommandHistory.resetPosition();

    setTerminalHistory((prev) => [
      ...prev,
      { type: 'command', text: commandText, timestamp }
    ]);

    if (isConnected) {
      sendMessage({ type: 'shell_command', payload: { id: commandId, command: trimmedInput } });
    } else {
      setTerminalHistory((prev) => [
        ...prev,
        { type: 'error', text: 'WebSocket disconnected. Unable to run command.', timestamp: Date.now() }
      ]);
    }

    setTerminalInput('');
  };

  const handleModelChange = (model: AIModel) => {
    console.log('[App] Model changed to:', model);
    setCurrentModel(model);
    // Send model change to backend
    if (isConnected) {
      sendMessage({ type: 'set_model', payload: { model } });
    }
  };

  const handleAddModel = (model: string) => {
    console.log('[App] Pulling model:', model);
    if (isConnected) {
      setModelPullStatus({ model, status: 'started', message: 'Starting pull...' });
      sendMessage({ type: 'pull_model', payload: { model } });
    }
  };

  const handleProfileSave = (profile: UserProfileData) => {
    console.log('[App] User profile saved:', profile);
    setUserProfile(profile);
    // Send profile to backend
    if (isConnected) {
      sendMessage({ type: 'update_profile', payload: profile });
    }
    // Save to localStorage for persistence
    localStorage.setItem('atlas_user_profile', JSON.stringify(profile));
  };

  const handleClearChat = () => {
    console.log('[App] Clearing chat');
    setChatHistory(DEFAULT_CHAT);
    setStreamingResponse('');
  };

  const handleClearTerminal = () => {
    console.log('[App] Clearing terminal');
    setTerminalHistory(DEFAULT_TERMINAL);
  };

  // Load profile from localStorage on mount
  useEffect(() => {
    const savedProfile = localStorage.getItem('atlas_user_profile');
    if (savedProfile) {
      try {
        setUserProfile(JSON.parse(savedProfile));
      } catch (error) {
        console.error('Failed to load user profile:', error);
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-atlas-black text-atlas-green-500 flex flex-col">
      <Header
        time={time}
        isConnected={isConnected}
        currentModel={currentModel}
        onModelChange={handleModelChange}
        installedModels={installedModels}
        availableModels={availableModels}
        onAddModel={handleAddModel}
        modelPullStatus={modelPullStatus}
        onOpenProfile={() => setIsProfileOpen(true)}
        agentStatus={agentStatus}
      />
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
            <ChatTab
              ref={chatTabRef}
              history={chatHistory}
              input={chatInput}
              setInput={setChatInput}
              onCommand={handleChatCommand}
              streamingText={streamingResponse}
              onNavigateHistory={commandHistory.navigateHistory}
              onClear={handleClearChat}
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
      <TerminalFooter
        history={terminalHistory}
        input={terminalInput}
        setInput={setTerminalInput}
        onCommand={handleTerminalCommand}
        onClear={handleClearTerminal}
        isExpanded={isTerminalExpanded}
        onToggle={() => setIsTerminalExpanded(!isTerminalExpanded)}
        onNavigateHistory={terminalCommandHistory.navigateHistory}
      />

      <UserProfile
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
        profile={userProfile}
        onSave={handleProfileSave}
      />
    </div>
  );
};

export default App;
