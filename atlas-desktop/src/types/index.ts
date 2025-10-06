export interface SystemMetrics {
  cpu: number;
  memory: number;
  network: number;
  disk: number;
}

export interface AtlasMetrics {
  tokens: number;
  operations: number;
  inference: number;
}

export interface ContextUsage {
  current: number;
  max: number;
  percentage: number;
}

export interface MemoryLayers {
  episodes: number;
  facts: number;
  insights: number;
}

export interface MemoryEvent {
  time: string;
  type: string;
  detail: string;
}

export interface ToolRun {
  id: string;
  name: string;
  summary: string;
  time: string;
}

export interface TerminalEntry {
  type: 'system' | 'command' | 'success' | 'error' | 'warn' | 'info';
  text: string;
}

export interface TopicDistribution {
  topic: string;
  percentage: number;
}

export interface ToolUsageStats {
  tool: string;
  count: number;
}

export interface GraphNode {
  id: number;
  label: string;
  x: number;
  y: number;
  size: 'large' | 'medium' | 'small';
}

export interface GraphEdge {
  from: number;
  to: number;
}

export interface Process {
  name: string;
  cpu: number;
  mem: number;
}

export interface FileAccess {
  path: string;
  action: string;
  time: string;
}

export interface WebSocketMessage {
  type: string;
  payload: unknown;
}
