import React, { useState } from 'react';
import type { ContextUsage, MemoryEvent, MemoryLayers, ToolRun } from '../types';

interface RightSidebarProps {
  contextUsage: ContextUsage;
  memoryLayers: MemoryLayers;
  memoryEvents: MemoryEvent[];
  toolRuns: ToolRun[];
}

export const RightSidebar: React.FC<RightSidebarProps> = ({
  contextUsage,
  memoryLayers,
  memoryEvents,
  toolRuns
}) => {
  const [expandedToolId, setExpandedToolId] = useState<string | null>(null);

  const getContextColor = (percentage: number) => {
    if (percentage > 80) return 'text-atlas-red-400';
    if (percentage > 60) return 'text-yellow-400';
    return 'text-atlas-cyan-400';
  };

  const filled = Math.min(10, Math.max(0, Math.floor(contextUsage.percentage / 10)));
  const progressBar = '█'.repeat(filled) + '░'.repeat(10 - filled);

  return (
    <div className="col-span-3 flex flex-col overflow-hidden">
      <div className="border-b border-atlas-green-900">
        <div className="text-xs font-semibold text-atlas-cyan-400 px-3 py-2 border-b border-atlas-green-900">
          Memory
        </div>

        <div className="px-3 py-2 border-b border-atlas-green-900">
          <div className="text-xs text-atlas-green-700 mb-2 font-semibold">CONTEXT USAGE</div>
          <div className="text-sm mb-1">
            <span className={getContextColor(contextUsage.percentage)}>
              Context: {contextUsage.current}K
            </span>
            <span className="text-atlas-green-700 ml-2">[{progressBar}]</span>
            <span className={`ml-2 ${getContextColor(contextUsage.percentage)}`}>
              {contextUsage.percentage}%
            </span>
          </div>
        </div>

        <div className="px-3 py-2 border-b border-atlas-green-900">
          <div className="text-xs text-atlas-green-700 mb-2 font-semibold">MEMORY LAYERS</div>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-atlas-green-600">Episodes:</span>
              <span className="text-atlas-green-400">{memoryLayers.episodes}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-atlas-green-600">Facts:</span>
              <span className="text-atlas-green-400">{memoryLayers.facts}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-atlas-green-600">Insights:</span>
              <span className="text-atlas-green-400">{memoryLayers.insights}</span>
            </div>
          </div>
        </div>

        <div className="px-3 py-2 border-b border-atlas-green-900">
          <div className="text-xs text-atlas-green-700 mb-2 font-semibold">MEMORY EVENTS</div>
          <div className="space-y-2 text-xs">
            {memoryEvents.slice(0, 5).map((event, i) => (
              <div key={i} className="border-l-2 border-atlas-green-700 pl-2 py-1">
                <div className="text-atlas-green-400">
                  {event.time} {event.type}
                </div>
                <div className="text-atlas-green-700 truncate">
                  {event.detail}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border-b border-atlas-green-900">
        <div className="text-xs font-semibold text-atlas-cyan-400 px-3 py-2 border-b border-atlas-green-900">
          Tool Drawer
        </div>
        <div className="px-3 py-2 text-sm">
          {toolRuns.length === 0 ? (
            <div className="text-atlas-green-600">No tool runs yet.</div>
          ) : (
            <div className="space-y-2">
              {toolRuns.slice(0, 5).map((tool, i) => {
                const isExpanded = expandedToolId === tool.id;
                const hasError = tool.error;
                const borderColor = hasError ? 'border-atlas-red-500' : 'border-atlas-green-700';

                return (
                  <div
                    key={i}
                    className={`border-l-2 ${borderColor} pl-2 py-1 ${hasError ? 'bg-atlas-red-950/20' : ''}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className={`text-xs ${hasError ? 'text-atlas-red-400' : 'text-atlas-green-400'}`}>
                        {tool.name}
                        {hasError && <span className="ml-1">⚠</span>}
                      </div>
                      {tool.execution_time !== undefined && (
                        <div className="text-atlas-green-600 text-xs">
                          {tool.execution_time.toFixed(3)}s
                        </div>
                      )}
                    </div>
                    <div className={`text-xs truncate ${hasError ? 'text-atlas-red-600' : 'text-atlas-green-700'}`}>
                      {tool.summary}
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="text-atlas-green-800 text-xs">
                        ID: {tool.id} • {tool.time}
                      </div>
                      {hasError && (
                        <button
                          onClick={() => setExpandedToolId(isExpanded ? null : tool.id)}
                          className="text-xs text-atlas-red-500 hover:text-atlas-red-400 underline"
                        >
                          {isExpanded ? 'Hide' : 'Details'}
                        </button>
                      )}
                    </div>
                    {hasError && isExpanded && (
                      <div className="mt-2 p-2 bg-atlas-red-950/30 border border-atlas-red-800 rounded text-xs">
                        <div className="text-atlas-red-400 font-semibold mb-1">
                          {tool.error_type || 'Error'}
                        </div>
                        <div className="text-atlas-red-600 break-words">
                          {tool.error_message || 'An error occurred'}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
