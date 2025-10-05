import React from 'react';
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
        <div className="text-[10px] text-atlas-cyan-400 px-2 py-1 border-b border-atlas-green-900">
          Memory
        </div>

        <div className="px-2 py-1 border-b border-atlas-green-900">
          <div className="text-[9px] text-atlas-green-700 mb-1">CONTEXT USAGE</div>
          <div className="text-[10px] mb-0.5">
            <span className={getContextColor(contextUsage.percentage)}>
              Context: {contextUsage.current}K
            </span>
            <span className="text-atlas-green-700 ml-1">[{progressBar}]</span>
            <span className={`ml-1 ${getContextColor(contextUsage.percentage)}`}>
              {contextUsage.percentage}%
            </span>
          </div>
        </div>

        <div className="px-2 py-1 border-b border-atlas-green-900">
          <div className="text-[9px] text-atlas-green-700 mb-1">MEMORY LAYERS</div>
          <div className="space-y-0.5 text-[10px]">
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

        <div className="px-2 py-1 border-b border-atlas-green-900">
          <div className="text-[9px] text-atlas-green-700 mb-1">MEMORY EVENTS</div>
          <div className="space-y-1 text-[10px]">
            {memoryEvents.slice(0, 5).map((event, i) => (
              <div key={i} className="border-l border-atlas-green-700 pl-1 py-0.5">
                <div className="text-atlas-green-400 text-[9px]">
                  {event.time} {event.type}
                </div>
                <div className="text-atlas-green-700 text-[9px] truncate">
                  {event.detail}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border-b border-atlas-green-900">
        <div className="text-[10px] text-atlas-cyan-400 px-2 py-1 border-b border-atlas-green-900">
          Tool Drawer
        </div>
        <div className="px-2 py-1 text-[10px]">
          {toolRuns.length === 0 ? (
            <div className="text-atlas-green-600">No tool runs yet.</div>
          ) : (
            <div className="space-y-1">
              {toolRuns.slice(0, 3).map((tool, i) => (
                <div key={i} className="border-l border-atlas-green-700 pl-1 py-0.5">
                  <div className="text-atlas-green-400 text-[9px]">{tool.name}</div>
                  <div className="text-atlas-green-700 text-[9px] truncate">{tool.summary}</div>
                  <div className="text-atlas-green-800 text-[8px]">
                    ID: {tool.id} • {tool.time}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
