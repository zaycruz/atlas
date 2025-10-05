import React from 'react';
import { TrendingUp } from 'lucide-react';
import type {
  ContextUsage,
  MemoryLayers,
  TopicDistribution,
  ToolUsageStats
} from '../types';

interface AnalyticsTabProps {
  topicDistribution: TopicDistribution[];
  toolUsage: ToolUsageStats[];
  memoryLayers: MemoryLayers;
  contextUsage: ContextUsage;
}

export const AnalyticsTab: React.FC<AnalyticsTabProps> = ({
  topicDistribution,
  toolUsage,
  memoryLayers,
  contextUsage
}) => {
  const getContextColor = (percentage: number) => {
    if (percentage > 80) return 'text-atlas-red-400';
    if (percentage > 60) return 'text-yellow-400';
    return 'text-atlas-cyan-400';
  };

  const filled = Math.min(20, Math.max(0, Math.floor(contextUsage.percentage / 5)));
  const progressBar = '█'.repeat(filled) + '░'.repeat(20 - filled);

  return (
    <div className="h-full overflow-y-auto">
      <div className="flex items-center gap-2 px-3 py-1 border-b border-atlas-green-900">
        <TrendingUp size={12} className="text-atlas-yellow-400" />
        <span className="text-[11px] text-atlas-yellow-400">USAGE ANALYTICS</span>
      </div>

      <div className="p-3 space-y-3">
        <div className="border border-atlas-green-900 p-2">
          <div className="text-[10px] text-atlas-cyan-400 mb-2">TOPIC DISTRIBUTION</div>
          <div className="space-y-1">
            {topicDistribution.map((topic, i) => (
              <div key={i}>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-atlas-green-500">{topic.topic}</span>
                  <span className="text-atlas-green-400">{topic.percentage}%</span>
                </div>
                <div className="w-full bg-atlas-green-950 h-1">
                  <div className="bg-atlas-cyan-500 h-1" style={{ width: `${topic.percentage}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-atlas-green-900 p-2">
          <div className="text-[10px] text-atlas-cyan-400 mb-2">TOOL USAGE</div>
          <div className="space-y-1 text-[10px]">
            {toolUsage.map((tool, i) => (
              <div key={i} className="flex justify-between border-b border-atlas-green-950 pb-0.5">
                <span className="text-atlas-green-500">{tool.tool}</span>
                <span className="text-atlas-green-400">{tool.count} calls</span>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-atlas-green-900 p-2">
          <div className="text-[10px] text-atlas-cyan-400 mb-2">MEMORY GROWTH</div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="border border-atlas-green-900 p-2">
              <div className="text-[9px] text-atlas-green-700">Episodes</div>
              <div className="text-xl text-atlas-cyan-400">{memoryLayers.episodes}</div>
            </div>
            <div className="border border-atlas-green-900 p-2">
              <div className="text-[9px] text-atlas-green-700">Facts</div>
              <div className="text-xl text-atlas-cyan-400">{memoryLayers.facts}</div>
            </div>
            <div className="border border-atlas-green-900 p-2">
              <div className="text-[9px] text-atlas-green-700">Insights</div>
              <div className="text-xl text-atlas-cyan-400">{memoryLayers.insights}</div>
            </div>
          </div>
        </div>

        <div className="border border-atlas-green-900 p-2">
          <div className="text-[10px] text-atlas-cyan-400 mb-2">CONTEXT PRESSURE</div>
          <div className="text-[10px]">
            <span className={getContextColor(contextUsage.percentage)}>
              {contextUsage.current}K / {contextUsage.max}K tokens
            </span>
            <span className="text-atlas-green-700 ml-2">[{progressBar}]</span>
          </div>
        </div>
      </div>
    </div>
  );
};
