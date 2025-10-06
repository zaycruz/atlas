import React from 'react';
import { Bot, Brain, Zap, Circle } from 'lucide-react';

export type AgentState = 'idle' | 'thinking' | 'processing' | 'responding';

interface AgentStatusProps {
  status: AgentState;
}

export const AgentStatus: React.FC<AgentStatusProps> = ({ status }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'idle':
        return {
          icon: Circle,
          text: 'IDLE',
          color: 'text-atlas-green-700',
          bgColor: 'bg-atlas-green-950/30',
          pulse: false
        };
      case 'thinking':
        return {
          icon: Brain,
          text: 'THINKING',
          color: 'text-atlas-yellow-400',
          bgColor: 'bg-yellow-900/20',
          pulse: true
        };
      case 'processing':
        return {
          icon: Zap,
          text: 'PROCESSING',
          color: 'text-atlas-cyan-400',
          bgColor: 'bg-cyan-900/20',
          pulse: true
        };
      case 'responding':
        return {
          icon: Bot,
          text: 'RESPONDING',
          color: 'text-atlas-green-400',
          bgColor: 'bg-atlas-green-900/30',
          pulse: true
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1 rounded-md ${config.bgColor} border border-transparent`}
    >
      <Icon
        size={12}
        className={`${config.color} ${config.pulse ? 'animate-pulse' : ''}`}
      />
      <span className={`text-xs font-semibold ${config.color}`}>
        {config.text}
      </span>
      {config.pulse && (
        <div className="flex gap-1">
          <div className={`w-1 h-1 rounded-full ${config.color} animate-bounce delay-0`} />
          <div className={`w-1 h-1 rounded-full ${config.color} animate-bounce delay-75`} />
          <div className={`w-1 h-1 rounded-full ${config.color} animate-bounce delay-150`} />
        </div>
      )}
    </div>
  );
};
