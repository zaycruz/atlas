import React, { useState } from 'react';
import { Cpu, Network, Terminal, TrendingUp, ChevronDown } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface Module {
  id: string;
  icon: LucideIcon;
  label: string;
}

interface LeftSidebarProps {
  activeModule: string;
  setActiveModule: (id: string) => void;
  systemMetrics: {
    cpu: number;
    memory: number;
    network: number;
    disk: number;
  };
}

export const LeftSidebar: React.FC<LeftSidebarProps> = ({
  activeModule,
  setActiveModule,
  systemMetrics
}) => {
  const [isSystemExpanded, setIsSystemExpanded] = useState(true);
  const modules: Module[] = [
    { id: 'terminal', icon: Terminal, label: 'CHAT' },
    { id: 'analytics', icon: TrendingUp, label: 'ANALYTICS' },
    { id: 'network', icon: Network, label: 'NETWORK' },
    { id: 'system', icon: Cpu, label: 'SYSTEM' }
  ];

  return (
    <div className="col-span-2 border-r border-atlas-green-900 flex flex-col overflow-hidden">
      <div className="border-b border-atlas-green-900">
        <div className="text-xs font-semibold text-atlas-yellow-400 px-3 py-2 border-b border-atlas-green-900">
          MODULES
        </div>
        {modules.map((mod) => {
          const Icon = mod.icon;
          const isActive = activeModule === mod.id;
          return (
            <button
              key={mod.id}
              onClick={() => setActiveModule(mod.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 text-sm border-b border-atlas-green-900 transition-colors ${
                isActive
                  ? 'bg-white text-black'
                  : 'bg-black text-atlas-green-500 hover:bg-atlas-green-950'
              }`}
            >
              <Icon size={16} />
              {mod.label}
            </button>
          );
        })}
      </div>

      <div className="border-b border-atlas-green-900">
        <button
          onClick={() => setIsSystemExpanded(!isSystemExpanded)}
          className="w-full text-xs font-semibold text-atlas-yellow-400 px-3 py-2 border-b border-atlas-green-900 flex items-center justify-between hover:bg-atlas-green-950/30 transition-colors"
        >
          <span>SYSTEM</span>
          <ChevronDown
            size={14}
            className={`transition-transform ${isSystemExpanded ? 'rotate-180' : ''}`}
          />
        </button>
        {isSystemExpanded && (
          <div className="px-3 py-2 space-y-2 text-xs">
            {Object.entries(systemMetrics).map(([key, value]) => (
              <div key={key}>
                <div className="flex justify-between mb-1">
                  <span className="text-atlas-green-700 uppercase">{key}</span>
                  <span className="text-atlas-green-400">{Math.round(value)}%</span>
                </div>
                <div className="w-full bg-atlas-green-950 h-1 rounded-full">
                  <div className="bg-atlas-green-500 h-1 rounded-full" style={{ width: `${Math.min(100, value)}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
