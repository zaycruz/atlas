import React from 'react';
import { Cpu, Network, Terminal, TrendingUp } from 'lucide-react';

interface Module {
  id: string;
  icon: React.ComponentType<{ size: number }>;
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
  atlasMetrics: {
    tokens: number;
    operations: number;
    inference: number;
  };
}

export const LeftSidebar: React.FC<LeftSidebarProps> = ({
  activeModule,
  setActiveModule,
  systemMetrics,
  atlasMetrics
}) => {
  const modules: Module[] = [
    { id: 'terminal', icon: Terminal, label: 'TERMINAL' },
    { id: 'analytics', icon: TrendingUp, label: 'ANALYTICS' },
    { id: 'network', icon: Network, label: 'NETWORK' },
    { id: 'system', icon: Cpu, label: 'SYSTEM' }
  ];

  return (
    <div className="col-span-2 border-r border-atlas-green-900 flex flex-col overflow-hidden">
      <div className="border-b border-atlas-green-900">
        <div className="text-[10px] text-atlas-yellow-400 px-2 py-1 border-b border-atlas-green-900">
          MODULES
        </div>
        {modules.map((mod) => {
          const Icon = mod.icon;
          const isActive = activeModule === mod.id;
          return (
            <button
              key={mod.id}
              onClick={() => setActiveModule(mod.id)}
              className={`w-full flex items-center gap-2 px-2 py-1 text-[11px] border-b border-atlas-green-900 transition-colors ${
                isActive
                  ? 'bg-white text-black'
                  : 'bg-black text-atlas-green-500 hover:bg-atlas-green-950'
              }`}
            >
              <Icon size={12} />
              {mod.label}
            </button>
          );
        })}
      </div>

      <div className="border-b border-atlas-green-900">
        <div className="text-[10px] text-atlas-yellow-400 px-2 py-1 border-b border-atlas-green-900">
          SYSTEM
        </div>
        <div className="px-2 py-1 space-y-1 text-[10px]">
          {Object.entries(systemMetrics).map(([key, value]) => (
            <div key={key}>
              <div className="flex justify-between mb-0.5">
                <span className="text-atlas-green-700 uppercase">{key}</span>
                <span className="text-atlas-green-400">{Math.round(value)}%</span>
              </div>
              <div className="w-full bg-atlas-green-950 h-0.5">
                <div className="bg-atlas-green-500 h-0.5" style={{ width: `${Math.min(100, value)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="border-b border-atlas-green-900">
        <div className="text-[10px] text-atlas-yellow-400 px-2 py-1 border-b border-atlas-green-900">
          ATLAS
        </div>
        <div className="px-2 py-1 space-y-1 text-[10px]">
          <div className="flex justify-between">
            <span className="text-atlas-green-700">Tokens</span>
            <span className="text-atlas-cyan-400">{atlasMetrics.tokens.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-atlas-green-700">Operations</span>
            <span className="text-atlas-cyan-400">{atlasMetrics.operations}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-atlas-green-700">Inference</span>
            <span className="text-atlas-cyan-400">{atlasMetrics.inference}ms</span>
          </div>
        </div>
      </div>
    </div>
  );
};
