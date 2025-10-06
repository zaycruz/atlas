import React from 'react';
import { Cpu, Database, HardDrive, Terminal, Zap } from 'lucide-react';
import type { FileAccess, Process } from '../types';

interface SystemTabProps {
  processes: Process[];
  fileAccess: FileAccess[];
}

export const SystemTab: React.FC<SystemTabProps> = ({ processes, fileAccess }) => {
  const quickActions = [
    { label: 'Clear Cache', icon: Database },
    { label: 'Restart Agent', icon: Zap },
    { label: 'View Logs', icon: Terminal },
    { label: 'Disk Usage', icon: HardDrive }
  ];

  return (
    <div className="h-full overflow-y-auto">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-atlas-green-900">
        <Cpu size={16} className="text-atlas-yellow-400" />
        <span className="text-sm font-semibold text-atlas-yellow-400">SYSTEM MONITOR</span>
      </div>

      <div className="p-4 space-y-4">
        <div className="border border-atlas-green-900 p-3">
          <div className="text-xs font-semibold text-atlas-cyan-400 mb-3">ATLAS PROCESSES</div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between text-xs text-atlas-green-700 border-b border-atlas-green-950 pb-1">
              <span className="flex-1">Process</span>
              <span className="w-20 text-right">CPU</span>
              <span className="w-20 text-right">Memory</span>
            </div>
            {processes.map((proc) => (
              <div key={proc.name} className="flex justify-between border-b border-atlas-green-950 pb-1">
                <span className="flex-1 text-atlas-green-500">{proc.name}</span>
                <span className="w-20 text-right text-atlas-cyan-400">{proc.cpu}%</span>
                <span className="w-20 text-right text-atlas-cyan-400">{proc.mem}MB</span>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-atlas-green-900 p-3">
          <div className="text-xs font-semibold text-atlas-cyan-400 mb-3">RECENT FILE ACCESS</div>
          <div className="space-y-2 text-sm">
            {fileAccess.map((file, i) => (
              <div key={`${file.path}-${i}`} className="border-l-2 border-atlas-green-700 pl-2 py-1">
                <div className="text-atlas-green-500">{file.path}</div>
                <div className="flex justify-between text-atlas-green-700 text-xs">
                  <span>{file.action}</span>
                  <span>{file.time}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-atlas-green-900 p-3">
          <div className="text-xs font-semibold text-atlas-cyan-400 mb-3">QUICK ACTIONS</div>
          <div className="grid grid-cols-2 gap-2">
            {quickActions.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.label}
                  className="border border-atlas-green-900 px-3 py-2 text-sm text-atlas-green-500 hover:bg-atlas-green-950 hover:text-atlas-cyan-400 transition-colors flex items-center gap-2"
                  type="button"
                >
                  <Icon size={14} />
                  {action.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
