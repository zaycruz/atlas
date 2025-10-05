import React from 'react';
import { Terminal } from 'lucide-react';
import type { TerminalEntry } from '../types';

interface TerminalTabProps {
  history: TerminalEntry[];
  input: string;
  setInput: (value: string) => void;
  onCommand: () => void;
  streamingText?: string;
}

export const TerminalTab: React.FC<TerminalTabProps> = ({
  history,
  input,
  setInput,
  onCommand,
  streamingText = ''
}) => {
  const getColorClass = (type: TerminalEntry['type']) => {
    switch (type) {
      case 'system':
        return 'text-atlas-yellow-400';
      case 'command':
        return 'text-atlas-green-500';
      case 'success':
        return 'text-atlas-green-400';
      case 'error':
        return 'text-atlas-red-400';
      case 'warn':
        return 'text-yellow-400';
      default:
        return 'text-atlas-green-500';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="bg-black px-3 py-1 border-b border-atlas-green-900 flex items-center gap-2">
        <Terminal size={12} className="text-atlas-yellow-400" />
        <span className="text-[11px] text-atlas-yellow-400">COMMAND TERMINAL</span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5 text-[11px]">
        {history.map((entry, i) => (
          <div key={i} className={getColorClass(entry.type)}>
            {entry.text}
          </div>
        ))}
        {streamingText && (
          <div className="text-atlas-green-400 opacity-80">
            {streamingText}
            <span className="animate-pulse">▋</span>
          </div>
        )}
      </div>

      <div className="border-t border-atlas-green-900 px-3 py-1">
        <div className="flex items-center gap-2">
          <span className="text-atlas-yellow-400 text-[11px]">$</span>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                onCommand();
              }
            }}
            className="flex-1 bg-transparent outline-none text-atlas-green-400 text-[11px]"
            placeholder="Enter command..."
          />
        </div>
      </div>
    </div>
  );
};
