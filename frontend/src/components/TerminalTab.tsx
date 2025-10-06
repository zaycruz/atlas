import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Terminal, User, Bot } from 'lucide-react';
import type { TerminalEntry } from '../types';

interface TerminalTabProps {
  history: TerminalEntry[];
  input: string;
  setInput: (value: string) => void;
  onCommand: () => void;
  streamingText?: string;
  onNavigateHistory?: (direction: 'up' | 'down') => string | null;
}

export const TerminalTab: React.FC<TerminalTabProps> = ({
  history,
  input,
  setInput,
  onCommand,
  streamingText = '',
  onNavigateHistory
}) => {
  const historyRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    event.stopPropagation();
  }, []);

  // Check if user is at bottom of scroll
  const checkScrollPosition = useCallback(() => {
    if (!historyRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = historyRef.current;
    const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;
    setAutoScroll(isAtBottom);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [history, streamingText, autoScroll]);

  // Monitor scroll position
  const handleScroll = useCallback(() => {
    checkScrollPosition();
  }, [checkScrollPosition]);

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
      <div className="bg-black px-4 py-2 border-b border-atlas-green-900 flex items-center gap-2">
        <Terminal size={16} className="text-atlas-yellow-400" />
        <span className="text-sm font-semibold text-atlas-yellow-400">COMMAND TERMINAL</span>
      </div>

      <div
        ref={historyRef}
        onWheel={handleWheel}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-3 space-y-2 text-sm"
        style={{ overscrollBehavior: 'contain' }}
      >
        {history.map((entry, i) => {
          const isUserCommand = entry.type === 'command';
          const isAIResponse = entry.type === 'success';
          const isSystem = entry.type === 'system';

          return (
            <div
              key={i}
              className={`flex gap-3 ${
                isUserCommand
                  ? 'bg-atlas-green-950/30 border-l-2 border-atlas-cyan-400 pl-3 py-2'
                  : isAIResponse
                  ? 'bg-atlas-green-900/20 border-l-2 border-atlas-yellow-400 pl-3 py-2'
                  : 'pl-1 py-1'
              }`}
            >
              {isUserCommand && (
                <User size={16} className="text-atlas-cyan-400 flex-shrink-0 mt-0.5" />
              )}
              {isAIResponse && (
                <Bot size={16} className="text-atlas-yellow-400 flex-shrink-0 mt-0.5" />
              )}
              <div className={`flex-1 ${getColorClass(entry.type)}`}>
                {entry.text}
              </div>
            </div>
          );
        })}
        {streamingText && (
          <div className="flex gap-3 bg-atlas-green-900/20 border-l-2 border-atlas-yellow-400 pl-3 py-2">
            <Bot size={16} className="text-atlas-yellow-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1 text-atlas-green-400 opacity-80">
              {streamingText}
              <span className="animate-pulse">▋</span>
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-atlas-green-900 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-atlas-yellow-400 text-sm font-bold">$</span>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                onCommand();
              } else if (e.key === 'ArrowUp' && onNavigateHistory) {
                e.preventDefault();
                const command = onNavigateHistory('up');
                if (command !== null) {
                  setInput(command);
                }
              } else if (e.key === 'ArrowDown' && onNavigateHistory) {
                e.preventDefault();
                const command = onNavigateHistory('down');
                if (command !== null) {
                  setInput(command);
                }
              }
            }}
            className="flex-1 bg-transparent outline-none text-atlas-green-400 text-sm"
            placeholder="Enter command..."
          />
        </div>
      </div>
    </div>
  );
};
