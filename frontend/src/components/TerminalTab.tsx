import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Terminal, User, Bot, Trash2, Copy, Download } from 'lucide-react';
import type { TerminalEntry } from '../types';
import { Tooltip } from './Tooltip';

interface TerminalTabProps {
  history: TerminalEntry[];
  input: string;
  setInput: (value: string) => void;
  onCommand: () => void;
  streamingText?: string;
  onNavigateHistory?: (direction: 'up' | 'down') => string | null;
  onClear?: () => void;
}

export const TerminalTab: React.FC<TerminalTabProps> = ({
  history,
  input,
  setInput,
  onCommand,
  streamingText = '',
  onNavigateHistory,
  onClear
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

  // Copy terminal content to clipboard
  const handleCopy = useCallback(() => {
    const content = history.map((entry) => entry.text).join('\n');
    navigator.clipboard.writeText(content).then(() => {
      console.log('[Terminal] Copied to clipboard');
    }).catch((err) => {
      console.error('[Terminal] Failed to copy:', err);
    });
  }, [history]);

  // Export terminal content as text file
  const handleExport = useCallback(() => {
    const content = history.map((entry) => entry.text).join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `atlas-session-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [history]);

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
      <div className="bg-black px-4 py-2 border-b border-atlas-green-900 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal size={16} className="text-atlas-yellow-400" />
          <span className="text-sm font-semibold text-atlas-yellow-400">COMMAND TERMINAL</span>
        </div>

        <div className="flex items-center gap-2">
          <Tooltip content="Copy terminal content" position="bottom">
            <button
              onClick={handleCopy}
              className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
              aria-label="Copy"
            >
              <Copy size={14} className="text-atlas-green-500" />
            </button>
          </Tooltip>

          <Tooltip content="Export as text file" position="bottom">
            <button
              onClick={handleExport}
              className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
              aria-label="Export"
            >
              <Download size={14} className="text-atlas-green-500" />
            </button>
          </Tooltip>

          {onClear && (
            <Tooltip content="Clear terminal" position="bottom">
              <button
                onClick={onClear}
                className="p-1.5 hover:bg-red-900/30 rounded transition-colors"
                aria-label="Clear"
              >
                <Trash2 size={14} className="text-atlas-red-400" />
              </button>
            </Tooltip>
          )}
        </div>
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
