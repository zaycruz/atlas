import React, { useRef, useCallback, useEffect, useState } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2, Copy, Download } from 'lucide-react';
import type { TerminalEntry } from '../types';
import { Tooltip } from './Tooltip';

interface TerminalFooterProps {
  history: TerminalEntry[];
  onClear?: () => void;
  isExpanded: boolean;
  onToggle: () => void;
}

export const TerminalFooter: React.FC<TerminalFooterProps> = ({
  history,
  onClear,
  isExpanded,
  onToggle
}) => {
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Check if user is at bottom of scroll
  const checkScrollPosition = useCallback(() => {
    if (!terminalRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = terminalRef.current;
    const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;
    setAutoScroll(isAtBottom);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [history, autoScroll]);

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
    link.download = `atlas-terminal-${new Date().toISOString().slice(0, 10)}.txt`;
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
    <div
      className={`border-t border-atlas-green-900 bg-black transition-all duration-300 ${
        isExpanded ? 'h-80' : 'h-12'
      }`}
    >
      {/* Header */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-atlas-green-900">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-atlas-green-500" />
          <span className="text-xs font-semibold text-atlas-green-500">SYSTEM TERMINAL</span>
          <span className="text-xs text-atlas-green-700">({history.length} entries)</span>
        </div>

        <div className="flex items-center gap-2">
          {isExpanded && (
            <>
              <Tooltip content="Copy terminal content" position="top">
                <button
                  onClick={handleCopy}
                  className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
                  aria-label="Copy"
                >
                  <Copy size={12} className="text-atlas-green-500" />
                </button>
              </Tooltip>

              <Tooltip content="Export as text file" position="top">
                <button
                  onClick={handleExport}
                  className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
                  aria-label="Export"
                >
                  <Download size={12} className="text-atlas-green-500" />
                </button>
              </Tooltip>

              {onClear && (
                <Tooltip content="Clear terminal" position="top">
                  <button
                    onClick={onClear}
                    className="p-1.5 hover:bg-red-900/30 rounded transition-colors"
                    aria-label="Clear"
                  >
                    <Trash2 size={12} className="text-atlas-red-400" />
                  </button>
                </Tooltip>
              )}

              <div className="w-px h-4 bg-atlas-green-900 mx-1" />
            </>
          )}

          <Tooltip content={isExpanded ? 'Collapse terminal' : 'Expand terminal'} position="top">
            <button
              onClick={onToggle}
              className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? (
                <ChevronDown size={14} className="text-atlas-green-500" />
              ) : (
                <ChevronUp size={14} className="text-atlas-green-500" />
              )}
            </button>
          </Tooltip>
        </div>
      </div>

      {/* Terminal Content */}
      {isExpanded && (
        <div
          ref={terminalRef}
          onScroll={handleScroll}
          className="h-[calc(100%-3rem)] overflow-y-auto px-4 py-2 space-y-1 text-xs font-mono"
          style={{ overscrollBehavior: 'contain' }}
        >
          {history.map((entry, i) => (
            <div key={i} className={getColorClass(entry.type)}>
              {entry.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
