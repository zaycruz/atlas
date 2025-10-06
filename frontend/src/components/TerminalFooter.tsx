import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2, Copy, Download, Search, X } from 'lucide-react';
import type { TerminalEntry } from '../types';
import { Tooltip } from './Tooltip';

interface TerminalFooterProps {
  history: TerminalEntry[];
  input: string;
  setInput: (value: string) => void;
  onCommand: () => void;
  onClear?: () => void;
  isExpanded: boolean;
  onToggle: () => void;
  onNavigateHistory?: (direction: 'up' | 'down') => string | null;
}

export const TerminalFooter: React.FC<TerminalFooterProps> = ({
  history,
  input,
  setInput,
  onCommand,
  onClear,
  isExpanded,
  onToggle,
  onNavigateHistory
}) => {
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchOpen, setIsSearchOpen] = useState(false);

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
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
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
    setExportSuccess(true);
    setTimeout(() => setExportSuccess(false), 2000);
  }, [history]);

  // Filter history based on search query
  const filteredHistory = useMemo(() => {
    if (!searchQuery.trim()) return history;
    const query = searchQuery.toLowerCase();
    return history.filter((entry) => entry.text.toLowerCase().includes(query));
  }, [history, searchQuery]);

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

  const handleClearSearch = () => {
    setSearchQuery('');
    setIsSearchOpen(false);
  };

  return (
    <div
      className={`border-t border-atlas-green-900 bg-black transition-all duration-300 ${
        isExpanded ? 'h-96' : 'h-12'
      }`}
    >
      {/* Header */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-atlas-green-900">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-atlas-green-500" />
          <span className="text-xs font-semibold text-atlas-green-500">SYSTEM TERMINAL</span>
          <span className="text-xs text-atlas-green-700">
            ({searchQuery ? `${filteredHistory.length}/${history.length}` : `${history.length} entries`})
          </span>
        </div>

        <div className="flex items-center gap-2">
          {isExpanded && (
            <>
              {isSearchOpen ? (
                <div className="flex items-center gap-1 bg-atlas-green-950/50 border border-atlas-green-900 rounded px-2 py-1">
                  <Search size={10} className="text-atlas-green-500" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search..."
                    className="bg-transparent outline-none text-xs text-atlas-green-400 placeholder-atlas-green-700 w-32"
                    autoFocus
                  />
                  <button
                    onClick={handleClearSearch}
                    className="p-0.5 hover:bg-atlas-green-900 rounded"
                  >
                    <X size={10} className="text-atlas-green-500" />
                  </button>
                </div>
              ) : (
                <Tooltip content="Search terminal" position="top">
                  <button
                    onClick={() => setIsSearchOpen(true)}
                    className="p-1.5 hover:bg-atlas-green-950 rounded transition-colors"
                    aria-label="Search"
                  >
                    <Search size={12} className="text-atlas-green-500" />
                  </button>
                </Tooltip>
              )}

              <Tooltip content={copySuccess ? "Copied!" : "Copy terminal content"} position="top">
                <button
                  onClick={handleCopy}
                  className={`p-1.5 rounded transition-colors ${
                    copySuccess
                      ? 'bg-atlas-green-500/20'
                      : 'hover:bg-atlas-green-950'
                  }`}
                  aria-label="Copy"
                >
                  <Copy size={12} className={copySuccess ? 'text-atlas-cyan-400' : 'text-atlas-green-500'} />
                </button>
              </Tooltip>

              <Tooltip content={exportSuccess ? "Exported!" : "Export as text file"} position="top">
                <button
                  onClick={handleExport}
                  className={`p-1.5 rounded transition-colors ${
                    exportSuccess
                      ? 'bg-atlas-green-500/20'
                      : 'hover:bg-atlas-green-950'
                  }`}
                  aria-label="Export"
                >
                  <Download size={12} className={exportSuccess ? 'text-atlas-cyan-400' : 'text-atlas-green-500'} />
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
        <>
          <div
            ref={terminalRef}
            onScroll={handleScroll}
            className="h-[calc(100%-6rem)] overflow-y-auto px-4 py-2 space-y-1 text-xs font-mono"
            style={{ overscrollBehavior: 'contain' }}
          >
            {filteredHistory.length > 0 ? (
              filteredHistory.map((entry, i) => (
                <div key={i} className={getColorClass(entry.type)}>
                  {entry.text}
                </div>
              ))
            ) : (
              <div className="text-atlas-green-700 text-center py-4">
                {searchQuery ? 'No matching entries found' : 'No terminal output'}
              </div>
            )}
          </div>

          {/* Terminal Input */}
          <div className="h-12 px-4 py-2 border-t border-atlas-green-900 bg-black">
            <div className="flex items-center gap-2 h-full">
              <span className="text-atlas-green-500 font-mono text-xs">$</span>
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
                className="flex-1 bg-transparent outline-none text-atlas-green-400 font-mono text-xs placeholder-atlas-green-700"
                placeholder="Enter terminal command..."
                autoComplete="off"
                spellCheck={false}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};
