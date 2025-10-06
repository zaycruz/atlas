import React, { useRef, useCallback, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import { MessageSquare, User, Bot, Trash2 } from 'lucide-react';
import type { TerminalEntry } from '../types';
import { Tooltip } from './Tooltip';

interface ChatTabProps {
  history: TerminalEntry[];
  input: string;
  setInput: (value: string) => void;
  onCommand: () => void;
  streamingText?: string;
  onNavigateHistory?: (direction: 'up' | 'down') => string | null;
  onClear?: () => void;
}

export interface ChatTabRef {
  focusInput: () => void;
}

export const ChatTab = forwardRef<ChatTabRef, ChatTabProps>(({
  history,
  input,
  setInput,
  onCommand,
  streamingText = '',
  onNavigateHistory,
  onClear
}, ref) => {
  const chatRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useImperativeHandle(ref, () => ({
    focusInput: () => {
      inputRef.current?.focus();
    }
  }));

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    event.stopPropagation();
  }, []);

  // Check if user is at bottom of scroll
  const checkScrollPosition = useCallback(() => {
    if (!chatRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = chatRef.current;
    const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;
    setAutoScroll(isAtBottom);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [history, streamingText, autoScroll]);

  // Monitor scroll position
  const handleScroll = useCallback(() => {
    checkScrollPosition();
  }, [checkScrollPosition]);

  const formatTime = (timestamp?: number): string => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderMessage = (entry: TerminalEntry, index: number) => {
    const isUserMessage = entry.type === 'command';
    const isAIMessage = entry.type === 'success';
    const isSystemMessage = entry.type === 'system';

    // Remove the $ prefix from user commands for chat display
    const messageText = isUserMessage ? entry.text.replace(/^\$\s*/, '') : entry.text;

    if (isSystemMessage) {
      return (
        <div key={index} className="flex justify-center py-2">
          <div className="text-xs text-atlas-green-700 bg-atlas-green-950/30 px-3 py-1 rounded-full">
            {messageText}
          </div>
        </div>
      );
    }

    return (
      <div
        key={index}
        className={`flex gap-3 mb-4 ${isUserMessage ? 'justify-end' : 'justify-start'}`}
      >
        {isAIMessage && (
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-atlas-yellow-400/20 flex items-center justify-center">
            <Bot size={18} className="text-atlas-yellow-400" />
          </div>
        )}

        <div
          className={`max-w-[70%] rounded-2xl px-4 py-3 ${
            isUserMessage
              ? 'bg-atlas-cyan-400/20 border border-atlas-cyan-400/30 text-atlas-green-400'
              : 'bg-atlas-green-900/30 border border-atlas-green-800 text-atlas-green-400'
          }`}
        >
          <div className="text-sm whitespace-pre-wrap break-words">{messageText}</div>
          {entry.timestamp && (
            <div className="text-xs text-atlas-green-700 mt-1 opacity-70">
              {formatTime(entry.timestamp)}
            </div>
          )}
        </div>

        {isUserMessage && (
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-atlas-cyan-400/20 flex items-center justify-center">
            <User size={18} className="text-atlas-cyan-400" />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-atlas-black">
      <div className="bg-black px-4 py-3 border-b border-atlas-green-900 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare size={16} className="text-atlas-yellow-400" />
          <span className="text-sm font-semibold text-atlas-yellow-400">ATLAS INTERFACE</span>
        </div>
        {onClear && (
          <Tooltip content="Clear chat" position="bottom">
            <button
              onClick={onClear}
              className="p-1.5 hover:bg-red-900/30 rounded transition-colors"
              aria-label="Clear"
            >
              <Trash2 size={12} className="text-atlas-red-400" />
            </button>
          </Tooltip>
        )}
      </div>

      <div
        ref={chatRef}
        onWheel={handleWheel}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-4"
        style={{ overscrollBehavior: 'contain' }}
      >
        <div className="max-w-4xl mx-auto">
          {history.map((entry, i) => renderMessage(entry, i))}

          {streamingText && (
            <div className="flex gap-3 mb-4 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-atlas-yellow-400/20 flex items-center justify-center">
                <Bot size={18} className="text-atlas-yellow-400" />
              </div>

              <div className="max-w-[70%] rounded-2xl px-4 py-3 bg-atlas-green-900/30 border border-atlas-green-800 text-atlas-green-400">
                <div className="text-sm whitespace-pre-wrap break-words">
                  {streamingText}
                  <span className="inline-block animate-pulse ml-1">▋</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-atlas-green-900 px-4 py-4 bg-atlas-green-950/20">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 bg-atlas-green-950/50 border border-atlas-green-900 rounded-2xl px-4 py-3 focus-within:border-atlas-cyan-400 transition-colors">
            <User size={16} className="text-atlas-cyan-400 flex-shrink-0" />
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
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
              className="flex-1 bg-transparent outline-none text-atlas-green-400 text-sm placeholder-atlas-green-700"
              placeholder="Ask ATLAS anything..."
            />
          </div>
          <div className="text-xs text-atlas-green-700 mt-2 text-center">
            Press Enter to send • Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
});
