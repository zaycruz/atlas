import { useState, useCallback, useEffect } from 'react';

interface UseCommandHistoryReturn {
  addCommand: (command: string) => void;
  navigateHistory: (direction: 'up' | 'down') => string | null;
  resetPosition: () => void;
  clearHistory: () => void;
  history: string[];
}

export const useCommandHistory = (maxSize: number = 100): UseCommandHistoryReturn => {
  const [history, setHistory] = useState<string[]>([]);
  const [position, setPosition] = useState<number>(-1);

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('atlas_command_history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      } catch (error) {
        console.error('Failed to load command history:', error);
      }
    }
  }, []);

  const addCommand = useCallback(
    (command: string) => {
      if (!command.trim()) return;

      setHistory((prev) => {
        // Don't add duplicate consecutive commands
        if (prev.length > 0 && prev[prev.length - 1] === command) {
          return prev;
        }

        // Add command and maintain max size
        const newHistory = [...prev, command];
        if (newHistory.length > maxSize) {
          newHistory.shift();
        }

        // Save to localStorage
        localStorage.setItem('atlas_command_history', JSON.stringify(newHistory));

        return newHistory;
      });

      // Reset position after adding
      setPosition(-1);
    },
    [maxSize]
  );

  const navigateHistory = useCallback(
    (direction: 'up' | 'down'): string | null => {
      if (history.length === 0) return null;

      setPosition((prevPosition) => {
        let newPosition = prevPosition;

        if (direction === 'up') {
          // Moving backward in history (older commands)
          if (prevPosition === -1) {
            newPosition = history.length - 1;
          } else if (prevPosition > 0) {
            newPosition = prevPosition - 1;
          }
        } else {
          // Moving forward in history (newer commands)
          if (prevPosition < history.length - 1 && prevPosition !== -1) {
            newPosition = prevPosition + 1;
          } else {
            newPosition = -1;
          }
        }

        return newPosition;
      });

      // Return the command at the new position
      if (direction === 'up') {
        if (position === -1) {
          return history[history.length - 1];
        } else if (position > 0) {
          return history[position - 1];
        } else {
          return history[position];
        }
      } else {
        if (position < history.length - 1 && position !== -1) {
          return history[position + 1];
        } else {
          return '';
        }
      }
    },
    [history, position]
  );

  const resetPosition = useCallback(() => {
    setPosition(-1);
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setPosition(-1);
    localStorage.removeItem('atlas_command_history');
  }, []);

  return {
    addCommand,
    navigateHistory,
    resetPosition,
    clearHistory,
    history
  };
};
