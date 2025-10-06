import { useEffect } from 'react';

interface ShortcutHandler {
  key: string;
  ctrl?: boolean;
  meta?: boolean;
  shift?: boolean;
  handler: () => void;
  description: string;
}

export const useKeyboardShortcuts = (shortcuts: ShortcutHandler[]) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
        const metaMatch = shortcut.meta ? event.metaKey : !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

        // On Mac, Cmd is used instead of Ctrl
        const modifierMatch = shortcut.ctrl || shortcut.meta
          ? (event.ctrlKey || event.metaKey) && shiftMatch
          : ctrlMatch && metaMatch && shiftMatch;

        if (keyMatch && modifierMatch) {
          // Don't prevent default if user is typing in an input
          const target = event.target as HTMLElement;
          const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';

          // Allow Escape key even in inputs
          if (shortcut.key === 'Escape' || !isInput) {
            event.preventDefault();
            shortcut.handler();
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

export const getShortcutLabel = (shortcut: ShortcutHandler): string => {
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const parts: string[] = [];

  if (shortcut.ctrl || shortcut.meta) {
    parts.push(isMac ? '⌘' : 'Ctrl');
  }
  if (shortcut.shift) {
    parts.push(isMac ? '⇧' : 'Shift');
  }
  parts.push(shortcut.key.toUpperCase());

  return parts.join('+');
};
