import React, { useState, useRef, useEffect } from 'react';

interface TooltipProps {
  content: string | React.ReactNode;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'top',
  delay = 300
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const timeoutRef = useRef<NodeJS.Timeout>();
  const triggerRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      if (triggerRef.current) {
        const rect = triggerRef.current.getBoundingClientRect();
        let x = 0;
        let y = 0;

        switch (position) {
          case 'top':
            x = rect.left + rect.width / 2;
            y = rect.top;
            break;
          case 'bottom':
            x = rect.left + rect.width / 2;
            y = rect.bottom;
            break;
          case 'left':
            x = rect.left;
            y = rect.top + rect.height / 2;
            break;
          case 'right':
            x = rect.right;
            y = rect.top + rect.height / 2;
            break;
        }

        setCoords({ x, y });
        setIsVisible(true);
      }
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const getPositionClasses = () => {
    switch (position) {
      case 'top':
        return '-translate-x-1/2 -translate-y-full mb-2';
      case 'bottom':
        return '-translate-x-1/2 mt-2';
      case 'left':
        return '-translate-y-1/2 -translate-x-full mr-2';
      case 'right':
        return '-translate-y-1/2 ml-2';
      default:
        return '-translate-x-1/2 -translate-y-full mb-2';
    }
  };

  return (
    <div className="relative inline-block">
      <div
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleMouseEnter}
        onBlur={handleMouseLeave}
        tabIndex={0}
      >
        {children}
      </div>
      {isVisible && (
        <div
          className={`fixed z-50 px-3 py-2 bg-atlas-green-950 border border-atlas-green-700 rounded-md shadow-lg text-xs text-atlas-green-400 pointer-events-none whitespace-nowrap ${getPositionClasses()}`}
          style={{
            left: `${coords.x}px`,
            top: `${coords.y}px`
          }}
        >
          {content}
          <div
            className={`absolute w-2 h-2 bg-atlas-green-950 border-atlas-green-700 transform rotate-45 ${
              position === 'top'
                ? 'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 border-r border-b'
                : position === 'bottom'
                ? 'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 border-l border-t'
                : position === 'left'
                ? 'right-0 top-1/2 translate-x-1/2 -translate-y-1/2 border-r border-t'
                : 'left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 border-l border-b'
            }`}
          />
        </div>
      )}
    </div>
  );
};
