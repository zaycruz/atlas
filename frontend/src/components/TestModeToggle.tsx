import React from 'react';
import { FlaskConical, Activity } from 'lucide-react';
import { Tooltip } from './Tooltip';

interface TestModeToggleProps {
  enabled: boolean;
  onToggle: () => void;
}

export const TestModeToggle: React.FC<TestModeToggleProps> = ({
  enabled,
  onToggle
}) => {
  return (
    <Tooltip
      content={enabled ? "Test Mode ON - Interactions not logged" : "Test Mode OFF - Interactions logged to memory"}
      position="bottom"
    >
      <button
        onClick={onToggle}
        className={`flex items-center gap-2 px-3 py-1.5 border rounded-md transition-colors ${
          enabled
            ? 'bg-atlas-yellow-900/30 border-atlas-yellow-600 hover:border-atlas-yellow-400'
            : 'bg-atlas-green-950/30 border-atlas-green-900 hover:border-atlas-cyan-400'
        }`}
      >
        {enabled ? (
          <FlaskConical size={14} className="text-atlas-yellow-400" />
        ) : (
          <Activity size={14} className="text-atlas-green-400" />
        )}
        <span className={`text-xs font-semibold ${enabled ? 'text-atlas-yellow-400' : 'text-atlas-green-400'}`}>
          {enabled ? 'TEST MODE' : 'LIVE MODE'}
        </span>
      </button>
    </Tooltip>
  );
};
