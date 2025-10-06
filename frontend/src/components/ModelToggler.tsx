import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';

export type AIModel = string;

interface ModelTogglerProps {
  currentModel: AIModel;
  installed: string[];
  onModelChange: (model: AIModel) => void;
  disabled?: boolean;
}

const formatModelName = (model: string) =>
  model.replace(/:latest$/i, '').replace(/_/g, ' ').replace(/\b([a-z])/g, (m) => m.toUpperCase());

export const ModelToggler: React.FC<ModelTogglerProps> = ({
  currentModel,
  installed,
  onModelChange,
  disabled = false
}) => {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const installedModels = useMemo(
    () => Array.from(new Set(installed)).sort(),
    [installed]
  );

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, []);

  useEffect(() => {
    if (disabled) {
      setOpen(false);
    }
  }, [disabled]);

  const handleSelect = (modelId: AIModel) => {
    setOpen(false);
    if (modelId !== currentModel) {
      onModelChange(modelId);
    }
  };

  const activeLabel = formatModelName(currentModel || installedModels[0] || 'Unknown');

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setOpen((prev) => !prev)}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-1.5 bg-atlas-green-950/30 border border-atlas-green-900 rounded-md transition-colors ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-atlas-cyan-400'
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="text-xs font-semibold text-atlas-green-400">{activeLabel}</span>
        <ChevronDown
          size={14}
          className={`text-atlas-green-500 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div
          className="absolute right-0 z-40 mt-1 w-64 rounded-md border border-atlas-green-900 bg-atlas-black shadow-lg"
          role="listbox"
        >
          <div className="px-3 py-2">
            <div className="text-[10px] font-semibold text-atlas-green-600 mb-2">INSTALLED MODELS</div>
            <div className="space-y-1">
              {installedModels.length === 0 && (
                <div className="text-xs text-atlas-green-600">No models installed yet.</div>
              )}
              {installedModels.map((model) => (
                <button
                  key={model}
                  onClick={() => handleSelect(model)}
                  className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                    model === currentModel
                      ? 'bg-atlas-green-900/70 text-white border border-atlas-cyan-400'
                      : 'bg-atlas-green-950/30 border border-atlas-green-900 text-atlas-green-400 hover:border-atlas-cyan-400'
                  }`}
                  role="option"
                  aria-selected={model === currentModel}
                >
                  {formatModelName(model)}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
