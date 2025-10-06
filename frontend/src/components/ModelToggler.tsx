import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Plus } from 'lucide-react';

export type AIModel = string;

interface ModelTogglerProps {
  currentModel: AIModel;
  installed: string[];
  available: string[];
  onModelChange: (model: AIModel) => void;
  onAddModel: (model: AIModel) => void;
  pullStatus?: {
    model: string | null;
    status: 'idle' | 'started' | 'progress' | 'completed' | 'error';
    message?: string;
  };
  disabled?: boolean;
}

const formatModelName = (model: string) =>
  model.replace(/:latest$/i, '').replace(/_/g, ' ').replace(/\b([a-z])/g, (m) => m.toUpperCase());

export const ModelToggler: React.FC<ModelTogglerProps> = ({
  currentModel,
  installed,
  available,
  onModelChange,
  onAddModel,
  pullStatus,
  disabled = false
}) => {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const installedModels = useMemo(
    () => Array.from(new Set(installed)).sort(),
    [installed]
  );

  const addableModels = useMemo(
    () => Array.from(new Set(available)).filter((model) => !installedModels.includes(model)).sort(),
    [available, installedModels]
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

  const isPulling = pullStatus && pullStatus.status !== 'idle' && pullStatus.model;

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setOpen((prev) => !prev)}
        disabled={disabled}
        className={`flex items-center gap-3 px-3 py-2 bg-atlas-green-950/30 border border-atlas-green-900 rounded-md transition-colors ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-atlas-cyan-400'
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <div className="flex flex-col items-start text-left">
          <span className="text-xs font-semibold text-atlas-green-400">Model</span>
          <span className="text-sm text-white">{activeLabel}</span>
          {isPulling && pullStatus?.model === currentModel && (
            <span className="text-[10px] text-atlas-green-600">{pullStatus?.message || 'Updating...'}</span>
          )}
        </div>
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
          <div className="px-3 py-2 border-b border-atlas-green-900">
            <div className="text-[10px] font-semibold text-atlas-green-600">INSTALLED MODELS</div>
            <div className="mt-2 space-y-1">
              {installedModels.length === 0 && (
                <div className="text-xs text-atlas-green-600">No models installed yet.</div>
              )}
              {installedModels.map((model) => (
                <button
                  key={model}
                  onClick={() => handleSelect(model)}
                  className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
                    model === currentModel ? 'bg-atlas-green-900/70 text-white' : 'hover:bg-atlas-green-950 text-atlas-green-400'
                  }`}
                  role="option"
                  aria-selected={model === currentModel}
                >
                  {formatModelName(model)}
                </button>
              ))}
            </div>
          </div>

          <div className="px-3 py-2">
            <div className="text-[10px] font-semibold text-atlas-green-600">AVAILABLE TO ADD</div>
            <div className="mt-2 space-y-1">
              {addableModels.length === 0 && (
                <div className="text-xs text-atlas-green-600">All known models installed.</div>
              )}
              {addableModels.map((model) => {
                const isPullingThis = pullStatus?.model === model && pullStatus.status !== 'completed' && pullStatus.status !== 'error';
                return (
                  <button
                    key={model}
                    onClick={() => {
                      setOpen(false);
                      onAddModel(model);
                    }}
                    className="flex w-full items-center justify-between px-2 py-1.5 rounded text-sm text-atlas-green-400 hover:bg-atlas-green-950 transition-colors"
                    role="option"
                  >
                    <span>{formatModelName(model)}</span>
                    <span className="flex items-center gap-1 text-[11px] text-atlas-cyan-400">
                      <Plus size={12} />
                      {isPullingThis ? 'Pulling...' : 'Add'}
                    </span>
                  </button>
                );
              })}
            </div>
            {pullStatus?.status === 'error' && pullStatus.model && (
              <div className="mt-2 text-[11px] text-atlas-red-400">
                Failed to pull {formatModelName(pullStatus.model)}: {pullStatus.message || 'Unknown error'}
              </div>
            )}
            {pullStatus?.status === 'completed' && pullStatus.model && (
              <div className="mt-2 text-[11px] text-atlas-green-500">
                Added {formatModelName(pullStatus.model)} successfully.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
