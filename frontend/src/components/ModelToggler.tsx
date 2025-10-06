import React from 'react';
import { Cpu, Zap, Brain } from 'lucide-react';
import { Tooltip } from './Tooltip';

export type AIModel = 'qwen2.5:latest' | 'qwen3:latest' | 'gpt-oss:latest';

interface ModelTogglerProps {
  currentModel: AIModel;
  onModelChange: (model: AIModel) => void;
  disabled?: boolean;
}

const models = [
  {
    id: 'qwen2.5:latest' as AIModel,
    name: 'Qwen 2.5',
    icon: Zap,
    description: 'Fast & efficient Qwen model',
    color: 'text-atlas-cyan-400'
  },
  {
    id: 'qwen3:latest' as AIModel,
    name: 'Qwen 3',
    icon: Brain,
    description: 'Latest Qwen generation',
    color: 'text-atlas-yellow-400'
  },
  {
    id: 'gpt-oss:latest' as AIModel,
    name: 'GPT-OSS',
    icon: Cpu,
    description: 'Open source GPT model',
    color: 'text-atlas-green-400'
  }
];

export const ModelToggler: React.FC<ModelTogglerProps> = ({
  currentModel,
  onModelChange,
  disabled = false
}) => {
  return (
    <div className="flex items-center gap-1 bg-atlas-green-950/30 border border-atlas-green-900 rounded-md p-1">
      {models.map((model) => {
        const Icon = model.icon;
        const isActive = currentModel === model.id;

        return (
          <Tooltip
            key={model.id}
            content={
              <div>
                <div className="font-semibold">{model.name}</div>
                <div className="text-xs text-atlas-green-600">{model.description}</div>
              </div>
            }
            position="bottom"
          >
            <button
              onClick={() => onModelChange(model.id)}
              disabled={disabled}
              className={`flex items-center gap-2 px-3 py-1.5 rounded transition-all ${
                isActive
                  ? 'bg-white text-black shadow-md'
                  : `${model.color} hover:bg-atlas-green-900/50`
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <Icon size={14} className={isActive ? 'text-black' : ''} />
              <span className={`text-xs font-semibold ${isActive ? 'text-black' : ''}`}>
                {model.name}
              </span>
            </button>
          </Tooltip>
        );
      })}
    </div>
  );
};
