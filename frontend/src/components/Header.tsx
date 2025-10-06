import React from 'react';
import { User } from 'lucide-react';
import { ConnectionStatus } from './ConnectionStatus';
import { ModelToggler, type AIModel } from './ModelToggler';
import { AgentStatus, type AgentState } from './AgentStatus';
import { Tooltip } from './Tooltip';

interface HeaderProps {
  time: Date;
  isConnected: boolean;
  currentModel: AIModel;
  onModelChange: (model: AIModel) => void;
  installedModels: string[];
  availableModels: string[];
  onAddModel: (model: string) => void;
  modelPullStatus?: {
    model: string | null;
    status: 'idle' | 'started' | 'progress' | 'completed' | 'error';
    message?: string;
  };
  onOpenProfile: () => void;
  agentStatus: AgentState;
}

export const Header: React.FC<HeaderProps> = ({
  time,
  isConnected,
  currentModel,
  onModelChange,
  installedModels,
  availableModels,
  onAddModel,
  modelPullStatus,
  onOpenProfile,
  agentStatus
}) => {
  return (
    <div className="px-4 py-3 border-b border-atlas-green-900 flex justify-between items-center">
      <div>
        <h1 className="text-2xl font-bold text-atlas-yellow-400 tracking-wider leading-tight">
          A.T.L.A.S.
        </h1>
        <p className="text-xs text-atlas-green-700 leading-tight">
          Advanced Tactical Logistics and Analysis System
        </p>
      </div>

      <div className="flex items-center gap-4">
        <AgentStatus status={agentStatus} />
        <Tooltip content="User Profile & Preferences" position="bottom">
          <button
            onClick={onOpenProfile}
            className="flex items-center gap-2 px-3 py-1.5 bg-atlas-green-950/30 border border-atlas-green-900 rounded-md hover:border-atlas-cyan-400 transition-colors"
          >
            <User size={14} className="text-atlas-green-500" />
            <span className="text-xs font-semibold text-atlas-green-500">Profile</span>
          </button>
        </Tooltip>
        <ModelToggler
          currentModel={currentModel}
          installed={installedModels}
          available={availableModels}
          onModelChange={onModelChange}
          onAddModel={onAddModel}
          pullStatus={modelPullStatus}
        />
        <ConnectionStatus isConnected={isConnected} />
        <div className="text-right">
          <div className="text-xl text-atlas-yellow-400 leading-tight">
            {time.toLocaleTimeString()}
          </div>
          <div className="text-xs text-atlas-green-700 leading-tight">
            {time.toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
};
