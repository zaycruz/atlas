import React from 'react';
import { ConnectionStatus } from './ConnectionStatus';
import { ModelToggler, type AIModel } from './ModelToggler';
import { AgentStatus, type AgentState } from './AgentStatus';
import { TestModeToggle } from './TestModeToggle';

interface HeaderProps {
  time: Date;
  isConnected: boolean;
  currentModel: AIModel;
  onModelChange: (model: AIModel) => void;
  installedModels: string[];
  agentStatus: AgentState;
  testMode: boolean;
  onTestModeToggle: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  time,
  isConnected,
  currentModel,
  onModelChange,
  installedModels,
  agentStatus,
  testMode,
  onTestModeToggle
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
        <TestModeToggle enabled={testMode} onToggle={onTestModeToggle} />
        <ModelToggler
          currentModel={currentModel}
          installed={installedModels}
          onModelChange={onModelChange}
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
