import React from 'react';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';

interface ConnectionStatusProps {
  isConnected: boolean;
  isConnecting?: boolean;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  isConnecting = false
}) => {
  const getStatusConfig = () => {
    if (isConnecting) {
      return {
        icon: Loader2,
        text: 'CONNECTING',
        bgColor: 'bg-yellow-900/30',
        borderColor: 'border-yellow-600',
        textColor: 'text-yellow-400',
        iconClass: 'animate-spin'
      };
    }

    if (isConnected) {
      return {
        icon: Wifi,
        text: 'CONNECTED',
        bgColor: 'bg-atlas-green-900/30',
        borderColor: 'border-atlas-green-500',
        textColor: 'text-atlas-green-400',
        iconClass: ''
      };
    }

    return {
      icon: WifiOff,
      text: 'DISCONNECTED',
      bgColor: 'bg-red-900/30',
      borderColor: 'border-red-600',
      textColor: 'text-red-400',
      iconClass: ''
    };
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 border ${config.borderColor} ${config.bgColor} rounded-md`}
    >
      <Icon size={14} className={`${config.textColor} ${config.iconClass}`} />
      <span className={`text-xs font-semibold ${config.textColor}`}>
        {config.text}
      </span>
      {isConnected && (
        <div className="flex gap-1">
          <div className="w-1 h-1 rounded-full bg-atlas-green-500 animate-pulse" />
          <div className="w-1 h-1 rounded-full bg-atlas-green-500 animate-pulse delay-75" />
          <div className="w-1 h-1 rounded-full bg-atlas-green-500 animate-pulse delay-150" />
        </div>
      )}
    </div>
  );
};
