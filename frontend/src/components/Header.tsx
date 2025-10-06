import React from 'react';

interface HeaderProps {
  time: Date;
}

export const Header: React.FC<HeaderProps> = ({ time }) => {
  return (
    <div className="px-4 py-3 border-b border-atlas-green-900 flex justify-between items-start">
      <div>
        <h1 className="text-2xl font-bold text-atlas-yellow-400 tracking-wider leading-tight">
          A.T.L.A.S.
        </h1>
        <p className="text-xs text-atlas-green-700 leading-tight">
          Advanced Tactical Logistics and Analysis System
        </p>
      </div>
      <div className="text-right">
        <div className="text-xl text-atlas-yellow-400 leading-tight">
          {time.toLocaleTimeString()}
        </div>
        <div className="text-xs text-atlas-green-700 leading-tight">
          {time.toLocaleDateString()}
        </div>
      </div>
    </div>
  );
};
