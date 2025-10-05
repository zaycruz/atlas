import React from 'react';

interface HeaderProps {
  time: Date;
}

export const Header: React.FC<HeaderProps> = ({ time }) => {
  return (
    <div className="px-2 py-1 border-b border-atlas-green-900 flex justify-between items-start">
      <div>
        <h1 className="text-xl font-bold text-atlas-yellow-400 tracking-wider leading-tight">
          A.T.L.A.S.
        </h1>
        <p className="text-[10px] text-atlas-green-700 leading-tight">
          Advanced Tactical Logistics and Analysis System
        </p>
      </div>
      <div className="text-right">
        <div className="text-lg text-atlas-yellow-400 leading-tight">
          {time.toLocaleTimeString()}
        </div>
        <div className="text-[10px] text-atlas-green-700 leading-tight">
          {time.toLocaleDateString()}
        </div>
      </div>
    </div>
  );
};
