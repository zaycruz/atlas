import React from 'react';
import { Network } from 'lucide-react';
import type { GraphEdge, GraphNode } from '../types';

interface NetworkTabProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export const NetworkTab: React.FC<NetworkTabProps> = ({ nodes, edges }) => {
  const getNodeRadius = (size: GraphNode['size']) => {
    switch (size) {
      case 'large':
        return 12;
      case 'medium':
        return 8;
      default:
        return 5;
    }
  };

  const getNodeColor = (size: GraphNode['size']) => {
    switch (size) {
      case 'large':
        return 'text-atlas-yellow-400';
      case 'medium':
        return 'text-atlas-cyan-400';
      default:
        return 'text-atlas-green-500';
    }
  };

  return (
    <div className="h-full">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-atlas-green-900">
        <Network size={16} className="text-atlas-yellow-400" />
        <span className="text-sm font-semibold text-atlas-yellow-400">KNOWLEDGE GRAPH</span>
      </div>

      <div className="relative h-full flex items-center justify-center p-4">
        {nodes.length === 0 ? (
          <div className="text-atlas-green-600 text-sm">
            Knowledge graph is not available yet.
          </div>
        ) : (
          <svg className="w-full h-full">
          {edges.map((edge, i) => {
            const from = nodes.find((n) => n.id === edge.from);
            const to = nodes.find((n) => n.id === edge.to);
            if (!from || !to) return null;
            return (
              <line
                key={i}
                x1={`${from.x}%`}
                y1={`${from.y}%`}
                x2={`${to.x}%`}
                y2={`${to.y}%`}
                stroke="currentColor"
                strokeWidth="1"
                className="text-atlas-green-800"
              />
            );
          })}

          {nodes.map((node) => (
            <g key={node.id}>
              <circle
                cx={`${node.x}%`}
                cy={`${node.y}%`}
                r={getNodeRadius(node.size)}
                fill="currentColor"
                className={getNodeColor(node.size)}
              />
              <text
                x={`${node.x}%`}
                y={`${node.y + 4}%`}
                textAnchor="middle"
                className="text-xs fill-current text-atlas-green-400"
              >
                {node.label}
              </text>
            </g>
          ))}
          </svg>
        )}

        <div className="absolute bottom-4 left-4 text-xs space-y-1">
          <div className="text-atlas-cyan-400 mb-2 font-semibold">Legend:</div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-atlas-yellow-400" />
            <span className="text-atlas-green-500">Core Concept</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-atlas-cyan-400" />
            <span className="text-atlas-green-500">Major Topic</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-atlas-green-500" />
            <span className="text-atlas-green-500">Related Concept</span>
          </div>
        </div>
      </div>
    </div>
  );
};
