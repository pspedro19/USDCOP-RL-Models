import React from 'react';
import { GitBranch } from 'lucide-react';

interface TraceabilityPanelProps {
  traceability: {
    datasetHashes: Record<string, string>;
    gitCommits: Record<string, string>;
    mlflowRuns: Array<{
      id: string;
      experiment: string;
      timestamp: string;
      status: string;
    }>;
  };
}

export const TraceabilityPanel: React.FC<TraceabilityPanelProps> = ({ traceability }) => {
  return (
    <div className="glass-surface p-6 rounded-xl">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <GitBranch className="w-5 h-5 text-fintech-cyan-500" />
        Complete Traceability
      </h3>

      <div className="space-y-6">
        <div>
          <h4 className="text-sm font-semibold text-white mb-2">Dataset SHA256 Hashes</h4>
          <div className="space-y-2">
            {Object.entries(traceability.datasetHashes).map(([dataset, hash]) => (
              <div key={dataset} className="flex items-center justify-between p-2 bg-fintech-dark-800/30 rounded">
                <span className="text-sm text-fintech-dark-300">{dataset}:</span>
                <span className="text-xs font-mono text-fintech-cyan-400">{hash}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-white mb-2">Git Commits</h4>
          <div className="space-y-2">
            {Object.entries(traceability.gitCommits).map(([repo, commit]) => (
              <div key={repo} className="flex items-center justify-between p-2 bg-fintech-dark-800/30 rounded">
                <span className="text-sm text-fintech-dark-300">{repo}:</span>
                <span className="text-xs font-mono text-fintech-purple-400">{commit}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-white mb-2">MLflow Run IDs</h4>
          <div className="space-y-2">
            {traceability.mlflowRuns.map((run) => (
              <div key={run.id} className="flex items-center justify-between p-2 bg-fintech-dark-800/30 rounded">
                <div>
                  <span className="text-sm text-white">{run.experiment}</span>
                  <div className="text-xs text-fintech-dark-400">{run.id}</div>
                </div>
                <div className={`text-xs px-2 py-1 rounded ${
                  run.status === 'FINISHED' ? 'bg-market-up/20 text-market-up' :
                  run.status === 'RUNNING' ? 'bg-fintech-purple-400/20 text-fintech-purple-400' :
                  'bg-market-down/20 text-market-down'
                }`}>
                  {run.status}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
