'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Database, Layers, AlertCircle, CheckCircle } from 'lucide-react';

interface DatasetSplit {
  split: string;
  num_episodes: number;
  num_timesteps: number;
  avg_episode_length: number;
  reward_mean: number;
  reward_std: number;
}

interface DatasetResponse {
  splits: DatasetSplit[];
  total_episodes: number;
  total_timesteps: number;
  feature_count: number;
  action_space_size: number;
}

export default function L4RLReadyData() {
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l4/dataset?split=test');
        if (!response.ok) throw new Error('Failed to fetch dataset');
        const result = await response.json();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-cyan-500/20 border-t-cyan-500 mx-auto mb-4"></div>
          <p className="text-cyan-500 font-mono text-sm">Loading RL dataset...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <p className="font-mono">Error: {error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      <div className="border-b border-cyan-500/20 pb-4">
        <h1 className="text-2xl font-bold text-cyan-500 font-mono">L4 RL-READY DATASET</h1>
        <p className="text-slate-400 text-sm mt-1">
          Train/Val/Test Splits • {data.total_episodes} Total Episodes •
          {data.feature_count} Features • Action Space: {data.action_space_size}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Database className="h-5 w-5 text-cyan-400" />
            <p className="text-sm text-slate-400 font-mono">Total Episodes</p>
          </div>
          <p className="text-3xl font-bold text-cyan-400 font-mono">{data.total_episodes}</p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Layers className="h-5 w-5 text-green-400" />
            <p className="text-sm text-slate-400 font-mono">Total Timesteps</p>
          </div>
          <p className="text-3xl font-bold text-green-400 font-mono">
            {data.total_timesteps.toLocaleString()}
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="h-5 w-5 text-purple-400" />
            <p className="text-sm text-slate-400 font-mono">Feature Count</p>
          </div>
          <p className="text-3xl font-bold text-purple-400 font-mono">{data.feature_count}</p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <AlertCircle className="h-5 w-5 text-yellow-400" />
            <p className="text-sm text-slate-400 font-mono">Action Space</p>
          </div>
          <p className="text-3xl font-bold text-yellow-400 font-mono">{data.action_space_size}</p>
        </Card>
      </div>

      <Card className="bg-slate-900 border-cyan-500/20 p-6">
        <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">Dataset Splits</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left p-3 text-slate-400 font-mono">Split</th>
                <th className="text-left p-3 text-slate-400 font-mono">Episodes</th>
                <th className="text-left p-3 text-slate-400 font-mono">Timesteps</th>
                <th className="text-left p-3 text-slate-400 font-mono">Avg Length</th>
                <th className="text-left p-3 text-slate-400 font-mono">Reward Mean</th>
                <th className="text-left p-3 text-slate-400 font-mono">Reward Std</th>
                <th className="text-left p-3 text-slate-400 font-mono">Percentage</th>
              </tr>
            </thead>
            <tbody>
              {data.splits.map((split) => {
                const percentage = (split.num_episodes / data.total_episodes) * 100;
                return (
                  <tr
                    key={split.split}
                    className="border-b border-slate-800 hover:bg-slate-800/50"
                  >
                    <td className="p-3">
                      <span className={`font-mono font-bold ${
                        split.split === 'train' ? 'text-cyan-400' :
                        split.split === 'val' ? 'text-purple-400' :
                        'text-green-400'
                      }`}>
                        {split.split.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 font-mono text-white font-bold">
                      {split.num_episodes.toLocaleString()}
                    </td>
                    <td className="p-3 font-mono text-slate-300">
                      {split.num_timesteps.toLocaleString()}
                    </td>
                    <td className="p-3 font-mono text-slate-300">
                      {split.avg_episode_length?.toFixed(1) || 'N/A'}
                    </td>
                    <td className="p-3 font-mono text-green-400">
                      {split.reward_mean?.toFixed(6) || 'N/A'}
                    </td>
                    <td className="p-3 font-mono text-yellow-400">
                      {split.reward_std?.toFixed(6) || 'N/A'}
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-slate-800 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              split.split === 'train' ? 'bg-cyan-400' :
                              split.split === 'val' ? 'bg-purple-400' :
                              'bg-green-400'
                            }`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <span className="font-mono text-slate-400 text-xs">
                          {percentage.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {data.splits.map((split) => (
          <Card key={split.split} className="bg-slate-900 border-cyan-500/20 p-6">
            <h3 className={`text-lg font-bold font-mono mb-4 ${
              split.split === 'train' ? 'text-cyan-400' :
              split.split === 'val' ? 'text-purple-400' :
              'text-green-400'
            }`}>
              {split.split.toUpperCase()} SPLIT
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
                <span className="text-slate-300 font-mono text-sm">Episodes</span>
                <span className="text-white font-mono font-bold">{split.num_episodes}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
                <span className="text-slate-300 font-mono text-sm">Timesteps</span>
                <span className="text-white font-mono font-bold">
                  {split.num_timesteps.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
                <span className="text-slate-300 font-mono text-sm">Avg Length</span>
                <span className="text-white font-mono font-bold">
                  {split.avg_episode_length?.toFixed(1) || 'N/A'}
                </span>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
