'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Brain, Zap, CheckCircle, AlertCircle, Clock, Cpu, Database, AlertTriangle } from 'lucide-react';

interface Model {
  model_id: string;
  model_name: string;
  version: string;
  format: string;
  size_mb: number;
  created_at: string;
  training_episodes: number;
  val_reward_mean: number;
  checkpoint_path: string;
}

interface ModelsResponse {
  models: Model[];
  total_models: number;
  latest_model: Model | null;
  avg_size_mb: number;
}

export default function L5ModelDashboard() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l5/models');
        if (!response.ok) throw new Error('Failed to fetch models');
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
          <p className="text-cyan-500 font-mono text-sm">Loading models...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 bg-red-900/20 rounded-lg border border-red-500/30">
        <div className="flex items-center gap-3 text-red-400 mb-2">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-semibold">L5 Data Unavailable</span>
        </div>
        <p className="text-fintech-dark-300 text-sm">
          {error || 'Failed to load pipeline data. Check backend connectivity.'}
        </p>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 bg-fintech-dark-700 hover:bg-fintech-dark-600 rounded text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data || !data.models || data.models.length === 0) {
    return (
      <div className="p-8 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700">
        <div className="text-center text-fintech-dark-400">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No L5 Models Available</p>
          <p className="text-sm mt-2">Run the L5 training DAG to train and export RL models</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      <div className="border-b border-cyan-500/20 pb-4">
        <h1 className="text-2xl font-bold text-cyan-500 font-mono">L5 MODEL DASHBOARD</h1>
        <p className="text-slate-400 text-sm mt-1">
          ONNX Models & Checkpoints • {data.total_models} Total Models •
          Avg Size: {data.avg_size_mb.toFixed(1)} MB
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Brain className="h-5 w-5 text-cyan-400" />
            <p className="text-sm text-slate-400 font-mono">Total Models</p>
          </div>
          <p className="text-3xl font-bold text-cyan-400 font-mono">{data.total_models}</p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Cpu className="h-5 w-5 text-green-400" />
            <p className="text-sm text-slate-400 font-mono">Avg Size</p>
          </div>
          <p className="text-3xl font-bold text-green-400 font-mono">
            {data.avg_size_mb.toFixed(1)} MB
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Zap className="h-5 w-5 text-purple-400" />
            <p className="text-sm text-slate-400 font-mono">ONNX Models</p>
          </div>
          <p className="text-3xl font-bold text-purple-400 font-mono">
            {data.models.filter(m => m.format === 'ONNX').length}
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="h-5 w-5 text-yellow-400" />
            <p className="text-sm text-slate-400 font-mono">Latest Version</p>
          </div>
          <p className="text-3xl font-bold text-yellow-400 font-mono">
            {data.latest_model?.version || 'N/A'}
          </p>
        </Card>
      </div>

      {data.latest_model && (
        <Card className="bg-slate-900 border-green-500/50 p-6">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="h-6 w-6 text-green-400" />
            <h3 className="text-lg font-bold text-green-400 font-mono">LATEST MODEL</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Model Name</p>
              <p className="text-sm text-white font-mono font-bold">{data.latest_model.model_name}</p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Version</p>
              <p className="text-sm text-cyan-400 font-mono font-bold">{data.latest_model.version}</p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Format</p>
              <p className="text-sm text-purple-400 font-mono font-bold">{data.latest_model.format}</p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Size</p>
              <p className="text-sm text-green-400 font-mono font-bold">
                {data.latest_model.size_mb.toFixed(2)} MB
              </p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Training Episodes</p>
              <p className="text-sm text-white font-mono font-bold">
                {data.latest_model.training_episodes.toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded">
              <p className="text-xs text-slate-400 font-mono mb-1">Val Reward</p>
              <p className="text-sm text-yellow-400 font-mono font-bold">
                {data.latest_model.val_reward_mean.toFixed(6)}
              </p>
            </div>
            <div className="p-4 bg-slate-800/50 rounded col-span-2">
              <p className="text-xs text-slate-400 font-mono mb-1">Created</p>
              <p className="text-sm text-white font-mono">
                {new Date(data.latest_model.created_at).toLocaleString()}
              </p>
            </div>
          </div>
        </Card>
      )}

      <Card className="bg-slate-900 border-cyan-500/20 p-6">
        <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">All Models</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left p-3 text-slate-400 font-mono">Model Name</th>
                <th className="text-left p-3 text-slate-400 font-mono">Version</th>
                <th className="text-left p-3 text-slate-400 font-mono">Format</th>
                <th className="text-left p-3 text-slate-400 font-mono">Size (MB)</th>
                <th className="text-left p-3 text-slate-400 font-mono">Episodes</th>
                <th className="text-left p-3 text-slate-400 font-mono">Val Reward</th>
                <th className="text-left p-3 text-slate-400 font-mono">Created</th>
              </tr>
            </thead>
            <tbody>
              {data.models.map((model) => (
                <tr
                  key={model.model_id}
                  className="border-b border-slate-800 hover:bg-slate-800/50"
                >
                  <td className="p-3 font-mono text-cyan-400 font-bold">
                    {model.model_name}
                  </td>
                  <td className="p-3 font-mono text-purple-400">
                    {model.version}
                  </td>
                  <td className="p-3">
                    <span className="text-xs px-2 py-1 bg-purple-900/30 text-purple-400 rounded font-mono">
                      {model.format}
                    </span>
                  </td>
                  <td className="p-3 font-mono text-green-400">
                    {model.size_mb.toFixed(2)}
                  </td>
                  <td className="p-3 font-mono text-slate-300">
                    {model.training_episodes.toLocaleString()}
                  </td>
                  <td className="p-3 font-mono text-yellow-400">
                    {model.val_reward_mean.toFixed(6)}
                  </td>
                  <td className="p-3 font-mono text-slate-300 text-xs">
                    {new Date(model.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
