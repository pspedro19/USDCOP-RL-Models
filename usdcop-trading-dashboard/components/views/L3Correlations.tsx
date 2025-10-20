'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Network, TrendingUp, AlertCircle, BarChart3 } from 'lucide-react';

interface Feature {
  feature_name: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  ic_mean: number;
  ic_std: number;
  rank_ic: number;
  correlation_with_target: number;
}

interface FeaturesResponse {
  features: Feature[];
  total_features: number;
  avg_ic: number;
  high_ic_count: number;
}

export default function L3Correlations() {
  const [data, setData] = useState<FeaturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l3/features?limit=100');
        if (!response.ok) throw new Error('Failed to fetch features');
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
          <p className="text-cyan-500 font-mono text-sm">Loading feature correlations...</p>
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
        <h1 className="text-2xl font-bold text-cyan-500 font-mono">L3 CORRELATIONS & IC ANALYSIS</h1>
        <p className="text-slate-400 text-sm mt-1">
          17 Features • Avg IC: {data.avg_ic.toFixed(4)} •
          High IC Features: {data.high_ic_count}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Network className="h-5 w-5 text-cyan-400" />
            <p className="text-sm text-slate-400 font-mono">Total Features</p>
          </div>
          <p className="text-3xl font-bold text-cyan-400 font-mono">{data.total_features}</p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <p className="text-sm text-slate-400 font-mono">Avg IC</p>
          </div>
          <p className="text-3xl font-bold text-green-400 font-mono">
            {data.avg_ic.toFixed(4)}
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="h-5 w-5 text-purple-400" />
            <p className="text-sm text-slate-400 font-mono">High IC Count</p>
          </div>
          <p className="text-3xl font-bold text-purple-400 font-mono">
            {data.high_ic_count}
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <AlertCircle className="h-5 w-5 text-yellow-400" />
            <p className="text-sm text-slate-400 font-mono">Low IC Count</p>
          </div>
          <p className="text-3xl font-bold text-yellow-400 font-mono">
            {data.features.filter(f => Math.abs(f.ic_mean) < 0.01).length}
          </p>
        </Card>
      </div>

      <Card className="bg-slate-900 border-cyan-500/20 p-6">
        <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">Feature IC Rankings</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left p-3 text-slate-400 font-mono">Feature Name</th>
                <th className="text-left p-3 text-slate-400 font-mono">IC Mean</th>
                <th className="text-left p-3 text-slate-400 font-mono">IC Std</th>
                <th className="text-left p-3 text-slate-400 font-mono">Rank IC</th>
                <th className="text-left p-3 text-slate-400 font-mono">Correlation</th>
                <th className="text-left p-3 text-slate-400 font-mono">Mean</th>
                <th className="text-left p-3 text-slate-400 font-mono">Std</th>
                <th className="text-left p-3 text-slate-400 font-mono">Range</th>
              </tr>
            </thead>
            <tbody>
              {data.features
                .sort((a, b) => Math.abs(b.ic_mean) - Math.abs(a.ic_mean))
                .map((feature, idx) => (
                <tr
                  key={feature.feature_name}
                  className="border-b border-slate-800 hover:bg-slate-800/50"
                >
                  <td className="p-3 font-mono text-cyan-400 font-bold">
                    {feature.feature_name}
                  </td>
                  <td className="p-3">
                    <span className={`font-mono font-bold ${
                      Math.abs(feature.ic_mean) >= 0.05 ? 'text-green-400' :
                      Math.abs(feature.ic_mean) >= 0.02 ? 'text-yellow-400' :
                      'text-slate-400'
                    }`}>
                      {feature.ic_mean.toFixed(5)}
                    </span>
                  </td>
                  <td className="p-3 font-mono text-slate-300">
                    {feature.ic_std.toFixed(5)}
                  </td>
                  <td className="p-3">
                    <span className={`font-mono ${
                      Math.abs(feature.rank_ic) >= 0.05 ? 'text-purple-400' :
                      'text-slate-400'
                    }`}>
                      {feature.rank_ic.toFixed(5)}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className={`font-mono ${
                      Math.abs(feature.correlation_with_target) >= 0.3 ? 'text-red-400' :
                      Math.abs(feature.correlation_with_target) >= 0.1 ? 'text-orange-400' :
                      'text-slate-400'
                    }`}>
                      {feature.correlation_with_target.toFixed(4)}
                    </span>
                  </td>
                  <td className="p-3 font-mono text-slate-300 text-xs">
                    {feature.mean.toFixed(4)}
                  </td>
                  <td className="p-3 font-mono text-slate-300 text-xs">
                    {feature.std.toFixed(4)}
                  </td>
                  <td className="p-3 font-mono text-slate-300 text-xs">
                    [{feature.min.toFixed(2)}, {feature.max.toFixed(2)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="text-center py-4 border-t border-cyan-500/20">
        <p className="text-slate-500 text-xs font-mono">
          IC = Information Coefficient • Rank IC = Spearman Correlation •
          High IC: |IC| ≥ 0.05
        </p>
      </div>
    </div>
  );
}
