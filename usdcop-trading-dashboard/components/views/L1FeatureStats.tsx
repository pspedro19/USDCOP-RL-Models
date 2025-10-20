'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { FileText, CheckCircle, XCircle, AlertCircle, TrendingUp } from 'lucide-react';

interface Episode {
  episode_id: string;
  timestamp: string;
  duration_minutes: number;
  data_points: number;
  quality_score: number;
  completeness: number;
  has_gaps: boolean;
  session_type: string;
}

interface EpisodesResponse {
  episodes: Episode[];
  total_count: number;
  avg_quality_score: number;
  avg_completeness: number;
}

export default function L1FeatureStats() {
  const [data, setData] = useState<EpisodesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l1/episodes?limit=100');
        if (!response.ok) throw new Error('Failed to fetch episodes');
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
          <p className="text-cyan-500 font-mono text-sm">Loading episodes...</p>
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
        <h1 className="text-2xl font-bold text-cyan-500 font-mono">L1 FEATURE STATISTICS</h1>
        <p className="text-slate-400 text-sm mt-1">
          Raw Data Episodes • {data.total_count} Total Episodes •
          Avg Quality: {(data.avg_quality_score * 100).toFixed(1)}%
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <FileText className="h-5 w-5 text-cyan-400" />
            <p className="text-sm text-slate-400 font-mono">Total Episodes</p>
          </div>
          <p className="text-3xl font-bold text-cyan-400 font-mono">{data.total_count}</p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <p className="text-sm text-slate-400 font-mono">Avg Quality</p>
          </div>
          <p className="text-3xl font-bold text-green-400 font-mono">
            {(data.avg_quality_score * 100).toFixed(1)}%
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="h-5 w-5 text-purple-400" />
            <p className="text-sm text-slate-400 font-mono">Completeness</p>
          </div>
          <p className="text-3xl font-bold text-purple-400 font-mono">
            {(data.avg_completeness * 100).toFixed(1)}%
          </p>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <div className="flex items-center gap-3 mb-2">
            <AlertCircle className="h-5 w-5 text-yellow-400" />
            <p className="text-sm text-slate-400 font-mono">Episodes w/ Gaps</p>
          </div>
          <p className="text-3xl font-bold text-yellow-400 font-mono">
            {data.episodes.filter(e => e.has_gaps).length}
          </p>
        </Card>
      </div>

      <Card className="bg-slate-900 border-cyan-500/20 p-6">
        <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">Recent Episodes</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left p-3 text-slate-400 font-mono">Episode ID</th>
                <th className="text-left p-3 text-slate-400 font-mono">Timestamp</th>
                <th className="text-left p-3 text-slate-400 font-mono">Duration</th>
                <th className="text-left p-3 text-slate-400 font-mono">Data Points</th>
                <th className="text-left p-3 text-slate-400 font-mono">Quality</th>
                <th className="text-left p-3 text-slate-400 font-mono">Completeness</th>
                <th className="text-left p-3 text-slate-400 font-mono">Session</th>
                <th className="text-left p-3 text-slate-400 font-mono">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.episodes.map((episode) => (
                <tr
                  key={episode.episode_id}
                  className="border-b border-slate-800 hover:bg-slate-800/50"
                >
                  <td className="p-3 font-mono text-cyan-400 text-xs">
                    {episode.episode_id.substring(0, 8)}
                  </td>
                  <td className="p-3 font-mono text-slate-300 text-xs">
                    {new Date(episode.timestamp).toLocaleString()}
                  </td>
                  <td className="p-3 font-mono text-slate-300">
                    {episode.duration_minutes}m
                  </td>
                  <td className="p-3 font-mono text-slate-300">
                    {episode.data_points.toLocaleString()}
                  </td>
                  <td className="p-3">
                    <span className={`font-mono ${
                      episode.quality_score >= 0.9 ? 'text-green-400' :
                      episode.quality_score >= 0.7 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {(episode.quality_score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="p-3">
                    <span className={`font-mono ${
                      episode.completeness >= 0.95 ? 'text-green-400' :
                      episode.completeness >= 0.85 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {(episode.completeness * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-xs px-2 py-1 bg-slate-800 text-purple-400 rounded font-mono">
                      {episode.session_type}
                    </span>
                  </td>
                  <td className="p-3">
                    {episode.has_gaps ? (
                      <XCircle className="h-4 w-4 text-red-400" />
                    ) : (
                      <CheckCircle className="h-4 w-4 text-green-400" />
                    )}
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
