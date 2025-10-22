'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Database, Activity, RefreshCw, TrendingUp, Server, CheckCircle } from 'lucide-react';
import { useDbStats } from '@/hooks/useDbStats';

interface L0Record {
  timestamp: string;
  symbol: string;
  close: number;
  bid?: number;
  ask?: number;
  volume?: number;
  source: string;
}

interface L0Statistics {
  overview: {
    totalRecords: number;
    dateRange: {
      earliest: string;
      latest: string;
      tradingDays?: number;
    };
    priceMetrics: {
      min: number;
      max: number;
      avg: number;
      stddev?: number;
    };
  };
  sources?: Array<{
    source: string;
    count: number;
    percentage: string;
  }>;
}

export default function L0RawDataDashboard() {
  const [data, setData] = useState<L0Record[]>([]);
  const [stats, setStats] = useState<L0Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const { stats: dbStats } = useDbStats(60000); // Refresh every 60 seconds

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch raw data from PostgreSQL (real API endpoint)
      const response = await fetch('/api/pipeline/l0/raw-data?limit=1000');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setData(result.data || []);
      } else {
        throw new Error(result.error || 'Failed to fetch data');
      }

      // Fetch statistics
      const statsResponse = await fetch('/api/pipeline/l0/statistics');
      if (statsResponse.ok) {
        const statsResult = await statsResponse.json();
        if (statsResult.success) {
          setStats(statsResult.statistics);
        }
      }

      setLastRefresh(new Date());
      setLoading(false);
    } catch (err) {
      console.error('[L0 Dashboard] Error fetching data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Refresh every 60 seconds
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !data.length) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-400" />
          <p className="text-lg text-slate-300">Loading L0 data from PostgreSQL...</p>
          <p className="text-sm text-slate-500 mt-2">Fetching {dbStats.totalRecords.toLocaleString()} real OHLC records</p>
        </div>
      </div>
    );
  }

  if (error && !data.length) {
    return (
      <div className="p-8">
        <div className="bg-red-900/20 border border-red-500 rounded-lg p-6">
          <h2 className="text-red-400 font-bold text-lg mb-2">Error Loading Data</h2>
          <p className="text-red-300">{error}</p>
          <button
            onClick={fetchData}
            className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const sourceDistribution = stats?.sources || [];
  const overview = stats?.overview;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Database className="w-8 h-8 text-blue-400" />
            L0 - Raw Market Data
          </h1>
          <p className="text-slate-400 mt-1">
            Real-time USD/COP OHLC data from PostgreSQL/TimescaleDB
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right text-sm">
            <p className="text-slate-400">Last Updated</p>
            <p className="text-white font-mono">{lastRefresh.toLocaleTimeString()}</p>
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh data"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      {overview && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Records */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Total Records</p>
                <p className="text-3xl font-bold text-white">
                  {overview.totalRecords?.toLocaleString()}
                </p>
                <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  PostgreSQL Connected
                </p>
              </div>
              <Server className="w-12 h-12 text-blue-400 opacity-50" />
            </div>
          </Card>

          {/* Date Range */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Date Range</p>
                <p className="text-lg font-bold text-white">
                  {overview.dateRange?.earliest?.split('T')[0]}
                </p>
                <p className="text-sm text-slate-300">
                  to {overview.dateRange?.latest?.split('T')[0]}
                </p>
                {overview.dateRange?.tradingDays && (
                  <p className="text-xs text-slate-400 mt-1">
                    {overview.dateRange.tradingDays} trading days
                  </p>
                )}
              </div>
              <Activity className="w-12 h-12 text-purple-400 opacity-50" />
            </div>
          </Card>

          {/* Average Price */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Average Price</p>
                <p className="text-3xl font-bold text-white">
                  ${overview.priceMetrics?.avg?.toFixed(2)}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Range: ${overview.priceMetrics?.min?.toFixed(2)} - $
                  {overview.priceMetrics?.max?.toFixed(2)}
                </p>
              </div>
              <TrendingUp className="w-12 h-12 text-green-400 opacity-50" />
            </div>
          </Card>

          {/* Data Source */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Primary Source</p>
                <p className="text-xl font-bold text-white">
                  {sourceDistribution[0]?.source || 'PostgreSQL'}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {sourceDistribution[0]?.count?.toLocaleString()} records (
                  {sourceDistribution[0]?.percentage})
                </p>
              </div>
              <Database className="w-12 h-12 text-cyan-400 opacity-50" />
            </div>
          </Card>
        </div>
      )}

      {/* Data Table */}
      <Card className="bg-slate-800/50 border-slate-700">
        <div className="p-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            Latest OHLC Records
            <span className="text-sm text-slate-400 ml-2">
              (Showing {Math.min(data.length, 50)} of {data.length})
            </span>
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Close
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Bid
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Ask
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Volume
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Source
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {data.slice(0, 50).map((row, idx) => (
                <tr
                  key={idx}
                  className="hover:bg-slate-750 transition-colors"
                >
                  <td className="px-4 py-3 text-sm text-slate-300">
                    {new Date(row.timestamp).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-sm text-white font-mono text-right">
                    {row.close?.toFixed(4)}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 font-mono text-right">
                    {row.bid?.toFixed(4) || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 font-mono text-right">
                    {row.ask?.toFixed(4) || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 text-right">
                    {row.volume?.toLocaleString() || '-'}
                  </td>
                  <td className="px-4 py-3 text-xs">
                    <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded">
                      {row.source}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="p-4 border-t border-slate-700 bg-slate-800/30">
          <p className="text-sm text-slate-400">
            ✓ Connected to PostgreSQL/TimescaleDB |
            ✓ Real data (no mock) |
            ✓ Auto-refresh every 60 seconds |
            ✓ {data.length} records loaded
          </p>
        </div>
      </Card>

      {error && (
        <div className="bg-yellow-900/20 border border-yellow-500 rounded-lg p-4">
          <p className="text-yellow-300 text-sm">
            ⚠️ Warning: {error} (showing cached data)
          </p>
        </div>
      )}
    </div>
  );
}
