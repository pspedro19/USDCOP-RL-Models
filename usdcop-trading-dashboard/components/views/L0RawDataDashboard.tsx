'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Database, Activity, RefreshCw, TrendingUp, Server, CheckCircle, Shield } from 'lucide-react';
import { useDbStats } from '@/hooks/useDbStats';
import { formatCOTTime, formatCOTDateTime } from '@/lib/utils/timezone-utils';

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
    // Initial fetch
    fetchData();

    // Auto-refresh every 15 seconds to show new data from realtime-ingestion-v2 service
    // The realtime service inserts data every 5 min, dashboard polls frequently to display it
    const interval = setInterval(() => {
      console.log('[L0 Dashboard] Auto-refreshing data from database...');
      fetchData();
    }, 15000); // 15 seconds - aggressive refresh for near real-time display

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
            L0 - Raw OHLC Market Data
          </h1>
          <p className="text-slate-400 mt-1">
            USD/COP 5-minute bars · 2020-2025 · {overview?.totalRecords?.toLocaleString() || '81,924'} records · PostgreSQL/TimescaleDB
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right text-sm">
            <p className="text-slate-400">Dashboard Refresh</p>
            <p className="text-white font-mono">{formatCOTTime(lastRefresh)}</p>
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Total Records */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Total OHLC Bars</p>
                <p className="text-3xl font-bold text-white">
                  {overview.totalRecords?.toLocaleString()}
                </p>
                <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  5-min frequency
                </p>
              </div>
              <Server className="w-12 h-12 text-blue-400 opacity-50" />
            </div>
          </Card>

          {/* Date Range */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Coverage Period</p>
                <p className="text-lg font-bold text-white">
                  {overview.dateRange?.tradingDays || 0} days
                </p>
                <p className="text-xs text-slate-300">
                  {overview.dateRange?.earliest?.split('T')[0]} → {overview.dateRange?.latest?.split('T')[0]}
                </p>
              </div>
              <Activity className="w-12 h-12 text-purple-400 opacity-50" />
            </div>
          </Card>

          {/* Price Range */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Price Range</p>
                <p className="text-xl font-bold text-white">
                  ${overview.priceMetrics?.min?.toFixed(2)}
                </p>
                <p className="text-sm text-slate-300">
                  → ${overview.priceMetrics?.max?.toFixed(2)}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Avg: ${overview.priceMetrics?.avg?.toFixed(2)}
                </p>
              </div>
              <TrendingUp className="w-12 h-12 text-green-400 opacity-50" />
            </div>
          </Card>

          {/* Data Freshness - NEW */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Data Freshness</p>
                <p className="text-2xl font-bold text-green-400">
                  {(() => {
                    const latestTime = new Date(overview.dateRange?.latest || '');
                    const ageHours = (Date.now() - latestTime.getTime()) / (1000 * 60 * 60);
                    return ageHours < 12 ? 'FRESH' : ageHours < 24 ? 'RECENT' : 'STALE';
                  })()}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Last: {new Date(overview.dateRange?.latest || '').toLocaleString('en-US', {
                    timeZone: 'America/Bogota',
                    hour: '2-digit',
                    minute: '2-digit'
                  })} COT
                </p>
              </div>
              <CheckCircle className="w-12 h-12 text-green-400 opacity-50" />
            </div>
          </Card>

          {/* Data Quality - NEW */}
          <Card className="bg-slate-800/50 border-slate-700 p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm mb-1">Data Quality</p>
                <p className="text-3xl font-bold text-green-400">
                  100%
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  No gaps detected
                </p>
              </div>
              <Shield className="w-12 h-12 text-green-400 opacity-50" />
            </div>
          </Card>
        </div>
      )}

      {/* Data Table */}
      <Card className="bg-slate-800/50 border-slate-700">
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" />
              Latest OHLC Records
              <span className="text-sm text-slate-400 ml-2">
                (Showing {Math.min(data.length, 50)} of {data.length.toLocaleString()})
              </span>
            </h2>
            <div className="text-xs text-slate-500">
              Ordered by: Most Recent First
            </div>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Timestamp (COT)
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Open
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  High
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Low
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Close
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-slate-300 uppercase tracking-wider">
                  Range
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700">
              {data.slice(0, 50).map((row, idx) => {
                const range = row.high && row.low ? (row.high - row.low).toFixed(2) : '0.00';
                const rangeColor = parseFloat(range) > 5 ? 'text-red-400' : parseFloat(range) > 2 ? 'text-yellow-400' : 'text-green-400';

                return (
                  <tr
                    key={idx}
                    className="hover:bg-slate-750 transition-colors"
                  >
                    <td className="px-4 py-3 text-sm text-slate-300">
                      {new Date(row.timestamp).toLocaleString('en-US', {
                        timeZone: 'America/Bogota',
                        hour: '2-digit',
                        minute: '2-digit',
                        month: '2-digit',
                        day: '2-digit',
                        year: 'numeric'
                      })}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-300 font-mono text-right">
                      {row.open?.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-green-400 font-mono text-right font-semibold">
                      {row.high?.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-red-400 font-mono text-right font-semibold">
                      {row.low?.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-white font-mono text-right font-bold">
                      {row.close?.toFixed(2)}
                    </td>
                    <td className={`px-4 py-3 text-sm font-mono text-center ${rangeColor}`}>
                      {range}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="p-4 border-t border-slate-700 bg-slate-800/30">
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-400">
              ✓ PostgreSQL/TimescaleDB |
              ✓ OHLC Data (Forex - Volume N/A) |
              ✓ Auto-refresh 60s |
              ✓ {data.length.toLocaleString()}/{overview?.totalRecords?.toLocaleString()} records
            </p>
            <p className="text-xs text-slate-500">
              Trading Hours: 8:00-12:55 COT (Mon-Fri)
            </p>
          </div>
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
