'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import {
  Database, Activity, RefreshCw, CheckCircle, XCircle, AlertTriangle,
  Shield, Filter, BarChart3, Calendar, Target, TrendingUp, Clock
} from 'lucide-react';

interface L1Metadata {
  rows_all: number;
  rows_accepted: number;
  episodes_all: number;
  episodes_accepted: number;
  quality_summary: {
    ok: number;
    warn: number;
    fail: number;
    total_days: number;
  };
  calendar: {
    holiday_episodes_rejected: number;
    repeated_ohlc_episodes_rejected: number;
  };
  source: string;
  timestamp: string;
}

interface L1Status {
  success: boolean;
  pipeline: string;
  status: {
    health: string;
    lastExecution: {
      rowsProcessed: number;
      rowsAccepted: number;
      episodesAccepted: number;
      acceptanceRate: number;
    };
    dataQuality: {
      okEpisodes: number;
      warnEpisodes: number;
      failEpisodes: number;
      rejectionReasons: {
        repeatedOHLC: number;
        insufficientBars: number;
        holidays: number;
      };
    };
  };
  outputs: any;
  contracts: any;
}

export default function L1StandardizedData() {
  const [status, setStatus] = useState<L1Status | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/pipeline/l1/status');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        setStatus(result);
      } else {
        throw new Error(result.error || 'Failed to fetch L1 status');
      }

      setLastRefresh(new Date());
      setLoading(false);
    } catch (err) {
      console.error('[L1 Dashboard] Error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-400" />
          <p className="text-lg text-slate-300">Loading L1 Standardized Data...</p>
        </div>
      </div>
    );
  }

  if (error && !status) {
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

  if (!status) return null;

  const acceptanceRate = status.status.lastExecution.acceptanceRate;
  const quality = status.status.dataQuality;
  const rejections = quality.rejectionReasons;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Filter className="w-8 h-8 text-purple-400" />
            L1 - Standardized & Quality Filtered Data
          </h1>
          <p className="text-slate-400 mt-1">
            Clean OHLC data · Quality filtering · Holiday detection · {status.status.lastExecution.episodesAccepted} perfect days
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right text-sm">
            <p className="text-slate-400">Last Refresh</p>
            <p className="text-white font-mono">{lastRefresh.toLocaleTimeString()}</p>
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh data"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Episodes Processed */}
        <Card className="bg-slate-800/50 border-slate-700 p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm mb-1">Total Episodes</p>
              <p className="text-3xl font-bold text-white">
                {quality.okEpisodes + quality.warnEpisodes + quality.failEpisodes}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                {status.status.lastExecution.rowsProcessed.toLocaleString()} bars processed
              </p>
            </div>
            <Database className="w-12 h-12 text-slate-400 opacity-50" />
          </div>
        </Card>

        {/* Accepted Episodes */}
        <Card className="bg-slate-800/50 border-slate-700 p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm mb-1">Accepted Episodes</p>
              <p className="text-3xl font-bold text-green-400">
                {quality.okEpisodes}
              </p>
              <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                {acceptanceRate.toFixed(1)}% acceptance rate
              </p>
            </div>
            <CheckCircle className="w-12 h-12 text-green-400 opacity-50" />
          </div>
        </Card>

        {/* Quality Score */}
        <Card className="bg-slate-800/50 border-slate-700 p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm mb-1">Accepted Records</p>
              <p className="text-3xl font-bold text-cyan-400">
                {status.status.lastExecution.rowsAccepted.toLocaleString()}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                60 bars/day × {quality.okEpisodes} days
              </p>
            </div>
            <Target className="w-12 h-12 text-cyan-400 opacity-50" />
          </div>
        </Card>

        {/* Rejections */}
        <Card className="bg-slate-800/50 border-slate-700 p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm mb-1">Rejected Episodes</p>
              <p className="text-3xl font-bold text-red-400">
                {quality.failEpisodes}
              </p>
              <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                <XCircle className="w-3 h-3" />
                {((quality.failEpisodes / (quality.okEpisodes + quality.warnEpisodes + quality.failEpisodes)) * 100).toFixed(1)}% rejected
              </p>
            </div>
            <XCircle className="w-12 h-12 text-red-400 opacity-50" />
          </div>
        </Card>
      </div>

      {/* Quality Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quality Distribution */}
        <Card className="bg-slate-800/50 border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            Quality Distribution
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-slate-300">Perfect Days (OK)</span>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold text-green-400">{quality.okEpisodes}</p>
                <p className="text-xs text-slate-400">
                  {((quality.okEpisodes / (quality.okEpisodes + quality.warnEpisodes + quality.failEpisodes)) * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                <span className="text-slate-300">Warning Days</span>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold text-yellow-400">{quality.warnEpisodes}</p>
                <p className="text-xs text-slate-400">
                  {((quality.warnEpisodes / (quality.okEpisodes + quality.warnEpisodes + quality.failEpisodes)) * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <XCircle className="w-4 h-4 text-red-400" />
                <span className="text-slate-300">Failed Days</span>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold text-red-400">{quality.failEpisodes}</p>
                <p className="text-xs text-slate-400">
                  {((quality.failEpisodes / (quality.okEpisodes + quality.warnEpisodes + quality.failEpisodes)) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Rejection Reasons */}
        <Card className="bg-slate-800/50 border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <XCircle className="w-5 h-5 text-red-400" />
            Rejection Breakdown
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-300">Repeated OHLC (Stale Data)</span>
              <div className="text-right">
                <p className="text-xl font-bold text-orange-400">{rejections.repeatedOHLC}</p>
                <p className="text-xs text-slate-400">
                  {((rejections.repeatedOHLC / quality.failEpisodes) * 100).toFixed(1)}% of failures
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-300">Insufficient Bars</span>
              <div className="text-right">
                <p className="text-xl font-bold text-red-400">{rejections.insufficientBars}</p>
                <p className="text-xs text-slate-400">
                  {((rejections.insufficientBars / quality.failEpisodes) * 100).toFixed(1)}% of failures
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-300">Holidays</span>
              <div className="text-right">
                <p className="text-xl font-bold text-yellow-400">{rejections.holidays}</p>
                <p className="text-xs text-slate-400">
                  {((rejections.holidays / quality.failEpisodes) * 100).toFixed(1)}% of failures
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Contract Assertions */}
      <Card className="bg-slate-800/50 border-slate-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-green-400" />
          L1 Contract Assertions (Accepted Data Only)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(status.contracts.assertions).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 text-sm">
              {value ? (
                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              )}
              <span className={value ? 'text-green-300' : 'text-red-300'}>
                {key.replace(/([A-Z])/g, ' $1').replace(/^./, (str) => str.toUpperCase())}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-4 border-t border-slate-700">
          <p className="text-xs text-slate-400">
            ✓ All assertions validated on accepted dataset |
            ✓ Schema: {status.contracts.schemaVersion} |
            ✓ Calendar: {status.contracts.calendarVersion}
          </p>
        </div>
      </Card>

      {/* Outputs Information */}
      <Card className="bg-slate-800/50 border-slate-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-cyan-400" />
          Generated Outputs
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
            <p className="text-sm text-slate-400 mb-2">Main Datasets</p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-300">All Data (Cleaned)</span>
                <span className="text-white font-mono">{status.outputs.statistics.totalRecords.toLocaleString()} rows</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-300">Accepted (Perfect)</span>
                <span className="text-green-400 font-mono font-bold">{status.outputs.statistics.acceptedRecords.toLocaleString()} rows</span>
              </div>
              <div className="text-xs text-slate-500 mt-2">
                Files: standardized_data_*.parquet, *.csv
              </div>
            </div>
          </div>

          <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
            <p className="text-sm text-slate-400 mb-2">Reports & Statistics</p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span className="text-slate-300">daily_quality_60.csv</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span className="text-slate-300">accepted_summary.csv</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span className="text-slate-300">failure_summary.csv</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-400" />
                <span className="text-slate-300">hod_baseline.parquet</span>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-700">
          <p className="text-sm text-slate-400">
            ✓ MinIO Bucket: <span className="font-mono text-cyan-400">{status.outputs.minio.bucket}</span> |
            ✓ Quality Score: <span className="text-green-400 font-bold">{status.outputs.statistics.qualityScore}%</span> |
            ✓ Source: PostgreSQL
          </p>
        </div>
      </Card>

      {/* Data Quality Metrics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-gradient-to-br from-green-900/20 to-green-800/10 border-green-500/30 p-5">
          <div className="flex items-center gap-3 mb-3">
            <CheckCircle className="w-6 h-6 text-green-400" />
            <p className="text-sm text-green-300 font-semibold">Perfect Days</p>
          </div>
          <p className="text-4xl font-bold text-green-400 mb-2">{quality.okEpisodes}</p>
          <ul className="text-xs text-green-300/80 space-y-1">
            <li>✓ Exactly 60 bars (08:00-12:55 COT)</li>
            <li>✓ No gaps, no duplicates</li>
            <li>✓ 0% repeated OHLC</li>
            <li>✓ Valid OHLC invariants</li>
            <li>✓ Not a holiday</li>
          </ul>
        </Card>

        <Card className="bg-gradient-to-br from-yellow-900/20 to-yellow-800/10 border-yellow-500/30 p-5">
          <div className="flex items-center gap-3 mb-3">
            <AlertTriangle className="w-6 h-6 text-yellow-400" />
            <p className="text-sm text-yellow-300 font-semibold">Warning Days</p>
          </div>
          <p className="text-4xl font-bold text-yellow-400 mb-2">{quality.warnEpisodes}</p>
          <ul className="text-xs text-yellow-300/80 space-y-1">
            <li>• Minor quality issues</li>
            <li>• Acceptable for some analysis</li>
            <li>• Not used for RL training</li>
          </ul>
        </Card>

        <Card className="bg-gradient-to-br from-red-900/20 to-red-800/10 border-red-500/30 p-5">
          <div className="flex items-center gap-3 mb-3">
            <XCircle className="w-6 h-6 text-red-400" />
            <p className="text-sm text-red-300 font-semibold">Rejected Days</p>
          </div>
          <p className="text-4xl font-bold text-red-400 mb-2">{quality.failEpisodes}</p>
          <ul className="text-xs text-red-300/80 space-y-1">
            <li>• {rejections.repeatedOHLC} repeated OHLC</li>
            <li>• {rejections.insufficientBars} insufficient bars</li>
            <li>• {rejections.holidays} holidays</li>
          </ul>
        </Card>
      </div>

      {/* Footer Info */}
      <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-4">
        <p className="text-sm text-slate-400 text-center">
          ✓ L1 Standardization ensures ONLY high-quality data proceeds to feature engineering |
          ✓ {quality.okEpisodes} perfect days ({(acceptanceRate).toFixed(1)}% acceptance rate) |
          ✓ Zero repeated OHLC in accepted dataset |
          ✓ Grid: 300s exact (5-min bars) |
          ✓ Window: 08:00-12:55 COT only
        </p>
      </div>
    </div>
  );
}
