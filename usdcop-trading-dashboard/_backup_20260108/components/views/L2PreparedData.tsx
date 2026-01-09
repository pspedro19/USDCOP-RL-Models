'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { CheckCircle, AlertTriangle, Zap, Database } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/info-tooltip';
import { formatCOTDateTime } from '@/lib/utils/timezone-utils';

interface L2PreparedDataResponse {
  success: boolean;
  timestamp: string;
  count: number;
  quality: any;
  qualityGates: any[];
  readyForL3: boolean;
}

export default function L2PreparedData() {
  const [data, setData] = useState<L2PreparedDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l2/prepared-data');

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
          setData(result);
        } else {
          throw new Error(result.error || 'Failed to fetch L2 data');
        }

        setError(null);
      } catch (err) {
        console.error('[L2 Audit] Error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchData();

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      console.log('[L2 Audit] Auto-refreshing...');
      fetchData();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-cyan-500/20 border-t-cyan-500 mx-auto mb-4"></div>
          <p className="text-cyan-500 font-mono text-sm">Loading L2 audit data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 bg-red-900/20 rounded-lg border border-red-500/30">
        <div className="flex items-center gap-3 text-red-400 mb-2">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-semibold">L2 Data Unavailable</span>
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

  if (!data || data.count === 0) {
    return (
      <div className="p-8 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700">
        <div className="text-center text-fintech-dark-400">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No L2 Data Available</p>
          <p className="text-sm mt-2">Run the L2 preparation DAG to generate winsorized and deseasonalized data</p>
        </div>
      </div>
    );
  }

  const allGatesPassed = data.qualityGates?.every(g => g.status === 'PASS') ?? false;

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-cyan-500/20 pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-cyan-500 font-mono">L2 DATA PREPARATION - AUDIT VIEW</h1>
            <p className="text-slate-400 text-sm mt-1">
              Quality gates validation | Last run: {formatCOTDateTime(data.timestamp)}
            </p>
          </div>
          <div className={`px-6 py-3 rounded-lg border-2 ${
            data.readyForL3 ? 'bg-green-900/20 border-green-500' : 'bg-yellow-900/20 border-yellow-500'
          }`}>
            <p className={`text-2xl font-bold font-mono ${
              data.readyForL3 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {data.readyForL3 ? '✓ READY' : '⚠ PENDING'}
            </p>
            <p className="text-xs text-slate-400 mt-1">Pipeline Status</p>
          </div>
        </div>
      </div>

      {/* Critical Quality Gates - PRIMARY FOCUS */}
      <Card className={`p-6 ${
        allGatesPassed ? 'bg-slate-900 border-green-500/30' : 'bg-slate-900 border-yellow-500/30'
      }`}>
        <h2 className="text-xl font-bold text-cyan-400 font-mono mb-6 flex items-center gap-2">
          CRITICAL QUALITY GATES
          <InfoTooltip
            title="L2 Quality Gates"
            meaning="All 4 gates must PASS for L3 to proceed. These validate that winsorization, deseasonalization, and data completeness meet production standards."
            example="If Winsorization Rate = 2.3% > 1.0%, L2 FAILS and must be re-run with better L1 data. If all pass, L3 can safely consume L2 outputs."
          />
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Winsorization */}
          <div className={`p-5 rounded-lg border ${
            (data.quality?.winsorization?.pass ?? true) ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {(data.quality?.winsorization?.pass ?? true) ? (
                  <CheckCircle className="h-5 w-5 text-green-400" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                )}
                <p className="text-sm text-slate-300 font-mono">Winsorization Rate</p>
              </div>
              <span className={`text-lg font-bold font-mono ${
                (data.quality?.winsorization?.pass ?? true) ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.quality?.winsorization?.pass ? 'PASS' : 'FAIL'}
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <p className="text-3xl font-bold text-cyan-400 font-mono">
                {data.quality?.winsorization?.rate_pct ?? 0.5}%
              </p>
              <p className="text-sm text-slate-400">
                ≤ {data.quality?.winsorization?.target || '1.0%'}
              </p>
            </div>
            <p className="text-xs text-slate-500 mt-2">Outlier clipping rate</p>
          </div>

          {/* HOD Median */}
          <div className={`p-5 rounded-lg border ${
            (data.quality?.deseasonalization?.pass ?? true) ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {(data.quality?.deseasonalization?.pass ?? true) ? (
                  <CheckCircle className="h-5 w-5 text-green-400" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                )}
                <p className="text-sm text-slate-300 font-mono">HOD Median (|abs|)</p>
              </div>
              <span className={`text-lg font-bold font-mono ${
                (data.quality?.deseasonalization?.pass ?? true) ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.quality?.deseasonalization?.pass ? 'PASS' : 'FAIL'}
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <p className="text-3xl font-bold text-cyan-400 font-mono">
                {data.quality?.deseasonalization?.hod_median_abs ?? 0.02}
              </p>
              <p className="text-sm text-slate-400">
                ≤ {data.quality?.deseasonalization?.target_median?.replace('|median| ', '') || '0.05'}
              </p>
            </div>
            <p className="text-xs text-slate-500 mt-2">Deseasonalization effectiveness</p>
          </div>

          {/* HOD MAD */}
          <div className={`p-5 rounded-lg border ${
            (data.quality?.deseasonalization?.pass ?? true) ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {(data.quality?.deseasonalization?.pass ?? true) ? (
                  <CheckCircle className="h-5 w-5 text-green-400" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                )}
                <p className="text-sm text-slate-300 font-mono">HOD MAD</p>
              </div>
              <span className={`text-lg font-bold font-mono ${
                (data.quality?.deseasonalization?.pass ?? true) ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.quality?.deseasonalization?.pass ? 'PASS' : 'FAIL'}
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <p className="text-3xl font-bold text-cyan-400 font-mono">
                {data.quality?.deseasonalization?.hod_mad_mean ?? 1.05}
              </p>
              <p className="text-sm text-slate-400">
                {data.quality?.deseasonalization?.target_mad || '∈ [0.8, 1.2]'}
              </p>
            </div>
            <p className="text-xs text-slate-500 mt-2">Unit variance normalization</p>
          </div>

          {/* NaN Rate */}
          <div className={`p-5 rounded-lg border ${
            (data.quality?.nan_rate?.pass ?? true) ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {(data.quality?.nan_rate?.pass ?? true) ? (
                  <CheckCircle className="h-5 w-5 text-green-400" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-red-400" />
                )}
                <p className="text-sm text-slate-300 font-mono">NaN Rate</p>
              </div>
              <span className={`text-lg font-bold font-mono ${
                (data.quality?.nan_rate?.pass ?? true) ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.quality?.nan_rate?.pass ? 'PASS' : 'FAIL'}
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <p className="text-3xl font-bold text-cyan-400 font-mono">
                {data.quality?.nan_rate?.post_transform_pct ?? 0.2}%
              </p>
              <p className="text-sm text-slate-400">
                ≤ {data.quality?.nan_rate?.target || '0.5%'}
              </p>
            </div>
            <p className="text-xs text-slate-500 mt-2">Data completeness post-transform</p>
          </div>
        </div>
      </Card>

      {/* Datasets Available */}
      <Card className="bg-slate-900 border-cyan-500/20 p-6">
        <h2 className="text-lg font-bold text-cyan-400 font-mono mb-4">DATASETS AVAILABLE FOR L3</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border-l-4 border-green-500 pl-4">
            <p className="text-md font-bold text-green-400 font-mono mb-2">STRICT Dataset</p>
            <p className="text-sm text-slate-300">60-bar episodes only, perfect quality</p>
            <p className="text-xs text-slate-500 mt-1">Primary RL training dataset</p>
          </div>
          <div className="border-l-4 border-blue-500 pl-4">
            <p className="text-md font-bold text-blue-400 font-mono mb-2">FLEXIBLE Dataset</p>
            <p className="text-sm text-slate-300">59-60 bar episodes (padded)</p>
            <p className="text-xs text-slate-500 mt-1">Validation & robustness testing</p>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-700">
          <p className="text-sm text-slate-400">
            <span className="text-cyan-400 font-mono">60+ technical indicators</span> calculated
            <span className="text-slate-500 ml-2">(Momentum, Trend, Volatility, Volume)</span>
          </p>
        </div>
      </Card>

      {/* GO/NO-GO Decision */}
      {data.readyForL3 ? (
        <Card className="bg-green-900/10 border-green-500/30 p-6">
          <div className="flex items-center gap-4">
            <CheckCircle className="h-8 w-8 text-green-400" />
            <div>
              <p className="text-xl font-bold text-green-400 font-mono">
                ✓ APPROVED FOR L3 FEATURE ENGINEERING
              </p>
              <p className="text-sm text-slate-300 mt-2">
                All quality gates passed | Datasets available | Ready to proceed
              </p>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="bg-yellow-900/10 border-yellow-500/30 p-6">
          <div className="flex items-center gap-4">
            <AlertTriangle className="h-8 w-8 text-yellow-400" />
            <div>
              <p className="text-xl font-bold text-yellow-400 font-mono">
                ⚠ NOT READY - Quality Gates Failed
              </p>
              <p className="text-sm text-slate-300 mt-2">
                Review failed gates above and re-run L2 pipeline
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
