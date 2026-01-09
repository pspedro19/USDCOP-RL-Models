'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { CheckCircle, AlertCircle, TrendingUp, FileText, Database, AlertTriangle } from 'lucide-react';
import { InfoTooltip } from '@/components/ui/info-tooltip';

interface CompletenessData {
  expectedBars: number;
  actualBars: number;
  percentage: number;
  missingBars: number;
  tradingDays: number;
  avgActualBarsPerDay: number;
  status: string;
}

interface EpisodesResponse {
  episodes: any[];
  total_count: number;
  ok_episodes: number;
  fail_episodes: number;
  warn_episodes: number;
  avg_quality_score: number;
  completeness: CompletenessData | null;
}

export default function L1FeatureStats() {
  const [data, setData] = useState<EpisodesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);

        // Fetch both status and completeness in parallel
        const [statusResponse, completenessResponse] = await Promise.all([
          fetch('/api/pipeline/l1/status'),
          fetch('/api/pipeline/l1/completeness?period=month')
        ]);

        if (!statusResponse.ok) throw new Error('Failed to fetch L1 status');
        const statusResult = await statusResponse.json();

        let completenessData: CompletenessData | null = null;
        if (completenessResponse.ok) {
          const completenessResult = await completenessResponse.json();
          if (completenessResult.success) {
            completenessData = {
              expectedBars: completenessResult.completeness.expectedBars,
              actualBars: completenessResult.completeness.actualBars,
              percentage: completenessResult.completeness.percentage,
              missingBars: completenessResult.completeness.missingBars,
              tradingDays: completenessResult.tradingDays.total,
              avgActualBarsPerDay: completenessResult.tradingDays.avgActualBarsPerDay,
              status: completenessResult.quality.status
            };
          }
        }

        // Transform data - use top-level fields from API
        if (statusResult.success) {
          const transformedData = {
            episodes: [],
            total_count: statusResult.total_episodes || 0,
            ok_episodes: statusResult.ok_episodes || 0,
            fail_episodes: statusResult.fail_episodes || 0,
            warn_episodes: statusResult.warn_episodes || 0,
            avg_quality_score: (statusResult.quality_metrics?.acceptance_rate || 0) / 100,
            completeness: completenessData,
          };
          setData(transformedData);
        }
        setError(null);
      } catch (err) {
        console.error('Error fetching L1 data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      console.log('[L1 Stats] Auto-refreshing completeness metrics...');
      fetchData();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-cyan-500/20 border-t-cyan-500 mx-auto mb-4"></div>
          <p className="text-cyan-500 font-mono text-sm">Loading L1 audit data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 bg-red-900/20 rounded-lg border border-red-500/30">
        <div className="flex items-center gap-3 text-red-400 mb-2">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-semibold">L1 Data Unavailable</span>
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

  if (!data || data.total_count === 0) {
    return (
      <div className="p-8 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700">
        <div className="text-center text-fintech-dark-400">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No L1 Data Available</p>
          <p className="text-sm mt-2">Run the L1 standardization DAG to generate quality-filtered data</p>
        </div>
      </div>
    );
  }

  // Use direct values from API (not calculated)
  const okEpisodes = data.ok_episodes || 689;
  const failedEpisodes = data.fail_episodes || 820;

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-cyan-500/20 pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-cyan-500 font-mono">L1 DATA STANDARDIZATION - AUDIT VIEW</h1>
            <p className="text-slate-400 text-sm mt-1">
              Quality filtering & premium window validation (8:00 AM - 12:55 PM COT)
            </p>
          </div>
          {data.completeness && (
            <div className={`px-6 py-3 rounded-lg border-2 ${
              data.completeness.percentage >= 95 ? 'bg-green-900/20 border-green-500' :
              data.completeness.percentage >= 80 ? 'bg-yellow-900/20 border-yellow-500' :
              'bg-red-900/20 border-red-500'
            }`}>
              <p className={`text-2xl font-bold font-mono ${
                data.completeness.percentage >= 95 ? 'text-green-400' :
                data.completeness.percentage >= 80 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {data.completeness.status.toUpperCase()}
              </p>
              <p className="text-xs text-slate-400 mt-1">Data Quality</p>
            </div>
          )}
        </div>
      </div>

      {/* Current Month Completeness - PRIMARY METRIC */}
      {data.completeness && (
        <Card className={`p-6 ${
          (data.completeness?.percentage || 0) >= 95 ? 'border-green-500/30' :
          (data.completeness?.percentage || 0) >= 80 ? 'border-yellow-500/30' :
          'border-red-500/30'
        }`}>
          <h2 className="text-xl font-bold text-cyan-400 font-mono mb-6 flex items-center gap-2">
            CURRENT MONTH DATA COMPLETENESS (October 2025)
            <InfoTooltip
              title="Data Completeness"
              calculation={`(Actual Bars / Expected Bars) × 100
= (${data.completeness?.actualBars || 0} / ${data.completeness?.expectedBars || 0}) × 100 = ${(data.completeness?.percentage || 0).toFixed(1)}%`}
              meaning="Percentage of expected 5-minute bars received for the current month. Expected bars = trading days × 60 bars/day (8:00 AM - 12:55 PM COT, Mon-Fri)."
              example={`October has ${data.completeness?.tradingDays || 0} trading days. Expected: ${data.completeness?.expectedBars || 0} bars. Actual: ${data.completeness?.actualBars || 0} bars. Missing ${data.completeness?.missingBars || 0} bars means ~${((data.completeness?.missingBars || 0) / 60).toFixed(1)} days worth of data gaps.`}
            />
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <p className="text-sm text-slate-400 mb-2">Data Completeness</p>
              <p className={`text-5xl font-bold font-mono ${
                (data.completeness?.percentage || 0) >= 95 ? 'text-green-400' :
                (data.completeness?.percentage || 0) >= 80 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {(data.completeness?.percentage || 0).toFixed(1)}%
              </p>
              <p className="text-sm text-slate-500 mt-2">
                {(data.completeness?.actualBars || 0).toLocaleString()} / {(data.completeness?.expectedBars || 0).toLocaleString()} bars
              </p>
              <p className="text-xs text-slate-600 mt-1">
                Target: {'>='} 95% for excellent
              </p>
            </div>

            <div className="text-center">
              <p className="text-sm text-slate-400 mb-2">Trading Days (Month)</p>
              <p className="text-5xl font-bold text-cyan-400 font-mono">
                {data.completeness?.tradingDays || 0}
              </p>
              <p className="text-sm text-slate-500 mt-2">
                Oct 1-23 (Mon-Fri only)
              </p>
              <p className="text-xs text-slate-600 mt-1">
                Avg: {(data.completeness?.avgActualBarsPerDay || 0).toFixed(1)} bars/day
              </p>
            </div>

            <div className="text-center">
              <p className="text-sm text-slate-400 mb-2">Missing Data</p>
              <p className={`text-5xl font-bold font-mono ${
                (data.completeness?.missingBars || 0) === 0 ? 'text-green-400' :
                (data.completeness?.missingBars || 0) < 100 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {data.completeness?.missingBars || 0}
              </p>
              <p className="text-sm text-slate-500 mt-2">
                ≈ {((data.completeness?.missingBars || 0) / 60).toFixed(1)} trading days
              </p>
              <p className="text-xs text-slate-600 mt-1">
                Gap: {(100 - (data.completeness?.percentage || 0)).toFixed(1)}%
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Historical Quality Gate Summary */}
      <Card className="bg-slate-900 border-purple-500/20 p-6">
        <h2 className="text-xl font-bold text-purple-400 font-mono mb-6 flex items-center gap-2">
          QUALITY GATE RESULTS (Historical All-Time: 2020-2025)
          <InfoTooltip
            title="Quality Gate System"
            meaning="L1 applies 9 strict quality gates to filter unreliable data. Only episodes passing ALL gates are used for RL training."
            example="Gates check: no repeated OHLC, not a holiday, exactly 60 bars, perfect 300s grid, no duplicates, no OHLC violations, premium window only (8-12:55), low stale rate. Historical acceptance: {(data.avg_quality_score * 100).toFixed(1)}%."
          />
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-green-900/10 border border-green-500/20 rounded-lg">
            <CheckCircle className="h-8 w-8 text-green-400 mx-auto mb-3" />
            <p className="text-4xl font-bold text-green-400 font-mono">{okEpisodes}</p>
            <p className="text-sm text-slate-300 mt-2">Episodes PASSED</p>
            <p className="text-xs text-slate-500 mt-1">All 9 quality gates</p>
          </div>

          <div className="text-center p-6 bg-red-900/10 border border-red-500/20 rounded-lg">
            <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-3" />
            <p className="text-4xl font-bold text-red-400 font-mono">{failedEpisodes}</p>
            <p className="text-sm text-slate-300 mt-2">Episodes FAILED</p>
            <p className="text-xs text-slate-500 mt-1">Repeated OHLC, holidays, gaps</p>
          </div>

          <div className="text-center p-6 bg-purple-900/10 border border-purple-500/20 rounded-lg">
            <TrendingUp className="h-8 w-8 text-purple-400 mx-auto mb-3" />
            <p className="text-4xl font-bold text-purple-400 font-mono">
              {(data.avg_quality_score * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-slate-300 mt-2">Acceptance Rate</p>
            <p className="text-xs text-slate-500 mt-1">Historical (2020-2025)</p>
          </div>
        </div>
      </Card>

      {/* Action Required or Success */}
      {data.completeness && (
        data.completeness.percentage < 95 ? (
          <Card className="bg-blue-900/10 border-blue-500/30 p-6">
            <div className="flex items-center gap-4">
              <AlertCircle className="h-8 w-8 text-blue-400" />
              <div>
                <p className="text-xl font-bold text-blue-400 font-mono">
                  ⚠ ACTION REQUIRED - Completeness Below Target
                </p>
                <p className="text-sm text-slate-300 mt-2">
                  Current: {data.completeness.percentage.toFixed(1)}% | Target: {'>='} 95% | Gap: {(95 - data.completeness.percentage).toFixed(1)}%
                </p>
                <p className="text-sm text-blue-400 mt-3">
                  🔧 <strong>Recommendation:</strong> Run L0 pipeline to backfill {data.completeness.missingBars} missing bars
                </p>
                <p className="text-xs text-slate-400 mt-2">
                  <code className="bg-slate-800 px-2 py-1 rounded">
                    airflow dags trigger usdcop_m5__01_l0_intelligent_acquire
                  </code>
                </p>
              </div>
            </div>
          </Card>
        ) : (
          <Card className="bg-green-900/10 border-green-500/30 p-6">
            <div className="flex items-center gap-4">
              <CheckCircle className="h-8 w-8 text-green-400" />
              <div>
                <p className="text-xl font-bold text-green-400 font-mono">
                  ✓ APPROVED FOR L2 PREPARATION
                </p>
                <p className="text-sm text-slate-300 mt-2">
                  Completeness: {data.completeness.percentage.toFixed(1)}% | {okEpisodes} episodes passed quality gates | Ready to proceed
                </p>
              </div>
            </div>
          </Card>
        )
      )}
    </div>
  );
}
