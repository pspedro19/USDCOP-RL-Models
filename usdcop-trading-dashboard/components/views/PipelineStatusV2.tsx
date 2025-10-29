'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Database, CheckCircle, XCircle, AlertTriangle, Activity, Layers,
  BarChart3, Target, RefreshCw, Clock, Zap, TrendingUp, Shield
} from 'lucide-react';

// Types
interface QualityMetrics {
  [key: string]: any;
}

interface LayerData {
  layer: string;
  name: string;
  status: 'pass' | 'fail' | 'warning';
  pass: boolean;
  quality_metrics?: QualityMetrics;
  last_update?: string;
}

interface PipelineData {
  success: boolean;
  timestamp: string;
  system_health: {
    health_percentage: number;
    passing_layers: number;
    total_layers: number;
    status: string;
  };
  layers: {
    l0?: LayerData;
    l1?: LayerData;
    l2?: LayerData;
    l3?: LayerData;
    l4?: LayerData;
    l5?: LayerData;
    l6?: LayerData;
  };
}

// Status badge component
const StatusBadge = ({ status }: { status: 'pass' | 'fail' | 'warning' | 'loading' }) => {
  const variants = {
    pass: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/50', icon: CheckCircle },
    fail: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', icon: XCircle },
    warning: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/50', icon: AlertTriangle },
    loading: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50', icon: RefreshCw }
  };

  const variant = variants[status];
  const Icon = variant.icon;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${variant.bg} ${variant.border}`}>
      <Icon className={`w-4 h-4 ${variant.text} ${status === 'loading' ? 'animate-spin' : ''}`} />
      <span className={`text-sm font-medium ${variant.text} uppercase`}>
        {status}
      </span>
    </div>
  );
};

// Metric card component
const MetricCard = ({ label, value, unit, icon: Icon }: any) => (
  <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
    <div className="flex items-center gap-3">
      {Icon && <Icon className="w-5 h-5 text-blue-400" />}
      <div>
        <div className="text-sm text-slate-400">{label}</div>
        <div className="text-lg font-semibold text-white">
          {value}{unit && <span className="text-sm text-slate-400 ml-1">{unit}</span>}
        </div>
      </div>
    </div>
  </div>
);

export default function PipelineStatus() {
  const [pipelineData, setPipelineData] = useState<PipelineData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch pipeline data from real API
  const fetchPipelineData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/pipeline/consolidated', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: PipelineData = await response.json();

      if (data.success) {
        setPipelineData(data);
        setLastUpdate(new Date());
      } else {
        throw new Error('API returned success: false');
      }
    } catch (err: any) {
      console.error('[PipelineStatus] Fetch error:', err);
      setError(err.message || 'Failed to fetch pipeline data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchPipelineData();
  }, [fetchPipelineData]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPipelineData();
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [fetchPipelineData]);

  const handleRefresh = () => {
    fetchPipelineData();
  };

  const getStatus = (data: LayerData | undefined): 'pass' | 'fail' | 'warning' | 'loading' => {
    if (!data) return 'loading';
    // If status is 'unknown', treat as loading/pending (not error)
    if (data.status === 'unknown') return 'loading';
    return data.status || 'loading';
  };

  const l0Data = pipelineData?.layers?.l0;
  const l1Data = pipelineData?.layers?.l1;
  const l2Data = pipelineData?.layers?.l2;
  const l3Data = pipelineData?.layers?.l3;
  const l4Data = pipelineData?.layers?.l4;
  const l5Data = pipelineData?.layers?.l5;
  const l6Data = pipelineData?.layers?.l6;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Pipeline Status</h1>
            <p className="text-slate-400">Real-time monitoring of data pipeline layers L0-L6</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-slate-400">
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 rounded-lg text-white font-medium transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && !pipelineData && (
        <div className="max-w-7xl mx-auto mb-6">
          <div className="bg-blue-900/20 border border-blue-500/50 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />
              <div>
                <h3 className="text-blue-400 font-medium">Pipelines Not Executed Yet</h3>
                <p className="text-sm text-blue-300">Run DAGs L1-L6 in Airflow to populate pipeline data</p>
                <code className="text-xs text-slate-400 block mt-1">http://localhost:8080</code>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* System Health Summary */}
      {pipelineData?.system_health && (
        <div className="max-w-7xl mx-auto mb-6">
          <div className={`bg-gradient-to-r backdrop-blur-sm border rounded-xl p-6 shadow-xl ${
            pipelineData.system_health.health_percentage >= 80
              ? 'from-green-900/50 to-emerald-900/50 border-green-500/30'
              : pipelineData.system_health.health_percentage > 0
              ? 'from-yellow-900/50 to-orange-900/50 border-yellow-500/30'
              : 'from-slate-900/50 to-slate-800/50 border-slate-500/30'
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-white mb-1">System Health</h3>
                <p className="text-sm text-slate-300">
                  {pipelineData.system_health.passing_layers} of {pipelineData.system_health.total_layers} layers passing
                </p>
                {pipelineData.system_health.passing_layers === 0 && (
                  <p className="text-xs text-blue-400 mt-2">
                    Pipeline not executed yet - Run DAGs in Airflow
                  </p>
                )}
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${
                  pipelineData.system_health.health_percentage >= 80 ? 'text-green-400' :
                  pipelineData.system_health.health_percentage > 0 ? 'text-yellow-400' : 'text-slate-400'
                }`}>
                  {pipelineData.system_health.health_percentage.toFixed(0)}%
                </div>
                <div className={`text-sm font-medium ${
                  pipelineData.system_health.health_percentage >= 80 ? 'text-green-400' :
                  pipelineData.system_health.health_percentage > 0 ? 'text-yellow-400' : 'text-slate-500'
                }`}>
                  {pipelineData.system_health.passing_layers === 0 ? 'PENDING' : pipelineData.system_health.status.toUpperCase()}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Layers Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* L0: Raw Data */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Database className="w-6 h-6 text-blue-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L0: Raw Data</h2>
                <p className="text-sm text-slate-400">Market data quality</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l0Data)} />
          </div>
          {l0Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Data Coverage"
                value={l0Data.quality_metrics.coverage_pct?.toFixed(1) || 'N/A'}
                unit="%"
                icon={Target}
              />
              <MetricCard
                label="Data Points"
                value={l0Data.quality_metrics.data_points?.toLocaleString() || 'N/A'}
                icon={BarChart3}
              />
              <MetricCard
                label="OHLC Violations"
                value={l0Data.quality_metrics.ohlc_violations || 0}
                icon={AlertTriangle}
              />
            </div>
          )}
        </motion.div>

        {/* L1: Standardized */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Layers className="w-6 h-6 text-cyan-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L1: Standardized</h2>
                <p className="text-sm text-slate-400">Clean data</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l1Data)} />
          </div>
          {l1Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Rows"
                value={l1Data.quality_metrics.rows?.toLocaleString() || 'N/A'}
                icon={BarChart3}
              />
              <MetricCard
                label="Columns"
                value={l1Data.quality_metrics.columns || 'N/A'}
                icon={Layers}
              />
              <MetricCard
                label="File Size"
                value={l1Data.quality_metrics.file_size_mb?.toFixed(2) || 'N/A'}
                unit="MB"
                icon={Database}
              />
            </div>
          )}
        </motion.div>

        {/* L2: Prepared */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Activity className="w-6 h-6 text-green-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L2: Prepared</h2>
                <p className="text-sm text-slate-400">Features & indicators</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l2Data)} />
          </div>
          {l2Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Indicators"
                value={l2Data.quality_metrics.indicators_count || 'N/A'}
                icon={TrendingUp}
              />
              <MetricCard
                label="Winsorization"
                value={l2Data.quality_metrics.winsorization_pct?.toFixed(2) || 'N/A'}
                unit="%"
                icon={Shield}
              />
              <MetricCard
                label="Missing Values"
                value={l2Data.quality_metrics.missing_values_pct?.toFixed(2) || 'N/A'}
                unit="%"
                icon={AlertTriangle}
              />
            </div>
          )}
        </motion.div>

        {/* L3: Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-yellow-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L3: Features</h2>
                <p className="text-sm text-slate-400">Engineered features</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l3Data)} />
          </div>
          {l3Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Features"
                value={l3Data.quality_metrics.features_count || 'N/A'}
                icon={Layers}
              />
              <MetricCard
                label="Correlation"
                value={l3Data.quality_metrics.correlations_computed ? 'Computed' : 'Pending'}
                icon={CheckCircle}
              />
              <MetricCard
                label="Rows"
                value={l3Data.quality_metrics.rows?.toLocaleString() || 'N/A'}
                icon={BarChart3}
              />
            </div>
          )}
        </motion.div>

        {/* L4: RL-Ready */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Target className="w-6 h-6 text-purple-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L4: RL-Ready</h2>
                <p className="text-sm text-slate-400">Training dataset</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l4Data)} />
          </div>
          {l4Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Max Clip Rate"
                value={l4Data.quality_metrics.max_clip_rate_pct?.toFixed(2) || 'N/A'}
                unit="%"
                icon={Shield}
              />
              <MetricCard
                label="Reward RMSE"
                value={l4Data.quality_metrics.reward_rmse?.toFixed(4) || 'N/A'}
                icon={Target}
              />
              <MetricCard
                label="Episodes"
                value={l4Data.quality_metrics.episodes?.toLocaleString() || 'N/A'}
                icon={Database}
              />
            </div>
          )}
        </motion.div>

        {/* L5: Serving */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.5 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-orange-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L5: Serving</h2>
                <p className="text-sm text-slate-400">Model deployment</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l5Data)} />
          </div>
          {l5Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Model"
                value={l5Data.quality_metrics.active_models ? 'Available' : 'N/A'}
                icon={CheckCircle}
              />
              <MetricCard
                label="Inference"
                value="Ready"
                icon={Zap}
              />
              <MetricCard
                label="Latency"
                value={l5Data.quality_metrics.inference_latency_ms || 'N/A'}
                unit="ms"
                icon={Clock}
              />
            </div>
          )}
        </motion.div>

        {/* L6: Backtest */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.6 }}
          className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="w-6 h-6 text-emerald-400" />
              <div>
                <h2 className="text-xl font-bold text-white">L6: Backtest</h2>
                <p className="text-sm text-slate-400">Model performance</p>
              </div>
            </div>
            <StatusBadge status={getStatus(l6Data)} />
          </div>
          {l6Data?.quality_metrics && (
            <div className="space-y-3">
              <MetricCard
                label="Sortino Ratio"
                value={l6Data.quality_metrics.sortino_ratio?.toFixed(2) || 'N/A'}
                icon={TrendingUp}
              />
              <MetricCard
                label="Sharpe Ratio"
                value={l6Data.quality_metrics.sharpe_ratio?.toFixed(2) || 'N/A'}
                icon={Target}
              />
              <MetricCard
                label="Win Rate"
                value={l6Data.quality_metrics.win_rate_pct?.toFixed(1) || 'N/A'}
                unit="%"
                icon={CheckCircle}
              />
            </div>
          )}
        </motion.div>
      </div>

      {/* Quality Gates Summary */}
      <div className="max-w-7xl mx-auto mt-6">
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl">
          <h3 className="text-lg font-bold text-white mb-4">Quality Gates Summary</h3>
          <div className="flex items-center gap-4 flex-wrap">
            {['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'].map((layer, idx) => {
              const layerKey = `l${idx}` as keyof typeof pipelineData.layers;
              const layerData = pipelineData?.layers?.[layerKey];
              const status = layerData?.pass ? 'pass' : 'fail';

              return (
                <div key={layer} className="flex items-center gap-2">
                  {status === 'pass' ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-400" />
                  )}
                  <span className="text-sm font-medium text-slate-300">{layer}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
