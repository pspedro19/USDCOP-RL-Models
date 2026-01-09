'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Database, CheckCircle, XCircle, AlertTriangle, Activity, Layers,
  BarChart3, Target, RefreshCw, Clock, Zap, TrendingUp, Shield
} from 'lucide-react';

// Status badge component
const StatusBadge = ({ status }: { status: 'pass' | 'fail' | 'warning' | 'loading' | 'unknown' }) => {
  const variants = {
    pass: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/50', icon: CheckCircle },
    fail: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/50', icon: XCircle },
    warning: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/50', icon: AlertTriangle },
    loading: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50', icon: RefreshCw },
    unknown: { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/50', icon: AlertTriangle }
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
const MetricCard = ({ label, value, unit, icon: Icon, trend }: any) => (
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
    {trend && (
      <div className={`text-sm ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
        {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
      </div>
    )}
  </div>
);

export default function PipelineStatus() {
  const [pipelineData, setPipelineData] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /**
   * Fetch real pipeline status from consolidated API
   * NO MOCK DATA - All data from /api/pipeline/consolidated endpoint
   */
  const fetchPipelineStatus = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/pipeline/consolidated', {
        cache: 'no-store'
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch pipeline status');
      }

      setPipelineData(data);
      setLastUpdate(new Date());
    } catch (err: any) {
      console.error('[PipelineStatus] Error fetching data:', err);
      setError(err.message || 'Failed to load pipeline status');
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchPipelineStatus();

    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchPipelineStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchPipelineStatus();
  };

  const getStatus = (data: any): 'pass' | 'fail' | 'warning' | 'loading' | 'unknown' => {
    if (loading) return 'loading';
    if (!data) return 'unknown';
    return data.status || 'unknown';
  };

  // Show loading state
  if (loading && !pipelineData) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-white text-lg font-medium">Loading Pipeline Status...</p>
          <p className="text-slate-400 text-sm mt-2">Fetching real-time data from all layers</p>
        </div>
      </div>
    );
  }

  // Show error state
  if (error && !pipelineData) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
        <div className="bg-red-900/20 border border-red-500/50 rounded-xl p-8 max-w-2xl">
          <div className="flex items-center gap-3 mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
            <h2 className="text-2xl font-bold text-red-400">Failed to Load Pipeline Status</h2>
          </div>
          <p className="text-red-200 mb-4">{error}</p>
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white font-medium transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

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

      {/* System Health Summary */}
      {pipelineData?.system_health && (
        <div className="max-w-7xl mx-auto mb-6">
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 backdrop-blur-sm border border-blue-500/30 rounded-xl p-6 shadow-xl">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-white mb-1">System Health</h3>
                <p className="text-sm text-slate-300">
                  {pipelineData.system_health.passing_layers} of {pipelineData.system_health.total_layers} layers passing
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold text-white">
                  {pipelineData.system_health.health_percentage.toFixed(0)}%
                </div>
                <div className={`text-sm font-medium ${
                  pipelineData.system_health.status === 'healthy' ? 'text-green-400' :
                  pipelineData.system_health.status === 'degraded' ? 'text-yellow-400' :
                  'text-red-400'
                }`}>
                  {pipelineData.system_health.status.toUpperCase()}
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
                icon={BarChart3}
              />
              <MetricCard
                label="Data Points"
                value={l0Data.quality_metrics.data_points || 0}
                icon={Database}
              />
              <MetricCard
                label="OHLC Violations"
                value={l0Data.quality_metrics.ohlc_violations || 0}
                icon={Shield}
              />
            </div>
          )}

          {l0Data?.error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
              <p className="text-red-300 text-sm">{l0Data.error}</p>
            </div>
          )}
        </motion.div>

        {/* L1-L6 cards */}
        {[l1Data, l2Data, l3Data, l4Data, l5Data, l6Data].map((layerData, idx) => {
          const layerNum = idx + 1;
          const layerNames = ['Standardized', 'Prepared', 'Features', 'RL-Ready', 'Serving', 'Backtest'];
          const layerIcons = [Layers, BarChart3, Target, Activity, Zap, TrendingUp];
          const LayerIcon = layerIcons[idx];

          return (
            <motion.div
              key={`L${layerNum}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * (idx + 1) }}
              className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <LayerIcon className="w-6 h-6 text-blue-400" />
                  <div>
                    <h2 className="text-xl font-bold text-white">L{layerNum}: {layerNames[idx]}</h2>
                    <p className="text-sm text-slate-400">{layerData?.name || layerNames[idx]}</p>
                  </div>
                </div>
                <StatusBadge status={getStatus(layerData)} />
              </div>

              {layerData?.quality_metrics && Object.keys(layerData.quality_metrics).length > 0 && (
                <div className="space-y-2">
                  {Object.entries(layerData.quality_metrics).slice(0, 3).map(([key, value]: [string, any]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-slate-400 capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-white font-mono">
                        {typeof value === 'number' ? value.toFixed(2) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {layerData?.error && (
                <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                  <p className="text-yellow-300 text-xs">{layerData.error}</p>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Quality Gates Summary */}
      {pipelineData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="max-w-7xl mx-auto mt-6"
        >
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-400" />
              Quality Gates Summary
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              {[l0Data, l1Data, l2Data, l3Data, l4Data, l5Data, l6Data].map((layer, idx) => (
                <div key={idx} className="text-center p-4 bg-slate-900/50 rounded-lg">
                  <div className={`text-2xl font-bold ${
                    layer?.pass ? 'text-green-400' :
                    layer?.status === 'unknown' ? 'text-gray-400' :
                    'text-red-400'
                  }`}>
                    {layer?.pass ? '✓' : layer?.status === 'unknown' ? '?' : '✗'}
                  </div>
                  <div className="text-sm text-slate-400 mt-1">L{idx}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Real Data Badge */}
      <div className="max-w-7xl mx-auto mt-6 text-center">
        <p className="text-xs text-slate-500 font-mono">
          ✅ 100% Real Data • No Mock Values • Auto-refresh: 30s •
          Data Source: /api/pipeline/consolidated • Last Update: {pipelineData?.timestamp}
        </p>
      </div>
    </div>
  );
}
