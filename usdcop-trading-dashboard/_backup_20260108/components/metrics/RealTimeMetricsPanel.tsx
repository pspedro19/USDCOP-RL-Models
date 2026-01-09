/**
 * Real-Time Metrics Panel
 * Displays comprehensive financial metrics with real-time updates
 */

'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  Target,
  BarChart3,
  AlertCircle,
  CheckCircle,
  Zap,
  Clock,
  Award,
  AlertTriangle,
  RefreshCw,
  Database,
} from 'lucide-react';
import { useFinancialMetrics } from '@/hooks/useFinancialMetrics';
import { FinancialMetrics } from '@/lib/services/financial-metrics/types';

interface RealTimeMetricsPanelProps {
  className?: string;
  initialCapital?: number;
  pollInterval?: number;
  showAdvanced?: boolean;
}

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'gray' | 'cyan';
  change?: number;
  isGood?: boolean;
  tooltip?: string;
  isLoading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color,
  change,
  isGood,
  tooltip,
  isLoading = false,
}) => {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/30 text-blue-400',
    green: 'from-emerald-500/20 to-emerald-600/20 border-emerald-500/30 text-emerald-400',
    red: 'from-red-500/20 to-red-600/20 border-red-500/30 text-red-400',
    yellow: 'from-yellow-500/20 to-yellow-600/20 border-yellow-500/30 text-yellow-400',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/30 text-purple-400',
    gray: 'from-slate-500/20 to-slate-600/20 border-slate-500/30 text-slate-400',
    cyan: 'from-cyan-500/20 to-cyan-600/20 border-cyan-500/30 text-cyan-400',
  };

  return (
    <motion.div
      className={`relative p-4 rounded-xl border bg-gradient-to-br ${colorClasses[color]} backdrop-blur`}
      whileHover={{ scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300 }}
      title={tooltip}
    >
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm rounded-xl flex items-center justify-center z-10">
          <RefreshCw className="h-5 w-5 text-white animate-spin" />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-white/90">{title}</span>
        </div>

        {change !== undefined && !isLoading && (
          <div
            className={`flex items-center gap-1 text-xs ${
              isGood ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {change > 0 ? (
              <TrendingUp className="h-3 w-3" />
            ) : (
              <TrendingDown className="h-3 w-3" />
            )}
            {Math.abs(change).toFixed(2)}%
          </div>
        )}
      </div>

      {/* Value */}
      <div className="mb-1">
        <span className="text-2xl font-mono font-bold text-white">
          {isLoading ? '---' : typeof value === 'number' ? value.toFixed(2) : value}
        </span>
      </div>

      {/* Subtitle */}
      {subtitle && <div className="text-xs text-white/60">{subtitle}</div>}
    </motion.div>
  );
};

export default function RealTimeMetricsPanel({
  className = '',
  initialCapital = 100000,
  pollInterval = 30000,
  showAdvanced = true,
}: RealTimeMetricsPanelProps) {
  const { metrics, summary, isLoading, error, lastUpdate, refresh } = useFinancialMetrics({
    initialCapital,
    pollInterval,
    autoRefresh: true,
    enableWebSocket: true,
  });

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Format percentage
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Determine risk level color
  const getRiskColor = (level: 'low' | 'medium' | 'high') => {
    switch (level) {
      case 'low':
        return 'text-emerald-400';
      case 'medium':
        return 'text-yellow-400';
      case 'high':
        return 'text-red-400';
    }
  };

  // Show error state
  if (error && !metrics) {
    return (
      <div className={`bg-slate-900/50 backdrop-blur border border-red-500/30 rounded-xl p-6 ${className}`}>
        <div className="flex items-center gap-3 text-red-400">
          <AlertCircle className="h-6 w-6" />
          <div>
            <h3 className="font-semibold">Error loading metrics</h3>
            <p className="text-sm text-red-400/80">{error.message}</p>
          </div>
        </div>
        <button
          onClick={refresh}
          className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg text-sm text-white flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    );
  }

  // Show loading state
  if (isLoading && !metrics) {
    return (
      <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6 ${className}`}>
        <div className="flex items-center gap-2 text-slate-400">
          <RefreshCw className="h-5 w-5 animate-spin" />
          <span>Loading financial metrics...</span>
        </div>
      </div>
    );
  }

  // Show empty state
  if (!metrics) {
    return (
      <div className={`bg-fintech-dark-900 rounded-lg border border-fintech-dark-700 text-center p-6 ${className}`}>
        <div className="text-fintech-dark-400">
          <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>No metrics data available</p>
          <button
            onClick={refresh}
            className="mt-4 px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg text-sm text-white flex items-center gap-2 mx-auto"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh Metrics
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Financial Performance Metrics
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Real-time trading performance • {metrics.totalTrades} trades analyzed
          </p>
        </div>

        {/* Status indicators */}
        <div className="flex items-center gap-4">
          {/* Risk indicator */}
          {summary && (
            <div className={`flex items-center gap-2 text-sm ${getRiskColor(summary.riskIndicators.riskLevel)}`}>
              {summary.riskIndicators.isHighRisk ? (
                <AlertTriangle className="h-4 w-4" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
              Risk: {summary.riskIndicators.riskLevel.toUpperCase()}
            </div>
          )}

          {/* Last update */}
          {lastUpdate && (
            <div className="text-xs text-slate-500">
              Updated: {new Date(lastUpdate).toLocaleTimeString()}
            </div>
          )}

          {/* Refresh button */}
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh metrics"
          >
            <RefreshCw className={`h-4 w-4 text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total P&L */}
        <MetricCard
          title="Total P&L"
          value={formatCurrency(metrics.totalPnL)}
          subtitle={`Return: ${formatPercent(metrics.totalReturn)}`}
          icon={<DollarSign className="h-4 w-4" />}
          color={metrics.totalPnL >= 0 ? 'green' : 'red'}
          change={metrics.totalReturn * 100}
          isGood={metrics.totalPnL >= 0}
          tooltip="Total profit/loss including realized and unrealized P&L"
          isLoading={isLoading}
        />

        {/* Sharpe Ratio */}
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpeRatio.toFixed(2)}
          subtitle="Risk-adjusted return"
          icon={<Award className="h-4 w-4" />}
          color={metrics.sharpeRatio > 1 ? 'green' : metrics.sharpeRatio > 0 ? 'yellow' : 'red'}
          tooltip="Higher is better. >1 is good, >2 is excellent"
          isLoading={isLoading}
        />

        {/* Win Rate */}
        <MetricCard
          title="Win Rate"
          value={formatPercent(metrics.winRate)}
          subtitle={`${metrics.winningTrades}W / ${metrics.losingTrades}L`}
          icon={<Target className="h-4 w-4" />}
          color={metrics.winRate > 0.5 ? 'green' : metrics.winRate > 0.4 ? 'yellow' : 'red'}
          tooltip="Percentage of winning trades"
          isLoading={isLoading}
        />

        {/* Max Drawdown */}
        <MetricCard
          title="Max Drawdown"
          value={`${Math.abs(metrics.maxDrawdownPercent).toFixed(2)}%`}
          subtitle={formatCurrency(Math.abs(metrics.maxDrawdown))}
          icon={<TrendingDown className="h-4 w-4" />}
          color={
            Math.abs(metrics.maxDrawdownPercent) < 10
              ? 'green'
              : Math.abs(metrics.maxDrawdownPercent) < 20
              ? 'yellow'
              : 'red'
          }
          tooltip="Largest peak-to-trough decline"
          isLoading={isLoading}
        />
      </div>

      {/* Secondary Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Profit Factor */}
        <MetricCard
          title="Profit Factor"
          value={metrics.profitFactor.toFixed(2)}
          subtitle="Gross profit / Gross loss"
          icon={<BarChart3 className="h-4 w-4" />}
          color={metrics.profitFactor > 2 ? 'green' : metrics.profitFactor > 1 ? 'yellow' : 'red'}
          tooltip="Ratio of gross profit to gross loss. >1 is profitable"
          isLoading={isLoading}
        />

        {/* Sortino Ratio */}
        <MetricCard
          title="Sortino Ratio"
          value={metrics.sortinoRatio.toFixed(2)}
          subtitle="Downside risk-adjusted"
          icon={<Activity className="h-4 w-4" />}
          color={metrics.sortinoRatio > 1 ? 'green' : metrics.sortinoRatio > 0 ? 'yellow' : 'red'}
          tooltip="Similar to Sharpe but only considers downside volatility"
          isLoading={isLoading}
        />

        {/* Expectancy */}
        <MetricCard
          title="Expectancy"
          value={formatCurrency(metrics.expectancy)}
          subtitle="Avg. per trade"
          icon={<Zap className="h-4 w-4" />}
          color={metrics.expectancy > 0 ? 'green' : 'red'}
          tooltip="Average profit/loss per trade"
          isLoading={isLoading}
        />

        {/* Open Positions */}
        <MetricCard
          title="Open Positions"
          value={metrics.openPositions}
          subtitle={`Unrealized: ${formatCurrency(metrics.unrealizedPnL)}`}
          icon={<Clock className="h-4 w-4" />}
          color={metrics.openPositions > 0 ? 'cyan' : 'gray'}
          tooltip="Currently open positions and unrealized P&L"
          isLoading={isLoading}
        />
      </div>

      {/* Advanced Metrics Section */}
      {showAdvanced && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Analysis */}
          <div className="bg-slate-800/30 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Risk Analysis
            </h4>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Current Drawdown</span>
                <span className="text-sm font-mono text-white">
                  {Math.abs(metrics.currentDrawdownPercent).toFixed(2)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">VaR (95%)</span>
                <span className="text-sm font-mono text-white">
                  {(metrics.valueAtRisk95 * 100).toFixed(2)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Expected Shortfall</span>
                <span className="text-sm font-mono text-white">
                  {(metrics.expectedShortfall * 100).toFixed(2)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Volatility (Annual)</span>
                <span className="text-sm font-mono text-white">
                  {(metrics.volatility * 100).toFixed(2)}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Calmar Ratio</span>
                <span className="text-sm font-mono text-white">{metrics.calmarRatio.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Trade Statistics */}
          <div className="bg-slate-800/30 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Trade Statistics
            </h4>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Avg. Win</span>
                <span className="text-sm font-mono text-emerald-400">{formatCurrency(metrics.avgWin)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Avg. Loss</span>
                <span className="text-sm font-mono text-red-400">{formatCurrency(metrics.avgLoss)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Largest Win</span>
                <span className="text-sm font-mono text-emerald-400">
                  {formatCurrency(metrics.largestWin)}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Largest Loss</span>
                <span className="text-sm font-mono text-red-400">
                  {formatCurrency(metrics.largestLoss)}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Avg. Duration</span>
                <span className="text-sm font-mono text-white">
                  {metrics.avgTradeDuration.toFixed(0)} min
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-400">Payoff Ratio</span>
                <span className="text-sm font-mono text-white">{metrics.payoffRatio.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Risk Warnings */}
      {summary && summary.riskIndicators.warnings.length > 0 && (
        <div className="mt-6 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <h4 className="text-yellow-400 font-medium mb-2 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Risk Warnings
          </h4>
          <ul className="space-y-1">
            {summary.riskIndicators.warnings.map((warning, index) => (
              <li key={index} className="text-sm text-yellow-400/90">
                • {warning}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
