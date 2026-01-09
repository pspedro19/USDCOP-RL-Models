/**
 * Financial Metrics Usage Example
 * Demonstrates how to use the financial metrics system
 */

'use client';

import React from 'react';
import { useFinancialMetrics, useRealtimeMetrics } from '@/hooks/useFinancialMetrics';
import RealTimeMetricsPanel from './RealTimeMetricsPanel';
import { AlertTriangle, Database } from 'lucide-react';

/**
 * Example 1: Basic Usage
 */
export function BasicMetricsExample() {
  const { metrics, isLoading, error } = useFinancialMetrics({
    initialCapital: 100000,
    pollInterval: 30000, // 30 seconds
  });

  if (isLoading) {
    return (
      <div className="p-6 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700 animate-pulse">
        <div className="h-4 bg-fintech-dark-700 rounded w-1/3 mb-4"></div>
        <div className="h-8 bg-fintech-dark-700 rounded w-1/2"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-900/20 rounded-lg border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400">
          <AlertTriangle className="w-5 h-5" />
          <span>Failed to load metrics: {error.message}</span>
        </div>
      </div>
    );
  }

  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <div className="p-6 bg-fintech-dark-900 rounded-lg border border-fintech-dark-700 text-center">
        <div className="text-fintech-dark-400">
          <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>No metrics data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Basic Metrics</h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="text-slate-400 text-sm">Total P&L</p>
          <p className="text-2xl font-bold text-white">${metrics.totalPnL.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="text-slate-400 text-sm">Win Rate</p>
          <p className="text-2xl font-bold text-white">{(metrics.winRate * 100).toFixed(1)}%</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="text-slate-400 text-sm">Sharpe Ratio</p>
          <p className="text-2xl font-bold text-white">{metrics.sharpeRatio.toFixed(2)}</p>
        </div>
        <div className="bg-slate-800 p-4 rounded-lg">
          <p className="text-slate-400 text-sm">Max Drawdown</p>
          <p className="text-2xl font-bold text-white">{metrics.maxDrawdownPercent.toFixed(2)}%</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Example 2: Real-time Metrics with WebSocket
 */
export function RealtimeMetricsExample() {
  const { metrics, lastUpdate, isLoading } = useRealtimeMetrics({
    initialCapital: 100000,
  });

  if (isLoading || !metrics) return <div>Connecting to real-time feed...</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">Real-time Metrics</h2>
        <div className="text-sm text-slate-400">
          Last update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'N/A'}
        </div>
      </div>

      <div className="bg-slate-800 p-4 rounded-lg">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-sm text-slate-400">Live Updates Active</span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-slate-400">Current P&L</span>
            <span className={`font-mono ${metrics.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${metrics.totalPnL.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Unrealized P&L</span>
            <span className={`font-mono ${metrics.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${metrics.unrealizedPnL.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Open Positions</span>
            <span className="font-mono text-white">{metrics.openPositions}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Total Trades</span>
            <span className="font-mono text-white">{metrics.totalTrades}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Example 3: Performance Summary with Risk Analysis
 */
export function PerformanceSummaryExample() {
  const { summary, isLoading } = useFinancialMetrics({
    initialCapital: 100000,
  });

  if (isLoading || !summary) return <div>Loading performance summary...</div>;

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'text-green-400';
      case 'medium':
        return 'text-yellow-400';
      case 'high':
        return 'text-red-400';
      default:
        return 'text-slate-400';
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Performance Summary</h2>

      {/* Risk Assessment */}
      <div className="bg-slate-800 p-4 rounded-lg">
        <h3 className="font-semibold text-white mb-3">Risk Assessment</h3>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-slate-400">Risk Level</span>
            <span className={`font-semibold ${getRiskColor(summary.riskIndicators.riskLevel)}`}>
              {summary.riskIndicators.riskLevel.toUpperCase()}
            </span>
          </div>

          {summary.riskIndicators.warnings.length > 0 && (
            <div className="mt-4 space-y-1">
              <p className="text-sm font-medium text-yellow-400">Warnings:</p>
              {summary.riskIndicators.warnings.map((warning, idx) => (
                <p key={idx} className="text-sm text-slate-400">
                  • {warning}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Top Trades */}
      <div className="bg-slate-800 p-4 rounded-lg">
        <h3 className="font-semibold text-white mb-3">Top 5 Trades</h3>
        <div className="space-y-2">
          {summary.topTrades.slice(0, 5).map((trade, idx) => (
            <div key={trade.id} className="flex justify-between items-center">
              <span className="text-slate-400 text-sm">
                #{idx + 1} - {new Date(trade.timestamp).toLocaleDateString()}
              </span>
              <span className="font-mono text-green-400">${trade.pnl.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-slate-800 p-4 rounded-lg">
        <h3 className="font-semibold text-white mb-3">Recent Activity</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-slate-400 text-sm">Last 24h</p>
            <p className="text-lg font-bold text-white">{summary.recentActivity.last24h} trades</p>
          </div>
          <div>
            <p className="text-slate-400 text-sm">Last 7d</p>
            <p className="text-lg font-bold text-white">{summary.recentActivity.last7d} trades</p>
          </div>
          <div>
            <p className="text-slate-400 text-sm">Last 30d</p>
            <p className="text-lg font-bold text-white">{summary.recentActivity.last30d} trades</p>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Example 4: Equity Curve Visualization
 */
export function EquityCurveExample() {
  const { metrics, isLoading } = useFinancialMetrics({
    initialCapital: 100000,
  });

  if (isLoading || !metrics) return <div>Loading equity curve...</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">Equity Curve</h2>

      <div className="bg-slate-800 p-4 rounded-lg">
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-slate-400 text-sm">Starting Capital</p>
              <p className="text-lg font-bold text-white">$100,000</p>
            </div>
            <div>
              <p className="text-slate-400 text-sm">Current Value</p>
              <p className="text-lg font-bold text-white">
                ${metrics.equityCurve[metrics.equityCurve.length - 1]?.value.toFixed(2) || 'N/A'}
              </p>
            </div>
          </div>

          {/* Equity Points */}
          <div>
            <p className="text-sm font-medium text-white mb-2">
              Equity Points: {metrics.equityCurve.length}
            </p>
            <div className="h-32 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-800">
                  <tr className="text-slate-400">
                    <th className="text-left py-1">Date</th>
                    <th className="text-right py-1">Value</th>
                    <th className="text-right py-1">Return</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.equityCurve.slice(-10).reverse().map((point, idx) => (
                    <tr key={idx} className="border-t border-slate-700">
                      <td className="py-1 text-slate-400">
                        {new Date(point.timestamp).toLocaleDateString()}
                      </td>
                      <td className="text-right font-mono text-white">${point.value.toFixed(2)}</td>
                      <td
                        className={`text-right font-mono ${
                          point.cumReturn >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}
                      >
                        {point.cumReturn.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Example 5: Complete Dashboard with RealTimeMetricsPanel
 */
export function CompleteDashboardExample() {
  return (
    <div className="space-y-6 p-6 bg-slate-900 min-h-screen">
      <h1 className="text-3xl font-bold text-white">Financial Metrics Dashboard</h1>

      {/* Main Metrics Panel */}
      <RealTimeMetricsPanel
        initialCapital={100000}
        pollInterval={30000}
        showAdvanced={true}
      />

      {/* Additional Examples */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RealtimeMetricsExample />
        <PerformanceSummaryExample />
      </div>
    </div>
  );
}

// Export default for easy importing
export default CompleteDashboardExample;
