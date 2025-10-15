/**
 * Performance Analytics Dashboard
 * ==============================
 *
 * Institutional-grade performance analytics with comprehensive metrics,
 * risk analysis, and interactive visualizations.
 */

'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { EChartsIndicator, CorrelationHeatmap } from '../visualization/EChartsIndicators';
import { BacktestResult, PerformanceMetrics, Trade } from '../types';

interface PerformanceAnalyticsProps {
  backtestResults: BacktestResult[];
  benchmarkData?: {
    name: string;
    returns: number[];
    timestamps: number[];
  };
  onExport?: (format: 'pdf' | 'excel' | 'csv') => void;
}

interface RiskMetric {
  name: string;
  value: number;
  benchmark?: number;
  format: 'percentage' | 'ratio' | 'currency' | 'days';
  description: string;
  category: 'return' | 'risk' | 'ratio' | 'period';
}

export const PerformanceAnalytics: React.FC<PerformanceAnalyticsProps> = ({
  backtestResults,
  benchmarkData,
  onExport
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(backtestResults[0]?.strategy || '');
  const [timeRange, setTimeRange] = useState<'1M' | '3M' | '6M' | '1Y' | 'ALL'>('ALL');
  const [metricCategory, setMetricCategory] = useState<'all' | 'return' | 'risk' | 'ratio' | 'period'>('all');

  const selectedResult = useMemo(() => {
    return backtestResults.find(result => result.strategy === selectedStrategy) || backtestResults[0];
  }, [backtestResults, selectedStrategy]);

  const filteredEquity = useMemo(() => {
    if (!selectedResult || timeRange === 'ALL') return selectedResult?.equity || [];

    const now = Date.now() / 1000;
    const ranges = {
      '1M': 30 * 24 * 60 * 60,
      '3M': 90 * 24 * 60 * 60,
      '6M': 180 * 24 * 60 * 60,
      '1Y': 365 * 24 * 60 * 60
    };

    const cutoff = now - ranges[timeRange];
    return selectedResult.equity.filter(point => point.timestamp >= cutoff);
  }, [selectedResult, timeRange]);

  const performanceMetrics = useMemo((): RiskMetric[] => {
    if (!selectedResult) return [];

    const metrics: RiskMetric[] = [
      // Return Metrics
      {
        name: 'Total Return',
        value: selectedResult.performance.returns.total,
        format: 'percentage',
        description: 'Total cumulative return over the backtest period',
        category: 'return'
      },
      {
        name: 'Annualized Return',
        value: selectedResult.performance.returns.annualized,
        format: 'percentage',
        description: 'Annualized return based on compounding',
        category: 'return'
      },
      {
        name: 'Volatility',
        value: selectedResult.performance.returns.volatility,
        format: 'percentage',
        description: 'Annualized standard deviation of returns',
        category: 'return'
      },

      // Risk Metrics
      {
        name: 'Sharpe Ratio',
        value: selectedResult.performance.risk.sharpe,
        format: 'ratio',
        description: 'Risk-adjusted return relative to volatility',
        category: 'risk'
      },
      {
        name: 'Sortino Ratio',
        value: selectedResult.performance.risk.sortino,
        format: 'ratio',
        description: 'Risk-adjusted return relative to downside deviation',
        category: 'risk'
      },
      {
        name: 'Calmar Ratio',
        value: selectedResult.performance.risk.calmar,
        format: 'ratio',
        description: 'Annualized return divided by maximum drawdown',
        category: 'risk'
      },
      {
        name: 'Maximum Drawdown',
        value: selectedResult.performance.risk.maxDrawdown,
        format: 'percentage',
        description: 'Largest peak-to-trough decline',
        category: 'risk'
      },
      {
        name: 'VaR (95%)',
        value: selectedResult.performance.risk.var95,
        format: 'percentage',
        description: 'Value at Risk at 95% confidence level',
        category: 'risk'
      },
      {
        name: 'CVaR (95%)',
        value: selectedResult.performance.risk.cvar95,
        format: 'percentage',
        description: 'Conditional Value at Risk (Expected Shortfall)',
        category: 'risk'
      },

      // Trading Metrics
      {
        name: 'Win Rate',
        value: selectedResult.performance.periods.winRate,
        format: 'percentage',
        description: 'Percentage of profitable trades',
        category: 'period'
      },
      {
        name: 'Profit Factor',
        value: selectedResult.performance.periods.profitFactor,
        format: 'ratio',
        description: 'Gross profit divided by gross loss',
        category: 'period'
      },
      {
        name: 'Average Win',
        value: selectedResult.performance.periods.averageWin,
        format: 'currency',
        description: 'Average profit per winning trade',
        category: 'period'
      },
      {
        name: 'Average Loss',
        value: Math.abs(selectedResult.performance.periods.averageLoss),
        format: 'currency',
        description: 'Average loss per losing trade',
        category: 'period'
      },
      {
        name: 'Max Consecutive Wins',
        value: selectedResult.performance.periods.consecutiveWins,
        format: 'ratio',
        description: 'Maximum number of consecutive winning trades',
        category: 'period'
      },
      {
        name: 'Max Consecutive Losses',
        value: selectedResult.performance.periods.consecutiveLosses,
        format: 'ratio',
        description: 'Maximum number of consecutive losing trades',
        category: 'period'
      }
    ];

    return metricCategory === 'all' ? metrics : metrics.filter(m => m.category === metricCategory);
  }, [selectedResult, metricCategory]);

  const formatValue = useCallback((value: number, format: string): string => {
    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(2)}%`;
      case 'ratio':
        return value.toFixed(2);
      case 'currency':
        return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      case 'days':
        return `${Math.round(value)} days`;
      default:
        return value.toFixed(2);
    }
  }, []);

  const equityChartData = useMemo(() => {
    if (!filteredEquity.length) return [];

    return [{
      name: 'Portfolio Value',
      data: filteredEquity.map(point => ({
        timestamp: point.timestamp * 1000,
        value: point.equity
      }))
    }];
  }, [filteredEquity]);

  const drawdownChartData = useMemo(() => {
    if (!filteredEquity.length) return [];

    return [{
      name: 'Drawdown',
      data: filteredEquity.map(point => ({
        timestamp: point.timestamp * 1000,
        value: -point.drawdown * 100 // Convert to negative percentage
      }))
    }];
  }, [filteredEquity]);

  const returnsDistribution = useMemo(() => {
    if (!selectedResult) return [];

    const returns = selectedResult.equity
      .slice(1)
      .map((point, index) => {
        const prevPoint = selectedResult.equity[index];
        return (point.equity - prevPoint.equity) / prevPoint.equity;
      });

    // Create histogram bins
    const bins = 50;
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const binWidth = (max - min) / bins;

    const histogram = Array(bins).fill(0);
    returns.forEach(ret => {
      const binIndex = Math.min(Math.floor((ret - min) / binWidth), bins - 1);
      histogram[binIndex]++;
    });

    return histogram.map((count, index) => ({
      timestamp: min + (index + 0.5) * binWidth,
      value: count
    }));
  }, [selectedResult]);

  const monthlyReturns = useMemo(() => {
    if (!selectedResult) return [];

    const monthlyData: { [key: string]: { returns: number; month: string } } = {};

    selectedResult.equity.forEach((point, index) => {
      if (index === 0) return;

      const date = new Date(point.timestamp * 1000);
      const monthKey = `${date.getFullYear()}-${date.getMonth().toString().padStart(2, '0')}`;
      const prevPoint = selectedResult.equity[index - 1];
      const monthlyReturn = (point.equity - prevPoint.equity) / prevPoint.equity;

      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = {
          returns: 0,
          month: date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' })
        };
      }

      monthlyData[monthKey].returns += monthlyReturn;
    });

    return Object.values(monthlyData).map((data, index) => ({
      timestamp: index,
      value: data.returns * 100,
      month: data.month
    }));
  }, [selectedResult]);

  const tradeAnalysis = useMemo(() => {
    if (!selectedResult?.trades) return { winners: [], losers: [], summary: {} };

    const winners = selectedResult.trades.filter(trade => trade.pnl > 0);
    const losers = selectedResult.trades.filter(trade => trade.pnl < 0);

    const summary = {
      totalTrades: selectedResult.trades.length,
      winningTrades: winners.length,
      losingTrades: losers.length,
      largestWin: Math.max(...winners.map(t => t.pnl), 0),
      largestLoss: Math.min(...losers.map(t => t.pnl), 0),
      averageTradeDuration: selectedResult.trades.reduce((sum, t) => sum + t.duration, 0) / selectedResult.trades.length / (24 * 60 * 60), // in days
      totalCommissions: selectedResult.trades.reduce((sum, t) => sum + t.commission, 0)
    };

    return { winners, losers, summary };
  }, [selectedResult]);

  const renderMetricCard = (metric: RiskMetric) => (
    <div key={metric.name} className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-sm font-medium text-gray-600">{metric.name}</h3>
        <span className={`px-2 py-1 text-xs rounded-full ${
          metric.category === 'return' ? 'bg-blue-100 text-blue-800' :
          metric.category === 'risk' ? 'bg-red-100 text-red-800' :
          metric.category === 'ratio' ? 'bg-green-100 text-green-800' :
          'bg-gray-100 text-gray-800'
        }`}>
          {metric.category}
        </span>
      </div>

      <div className="text-2xl font-bold text-gray-900 mb-1">
        {formatValue(metric.value, metric.format)}
      </div>

      {metric.benchmark && (
        <div className="text-sm text-gray-500 mb-2">
          Benchmark: {formatValue(metric.benchmark, metric.format)}
        </div>
      )}

      <p className="text-xs text-gray-500">{metric.description}</p>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Performance Analytics</h1>
          <p className="text-gray-600 mt-2">Comprehensive analysis of trading strategy performance</p>
        </div>

        <div className="flex space-x-4">
          <select
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {backtestResults.map(result => (
              <option key={result.strategy} value={result.strategy}>
                {result.strategy}
              </option>
            ))}
          </select>

          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="ALL">All Time</option>
            <option value="1Y">1 Year</option>
            <option value="6M">6 Months</option>
            <option value="3M">3 Months</option>
            <option value="1M">1 Month</option>
          </select>

          {onExport && (
            <div className="flex space-x-2">
              <button
                onClick={() => onExport('pdf')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Export PDF
              </button>
              <button
                onClick={() => onExport('excel')}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Export Excel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Key Performance Summary */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-6 rounded-xl text-white">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <div className="text-sm opacity-90">Total Return</div>
            <div className="text-2xl font-bold">
              {formatValue(selectedResult?.performance.returns.total || 0, 'percentage')}
            </div>
          </div>
          <div>
            <div className="text-sm opacity-90">Sharpe Ratio</div>
            <div className="text-2xl font-bold">
              {(selectedResult?.performance.risk.sharpe || 0).toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-sm opacity-90">Max Drawdown</div>
            <div className="text-2xl font-bold">
              {formatValue(selectedResult?.performance.risk.maxDrawdown || 0, 'percentage')}
            </div>
          </div>
          <div>
            <div className="text-sm opacity-90">Win Rate</div>
            <div className="text-2xl font-bold">
              {formatValue(selectedResult?.performance.periods.winRate || 0, 'percentage')}
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Equity Curve */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Equity Curve</h2>
          <EChartsIndicator
            data={equityChartData}
            type="area"
            config={{
              colors: ['#3b82f6'],
              opacity: 0.3,
              smooth: true,
              fill: true,
              title: 'Portfolio Value Over Time'
            }}
            height={300}
          />
        </div>

        {/* Drawdown Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Drawdown Analysis</h2>
          <EChartsIndicator
            data={drawdownChartData}
            type="area"
            config={{
              colors: ['#ef4444'],
              opacity: 0.3,
              smooth: true,
              fill: true,
              title: 'Drawdown Periods',
              yAxis: { max: 0 }
            }}
            height={300}
          />
        </div>

        {/* Returns Distribution */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Returns Distribution</h2>
          <EChartsIndicator
            data={returnsDistribution}
            type="histogram"
            config={{
              colors: ['#8b5cf6', '#ef4444'],
              title: 'Distribution of Daily Returns'
            }}
            height={300}
          />
        </div>

        {/* Monthly Returns */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Monthly Returns</h2>
          <EChartsIndicator
            data={monthlyReturns}
            type="histogram"
            config={{
              colors: ['#10b981', '#ef4444'],
              title: 'Monthly Performance'
            }}
            height={300}
          />
        </div>
      </div>

      {/* Performance Metrics */}
      <div>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-semibold">Performance Metrics</h2>
          <select
            value={metricCategory}
            onChange={(e) => setMetricCategory(e.target.value as any)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Metrics</option>
            <option value="return">Return Metrics</option>
            <option value="risk">Risk Metrics</option>
            <option value="ratio">Risk Ratios</option>
            <option value="period">Trading Metrics</option>
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {performanceMetrics.map(renderMetricCard)}
        </div>
      </div>

      {/* Trade Analysis */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h2 className="text-2xl font-semibold mb-6">Trade Analysis</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="text-center">
            <div className="text-3xl font-bold text-gray-900">{tradeAnalysis.summary.totalTrades}</div>
            <div className="text-sm text-gray-600">Total Trades</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">{tradeAnalysis.summary.winningTrades}</div>
            <div className="text-sm text-gray-600">Winning Trades</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600">{tradeAnalysis.summary.losingTrades}</div>
            <div className="text-sm text-gray-600">Losing Trades</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">
              {(tradeAnalysis.summary.averageTradeDuration || 0).toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Avg Duration (days)</div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-medium mb-4">Largest Winners</h3>
            <div className="space-y-2">
              {tradeAnalysis.winners
                .sort((a, b) => b.pnl - a.pnl)
                .slice(0, 5)
                .map((trade, index) => (
                  <div key={trade.id} className="flex justify-between items-center p-3 bg-green-50 rounded">
                    <div>
                      <div className="font-medium">{trade.symbol}</div>
                      <div className="text-sm text-gray-600">
                        {new Date(trade.timestamp * 1000).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-green-600">
                        {formatValue(trade.pnl, 'currency')}
                      </div>
                      <div className="text-sm text-gray-600">
                        {formatValue(trade.pnl / (trade.price * trade.quantity), 'percentage')}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-medium mb-4">Largest Losers</h3>
            <div className="space-y-2">
              {tradeAnalysis.losers
                .sort((a, b) => a.pnl - b.pnl)
                .slice(0, 5)
                .map((trade, index) => (
                  <div key={trade.id} className="flex justify-between items-center p-3 bg-red-50 rounded">
                    <div>
                      <div className="font-medium">{trade.symbol}</div>
                      <div className="text-sm text-gray-600">
                        {new Date(trade.timestamp * 1000).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-red-600">
                        {formatValue(trade.pnl, 'currency')}
                      </div>
                      <div className="text-sm text-gray-600">
                        {formatValue(trade.pnl / (trade.price * trade.quantity), 'percentage')}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Analysis */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h2 className="text-2xl font-semibold mb-6">Risk Analysis</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-medium mb-4">Drawdown Periods</h3>
            <div className="space-y-3">
              {selectedResult?.drawdowns.slice(0, 5).map((drawdown, index) => (
                <div key={index} className="p-4 bg-red-50 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">Drawdown {index + 1}</span>
                    <span className="text-red-600 font-bold">
                      {formatValue(drawdown.drawdown, 'percentage')}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>
                      Start: {new Date(drawdown.start * 1000).toLocaleDateString()}
                    </div>
                    <div>
                      End: {new Date(drawdown.end * 1000).toLocaleDateString()}
                    </div>
                    <div>
                      Duration: {Math.round(drawdown.duration / (24 * 60 * 60))} days
                    </div>
                    <div>
                      Recovery: {Math.round(drawdown.recovery / (24 * 60 * 60))} days
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-medium mb-4">Risk Metrics Summary</h3>
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">Value at Risk (95%)</div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatValue(selectedResult?.performance.risk.var95 || 0, 'percentage')}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Expected maximum loss on 95% of days
                </div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">Conditional VaR (95%)</div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatValue(selectedResult?.performance.risk.cvar95 || 0, 'percentage')}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Expected loss when VaR is exceeded
                </div>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">Volatility</div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatValue(selectedResult?.performance.returns.volatility || 0, 'percentage')}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Annualized standard deviation
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Strategy Comparison */}
      {backtestResults.length > 1 && (
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h2 className="text-2xl font-semibold mb-6">Strategy Comparison</h2>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Strategy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Total Return
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Sharpe Ratio
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Max Drawdown
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Win Rate
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Profit Factor
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {backtestResults.map((result, index) => (
                  <tr key={result.strategy} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {result.strategy}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatValue(result.performance.returns.total, 'percentage')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.performance.risk.sharpe.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatValue(result.performance.risk.maxDrawdown, 'percentage')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatValue(result.performance.periods.winRate, 'percentage')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.performance.periods.profitFactor.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};