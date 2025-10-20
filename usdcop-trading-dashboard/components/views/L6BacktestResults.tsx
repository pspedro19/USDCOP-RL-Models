'use client';

import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { TrendingUp, TrendingDown, Activity, DollarSign, Percent, AlertCircle } from 'lucide-react';

interface BacktestMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  total_return: number;
  avg_trade_pnl: number;
  winning_trades: number;
  losing_trades: number;
}

export default function L6BacktestResults() {
  const [data, setData] = useState<BacktestMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch('/api/pipeline/l6/backtest-results?split=test');
        if (!response.ok) throw new Error('Failed to fetch backtest results');
        const result = await response.json();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-2 border-cyan-500/20 border-t-cyan-500 mx-auto mb-4"></div>
          <p className="text-cyan-500 font-mono text-sm">Loading backtest results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <p className="font-mono">Error: {error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const metrics = [
    {
      label: 'Sharpe Ratio',
      value: data.sharpe_ratio.toFixed(2),
      icon: TrendingUp,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-400/10'
    },
    {
      label: 'Sortino Ratio',
      value: data.sortino_ratio.toFixed(2),
      icon: Activity,
      color: 'text-purple-400',
      bgColor: 'bg-purple-400/10'
    },
    {
      label: 'Calmar Ratio',
      value: data.calmar_ratio.toFixed(2),
      icon: TrendingUp,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    },
    {
      label: 'Max Drawdown',
      value: `${(data.max_drawdown * 100).toFixed(2)}%`,
      icon: TrendingDown,
      color: 'text-red-400',
      bgColor: 'bg-red-400/10'
    },
    {
      label: 'Win Rate',
      value: `${(data.win_rate * 100).toFixed(1)}%`,
      icon: Percent,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-400/10'
    },
    {
      label: 'Profit Factor',
      value: data.profit_factor.toFixed(2),
      icon: DollarSign,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10'
    },
    {
      label: 'Total Trades',
      value: data.total_trades.toLocaleString(),
      icon: Activity,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10'
    },
    {
      label: 'Total Return',
      value: `${(data.total_return * 100).toFixed(2)}%`,
      icon: TrendingUp,
      color: data.total_return >= 0 ? 'text-green-400' : 'text-red-400',
      bgColor: data.total_return >= 0 ? 'bg-green-400/10' : 'bg-red-400/10'
    },
  ];

  return (
    <div className="space-y-6 p-6 bg-slate-950 min-h-screen">
      <div className="border-b border-cyan-500/20 pb-4">
        <h1 className="text-2xl font-bold text-cyan-500 font-mono">L6 BACKTEST RESULTS</h1>
        <p className="text-slate-400 text-sm mt-1">
          Test Split Performance Metrics • {data.total_trades} Trades •
          {data.winning_trades} Wins / {data.losing_trades} Losses
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <Card
              key={index}
              className="bg-slate-900 border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg ${metric.bgColor}`}>
                  <Icon className={`h-6 w-6 ${metric.color}`} />
                </div>
              </div>
              <p className="text-sm text-slate-400 font-mono mb-2">{metric.label}</p>
              <p className={`text-3xl font-bold font-mono ${metric.color}`}>
                {metric.value}
              </p>
            </Card>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">Trade Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Total Trades</span>
              <span className="text-white font-mono font-bold">{data.total_trades}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Winning Trades</span>
              <span className="text-green-400 font-mono font-bold">{data.winning_trades}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Losing Trades</span>
              <span className="text-red-400 font-mono font-bold">{data.losing_trades}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Avg Trade P&L</span>
              <span className={`font-mono font-bold ${data.avg_trade_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {data.avg_trade_pnl >= 0 ? '+' : ''}{data.avg_trade_pnl.toFixed(4)}
              </span>
            </div>
          </div>
        </Card>

        <Card className="bg-slate-900 border-cyan-500/20 p-6">
          <h3 className="text-lg font-bold text-cyan-400 font-mono mb-4">Risk Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Sharpe Ratio</span>
              <span className="text-cyan-400 font-mono font-bold">{data.sharpe_ratio.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Sortino Ratio</span>
              <span className="text-purple-400 font-mono font-bold">{data.sortino_ratio.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Calmar Ratio</span>
              <span className="text-green-400 font-mono font-bold">{data.calmar_ratio.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded">
              <span className="text-slate-300 font-mono text-sm">Max Drawdown</span>
              <span className="text-red-400 font-mono font-bold">
                {(data.max_drawdown * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
