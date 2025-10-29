'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Target } from 'lucide-react';

interface PerformanceMetricsGridProps {
  strategies: Array<{
    strategy_code: string;
    strategy_name: string;
    strategy_type: 'RL' | 'ML' | 'LLM';
    current_equity: number;
    cash_balance: number;
    open_positions: number;
    total_return_pct: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown_pct: number;
    total_trades: number;
    win_rate: number;
    profit_factor: number;
  }>;
}

export default function PerformanceMetricsGrid({ strategies }: PerformanceMetricsGridProps) {

  const formatPercent = (value: number | null) => {
    if (value === null || value === undefined) return 'N/A';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const formatNumber = (value: number | null, decimals: number = 2) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(decimals);
  };

  const getReturnColor = (value: number | null) => {
    if (value === null || value === undefined) return 'text-slate-400';
    return value >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getStrategyBadgeColor = (type: 'RL' | 'ML' | 'LLM' | string) => {
    switch (type) {
      case 'RL':
        return 'bg-blue-500 text-white';
      case 'ML':
        return 'bg-purple-500 text-white';
      case 'LLM':
        return 'bg-amber-500 text-white';
      default:
        return 'bg-slate-500 text-white';
    }
  };

  if (strategies.length === 0) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardContent className="p-12 text-center">
          <p className="text-slate-500 font-mono">
            No performance data available. Execute L5-L6 pipelines to populate metrics.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-slate-900 border-cyan-500/20">
      <CardHeader>
        <CardTitle className="text-cyan-500 font-mono flex items-center gap-2">
          <Target className="h-5 w-5 text-green-400" />
          PERFORMANCE METRICS
        </CardTitle>
        <p className="text-slate-400 text-sm mt-1">
          Comparative analysis • Last 30 days
        </p>
      </CardHeader>

      <CardContent>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-700">
            <thead>
              <tr className="bg-slate-950/50">
                <th className="px-4 py-3 text-left text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Strategy
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Equity
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Return
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Sharpe
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Sortino
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Max DD
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Win Rate
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Trades
                </th>
                <th className="px-4 py-3 text-right text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                  Profit Factor
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-800">
              {strategies.map((strategy, index) => (
                <tr
                  key={strategy.strategy_code}
                  className="hover:bg-slate-800/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Badge
                        className={`${getStrategyBadgeColor(strategy.strategy_type)} font-mono text-xs`}
                      >
                        {strategy.strategy_code}
                      </Badge>
                      <span className="text-sm text-slate-300">
                        {strategy.strategy_name}
                      </span>
                    </div>
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-white font-bold">
                    ${strategy.current_equity?.toFixed(0) || '10,000'}
                  </td>

                  <td className={`px-4 py-3 text-right font-mono font-bold ${getReturnColor(strategy.total_return_pct)}`}>
                    {formatPercent(strategy.total_return_pct)}
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-cyan-400">
                    {formatNumber(strategy.sharpe_ratio)}
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-cyan-400">
                    {formatNumber(strategy.sortino_ratio)}
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-red-400">
                    {formatPercent(strategy.max_drawdown_pct)}
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-white">
                    {strategy.win_rate ? (strategy.win_rate * 100).toFixed(1) : 'N/A'}%
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-slate-400">
                    {strategy.total_trades || 0}
                  </td>

                  <td className="px-4 py-3 text-right font-mono text-white">
                    {formatNumber(strategy.profit_factor)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Summary Stats */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 border-t border-slate-700 pt-4">
          <div className="text-center">
            <p className="text-xs text-slate-400 font-mono">Best Performer</p>
            <p className="text-lg font-bold text-green-400 font-mono mt-1">
              {strategies.length > 0
                ? strategies.reduce((best, s) =>
                    (s.total_return_pct || 0) > (best.total_return_pct || 0) ? s : best
                  ).strategy_code
                : 'N/A'}
            </p>
          </div>

          <div className="text-center">
            <p className="text-xs text-slate-400 font-mono">Avg Sharpe</p>
            <p className="text-lg font-bold text-cyan-400 font-mono mt-1">
              {strategies.length > 0
                ? (strategies.reduce((sum, s) => sum + (s.sharpe_ratio || 0), 0) / strategies.length).toFixed(2)
                : 'N/A'}
            </p>
          </div>

          <div className="text-center">
            <p className="text-xs text-slate-400 font-mono">Total Trades</p>
            <p className="text-lg font-bold text-white font-mono mt-1">
              {strategies.reduce((sum, s) => sum + (s.total_trades || 0), 0)}
            </p>
          </div>

          <div className="text-center">
            <p className="text-xs text-slate-400 font-mono">Avg Win Rate</p>
            <p className="text-lg font-bold text-white font-mono mt-1">
              {strategies.length > 0
                ? (strategies.reduce((sum, s) => sum + (s.win_rate || 0), 0) / strategies.length * 100).toFixed(1)
                : 'N/A'}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
