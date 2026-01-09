'use client';

/**
 * ModelComparisonPanel Component
 * ==============================
 * Side-by-side comparison of all trading models with key metrics.
 * Highlights the best performer in each category.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Trophy,
  TrendingUp,
  TrendingDown,
  BarChart3,
  RefreshCw,
  Activity,
  Target,
  Percent,
  Clock,
  DollarSign,
  Award,
  Crown
} from 'lucide-react';

// ============================================================================
// INTERFACES
// ============================================================================

interface ModelPerformance {
  model_id: string;
  model_name: string;
  model_type: 'RL' | 'ML' | 'LLM';
  status: 'active' | 'paper' | 'inactive';
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  calmar_ratio: number | null;
  max_drawdown_pct: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  total_return_pct: number | null;
  today_return_pct: number | null;
  total_trades: number;
  avg_trade_duration: string | null;
  last_signal: string | null;
  equity: number;
}

interface ModelComparisonPanelProps {
  showTitle?: boolean;
  highlightBest?: boolean;
  compact?: boolean;
}

// ============================================================================
// API CONFIGURATION
// ============================================================================

const MODEL_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  RL: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400' },
  ML: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400' },
  LLM: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400' },
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const formatValue = (value: number | null, type: string): string => {
  if (value === null || value === undefined) return '--';

  switch (type) {
    case 'percent':
      return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    case 'ratio':
      return value.toFixed(2);
    case 'currency':
      return `$${value.toLocaleString('en-US', { minimumFractionDigits: 0 })}`;
    case 'count':
      return value.toFixed(0);
    default:
      return value.toString();
  }
};

const findBest = (
  models: ModelPerformance[],
  key: keyof ModelPerformance,
  higherIsBetter: boolean = true
): string | null => {
  const validModels = models.filter(
    (m) => m[key] !== null && m[key] !== undefined
  );
  if (validModels.length === 0) return null;

  const bestModel = validModels.reduce((best, current) => {
    const currentVal = current[key] as number;
    const bestVal = best[key] as number;

    if (higherIsBetter) {
      return currentVal > bestVal ? current : best;
    } else {
      return currentVal < bestVal ? current : best;
    }
  });

  return bestModel.model_id;
};

// ============================================================================
// METRIC CELL COMPONENT
// ============================================================================

function MetricCell({
  value,
  type,
  isBest,
  trend
}: {
  value: number | null;
  type: string;
  isBest: boolean;
  trend?: 'up' | 'down' | null;
}) {
  const formatted = formatValue(value, type);
  const isPositive = value !== null && value >= 0;

  return (
    <div
      className={`relative px-3 py-2 text-center rounded ${
        isBest ? 'bg-green-500/10 ring-1 ring-green-500/30' : ''
      }`}
    >
      {isBest && (
        <Crown className="absolute -top-1 -right-1 h-3 w-3 text-yellow-500" />
      )}
      <span
        className={`font-mono text-sm font-medium ${
          type === 'percent'
            ? isPositive
              ? 'text-green-400'
              : 'text-red-400'
            : 'text-slate-200'
        }`}
      >
        {formatted}
      </span>
      {trend && (
        <span className="ml-1">
          {trend === 'up' ? (
            <TrendingUp className="inline h-3 w-3 text-green-400" />
          ) : (
            <TrendingDown className="inline h-3 w-3 text-red-400" />
          )}
        </span>
      )}
    </div>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function ModelComparisonPanel({
  showTitle = true,
  highlightBest = true,
  compact = false
}: ModelComparisonPanelProps) {
  const [models, setModels] = useState<ModelPerformance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Fetch model data
  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/trading/performance/multi-strategy', {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const strategies = data.strategies || [];

      // Transform to ModelPerformance format
      const transformed: ModelPerformance[] = strategies.map((s: any) => ({
        model_id: s.strategy_code || s.model_id || s.name,
        model_name: s.strategy_name || s.name || s.model_id,
        model_type: determineModelType(s.strategy_code || s.model_id || ''),
        status: s.status || 'paper',
        sharpe_ratio: s.sharpe_ratio ?? s.sharpe ?? null,
        sortino_ratio: s.sortino_ratio ?? s.sortino ?? null,
        calmar_ratio: s.calmar_ratio ?? null,
        max_drawdown_pct: s.max_drawdown_pct ?? s.max_dd ?? null,
        win_rate: s.win_rate ?? null,
        profit_factor: s.profit_factor ?? null,
        total_return_pct: s.total_return_pct ?? s.pnl_month_pct ?? null,
        today_return_pct: s.today_return_pct ?? s.pnl_today_pct ?? null,
        total_trades: s.total_trades ?? 0,
        avg_trade_duration: s.avg_trade_duration ?? null,
        last_signal: s.last_signal ?? null,
        equity: s.current_equity ?? s.equity ?? 10000,
      }));

      setModels(transformed);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();

    // Refresh every 30 seconds
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  // Calculate best performers
  const bestPerformers = useMemo(() => {
    if (models.length === 0) return {};

    return {
      sharpe: findBest(models, 'sharpe_ratio', true),
      sortino: findBest(models, 'sortino_ratio', true),
      maxDD: findBest(models, 'max_drawdown_pct', false), // Lower is better
      winRate: findBest(models, 'win_rate', true),
      profitFactor: findBest(models, 'profit_factor', true),
      totalReturn: findBest(models, 'total_return_pct', true),
      todayReturn: findBest(models, 'today_return_pct', true),
    };
  }, [models]);

  // Overall best model
  const overallBest = useMemo(() => {
    if (models.length === 0) return null;

    // Score each model based on rankings
    const scores: Record<string, number> = {};
    models.forEach((m) => {
      scores[m.model_id] = 0;
    });

    Object.values(bestPerformers).forEach((modelId) => {
      if (modelId) {
        scores[modelId] = (scores[modelId] || 0) + 1;
      }
    });

    const bestId = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
    return models.find((m) => m.model_id === bestId);
  }, [models, bestPerformers]);

  // Loading state
  if (loading && models.length === 0) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-cyan-500" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-slate-900 border-cyan-500/20">
      {showTitle && (
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-cyan-500 font-mono flex items-center gap-2">
                <Trophy className="h-5 w-5" />
                MODEL COMPARISON ARENA
              </CardTitle>
              <p className="text-slate-400 text-sm mt-1">
                Side-by-side performance comparison • Best performers highlighted
              </p>
            </div>

            <div className="flex items-center gap-3">
              {overallBest && (
                <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/50 flex items-center gap-1">
                  <Award className="h-3 w-3" />
                  Best: {overallBest.model_name}
                </Badge>
              )}

              {lastUpdate && (
                <span className="text-xs text-slate-500 font-mono">
                  {lastUpdate.toLocaleTimeString()}
                </span>
              )}

              <Button
                onClick={fetchModels}
                size="sm"
                variant="outline"
                className="border-slate-600"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
      )}

      <CardContent className={compact ? 'pt-0' : ''}>
        {error ? (
          <div className="text-center py-8">
            <p className="text-red-400 mb-4">{error}</p>
            <Button onClick={fetchModels} variant="outline">
              Retry
            </Button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="text-left py-3 px-3 text-slate-400 text-xs font-mono uppercase sticky left-0 bg-slate-900 z-10">
                    Model
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    Status
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <BarChart3 className="h-3 w-3" />
                      Sharpe
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <BarChart3 className="h-3 w-3" />
                      Sortino
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <TrendingDown className="h-3 w-3" />
                      Max DD
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <Target className="h-3 w-3" />
                      Win Rate
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <Percent className="h-3 w-3" />
                      P.Factor
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <TrendingUp className="h-3 w-3" />
                      Total Return
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <Clock className="h-3 w-3" />
                      Today
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <Activity className="h-3 w-3" />
                      Trades
                    </div>
                  </th>
                  <th className="text-center py-3 px-3 text-slate-400 text-xs font-mono uppercase">
                    <div className="flex items-center justify-center gap-1">
                      <DollarSign className="h-3 w-3" />
                      Equity
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const colors = MODEL_COLORS[model.model_type] || MODEL_COLORS.ML;
                  const isOverallBest = overallBest?.model_id === model.model_id;

                  return (
                    <tr
                      key={model.model_id}
                      className={`border-b border-slate-800/50 hover:bg-slate-800/30 ${
                        isOverallBest ? 'bg-yellow-500/5' : ''
                      }`}
                    >
                      {/* Model Name */}
                      <td className="py-3 px-3 sticky left-0 bg-slate-900 z-10">
                        <div className="flex items-center gap-2">
                          {isOverallBest && (
                            <Trophy className="h-4 w-4 text-yellow-500" />
                          )}
                          <div
                            className={`w-2 h-2 rounded-full ${colors.text.replace(
                              'text-',
                              'bg-'
                            )}`}
                          />
                          <div>
                            <p className="text-slate-200 font-mono text-sm font-medium">
                              {model.model_name}
                            </p>
                            <p className="text-slate-500 text-xs">
                              {model.model_type}
                            </p>
                          </div>
                        </div>
                      </td>

                      {/* Status */}
                      <td className="py-3 px-3 text-center">
                        <Badge
                          className={`text-xs ${
                            model.status === 'active'
                              ? 'bg-green-500/20 text-green-400'
                              : model.status === 'paper'
                              ? 'bg-yellow-500/20 text-yellow-400'
                              : 'bg-slate-500/20 text-slate-400'
                          }`}
                        >
                          {model.status.toUpperCase()}
                        </Badge>
                      </td>

                      {/* Sharpe */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.sharpe_ratio}
                          type="ratio"
                          isBest={
                            highlightBest &&
                            bestPerformers.sharpe === model.model_id
                          }
                        />
                      </td>

                      {/* Sortino */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.sortino_ratio}
                          type="ratio"
                          isBest={
                            highlightBest &&
                            bestPerformers.sortino === model.model_id
                          }
                        />
                      </td>

                      {/* Max Drawdown */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.max_drawdown_pct}
                          type="percent"
                          isBest={
                            highlightBest &&
                            bestPerformers.maxDD === model.model_id
                          }
                        />
                      </td>

                      {/* Win Rate */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.win_rate}
                          type="percent"
                          isBest={
                            highlightBest &&
                            bestPerformers.winRate === model.model_id
                          }
                        />
                      </td>

                      {/* Profit Factor */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.profit_factor}
                          type="ratio"
                          isBest={
                            highlightBest &&
                            bestPerformers.profitFactor === model.model_id
                          }
                        />
                      </td>

                      {/* Total Return */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.total_return_pct}
                          type="percent"
                          isBest={
                            highlightBest &&
                            bestPerformers.totalReturn === model.model_id
                          }
                        />
                      </td>

                      {/* Today Return */}
                      <td className="py-3 px-3">
                        <MetricCell
                          value={model.today_return_pct}
                          type="percent"
                          isBest={
                            highlightBest &&
                            bestPerformers.todayReturn === model.model_id
                          }
                        />
                      </td>

                      {/* Total Trades */}
                      <td className="py-3 px-3">
                        <div className="text-center">
                          <span className="font-mono text-sm text-slate-300">
                            {model.total_trades}
                          </span>
                        </div>
                      </td>

                      {/* Equity */}
                      <td className="py-3 px-3">
                        <div className="text-center">
                          <span className="font-mono text-sm text-cyan-400">
                            {formatValue(model.equity, 'currency')}
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Legend */}
        <div className="mt-4 pt-4 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Crown className="h-3 w-3 text-yellow-500" />
              Best in category
            </span>
            <span className="flex items-center gap-1">
              <Trophy className="h-3 w-3 text-yellow-500" />
              Overall leader
            </span>
          </div>
          <div className="flex items-center gap-4">
            {Object.entries(MODEL_COLORS).map(([type, colors]) => (
              <span key={type} className="flex items-center gap-1">
                <div
                  className={`w-2 h-2 rounded-full ${colors.text.replace(
                    'text-',
                    'bg-'
                  )}`}
                />
                {type}
              </span>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function determineModelType(modelId: string): 'RL' | 'ML' | 'LLM' {
  const id = modelId.toLowerCase();
  if (id.includes('ppo') || id.includes('sac') || id.includes('td3') || id.includes('rl')) {
    return 'RL';
  }
  if (id.includes('lgbm') || id.includes('xgb') || id.includes('ml')) {
    return 'ML';
  }
  if (id.includes('llm') || id.includes('claude') || id.includes('gpt')) {
    return 'LLM';
  }
  return 'ML';
}
