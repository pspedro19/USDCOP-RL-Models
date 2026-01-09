'use client';

/**
 * LiveVsBacktestComparison Component
 * ===================================
 * Side-by-side comparison of live trading metrics vs backtest metrics.
 * Highlights discrepancies and provides delta calculations.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { motion, AnimatePresence } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Activity,
  Target,
  Percent
} from 'lucide-react';

// ============================================================================
// INTERFACES
// ============================================================================

interface MetricComparison {
  metric_name: string;
  live_value: number | null;
  backtest_value: number | null;
  delta: number | null;
  delta_pct: number | null;
  is_better: boolean | null;
  unit: string;
}

interface ModelComparison {
  model_id: string;
  model_name: string;
  model_type: 'RL' | 'ML' | 'LLM';
  live_status: 'active' | 'paper' | 'inactive';
  backtest_period: string;
  comparisons: MetricComparison[];
  overall_score: number;
  warnings: string[];
}

interface LiveVsBacktestComparisonProps {
  modelId?: string;
  showAllModels?: boolean;
  compact?: boolean;
}

// ============================================================================
// API CONFIGURATION
// ============================================================================

const MULTI_MODEL_API_URL = process.env.NEXT_PUBLIC_MULTI_MODEL_API_URL || 'http://localhost:8006';

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const formatValue = (value: number | null, unit: string): string => {
  if (value === null || value === undefined) return 'N/A';

  switch (unit) {
    case '%':
      return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    case 'ratio':
      return value.toFixed(2);
    case 'usd':
      return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    case 'count':
      return value.toFixed(0);
    default:
      return value.toFixed(2);
  }
};

const getMetricIcon = (metricName: string) => {
  const iconMap: Record<string, React.ReactNode> = {
    sharpe_ratio: <BarChart3 className="h-4 w-4" />,
    sortino_ratio: <BarChart3 className="h-4 w-4" />,
    calmar_ratio: <BarChart3 className="h-4 w-4" />,
    total_return: <TrendingUp className="h-4 w-4" />,
    max_drawdown: <TrendingDown className="h-4 w-4" />,
    win_rate: <Target className="h-4 w-4" />,
    profit_factor: <Percent className="h-4 w-4" />,
    total_trades: <Activity className="h-4 w-4" />
  };
  return iconMap[metricName] || <BarChart3 className="h-4 w-4" />;
};

const getModelColor = (modelType: string): string => {
  switch (modelType) {
    case 'RL':
      return 'blue';
    case 'ML':
      return 'purple';
    case 'LLM':
      return 'amber';
    default:
      return 'slate';
  }
};

// ============================================================================
// SINGLE MODEL COMPARISON CARD
// ============================================================================

function ModelComparisonCard({
  comparison,
  isExpanded,
  onToggle
}: {
  comparison: ModelComparison;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const color = getModelColor(comparison.model_type);
  const hasWarnings = comparison.warnings.length > 0;
  const scoreColor =
    comparison.overall_score >= 80
      ? 'text-green-400'
      : comparison.overall_score >= 60
      ? 'text-yellow-400'
      : 'text-red-400';

  return (
    <Card className={`bg-slate-900/80 border-${color}-500/30 transition-all duration-300`}>
      <CardHeader
        className="cursor-pointer hover:bg-slate-800/50 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full bg-${color}-500`} />
            <div>
              <CardTitle className="text-lg font-mono text-white">
                {comparison.model_name}
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5">
                {comparison.model_type} Model • {comparison.backtest_period}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Overall Score */}
            <div className="text-right">
              <p className="text-xs text-slate-500">Match Score</p>
              <p className={`text-xl font-bold font-mono ${scoreColor}`}>
                {comparison.overall_score}%
              </p>
            </div>

            {/* Status Badges */}
            <Badge
              className={`${
                comparison.live_status === 'active'
                  ? 'bg-green-500/20 text-green-400 border-green-500/50'
                  : comparison.live_status === 'paper'
                  ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
                  : 'bg-slate-500/20 text-slate-400 border-slate-500/50'
              }`}
            >
              {comparison.live_status.toUpperCase()}
            </Badge>

            {hasWarnings && (
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
            )}

            {isExpanded ? (
              <ChevronUp className="h-5 w-5 text-slate-400" />
            ) : (
              <ChevronDown className="h-5 w-5 text-slate-400" />
            )}
          </div>
        </div>
      </CardHeader>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <CardContent className="border-t border-slate-700/50 pt-4">
              {/* Warnings */}
              {hasWarnings && (
                <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                  <p className="text-yellow-400 font-mono text-sm font-bold mb-2">
                    Warnings
                  </p>
                  <ul className="text-yellow-300/80 text-xs space-y-1">
                    {comparison.warnings.map((warning, idx) => (
                      <li key={idx}>• {warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Metrics Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left py-2 px-3 text-slate-400 text-xs font-mono uppercase">
                        Metric
                      </th>
                      <th className="text-right py-2 px-3 text-cyan-400 text-xs font-mono uppercase">
                        Live
                      </th>
                      <th className="text-right py-2 px-3 text-purple-400 text-xs font-mono uppercase">
                        Backtest
                      </th>
                      <th className="text-right py-2 px-3 text-slate-400 text-xs font-mono uppercase">
                        Delta
                      </th>
                      <th className="text-center py-2 px-3 text-slate-400 text-xs font-mono uppercase">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.comparisons.map((metric, idx) => (
                      <tr
                        key={idx}
                        className="border-b border-slate-800/50 hover:bg-slate-800/30"
                      >
                        <td className="py-2.5 px-3">
                          <div className="flex items-center gap-2">
                            <span className="text-slate-500">
                              {getMetricIcon(metric.metric_name)}
                            </span>
                            <span className="text-slate-200 font-mono text-sm">
                              {metric.metric_name
                                .replace(/_/g, ' ')
                                .replace(/\b\w/g, (l) => l.toUpperCase())}
                            </span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <span className="text-cyan-400 font-mono text-sm">
                            {formatValue(metric.live_value, metric.unit)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <span className="text-purple-400 font-mono text-sm">
                            {formatValue(metric.backtest_value, metric.unit)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          {metric.delta !== null && (
                            <span
                              className={`font-mono text-sm ${
                                metric.is_better
                                  ? 'text-green-400'
                                  : metric.is_better === false
                                  ? 'text-red-400'
                                  : 'text-slate-400'
                              }`}
                            >
                              {metric.delta >= 0 ? '+' : ''}
                              {metric.delta_pct !== null
                                ? `${metric.delta_pct.toFixed(1)}%`
                                : formatValue(metric.delta, metric.unit)}
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          {metric.is_better === true ? (
                            <CheckCircle2 className="h-4 w-4 text-green-400 inline" />
                          ) : metric.is_better === false ? (
                            <AlertTriangle className="h-4 w-4 text-red-400 inline" />
                          ) : (
                            <div className="h-4 w-4 rounded-full bg-slate-600 inline-block" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function LiveVsBacktestComparison({
  modelId,
  showAllModels = true,
  compact = false
}: LiveVsBacktestComparisonProps) {
  const [comparisons, setComparisons] = useState<ModelComparison[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Fetch comparison data
  const fetchComparisons = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (modelId) {
        // Fetch single model
        const response = await fetch(`/api/models/${modelId}/comparison`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setComparisons([data.comparison]);
      } else if (showAllModels) {
        // Fetch all models
        const modelIds = ['ppo_v19', 'sac_v1', 'td3_v1', 'lgbm_v1', 'xgb_v1'];
        const results = await Promise.all(
          modelIds.map(async (id) => {
            try {
              const response = await fetch(`/api/models/${id}/comparison`);
              if (!response.ok) return null;
              const data = await response.json();
              return data.comparison;
            } catch {
              return null;
            }
          })
        );
        setComparisons(results.filter(Boolean));
      }

      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch comparisons');
    } finally {
      setLoading(false);
    }
  }, [modelId, showAllModels]);

  useEffect(() => {
    fetchComparisons();

    // Refresh every 60 seconds
    const interval = setInterval(fetchComparisons, 60000);
    return () => clearInterval(interval);
  }, [fetchComparisons]);

  // Toggle expanded state
  const toggleModel = (id: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Summary stats
  const summary = useMemo(() => {
    if (comparisons.length === 0) return null;

    const avgScore = comparisons.reduce((sum, c) => sum + c.overall_score, 0) / comparisons.length;
    const totalWarnings = comparisons.reduce((sum, c) => sum + c.warnings.length, 0);
    const activeModels = comparisons.filter((c) => c.live_status === 'active').length;

    return { avgScore, totalWarnings, activeModels, totalModels: comparisons.length };
  }, [comparisons]);

  // Loading state
  if (loading && comparisons.length === 0) {
    return (
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <RefreshCw className="h-8 w-8 animate-spin text-cyan-500 mx-auto mb-3" />
            <p className="text-slate-300 font-mono">Loading comparisons...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error && comparisons.length === 0) {
    return (
      <Card className="bg-slate-900 border-red-500/30">
        <CardContent className="py-8 text-center">
          <AlertTriangle className="h-8 w-8 text-red-500 mx-auto mb-3" />
          <p className="text-red-400 font-mono mb-4">{error}</p>
          <Button onClick={fetchComparisons} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card className="bg-slate-900 border-cyan-500/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-cyan-500 font-mono flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                LIVE VS BACKTEST COMPARISON
              </CardTitle>
              <p className="text-slate-400 text-sm mt-1">
                Compare live trading performance against historical backtests
              </p>
            </div>

            <div className="flex items-center gap-4">
              {summary && (
                <div className="flex items-center gap-4 text-sm">
                  <div className="text-center">
                    <p className="text-slate-500 text-xs">Avg Match</p>
                    <p
                      className={`font-mono font-bold ${
                        summary.avgScore >= 80
                          ? 'text-green-400'
                          : summary.avgScore >= 60
                          ? 'text-yellow-400'
                          : 'text-red-400'
                      }`}
                    >
                      {summary.avgScore.toFixed(0)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-slate-500 text-xs">Active</p>
                    <p className="text-cyan-400 font-mono font-bold">
                      {summary.activeModels}/{summary.totalModels}
                    </p>
                  </div>
                  {summary.totalWarnings > 0 && (
                    <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/50">
                      {summary.totalWarnings} Warnings
                    </Badge>
                  )}
                </div>
              )}

              {lastUpdate && (
                <span className="text-xs text-slate-500 font-mono">
                  {lastUpdate.toLocaleTimeString()}
                </span>
              )}

              <Button
                onClick={fetchComparisons}
                size="sm"
                variant="outline"
                className="border-slate-600"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Model Comparison Cards */}
      <div className="space-y-3">
        {comparisons.map((comparison) => (
          <ModelComparisonCard
            key={comparison.model_id}
            comparison={comparison}
            isExpanded={expandedModels.has(comparison.model_id)}
            onToggle={() => toggleModel(comparison.model_id)}
          />
        ))}
      </div>

      {/* Empty State */}
      {comparisons.length === 0 && !loading && !error && (
        <Card className="bg-slate-900 border-slate-700">
          <CardContent className="py-12 text-center">
            <BarChart3 className="h-12 w-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400 font-mono">No comparison data available</p>
            <p className="text-slate-500 text-sm mt-2">
              Execute backtests and enable live trading to see comparisons
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
