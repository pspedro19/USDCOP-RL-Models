'use client';

/**
 * useModelMetrics Hook
 * =====================
 * Fetches performance metrics for the selected model.
 * Compares live metrics vs backtest metrics.
 *
 * Data source: /api/models/{modelId}/metrics
 * Fallback: /api/trading/performance/multi-strategy
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import type { ModelMetrics, PeriodOption } from '@/lib/config/models.config';
import { modelRefreshIntervals } from '@/lib/config/models.config';

// ============================================================================
// Types
// ============================================================================

interface UseModelMetricsOptions {
  period?: PeriodOption;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface MetricComparison {
  live: number | null;
  backtest: number;
  delta: number | null;
  isBetter: boolean | null;
}

interface UseModelMetricsReturn {
  metrics: ModelMetrics | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
  // Comparison helpers
  comparisons: {
    sharpe: MetricComparison;
    maxDrawdown: MetricComparison;
    winRate: MetricComparison;
    holdPercent: MetricComparison;
  } | null;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useModelMetrics(
  options: UseModelMetricsOptions = {}
): UseModelMetricsReturn {
  const {
    period = '30d',
    autoRefresh = true,
    refreshInterval = modelRefreshIntervals.metrics,
  } = options;

  const { modelId, model, isLoading: isModelLoading } = useSelectedModel();

  // State
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch metrics from API
  const fetchMetrics = useCallback(async () => {
    if (!modelId) {
      setMetrics(null);
      setIsLoading(false);
      return;
    }

    // Abort previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      setError(null);

      // Try model-specific endpoint first
      let response = await fetch(
        `/api/models/${modelId}/metrics?period=${period}`,
        {
          signal: abortControllerRef.current.signal,
          headers: { 'Content-Type': 'application/json' },
        }
      );

      // Fallback to multi-strategy performance endpoint
      if (!response.ok && response.status === 404) {
        response = await fetch(
          `/api/trading/performance/multi-strategy`,
          {
            signal: abortControllerRef.current.signal,
            headers: { 'Content-Type': 'application/json' },
          }
        );

        if (response.ok) {
          const data = await response.json();
          const strategies = data.strategies || [];

          // Find the strategy that matches our model
          const modelStrategy = strategies.find((s: any) =>
            s.model_id === modelId ||
            s.name?.toLowerCase().includes(modelId.toLowerCase().replace('_', ''))
          );

          if (modelStrategy) {
            const normalizedMetrics = normalizeFromMultiStrategy(
              modelStrategy,
              modelId,
              period,
              model?.backtest
            );
            setMetrics(normalizedMetrics);
            setLastUpdated(new Date());
            return;
          }
        }
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.status}`);
      }

      const responseData = await response.json();
      // Extract the inner data object (API returns { success, data: {...} })
      const data = responseData.data || responseData;
      setMetrics(normalizeMetrics(data, modelId, period));
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return;
      }

      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      console.error('Error fetching metrics:', err);
    } finally {
      setIsLoading(false);
    }
  }, [modelId, period, model?.backtest]);

  // Initial fetch
  useEffect(() => {
    if (!isModelLoading) {
      setIsLoading(true);
      fetchMetrics();
    }
  }, [modelId, isModelLoading, period, fetchMetrics]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh || !modelId) return;

    intervalRef.current = setInterval(fetchMetrics, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, modelId, fetchMetrics]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Calculate comparisons
  const comparisons = metrics
    ? {
        sharpe: calculateComparison(
          metrics.live.sharpe,
          metrics.backtest.sharpe,
          'higher'
        ),
        maxDrawdown: calculateComparison(
          metrics.live.maxDrawdown,
          metrics.backtest.maxDrawdown,
          'lower'
        ),
        winRate: calculateComparison(
          metrics.live.winRate,
          metrics.backtest.winRate,
          'higher'
        ),
        holdPercent: calculateComparison(
          metrics.live.holdPercent,
          metrics.backtest.holdPercent,
          'neutral'
        ),
      }
    : null;

  return {
    metrics,
    isLoading,
    error,
    lastUpdated,
    refresh: fetchMetrics,
    comparisons,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Normalize metrics from API
 * Handles both direct live data and nested metrics object
 */
function normalizeMetrics(
  data: any,
  modelId: string,
  period: string
): ModelMetrics {
  // Support both { live: {...} } and { metrics: {...}, live: {...} } structures
  const metrics = data.metrics || {};
  const liveData = data.live || {};

  return {
    modelId,
    period,
    // Also attach raw metrics for components that need direct access
    metrics,
    live: {
      sharpe: liveData.sharpe ?? metrics.sharpe_ratio ?? null,
      maxDrawdown: liveData.maxDrawdown ?? metrics.max_drawdown_pct ?? null,
      winRate: liveData.winRate ?? metrics.win_rate ?? null,
      holdPercent: liveData.holdPercent ?? metrics.hold_pct ?? null,
      totalTrades: liveData.totalTrades ?? metrics.total_trades ?? 0,
      pnlToday: liveData.pnlToday ?? metrics.pnl_today ?? null,
      pnlTodayPct: liveData.pnlTodayPct ?? metrics.pnl_today_pct ?? null,
      pnlMonth: liveData.pnlMonth ?? metrics.total_return ?? null,
      pnlMonthPct: liveData.pnlMonthPct ?? metrics.total_return_pct ?? null,
      // Additional fields from our API
      currentEquity: metrics.current_equity ?? null,
      startEquity: metrics.start_equity ?? 10000,
      tradingDays: metrics.trading_days ?? null,
      startDate: metrics.start_date ?? null,
      endDate: metrics.end_date ?? null,
    },
    backtest: {
      sharpe: data.backtest?.sharpe ?? 0,
      maxDrawdown: data.backtest?.maxDrawdown ?? data.backtest?.max_dd ?? 0,
      winRate: data.backtest?.winRate ?? data.backtest?.win_rate ?? 0,
      holdPercent: data.backtest?.holdPercent ?? data.backtest?.hold_pct ?? 0,
    },
  } as ModelMetrics;
}

/**
 * Normalize from multi-strategy endpoint
 */
function normalizeFromMultiStrategy(
  strategy: any,
  modelId: string,
  period: string,
  backtest?: any
): ModelMetrics {
  return {
    modelId,
    period,
    live: {
      sharpe: strategy.sharpe ?? strategy.sharpe_ratio ?? null,
      maxDrawdown: strategy.max_dd ?? strategy.maxDrawdown ?? null,
      winRate: strategy.win_rate ?? strategy.winRate ?? null,
      holdPercent: strategy.hold_pct ?? strategy.holdPercent ?? null,
      totalTrades: strategy.total_trades ?? strategy.totalTrades ?? 0,
      pnlToday: strategy.pnl_today ?? null,
      pnlTodayPct: strategy.pnl_today_pct ?? null,
      pnlMonth: strategy.pnl_month ?? null,
      pnlMonthPct: strategy.pnl_month_pct ?? null,
    },
    backtest: backtest || {
      sharpe: 0,
      maxDrawdown: 0,
      winRate: 0,
      holdPercent: 0,
    },
  };
}

/**
 * Calculate comparison between live and backtest
 */
function calculateComparison(
  live: number | null,
  backtest: number,
  betterWhen: 'higher' | 'lower' | 'neutral'
): MetricComparison {
  if (live === null) {
    return {
      live: null,
      backtest,
      delta: null,
      isBetter: null,
    };
  }

  const delta = live - backtest;
  let isBetter: boolean | null = null;

  if (betterWhen === 'higher') {
    isBetter = delta > 0;
  } else if (betterWhen === 'lower') {
    isBetter = delta < 0;
  }

  return {
    live,
    backtest,
    delta,
    isBetter,
  };
}

export default useModelMetrics;
