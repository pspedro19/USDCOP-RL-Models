/**
 * useFinancialMetrics Hook
 * React hook for fetching and calculating financial metrics
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Trade,
  Position,
  Signal,
  FinancialMetrics,
  PerformanceSummary,
  MetricsOptions,
} from '@/lib/services/financial-metrics/types';
import { MetricsCalculator } from '@/lib/services/financial-metrics/MetricsCalculator';
import { PerformanceAnalyzer } from '@/lib/services/financial-metrics/PerformanceAnalyzer';
import { EquityCurveBuilder } from '@/lib/services/financial-metrics/EquityCurveBuilder';

interface UseFinancialMetricsOptions extends MetricsOptions {
  pollInterval?: number; // milliseconds (default: 30000 = 30s)
  enableWebSocket?: boolean; // Enable real-time updates via WebSocket
  autoRefresh?: boolean; // Auto-refresh on interval
}

interface UseFinancialMetricsResult {
  metrics: FinancialMetrics | null;
  summary: PerformanceSummary | null;
  trades: Trade[];
  positions: Position[];
  isLoading: boolean;
  error: Error | null;
  lastUpdate: number | null;
  refresh: () => Promise<void>;
  refetch: () => Promise<void>;
}

const DEFAULT_OPTIONS: UseFinancialMetricsOptions = {
  pollInterval: 30000, // 30 seconds
  enableWebSocket: true,
  autoRefresh: true,
  riskFreeRate: 0.03,
  confidenceLevel: 0.95,
  tradingDaysPerYear: 252,
  initialCapital: 100000,
  includeOpenPositions: true,
};

export function useFinancialMetrics(
  options: UseFinancialMetricsOptions = {}
): UseFinancialMetricsResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // State
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [metrics, setMetrics] = useState<FinancialMetrics | null>(null);
  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const cacheRef = useRef<{
    trades: Trade[];
    positions: Position[];
    metrics: FinancialMetrics;
    timestamp: number;
  } | null>(null);

  /**
   * Fetch trades from API
   */
  const fetchTrades = useCallback(async (): Promise<Trade[]> => {
    try {
      const response = await fetch('/api/trading/trades', {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch trades: ${response.statusText}`);
      }

      const data = await response.json();

      // Transform API data to Trade format
      const transformedTrades: Trade[] = (data.trades || data || []).map((t: any) => ({
        id: t.id || t.trade_id || String(Math.random()),
        timestamp: new Date(t.timestamp || t.entry_time).getTime(),
        symbol: t.symbol || 'USDCOP',
        side: t.side || t.direction || 'buy',
        quantity: t.quantity || t.size || 0,
        entryPrice: t.entry_price || t.price || 0,
        exitPrice: t.exit_price || t.close_price,
        entryTime: new Date(t.entry_time || t.timestamp).getTime(),
        exitTime: t.exit_time ? new Date(t.exit_time).getTime() : undefined,
        pnl: t.pnl || t.profit || 0,
        pnlPercent: t.pnl_percent || t.return_pct || 0,
        commission: t.commission || t.fee || 0,
        status: t.status === 'closed' || t.exit_time ? 'closed' : 'open',
        duration: t.duration_minutes || t.duration,
        strategy: t.strategy || t.strategy_name,
      }));

      return transformedTrades;
    } catch (err) {
      console.error('Error fetching trades:', err);
      throw err;
    }
  }, []);

  /**
   * Fetch positions from API
   */
  const fetchPositions = useCallback(async (): Promise<Position[]> => {
    try {
      const response = await fetch('/api/trading/positions', {
        cache: 'no-store',
      });

      if (!response.ok) {
        // If positions endpoint doesn't exist, return empty array
        if (response.status === 404) {
          return [];
        }
        throw new Error(`Failed to fetch positions: ${response.statusText}`);
      }

      const data = await response.json();

      // Transform API data to Position format
      const transformedPositions: Position[] = (data.positions || data || []).map((p: any) => ({
        id: p.id || p.position_id || String(Math.random()),
        symbol: p.symbol || 'USDCOP',
        side: p.side || p.direction || 'long',
        quantity: p.quantity || p.size || 0,
        entryPrice: p.entry_price || p.avg_price || 0,
        currentPrice: p.current_price || p.last_price || 0,
        unrealizedPnL: p.unrealized_pnl || p.floating_pnl || 0,
        unrealizedPnLPercent: p.unrealized_pnl_percent || p.return_pct || 0,
        openTime: new Date(p.open_time || p.entry_time).getTime(),
        duration: p.duration_minutes || Math.floor((Date.now() - new Date(p.open_time || p.entry_time).getTime()) / 60000),
        strategy: p.strategy || p.strategy_name,
      }));

      return transformedPositions;
    } catch (err) {
      console.error('Error fetching positions:', err);
      return []; // Return empty array on error
    }
  }, []);

  /**
   * Calculate metrics from trades and positions
   */
  const calculateMetrics = useCallback(
    (trades: Trade[], positions: Position[]): { metrics: FinancialMetrics; summary: PerformanceSummary } => {
      // Check cache
      const now = Date.now();
      if (
        cacheRef.current &&
        cacheRef.current.trades === trades &&
        cacheRef.current.positions === positions &&
        now - cacheRef.current.timestamp < 5000 // 5 second cache
      ) {
        return {
          metrics: cacheRef.current.metrics,
          summary: PerformanceAnalyzer.generateSummary(trades, positions, opts),
        };
      }

      // Calculate metrics
      let metrics = MetricsCalculator.calculateMetrics(trades, positions, opts);

      // Update max drawdown and Calmar ratio
      const maxDrawdownInfo = EquityCurveBuilder.getMaxDrawdown(metrics.equityCurve);
      const currentDrawdownInfo = EquityCurveBuilder.getCurrentDrawdown(metrics.equityCurve);

      metrics = {
        ...metrics,
        maxDrawdown: maxDrawdownInfo.value,
        maxDrawdownPercent: maxDrawdownInfo.percent,
        currentDrawdown: currentDrawdownInfo.value,
        currentDrawdownPercent: currentDrawdownInfo.percent,
      };

      // Update Calmar ratio
      metrics = MetricsCalculator.updateCalmarRatio(metrics);

      // Generate summary
      const summary = PerformanceAnalyzer.generateSummary(trades, positions, opts);

      // Update cache
      cacheRef.current = {
        trades,
        positions,
        metrics,
        timestamp: now,
      };

      return { metrics, summary };
    },
    [opts]
  );

  /**
   * Fetch and calculate all metrics
   */
  const fetchAndCalculate = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Fetch data in parallel
      const [fetchedTrades, fetchedPositions] = await Promise.all([
        fetchTrades(),
        fetchPositions(),
      ]);

      setTrades(fetchedTrades);
      setPositions(fetchedPositions);

      // Calculate metrics
      const { metrics: calculatedMetrics, summary: calculatedSummary } = calculateMetrics(
        fetchedTrades,
        fetchedPositions
      );

      setMetrics(calculatedMetrics);
      setSummary(calculatedSummary);
      setLastUpdate(Date.now());
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
      console.error('Error fetching financial metrics:', err);
    } finally {
      setIsLoading(false);
    }
  }, [fetchTrades, fetchPositions, calculateMetrics]);

  /**
   * Setup WebSocket connection
   */
  const setupWebSocket = useCallback(() => {
    if (!opts.enableWebSocket) return;

    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/trades`);

      ws.onopen = () => {
        console.log('[useFinancialMetrics] WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'trade_update' || data.type === 'position_update') {
            // Trigger refresh on trade/position update
            fetchAndCalculate();
          }
        } catch (err) {
          console.error('[useFinancialMetrics] WebSocket message error:', err);
        }
      };

      ws.onerror = () => {
        // Silent when backend unavailable
      };

      ws.onclose = () => {
        console.log('[useFinancialMetrics] WebSocket disconnected');
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (opts.enableWebSocket) {
            setupWebSocket();
          }
        }, 5000);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[useFinancialMetrics] Failed to setup WebSocket:', err);
    }
  }, [opts.enableWebSocket, fetchAndCalculate]);

  /**
   * Setup polling interval
   */
  const setupPolling = useCallback(() => {
    if (!opts.autoRefresh || !opts.pollInterval) return;

    pollIntervalRef.current = setInterval(() => {
      fetchAndCalculate();
    }, opts.pollInterval);
  }, [opts.autoRefresh, opts.pollInterval, fetchAndCalculate]);

  /**
   * Cleanup function
   */
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  /**
   * Initialize on mount
   */
  useEffect(() => {
    fetchAndCalculate();
    setupWebSocket();
    setupPolling();

    return cleanup;
  }, [fetchAndCalculate, setupWebSocket, setupPolling, cleanup]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(async () => {
    await fetchAndCalculate();
  }, [fetchAndCalculate]);

  return {
    metrics,
    summary,
    trades,
    positions,
    isLoading,
    error,
    lastUpdate,
    refresh,
    refetch: refresh, // Alias for compatibility
  };
}

/**
 * Hook for real-time metric updates
 */
export function useRealtimeMetrics(options: UseFinancialMetricsOptions = {}) {
  return useFinancialMetrics({
    ...options,
    pollInterval: 5000, // 5 seconds for real-time
    enableWebSocket: true,
    autoRefresh: true,
  });
}

/**
 * Hook for cached metrics (less frequent updates)
 */
export function useCachedMetrics(options: UseFinancialMetricsOptions = {}) {
  return useFinancialMetrics({
    ...options,
    pollInterval: 60000, // 1 minute
    enableWebSocket: false,
    autoRefresh: true,
  });
}
