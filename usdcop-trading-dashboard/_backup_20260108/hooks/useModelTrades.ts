'use client';

/**
 * useModelTrades Hook
 * ====================
 * Fetches trade history for the selected model from the API.
 * Includes summary statistics (win rate, streak, P&L).
 *
 * Data source: /api/models/{modelId}/trades
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import type {
  ModelTrade,
  TradesSummary,
  SignalType,
  TradeStatus,
  PeriodOption,
} from '@/lib/config/models.config';
import { modelRefreshIntervals } from '@/lib/config/models.config';

// ============================================================================
// Types
// ============================================================================

interface UseModelTradesOptions {
  period?: PeriodOption;
  status?: TradeStatus | 'all';
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface UseModelTradesReturn {
  trades: ModelTrade[];
  summary: TradesSummary;
  openTrade: ModelTrade | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

// ============================================================================
// Default Summary
// ============================================================================

const defaultSummary: TradesSummary = {
  total: 0,
  wins: 0,
  losses: 0,
  holds: 0,
  winRate: 0,
  streak: 0,
  pnlTotal: 0,
  pnlPct: 0,
  avgDuration: null,
  bestTrade: null,
  worstTrade: null,
};

// ============================================================================
// Hook Implementation
// ============================================================================

export function useModelTrades(
  options: UseModelTradesOptions = {}
): UseModelTradesReturn {
  const {
    period = 'today',
    status = 'all',
    limit = 50,
    autoRefresh = true,
    refreshInterval = modelRefreshIntervals.trades,
  } = options;

  const { modelId, isLoading: isModelLoading } = useSelectedModel();

  // State
  const [trades, setTrades] = useState<ModelTrade[]>([]);
  const [summary, setSummary] = useState<TradesSummary>(defaultSummary);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch trades from API
  const fetchTrades = useCallback(async () => {
    if (!modelId) {
      setTrades([]);
      setSummary(defaultSummary);
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

      const params = new URLSearchParams({
        period,
        limit: limit.toString(),
      });

      if (status !== 'all') {
        params.append('status', status);
      }

      const response = await fetch(
        `/api/models/${modelId}/trades?${params}`,
        {
          signal: abortControllerRef.current.signal,
          headers: { 'Content-Type': 'application/json' },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch trades: ${response.status}`);
      }

      const data = await response.json();

      const normalizedTrades = (data.trades || []).map(normalizeTrade);
      const calculatedSummary = data.summary
        ? normalizeSummary(data.summary)
        : calculateSummary(normalizedTrades);

      setTrades(normalizedTrades);
      setSummary(calculatedSummary);
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return;
      }

      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      console.error('Error fetching trades:', err);
    } finally {
      setIsLoading(false);
    }
  }, [modelId, period, status, limit]);

  // Initial fetch when model changes
  useEffect(() => {
    if (!isModelLoading) {
      setIsLoading(true);
      fetchTrades();
    }
  }, [modelId, isModelLoading, period, status, fetchTrades]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh || !modelId) return;

    intervalRef.current = setInterval(fetchTrades, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, modelId, fetchTrades]);

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

  // Open trade (if any)
  const openTrade = useMemo(() => {
    return trades.find((t) => t.status === 'OPEN') || null;
  }, [trades]);

  return {
    trades,
    summary,
    openTrade,
    isLoading,
    error,
    lastUpdated,
    refresh: fetchTrades,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Normalize trade from API
 */
function normalizeTrade(trade: any): ModelTrade {
  return {
    tradeId: trade.trade_id || trade.tradeId || 0,
    modelId: trade.model_id || trade.modelId || '',
    openTime: trade.open_time || trade.openTime || '',
    closeTime: trade.close_time || trade.closeTime || null,
    signal: normalizeSignalType(trade.signal),
    entryPrice: trade.entry_price ?? trade.entryPrice ?? 0,
    exitPrice: trade.exit_price ?? trade.exitPrice ?? null,
    pnl: trade.pnl ?? null,
    pnlPct: trade.pnl_pct ?? trade.pnlPct ?? null,
    durationMinutes: trade.duration_minutes ?? trade.durationMinutes ?? null,
    status: normalizeTradeStatus(trade.status),
    confidence: trade.confidence ?? 0,
  };
}

/**
 * Normalize signal type
 */
function normalizeSignalType(type: string | undefined): SignalType {
  if (!type) return 'HOLD';
  const normalized = type.toUpperCase();
  if (normalized === 'LONG' || normalized === 'BUY') return 'LONG';
  if (normalized === 'SHORT' || normalized === 'SELL') return 'SHORT';
  return 'HOLD';
}

/**
 * Normalize trade status
 */
function normalizeTradeStatus(status: string | undefined): TradeStatus {
  if (!status) return 'CLOSED';
  const normalized = status.toUpperCase();
  if (normalized === 'OPEN') return 'OPEN';
  if (normalized === 'CANCELLED') return 'CANCELLED';
  return 'CLOSED';
}

/**
 * Normalize summary from API
 */
function normalizeSummary(summary: any): TradesSummary {
  return {
    total: summary.total ?? 0,
    wins: summary.wins ?? 0,
    losses: summary.losses ?? 0,
    holds: summary.holds ?? 0,
    winRate: summary.win_rate ?? summary.winRate ?? 0,
    streak: summary.streak ?? 0,
    pnlTotal: summary.pnl_total ?? summary.pnlTotal ?? 0,
    pnlPct: summary.pnl_pct ?? summary.pnlPct ?? 0,
    avgDuration: summary.avg_duration ?? summary.avgDuration ?? null,
    bestTrade: summary.best_trade ?? summary.bestTrade ?? null,
    worstTrade: summary.worst_trade ?? summary.worstTrade ?? null,
  };
}

/**
 * Calculate summary from trades (fallback if API doesn't provide)
 */
function calculateSummary(trades: ModelTrade[]): TradesSummary {
  if (trades.length === 0) {
    return defaultSummary;
  }

  const closedTrades = trades.filter((t) => t.status === 'CLOSED');
  const wins = closedTrades.filter((t) => (t.pnl ?? 0) > 0);
  const losses = closedTrades.filter((t) => (t.pnl ?? 0) < 0);
  const holds = trades.filter((t) => t.signal === 'HOLD');

  const pnls = closedTrades.map((t) => t.pnl ?? 0);
  const pnlTotal = pnls.reduce((sum, pnl) => sum + pnl, 0);

  const durations = closedTrades
    .map((t) => t.durationMinutes)
    .filter((d): d is number => d !== null);

  // Calculate streak (consecutive wins or losses from most recent)
  let streak = 0;
  for (const trade of closedTrades) {
    const pnl = trade.pnl ?? 0;
    if (streak === 0) {
      streak = pnl > 0 ? 1 : pnl < 0 ? -1 : 0;
    } else if (streak > 0 && pnl > 0) {
      streak++;
    } else if (streak < 0 && pnl < 0) {
      streak--;
    } else {
      break;
    }
  }

  return {
    total: trades.length,
    wins: wins.length,
    losses: losses.length,
    holds: holds.length,
    winRate: closedTrades.length > 0
      ? Math.round((wins.length / closedTrades.length) * 100)
      : 0,
    streak,
    pnlTotal,
    pnlPct: 0, // Would need initial capital to calculate
    avgDuration: durations.length > 0
      ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length)
      : null,
    bestTrade: pnls.length > 0 ? Math.max(...pnls) : null,
    worstTrade: pnls.length > 0 ? Math.min(...pnls) : null,
  };
}

export default useModelTrades;
