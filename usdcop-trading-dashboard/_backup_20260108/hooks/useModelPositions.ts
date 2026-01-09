'use client';

/**
 * useModelPositions Hook
 * =======================
 * Fetches current open positions for the selected model.
 * Includes unrealized P&L calculated on-the-fly.
 *
 * Data source: /api/models/positions/current
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import { modelRefreshIntervals } from '@/lib/config/models.config';
import type { FrontendPosition } from '@/lib/adapters/backend-adapter';

// ============================================================================
// Types
// ============================================================================

interface UseModelPositionsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface UseModelPositionsReturn {
  positions: FrontendPosition[];
  currentPosition: FrontendPosition | null;
  hasOpenPosition: boolean;
  unrealizedPnl: number;
  unrealizedPnlPct: number;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useModelPositions(
  options: UseModelPositionsOptions = {}
): UseModelPositionsReturn {
  const {
    autoRefresh = true,
    refreshInterval = modelRefreshIntervals.signals, // Update frequently like signals
  } = options;

  const { modelId, strategyCode, isLoading: isModelLoading } = useSelectedModel();

  // State
  const [positions, setPositions] = useState<FrontendPosition[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch positions from API
  const fetchPositions = useCallback(async () => {
    // Abort previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      setError(null);

      const params = new URLSearchParams();
      if (strategyCode) {
        params.append('strategy', strategyCode);
      }

      const url = `/api/models/positions/current${params.toString() ? `?${params}` : ''}`;

      const response = await fetch(url, {
        signal: abortControllerRef.current.signal,
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch positions: ${response.status}`);
      }

      const data = await response.json();
      const apiData = data.data || data;

      setPositions(apiData.positions || []);
      setLastUpdated(new Date());
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return;
      }

      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
      console.error('Error fetching positions:', err);
    } finally {
      setIsLoading(false);
    }
  }, [strategyCode]);

  // Initial fetch
  useEffect(() => {
    if (!isModelLoading) {
      setIsLoading(true);
      fetchPositions();
    }
  }, [modelId, isModelLoading, fetchPositions]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    intervalRef.current = setInterval(fetchPositions, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, fetchPositions]);

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

  // Derived values
  const currentPosition = positions.find((p) =>
    p.strategy_code === strategyCode && p.side !== 'flat'
  ) || null;

  const hasOpenPosition = currentPosition !== null && currentPosition.side !== 'flat';

  const unrealizedPnl = positions.reduce(
    (sum, p) => sum + (p.unrealized_pnl || 0),
    0
  );

  const unrealizedPnlPct = positions.reduce(
    (sum, p) => sum + (p.unrealized_pnl_pct || 0),
    0
  );

  return {
    positions,
    currentPosition,
    hasOpenPosition,
    unrealizedPnl,
    unrealizedPnlPct,
    isLoading,
    error,
    lastUpdated,
    refresh: fetchPositions,
  };
}

export default useModelPositions;
