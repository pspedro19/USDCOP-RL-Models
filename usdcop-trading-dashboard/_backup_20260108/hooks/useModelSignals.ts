'use client';

/**
 * useModelSignals Hook (SWR Implementation)
 * =========================================
 * Fetches trading signals for the selected model from the API using SWR for caching and revalidation.
 * 
 * Contracts Reference: CT-01 (types/contracts.ts)
 */

import useSWR from 'swr';
import { useSelectedModel } from '@/contexts/ModelContext';
import {
  LatestSignalsResponse,
  StrategySignal,
  APIError,
  MarketStatus,
  TradeSide as OrderSide
} from '@/types/contracts';

// Fallback types for backward compatibility
import type { ModelSignal } from '@/lib/config/models.config';
import { modelRefreshIntervals } from '@/lib/config/models.config';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Fetcher function
const fetcher = async (url: string): Promise<LatestSignalsResponse> => {
  const res = await fetch(url);

  if (!res.ok) {
    let errorMessage = 'Failed to fetch signals';
    try {
      const errorData: APIError = await res.json();
      errorMessage = errorData.message || errorMessage;
    } catch {
      // Ignore JSON parse error
    }
    throw new Error(errorMessage);
  }

  return res.json();
};

interface UseModelSignalsOptions {
  period?: string;
  limit?: number;
  refreshInterval?: number;
  enabled?: boolean;
}

interface UseModelSignalsReturn {
  signals: StrategySignal[];
  latestSignal: StrategySignal | null;
  marketPrice: number | null;
  marketStatus: MarketStatus | null;
  timestamp: string | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  lastUpdated: Date | null;
  mutate: () => Promise<LatestSignalsResponse | undefined>;
  // Legacy compatibility
  isConnected: boolean;
  latency: number;
}

export function useModelSignals(
  options: UseModelSignalsOptions = {}
): UseModelSignalsReturn {
  const {
    refreshInterval = modelRefreshIntervals.signals,
    enabled = true
  } = options;

  const { modelId } = useSelectedModel();

  // We use the consolidated endpoint for all models, then filter client-side if needed
  // In production, we might want to pass modelId as a query param if the API supports filtering
  const shouldFetch = enabled && modelId;
  const endpoint = `${API_BASE}/api/signals/latest`;

  const { data, error, isLoading, mutate } = useSWR<LatestSignalsResponse>(
    shouldFetch ? endpoint : null,
    fetcher,
    {
      refreshInterval,
      revalidateOnFocus: true,
      dedupingInterval: 2000,
      keepPreviousData: true,
    }
  );

  // Filter signals for the selected model
  // Note: The API currently returns "all latest signals". 
  // If we needed historical signals per model, we'd use /api/models/{modelId}/signals
  const allSignals = data?.signals || [];
  const modelSignals = modelId
    ? allSignals.filter(s => s.strategy_code === modelId)
    : allSignals;

  // Derive state
  return {
    signals: modelSignals,
    latestSignal: modelSignals.length > 0 ? modelSignals[0] : null,
    marketPrice: data?.market_price ?? null,
    marketStatus: data?.market_status ?? null,
    timestamp: data?.timestamp ?? null,
    isLoading,
    isError: !!error,
    error: error ?? null,
    lastUpdated: data?.timestamp ? new Date(data.timestamp) : null,
    mutate,
    // Legacy compatibility fields
    isConnected: !error && !isLoading,
    latency: 0, // SWR implementation abstracts this
  };
}

// Helper hook for single strategy
export function useModelSignal(strategyCode: string) {
  const { signals, ...rest } = useModelSignals();
  const signal = signals.find(s => s.strategy_code === strategyCode) ?? null;
  return { signal, ...rest };
}

// ============================================================================
// Legacy Adapters (to minimize breaking changes in existing components)
// ============================================================================

/**
 * Adapter fn to convert StrategySignal (Contract) to ModelSignal (Config)
 * Used for components that haven't been migrated to new types yet
 */
export function adaptSignal(signal: StrategySignal): ModelSignal {
  return {
    id: `${signal.strategy_code}-${signal.timestamp}`,
    modelId: signal.strategy_code,
    timestamp: signal.timestamp,
    signal: signal.signal === 'CLOSE' ? 'HOLD' : signal.signal, // Map CLOSE to HOLD for legacy
    actionRaw: signal.size * (signal.side === OrderSide.SHORT ? -1 : 1), // Approx
    confidence: signal.confidence,
    price: signal.entry_price || 0,
    features: {} // Not available in new contract yet
  };
}

export default useModelSignals;
