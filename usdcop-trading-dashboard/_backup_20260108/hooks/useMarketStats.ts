/**
 * Market Statistics Hook - Professional Grade
 * ==========================================
 *
 * Hook centralizado para obtener estadísticas de mercado en tiempo real
 * Integración completa con Trading API del backend (puerto 8000)
 *
 * Features:
 * - Carga inicial desde el backend
 * - Actualización automática cada 30 segundos
 * - Integración con real-time price updates
 * - Cálculo automático de tendencias
 * - Manejo robusto de errores con fallbacks
 * - Zero hardcoded values
 */

import { useState, useEffect, useCallback } from 'react';
import { MarketDataService } from '@/lib/services/market-data-service';

/**
 * Calculate liquidity score (0-1) based on multiple market factors
 *
 * @description
 * Liquidity is measured using three key factors:
 * 1. Spread-based liquidity (40% weight): Tighter spreads indicate better liquidity
 *    - USD/COP typical spreads: 0.01% excellent, 0.1% poor
 * 2. Volume-based liquidity (40% weight): Higher trading volume indicates better liquidity
 *    - Normalized: 1M+ COP volume considered excellent
 * 3. Volatility inverse (20% weight): Lower volatility suggests more stable liquidity
 *    - High volatility can indicate liquidity gaps
 *
 * @param params - Market statistics parameters
 * @param params.price - Current market price
 * @param params.spread - Bid-ask spread (high - low)
 * @param params.volume24h - 24-hour trading volume in COP
 * @param params.volatility - Volatility as decimal (0-1)
 *
 * @returns Liquidity score from 0 (illiquid) to 1 (highly liquid)
 */
const calculateLiquidityScore = (params: {
  price: number;
  spread: number;
  volume24h: number;
  volatility: number;
  high24h?: number;
  low24h?: number;
}): number => {
  let score = 0;
  let factors = 0;

  // Factor 1: Spread-based liquidity (40% weight)
  if (params.spread && params.price) {
    const spreadPct = (params.spread / params.price) * 100;
    // Typical USD/COP spread: 0.01% excellent, 0.1% poor
    const spreadScore = Math.max(0, Math.min(1, 1 - (spreadPct / 0.1)));
    score += spreadScore * 0.4;
    factors += 0.4;
  }

  // Factor 2: Volume-based liquidity (40% weight)
  if (params.volume24h) {
    // Normalize: 1M+ COP volume = excellent
    const volumeScore = Math.min(1, params.volume24h / 1000000);
    score += volumeScore * 0.4;
    factors += 0.4;
  }

  // Factor 3: Volatility inverse (20% weight) - lower vol = more stable liquidity
  if (params.volatility !== undefined) {
    const volScore = Math.max(0, Math.min(1, 1 - params.volatility));
    score += volScore * 0.2;
    factors += 0.2;
  }

  return factors > 0 ? score / factors : 0;
};

export interface MarketStats {
  currentPrice: number;
  change24h: number;
  changePercent: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  open24h: number;
  spread: number;
  volatility: number;
  liquidity: number;
  trend: 'up' | 'down' | 'neutral';
  timestamp: Date;
  source: string;
  sessionPnl?: number; // Session P&L from analytics API
}

interface UseMarketStatsReturn {
  stats: MarketStats | null;
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  lastUpdated: Date | null;
}

const DEFAULT_STATS: MarketStats = {
  currentPrice: 0,
  change24h: 0,
  changePercent: 0,
  volume24h: 0,
  high24h: 0,
  low24h: 0,
  open24h: 0,
  spread: 0,
  volatility: 0,
  liquidity: 0,
  trend: 'neutral',
  timestamp: new Date(),
  source: 'initializing'
};

export function useMarketStats(
  symbol: string = 'USDCOP',
  refreshInterval: number = 30000 // 30 seconds
): UseMarketStatsReturn {
  const [stats, setStats] = useState<MarketStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  /**
   * Fetch market statistics from backend
   */
  const fetchStats = useCallback(async () => {
    try {
      console.log(`[useMarketStats] Fetching stats for ${symbol}...`);

      // Get stats from backend (with candlestick fallback)
      const backendStats = await MarketDataService.getSymbolStats(symbol);

      // Calculate additional metrics
      const spread = backendStats.high_24h - backendStats.low_24h;
      const volatility = backendStats.high_24h > 0
        ? (spread / backendStats.high_24h) * 100
        : 0;

      // Determine trend
      let trend: 'up' | 'down' | 'neutral' = 'neutral';
      if (backendStats.change_percent_24h > 0.1) trend = 'up';
      else if (backendStats.change_percent_24h < -0.1) trend = 'down';

      // Calculate liquidity score using multi-factor analysis
      const liquidity = calculateLiquidityScore({
        price: backendStats.price,
        spread,
        volume24h: backendStats.volume_24h,
        volatility: volatility / 100, // Convert percentage to decimal
        high24h: backendStats.high_24h,
        low24h: backendStats.low_24h,
      });

      // Fetch session P&L from analytics API via local proxy
      let sessionPnl = 0;
      try {
        // Use local API proxy which forwards to Analytics API backend
        const pnlResponse = await fetch(`/api/analytics/session-pnl?symbol=${symbol}`);
        if (pnlResponse.ok) {
          const pnlData = await pnlResponse.json();
          sessionPnl = pnlData.session_pnl || 0;
        }
      } catch (pnlError) {
        console.warn('[useMarketStats] Failed to fetch session P&L:', pnlError);
      }

      const newStats: MarketStats = {
        currentPrice: backendStats.price,
        change24h: backendStats.change_24h,
        changePercent: backendStats.change_percent_24h,
        volume24h: backendStats.volume_24h,
        high24h: backendStats.high_24h,
        low24h: backendStats.low_24h,
        open24h: backendStats.open_24h,
        spread,
        volatility,
        liquidity,
        trend,
        timestamp: new Date(backendStats.timestamp),
        source: backendStats.source || 'backend_api',
        sessionPnl
      };

      setStats(newStats);
      setIsConnected(true);
      setError(null);
      setLastUpdated(new Date());

      console.log(`[useMarketStats] Stats updated successfully:`, {
        price: newStats.currentPrice,
        change: newStats.change24h,
        source: newStats.source,
        sessionPnl: newStats.sessionPnl
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch market stats';
      console.error('[useMarketStats] Error fetching stats:', err);

      setError(errorMessage);
      setIsConnected(false);

      // Keep previous stats if available, otherwise use defaults
      if (!stats) {
        setStats(DEFAULT_STATS);
      }
    } finally {
      setIsLoading(false);
    }
  }, [symbol, stats]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(async () => {
    setIsLoading(true);
    await fetchStats();
  }, [fetchStats]);

  /**
   * Initial fetch and periodic refresh
   */
  useEffect(() => {
    // Initial fetch
    fetchStats();

    // Set up periodic refresh
    const intervalId = setInterval(() => {
      fetchStats();
    }, refreshInterval);

    return () => {
      clearInterval(intervalId);
    };
  }, [fetchStats, refreshInterval]);

  /**
   * Listen to API health status
   */
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await MarketDataService.checkAPIHealth();
        setIsConnected(health.status === 'healthy');
      } catch {
        setIsConnected(false);
      }
    };

    // Check health on mount
    checkHealth();

    // Check health every minute
    const healthInterval = setInterval(checkHealth, 60000);

    return () => clearInterval(healthInterval);
  }, []);

  return {
    stats,
    isLoading,
    isConnected,
    error,
    refresh,
    lastUpdated
  };
}

/**
 * Format helpers for display
 */
export const formatMarketStats = {
  price: (price: number): string => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  },

  change: (change: number, withSign: boolean = true): string => {
    const sign = withSign && change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}`;
  },

  percent: (percent: number, withSign: boolean = true): string => {
    const sign = withSign && percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  },

  volume: (volume: number): string => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(2)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toFixed(0);
  },

  timestamp: (date: Date): string => {
    return new Intl.DateTimeFormat('es-CO', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }).format(date);
  }
};
