/**
 * Custom hooks for Trading Analytics API
 * All data is fetched from the backend analytics service via local API proxy
 */

import useSWR from 'swr';

// Use local API proxy which forwards requests to Analytics API backend
const ANALYTICS_API_URL = '/api/analytics';

// Fetcher function for SWR
const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Failed to fetch');
  }
  return res.json();
};

// ==========================================
// RL Metrics Hook
// ==========================================

export interface RLMetrics {
  tradesPerEpisode: number;
  avgHolding: number;
  actionBalance: {
    buy: number;
    sell: number;
    hold: number;
  };
  spreadCaptured: number;
  pegRate: number;
  vwapError: number;
}

export interface RLMetricsResponse {
  symbol: string;
  period_days: number;
  data_points: number;
  metrics: RLMetrics;
  timestamp: string;
}

export function useRLMetrics(symbol: string = 'USDCOP', days: number = 30) {
  const { data, error, isLoading } = useSWR<RLMetricsResponse>(
    `${ANALYTICS_API_URL}/rl-metrics?symbol=${symbol}&days=${days}`,
    fetcher,
    {
      refreshInterval: 60000, // Refresh every minute
      revalidateOnFocus: false,
    }
  );

  return {
    metrics: data?.metrics,
    isLoading,
    isError: error,
    raw: data,
  };
}

// ==========================================
// Performance KPIs Hook
// ==========================================

export interface PerformanceKPIs {
  sortinoRatio: number;
  calmarRatio: number;
  sharpeRatio: number;
  maxDrawdown: number;
  currentDrawdown: number;
  profitFactor: number;
  cagr: number;
  volatility: number;
  benchmarkSpread: number;
}

export interface PerformanceKPIsResponse {
  symbol: string;
  period_days: number;
  data_points: number;
  kpis: PerformanceKPIs;
  timestamp: string;
}

export function usePerformanceKPIs(symbol: string = 'USDCOP', days: number = 90) {
  const { data, error, isLoading, mutate } = useSWR<PerformanceKPIsResponse>(
    `${ANALYTICS_API_URL}/performance-kpis?symbol=${symbol}&days=${days}`,
    fetcher,
    {
      refreshInterval: 120000, // Refresh every 2 minutes
      revalidateOnFocus: false,
    }
  );

  return {
    kpis: data?.kpis,
    isLoading,
    isError: error,
    error: error?.message || null,
    refetch: mutate,
    raw: data,
  };
}

// ==========================================
// Production Gates Hook
// ==========================================

export interface ProductionGate {
  title: string;
  value: number;
  threshold: number;
  operator: string;
  status: boolean;
  description: string;
}

export interface ProductionGatesResponse {
  symbol: string;
  period_days: number;
  production_ready: boolean;
  passing_gates: number;
  total_gates: number;
  gates: ProductionGate[];
  timestamp: string;
}

export function useProductionGates(symbol: string = 'USDCOP', days: number = 90) {
  const { data, error, isLoading, mutate } = useSWR<ProductionGatesResponse>(
    `${ANALYTICS_API_URL}/production-gates?symbol=${symbol}&days=${days}`,
    fetcher,
    {
      refreshInterval: 120000, // Refresh every 2 minutes
      revalidateOnFocus: false,
    }
  );

  return {
    gates: data?.gates || [],
    productionReady: data?.production_ready || false,
    passingGates: data?.passing_gates || 0,
    totalGates: data?.total_gates || 0,
    isLoading,
    isError: error,
    error: error?.message || null,
    refetch: mutate,
    raw: data,
  };
}

// ==========================================
// Risk Metrics Hook
// ==========================================

export interface RiskMetrics {
  portfolioValue: number;
  grossExposure: number;
  netExposure: number;
  leverage: number;
  portfolioVaR95: number;
  portfolioVaR99: number;
  portfolioVaR95Percent: number;
  expectedShortfall95: number;
  portfolioVolatility: number;
  currentDrawdown: number;
  maximumDrawdown: number;
  liquidityScore: number;
  timeToLiquidate: number;
  bestCaseScenario: number;
  worstCaseScenario: number;
  stressTestResults: {
    [key: string]: number;
  };
}

export interface RiskMetricsResponse {
  symbol: string;
  period_days: number;
  portfolio_value: number;
  data_points: number;
  risk_metrics: RiskMetrics;
  timestamp: string;
}

export function useRiskMetrics(
  symbol: string = 'USDCOP',
  portfolioValue: number = 10000000,
  days: number = 30
) {
  const { data, error, isLoading } = useSWR<RiskMetricsResponse>(
    `${ANALYTICS_API_URL}/risk-metrics?symbol=${symbol}&portfolio_value=${portfolioValue}&days=${days}`,
    fetcher,
    {
      refreshInterval: 60000, // Refresh every minute
      revalidateOnFocus: false,
    }
  );

  return {
    metrics: data?.risk_metrics,
    isLoading,
    isError: error,
    raw: data,
  };
}

// ==========================================
// Session P&L Hook
// ==========================================

export interface SessionPnL {
  symbol: string;
  session_date: string;
  session_open?: number;
  session_close?: number;
  session_pnl: number;
  session_pnl_percent: number;
  has_data: boolean;
  timestamp: string;
}

export function useSessionPnL(symbol: string = 'USDCOP', sessionDate?: string) {
  const dateParam = sessionDate ? `&session_date=${sessionDate}` : '';

  const { data, error, isLoading } = useSWR<SessionPnL>(
    `${ANALYTICS_API_URL}/session-pnl?symbol=${symbol}${dateParam}`,
    fetcher,
    {
      refreshInterval: 30000, // Refresh every 30 seconds
      revalidateOnFocus: false,
    }
  );

  return {
    pnl: data?.session_pnl || 0,
    pnlPercent: data?.session_pnl_percent || 0,
    hasData: data?.has_data || false,
    isLoading,
    isError: error,
    raw: data,
  };
}

// ==========================================
// Combined Analytics Hook (All metrics at once)
// ==========================================

export function useAllAnalytics(symbol: string = 'USDCOP') {
  const rlMetrics = useRLMetrics(symbol, 30);
  const performanceKPIs = usePerformanceKPIs(symbol, 90);
  const productionGates = useProductionGates(symbol, 90);
  const riskMetrics = useRiskMetrics(symbol, 10000000, 30);
  const sessionPnL = useSessionPnL(symbol);

  return {
    rlMetrics,
    performanceKPIs,
    productionGates,
    riskMetrics,
    sessionPnL,
    isLoading:
      rlMetrics.isLoading ||
      performanceKPIs.isLoading ||
      productionGates.isLoading ||
      riskMetrics.isLoading ||
      sessionPnL.isLoading,
  };
}
