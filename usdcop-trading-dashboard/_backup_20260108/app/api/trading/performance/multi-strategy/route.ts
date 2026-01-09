import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * Multi-Strategy Performance Endpoint
 * Returns comparative performance metrics for all strategies
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

// Circuit breaker for Multi-Model API
const multiModelCircuitBreaker = getCircuitBreaker('multi-model-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();
  const { searchParams } = new URL(request.url);
  const periodDays = searchParams.get('period_days') || '30';

  try {
    const data = await multiModelCircuitBreaker.execute(async () => {
      const response = await fetch(
        `${MULTI_MODEL_API}/api/models/performance/comparison?period_days=${periodDays}`,
        {
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(10000),
          cache: 'no-store'
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    });

    const successResponse = createApiResponse(data, 'live');
    successResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(successResponse, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });

  } catch (error: any) {
    console.error('[Multi-Strategy Performance API] Error:', error.message);

    // Generate demo performance data when backend is unavailable
    const demoData = generateDemoPerformance();

    const demoResponse = createApiResponse(demoData, 'demo');
    demoResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(demoResponse, {
      status: 200,
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });
  }
});

// Generate demo performance metrics for when backend is unavailable
function generateDemoPerformance() {
  const strategies = [
    {
      strategy_code: 'RL_PPO',
      strategy_name: 'PPO USDCOP V1 (Production)',
      total_return_pct: 8.45,
      sharpe_ratio: 1.82,
      max_drawdown_pct: 4.2,
      win_rate: 58.5,
      total_trades: 156,
      profit_factor: 1.65,
      avg_trade_pnl: 12.50,
      volatility_pct: 8.2,
    },
    {
      strategy_code: 'ML_LGBM',
      strategy_name: 'LightGBM Ensemble V2',
      total_return_pct: 6.82,
      sharpe_ratio: 1.54,
      max_drawdown_pct: 5.1,
      win_rate: 55.2,
      total_trades: 142,
      profit_factor: 1.48,
      avg_trade_pnl: 9.80,
      volatility_pct: 9.5,
    },
    {
      strategy_code: 'ML_XGB',
      strategy_name: 'XGBoost Classifier V1',
      total_return_pct: 7.15,
      sharpe_ratio: 1.68,
      max_drawdown_pct: 4.8,
      win_rate: 56.8,
      total_trades: 138,
      profit_factor: 1.55,
      avg_trade_pnl: 11.20,
      volatility_pct: 8.8,
    },
  ];

  const portfolioTotal = {
    total_return_pct: 7.47,
    sharpe_ratio: 1.92,
    max_drawdown_pct: 3.5,
    win_rate: 56.8,
    total_trades: 436,
    profit_factor: 1.56,
    avg_trade_pnl: 11.17,
    volatility_pct: 7.2,
    correlation_benefit: 0.85,
  };

  return {
    strategies,
    portfolio_total: portfolioTotal,
    period_days: 30,
    timestamp: new Date().toISOString(),
    source: 'demo',
  };
}
