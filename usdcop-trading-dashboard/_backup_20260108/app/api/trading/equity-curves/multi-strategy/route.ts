import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';
import {
  transformToFrontendEquityCurves,
  BackendEquityCurve,
  getStrategyCode,
} from '@/lib/adapters/backend-adapter';

/**
 * Multi-Strategy Equity Curves Endpoint
 * Returns time-series equity data for all strategies
 * Transforms backend by-strategy format to frontend pivoted columns
 *
 * Backend Response Format (Confirmado):
 * {
 *   start_date: "2025-12-25T00:00:00Z",
 *   end_date: "2025-12-26T00:00:00Z",
 *   resolution: "5m",
 *   curves: [{
 *     strategy_code: "RL_PPO",
 *     strategy_name: "PPO USDCOP V1 (Production)",
 *     data: [{ timestamp, equity_value, return_pct, drawdown_pct }],
 *     summary: { starting_equity, ending_equity, total_return_pct }
 *   }]
 * }
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

// Circuit breaker for Multi-Model API
const multiModelCircuitBreaker = getCircuitBreaker('multi-model-equity', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();
  const { searchParams } = new URL(request.url);
  const hours = searchParams.get('hours') || '24';
  const resolution = searchParams.get('resolution') || '5m';
  const strategies = searchParams.get('strategies'); // e.g., "RL_PPO,ML_LGBM"

  try {
    // Build query params for backend
    const backendParams = new URLSearchParams({
      hours,
      resolution,
    });
    if (strategies) {
      backendParams.append('strategies', strategies);
    }

    const rawData = await multiModelCircuitBreaker.execute(async () => {
      const response = await fetch(
        `${MULTI_MODEL_API}/api/models/equity-curves?${backendParams}`,
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

    // Transform backend by-strategy format to frontend pivoted columns
    const backendCurves: BackendEquityCurve[] = rawData.curves || rawData.data?.curves || [];

    // Normalize backend curves to have strategy_code
    const normalizedCurves = backendCurves.map((curve: any) => ({
      ...curve,
      strategy_code: curve.strategy_code || getStrategyCode(curve.model_id || ''),
    }));

    // Transform to pivoted format expected by EquityCurveChart
    const pivotedData = transformToFrontendEquityCurves(normalizedCurves, 10000);

    const data = {
      equity_curves: pivotedData,
      curves: normalizedCurves, // Also include raw curves for SSE hook compatibility
      metadata: {
        hours: parseInt(hours),
        resolution,
        strategies: normalizedCurves.map((c: any) => c.strategy_code),
      },
      timestamp: new Date().toISOString(),
    };

    const successResponse = createApiResponse(data, 'live');
    successResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(successResponse, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });

  } catch (error: any) {
    console.error('[Multi-Strategy Equity Curves API] Error:', error.message);

    // Generate demo data when backend is unavailable
    const demoData = generateDemoEquityCurves(parseInt(hours));

    const demoResponse = createApiResponse(
      {
        equity_curves: demoData,
        curves: [],
        metadata: {
          hours: parseInt(hours),
          resolution,
          strategies: ['RL_PPO', 'ML_LGBM', 'ML_XGB', 'PORTFOLIO'],
          source: 'demo'
        },
        timestamp: new Date().toISOString()
      },
      'demo'
    );
    demoResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(demoResponse, {
      status: 200,
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });
  }
});

// Generate demo equity curves data
function generateDemoEquityCurves(hours: number = 24): any[] {
  const now = new Date();
  const startTime = new Date(now.getTime() - hours * 60 * 60 * 1000);
  const points: any[] = [];
  const interval = 5 * 60 * 1000; // 5 minutes
  const numPoints = Math.floor((hours * 60) / 5);

  let rl_ppo = 10000;
  let ml_lgbm = 10000;
  let ml_xgb = 10000;

  for (let i = 0; i <= numPoints; i++) {
    const timestamp = new Date(startTime.getTime() + i * interval);

    // Simulate realistic price movements
    const volatility = 0.002;
    const trend = 0.0001;

    rl_ppo *= 1 + (Math.random() - 0.48) * volatility + trend;
    ml_lgbm *= 1 + (Math.random() - 0.49) * volatility + trend * 0.8;
    ml_xgb *= 1 + (Math.random() - 0.47) * volatility + trend * 1.1;

    const portfolio = (rl_ppo + ml_lgbm + ml_xgb) / 3;

    points.push({
      timestamp: timestamp.toISOString(),
      RL_PPO: Math.round(rl_ppo * 100) / 100,
      ML_LGBM: Math.round(ml_lgbm * 100) / 100,
      ML_XGB: Math.round(ml_xgb * 100) / 100,
      PORTFOLIO: Math.round(portfolio * 100) / 100,
      CAPITAL_INICIAL: 10000
    });
  }

  return points;
}
