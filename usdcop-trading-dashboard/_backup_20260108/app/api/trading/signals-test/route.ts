import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * Trading Signals Test Endpoint
 *
 * This endpoint is used by SignalAlerts.tsx component to fetch
 * trading signals for the alerts system. It proxies to the
 * Multi-Model Trading API (port 8006) signals endpoint.
 *
 * Returns signals in a format compatible with the SignalAlerts component:
 * {
 *   signals: [{
 *     id: string,
 *     timestamp: string,
 *     type: 'BUY' | 'SELL' | 'HOLD',
 *     confidence: number,
 *     price: number,
 *     riskScore: number,
 *     modelSource: string
 *   }]
 * }
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

// Circuit breaker for Multi-Model API
const multiModelCircuitBreaker = getCircuitBreaker('multi-model-signals-test', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const rawData = await multiModelCircuitBreaker.execute(async () => {
      const response = await fetch(
        `${MULTI_MODEL_API}/api/models/signals/latest`,
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

    // Transform backend signals to SignalAlerts component format
    const backendSignals = rawData.signals || [];
    const transformedSignals = backendSignals.map((sig: any, idx: number) => {
      // Map signal types from backend format to component format
      const signalType = (sig.signal || sig.side || 'hold').toLowerCase();
      let type: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      if (signalType === 'long' || signalType === 'buy') {
        type = 'BUY';
      } else if (signalType === 'short' || signalType === 'sell') {
        type = 'SELL';
      }

      // Calculate risk score from confidence (inverse relationship)
      const confidence = (sig.confidence ?? 0.5) * 100;
      const riskScore = Math.max(1, Math.min(10, 10 - (confidence / 10)));

      return {
        id: sig.strategy_code || `signal_${idx}`,
        timestamp: sig.timestamp || new Date().toISOString(),
        type,
        confidence,
        price: sig.entry_price || rawData.market_price || 0,
        stopLoss: sig.stop_loss ?? null,
        takeProfit: sig.take_profit ?? null,
        riskScore,
        modelSource: sig.strategy_name || sig.strategy_code || 'Unknown',
        reasoning: sig.reasoning ? [sig.reasoning] : ['Signal from ML model'],
        expectedReturn: 0,
        timeHorizon: '15-30 min',
        latency: 0
      };
    });

    const data = {
      signals: transformedSignals,
      market_price: rawData.market_price || 0,
      market_status: rawData.market_status || 'unknown',
      timestamp: rawData.timestamp || new Date().toISOString(),
      source: 'Multi-Strategy API'
    };

    const successResponse = createApiResponse(data, 'live');
    successResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(successResponse, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });

  } catch (error: any) {
    const latency = Date.now() - startTime;

    if (error instanceof CircuitOpenError) {
      console.warn('[Signals-Test API] Circuit breaker OPEN - service unavailable');
      const errorResponse = createApiResponse(
        {
          signals: [],
          market_price: 0,
          timestamp: new Date().toISOString()
        },
        'none',
        'Service temporarily unavailable'
      );
      errorResponse.metadata.latency = latency;

      return NextResponse.json(errorResponse, {
        status: 503,
        headers: { 'Cache-Control': 'no-store' }
      });
    }

    console.error('[Signals-Test API] Error:', error);
    const errorResponse = createApiResponse(
      {
        signals: [],
        market_price: 0,
        timestamp: new Date().toISOString()
      },
      'none',
      error.message || 'Failed to fetch signals'
    );
    errorResponse.metadata.latency = latency;

    return NextResponse.json(errorResponse, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
});
