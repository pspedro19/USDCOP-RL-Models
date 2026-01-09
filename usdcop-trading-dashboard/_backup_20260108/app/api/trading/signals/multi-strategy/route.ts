import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';
import {
  transformToModelCardSignal,
  getStrategyCode,
  BackendInference,
} from '@/lib/adapters/backend-adapter';

/**
 * Multi-Strategy Signals Endpoint
 * Proxies to multi-model trading API (port 8006)
 * Transforms backend format (LONG/SHORT, model_id) to frontend format (long/short, strategy_code)
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

  try {
    const rawData = await multiModelCircuitBreaker.execute(async () => {
      const response = await fetch(
        `${MULTI_MODEL_API}/api/models/signals/latest?${searchParams}`,
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

    // Transform backend signals to frontend format
    const backendSignals = rawData.signals || rawData.data?.signals || [];
    const transformedSignals = backendSignals.map((sig: any) => {
      // Convert backend inference format to frontend model card signal format
      const backendInference: BackendInference = {
        inference_id: sig.inference_id || sig.signal_id || Date.now(),
        timestamp_utc: sig.timestamp_utc || sig.timestamp || new Date().toISOString(),
        model_id: sig.model_id || sig.strategy_code || '',
        action_raw: sig.action_raw ?? (sig.confidence || 0.5),
        action_discretized: sig.action_discretized || sig.signal?.toUpperCase() || 'HOLD',
        confidence: sig.confidence || 0,
        price_at_inference: sig.price_at_inference || sig.price || sig.entry_price || 0,
        features_snapshot: sig.features_snapshot,
      };

      return transformToModelCardSignal(backendInference, sig.strategy_name || sig.model_name);
    });

    const data = {
      signals: transformedSignals,
      market_price: rawData.market_price || rawData.current_price || 0,
      timestamp: rawData.timestamp || new Date().toISOString(),
    };

    const successResponse = createApiResponse(data, 'live');
    successResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(successResponse, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });

  } catch (error: any) {
    if (error instanceof CircuitOpenError) {
      const errorResponse = createApiResponse(
        {
          signals: [],
          market_price: 0,
          timestamp: new Date().toISOString()
        },
        'none',
        'Service temporarily unavailable'
      );
      errorResponse.metadata.latency = Date.now() - startTime;

      return NextResponse.json(errorResponse, {
        status: 503,
        headers: { 'Cache-Control': 'no-store' }
      });
    }
    console.error('[Multi-Strategy Signals API] Error:', error);

    const errorResponse = createApiResponse(
      {
        signals: [],
        market_price: 0,
        timestamp: new Date().toISOString()
      },
      'none',
      error.message || 'Failed to fetch signals'
    );
    errorResponse.metadata.latency = Date.now() - startTime;

    return NextResponse.json(errorResponse, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
});
