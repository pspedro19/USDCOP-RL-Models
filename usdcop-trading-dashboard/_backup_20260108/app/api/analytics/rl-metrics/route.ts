// RL Metrics API Route
// Returns RL trading metrics or empty defaults if data unavailable

import { NextRequest, NextResponse } from 'next/server';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { protectApiRoute } from '@/lib/auth/api-auth';

const ANALYTICS_API_URL = process.env.ANALYTICS_API_URL || 'http://localhost:8001';

// Initialize circuit breaker for Analytics API
const circuitBreaker = getCircuitBreaker('analytics-api-rl-metrics', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

// Disable caching for this route - always fetch fresh data
export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Default empty RL metrics response when no data is available
function getEmptyRLMetricsResponse(symbol: string, days: number) {
  return {
    symbol,
    period_days: days,
    data_points: 0,
    metrics: {
      tradesPerEpisode: 0,
      avgHolding: 0,
      actionBalance: {
        buy: 0,
        sell: 0,
        hold: 0,
      },
      spreadCaptured: 0,
      pegRate: 0,
      vwapError: 0,
    },
    timestamp: new Date().toISOString(),
    status: 'no_data',
    message: 'No RL metrics data available. This may occur if no RL models have been trained or no trading data exists.',
  };
}

export async function GET(request: NextRequest) {
  // Protect route with authentication
  const auth = await protectApiRoute(request);
  if (!auth.authenticated) {
    return NextResponse.json(
      { error: auth.error, timestamp: new Date().toISOString() },
      { status: auth.status || 401 }
    );
  }

  // Parse query parameters
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'USDCOP';
  const days = parseInt(searchParams.get('days') || '30', 10);

  try {
    const url = `${ANALYTICS_API_URL}/api/analytics/rl-metrics?symbol=${symbol}&days=${days}`;

    console.log('[RL Metrics API] Fetching from backend:', url);

    const { data, response } = await circuitBreaker.execute(async () => {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000),
      });

      // If response is not ok but not a server error, try to get error details
      if (!response.ok) {
        // For 404 or empty data scenarios, return empty response
        if (response.status === 404) {
          return { data: null, response };
        }

        // Try to parse error response
        try {
          const errorData = await response.json();
          return { data: errorData, response };
        } catch {
          return { data: null, response };
        }
      }

      const data = await response.json();
      return { data, response };
    });

    // If backend returned successfully with data
    if (response.ok && data) {
      return NextResponse.json(data, { status: 200 });
    }

    // If backend returned 404 or no data, return empty metrics
    if (response.status === 404 || !data) {
      console.log('[RL Metrics API] No data from backend, returning empty response');
      return NextResponse.json(getEmptyRLMetricsResponse(symbol, days), { status: 200 });
    }

    // For other errors, still return empty metrics with error info
    console.warn('[RL Metrics API] Backend returned error:', response.status);
    return NextResponse.json(getEmptyRLMetricsResponse(symbol, days), { status: 200 });

  } catch (error) {
    console.error('[RL Metrics API] Error:', error);

    if (error instanceof CircuitOpenError) {
      console.warn('[RL Metrics API] Circuit breaker open, returning empty response');
      return NextResponse.json(
        {
          ...getEmptyRLMetricsResponse(symbol, days),
          status: 'service_unavailable',
          message: 'Analytics API temporarily unavailable. Displaying default values.',
        },
        { status: 200 }
      );
    }

    // For any other error (network, timeout, etc.), return empty metrics
    // This prevents 500 errors and allows the UI to handle gracefully
    console.warn('[RL Metrics API] Returning empty response due to error');
    return NextResponse.json(
      {
        ...getEmptyRLMetricsResponse(symbol, days),
        status: 'error',
        message: 'Unable to fetch RL metrics. Displaying default values.',
      },
      { status: 200 }
    );
  }
}
