// Analytics API Proxy
// Proxies all requests to Analytics API backend (http://localhost:8001)

import { NextRequest, NextResponse } from 'next/server';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { protectApiRoute } from '@/lib/auth/api-auth';

const ANALYTICS_API_URL = process.env.ANALYTICS_API_URL || 'http://localhost:8001';

// Fallback data structures for endpoints that may not exist on the backend
// This prevents 404 errors from breaking the UI
const FALLBACK_RESPONSES: Record<string, object> = {
  'execution-metrics': {
    metrics: {
      vwap: 0,
      effective_spread_bps: 0,
      avg_slippage_bps: 0,
      turnover_cost_bps: 0,
      fill_ratio_pct: 0
    },
    data_available: false,
    message: 'Endpoint not available on backend - returning default values'
  },
  'order-flow': {
    data_available: false,
    order_flow: {
      buy_percent: 0,
      sell_percent: 0,
      imbalance: 0
    },
    message: 'Endpoint not available on backend - returning default values'
  },
  'rl-metrics': {
    metrics: {
      vwapError: 0,
      pegRate: 0
    },
    data_available: false,
    message: 'Endpoint not available on backend - returning default values'
  }
};

// Get fallback response for a given path
function getFallbackResponse(path: string): object | null {
  // Check exact match first
  if (FALLBACK_RESPONSES[path]) {
    return FALLBACK_RESPONSES[path];
  }
  // Check if path starts with any known fallback key
  for (const key of Object.keys(FALLBACK_RESPONSES)) {
    if (path.startsWith(key)) {
      return FALLBACK_RESPONSES[key];
    }
  }
  return null;
}

// Initialize circuit breaker for Analytics API
const circuitBreaker = getCircuitBreaker('analytics-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

// Disable caching for this route - always fetch fresh data
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  // Protect route with authentication
  const auth = await protectApiRoute(request);
  if (!auth.authenticated) {
    return NextResponse.json(
      { error: auth.error, timestamp: new Date().toISOString() },
      { status: auth.status || 401 }
    );
  }

  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    // Analytics API endpoints have /api/analytics/ prefix
    const url = `${ANALYTICS_API_URL}/api/analytics/${path}${searchParams ? `?${searchParams}` : ''}`;

    console.log('[Analytics Proxy] Forwarding GET request to:', url);

    const { data, response } = await circuitBreaker.execute(async () => {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000),
      });

      const data = await response.json();
      return { data, response };
    });

    // If backend returns 404, check for fallback response
    if (response.status === 404) {
      const fallback = getFallbackResponse(path);
      if (fallback) {
        console.log('[Analytics Proxy] Backend returned 404, using fallback for:', path);
        return NextResponse.json(fallback, { status: 200 });
      }
    }

    // Return the backend data directly - hooks expect the raw response structure
    // Don't wrap it to avoid breaking the existing hook contracts
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Analytics Proxy] Error:', error);

    // Try to get the path from resolvedParams for fallback lookup
    try {
      const resolvedParams = await params;
      const path = resolvedParams.path.join('/');
      const fallback = getFallbackResponse(path);
      if (fallback) {
        console.log('[Analytics Proxy] Error occurred, using fallback for:', path);
        return NextResponse.json(fallback, { status: 200 });
      }
    } catch {
      // Ignore errors while getting fallback
    }

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Analytics API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Analytics API: ${String(error)}` },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  // Protect route with authentication
  const auth = await protectApiRoute(request);
  if (!auth.authenticated) {
    return NextResponse.json(
      { error: auth.error, timestamp: new Date().toISOString() },
      { status: auth.status || 401 }
    );
  }

  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/');
    const body = await request.json();
    // Analytics API endpoints have /api/analytics/ prefix
    const url = `${ANALYTICS_API_URL}/api/analytics/${path}`;

    console.log('[Analytics Proxy] Forwarding POST request to:', url);

    const { data, response } = await circuitBreaker.execute(async () => {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000),
      });

      const data = await response.json();
      return { data, response };
    });

    // Return the backend data directly - hooks expect the raw response structure
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Analytics Proxy] Error:', error);

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Analytics API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Analytics API: ${String(error)}` },
      { status: 500 }
    );
  }
}
