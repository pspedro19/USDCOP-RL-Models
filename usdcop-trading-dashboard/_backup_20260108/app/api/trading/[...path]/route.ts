/**
 * Trading API Proxy
 * =================
 *
 * Proxies all requests to Trading API backend (http://localhost:8000)
 *
 * IMPORTANT: Returns data directly from Trading API without wrapping,
 * so frontend hooks can parse responses correctly.
 */

import { NextRequest, NextResponse } from 'next/server';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { protectApiRoute } from '@/lib/auth/api-auth';

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8000';

// Initialize circuit breaker for Trading API
const circuitBreaker = getCircuitBreaker('trading-api', {
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
    const url = `${TRADING_API_URL}/${path}${searchParams ? `?${searchParams}` : ''}`;

    console.log('[Trading Proxy] Forwarding GET request to:', url);

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

    // Return the backend data directly - hooks expect the raw response structure
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Trading API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Trading API: ${String(error)}` },
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
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding POST request to:', url);

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
    console.error('[Trading Proxy] Error:', error);

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Trading API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Trading API: ${String(error)}` },
      { status: 500 }
    );
  }
}

export async function PUT(
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
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding PUT request to:', url);

    const { data, response } = await circuitBreaker.execute(async () => {
      const response = await fetch(url, {
        method: 'PUT',
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
    console.error('[Trading Proxy] Error:', error);

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Trading API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Trading API: ${String(error)}` },
      { status: 500 }
    );
  }
}

export async function DELETE(
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
    const url = `${TRADING_API_URL}/${path}`;

    console.log('[Trading Proxy] Forwarding DELETE request to:', url);

    const { data, response } = await circuitBreaker.execute(async () => {
      const response = await fetch(url, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000),
      });

      const data = await response.json();
      return { data, response };
    });

    // Return the backend data directly - hooks expect the raw response structure
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('[Trading Proxy] Error:', error);

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        { detail: 'Trading API circuit breaker is open - service temporarily unavailable' },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { detail: `Failed to fetch from Trading API: ${String(error)}` },
      { status: 500 }
    );
  }
}
