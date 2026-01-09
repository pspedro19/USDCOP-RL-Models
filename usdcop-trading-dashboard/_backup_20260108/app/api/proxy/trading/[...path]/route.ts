/**
 * Trading API Proxy
 * =================
 *
 * Proxy requests to the Trading API to overcome CORS and firewall issues
 * Routes all /api/proxy/trading/* requests to the internal Trading API
 *
 * IMPORTANT: This proxy returns data directly from the Trading API without
 * additional wrapping, so frontend services can parse responses correctly.
 */

import { NextRequest, NextResponse } from 'next/server'
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker'
import { protectApiRoute } from '@/lib/auth/api-auth'

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8000'

// Initialize circuit breaker for Trading Proxy with more lenient settings
// to avoid blocking during temporary backend slowdowns
const circuitBreaker = getCircuitBreaker('trading-proxy', {
  failureThreshold: 10,  // Increased from 3 to be more tolerant
  resetTimeout: 15000,   // Reduced from 30s to recover faster
})

// Mock responses for when backend is unavailable
const MOCK_RESPONSES: Record<string, unknown> = {
  'health': {
    status: 'healthy',
    database: 'disconnected',
    total_records: 0,
    latest_data: '',
    timestamp: new Date().toISOString(),
    source: 'mock-data'
  },
  'latest': {
    symbol: 'USDCOP',
    price: 4288.00,
    timestamp: new Date().toISOString(),
    volume: 125000,
    bid: 4287.50,
    ask: 4288.50,
    source: 'mock-data'
  },
  'stats': {
    symbol: 'USDCOP',
    current_price: 4288.00,
    change_24h: 12.50,
    change_pct_24h: 0.29,
    high_24h: 4302.75,
    low_24h: 4275.25,
    volume_24h: 2850000,
    vwap_24h: 4285.50,
    timestamp: new Date().toISOString(),
    source: 'mock-data'
  },
  'candlesticks': {
    data: [
      { time: Date.now() - 300000, open: 4285.5, high: 4292.75, low: 4280.25, close: 4288.0, volume: 25000 },
      { time: Date.now() - 600000, open: 4280.0, high: 4286.0, low: 4278.5, close: 4285.5, volume: 22000 },
      { time: Date.now() - 900000, open: 4278.25, high: 4282.0, low: 4275.0, close: 4280.0, volume: 18000 },
    ],
    symbol: 'USDCOP',
    timeframe: '5m',
    count: 3
  },
  'signals': { signals: [], source: 'mock-data' },
  'positions': { positions: [], source: 'mock-data' },
  'trades': { trades: [], source: 'mock-data' },
};

// Fallback response for health endpoint when backend fails
const HEALTH_FALLBACK_RESPONSE = {
  status: 'degraded',
  database: 'disconnected',
  total_records: 0,
  latest_data: '',
  timestamp: new Date().toISOString(),
  source: 'fallback',
  error: 'Backend unavailable'
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  // Resolve params early (once) to avoid multiple awaits
  const resolvedParams = await params;
  const path = resolvedParams.path.join('/');

  // DEV MODE: Return mock data without backend
  if (process.env.NEXT_PUBLIC_SKIP_BACKEND === 'true') {
    // Find matching mock response
    for (const [key, value] of Object.entries(MOCK_RESPONSES)) {
      if (path.includes(key)) {
        return NextResponse.json(value);
      }
    }

    // Default mock response
    return NextResponse.json({ data: [], source: 'mock-data', path });
  }

  // Health endpoints should bypass authentication for monitoring purposes
  const isHealthEndpoint = path === 'health' || path.startsWith('health/');

  if (!isHealthEndpoint) {
    // Protect route with authentication for non-health endpoints
    const auth = await protectApiRoute(request);
    if (!auth.authenticated) {
      return NextResponse.json(
        { error: auth.error, timestamp: new Date().toISOString() },
        { status: auth.status || 401 }
      );
    }
  }

  try {
    const url = new URL(request.url)
    const queryString = url.search

    const targetUrl = `${TRADING_API_URL}/api/${path}${queryString}`

    console.log(`[Proxy] GET ${targetUrl}`)

    // Determine timeout based on endpoint type
    // Health checks need quick response, candlesticks need longer timeout
    const isCandlestickRequest = path.includes('candlesticks')
    const timeout = isHealthEndpoint ? 10000 : (isCandlestickRequest ? 60000 : 15000)

    const { response, data } = await circuitBreaker.execute(async () => {
      const response = await fetch(targetUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Connection': 'keep-alive',
        },
        signal: AbortSignal.timeout(timeout),
        // Next.js specific: disable internal caching to avoid stale data
        cache: 'no-store',
      })

      if (!response.ok) {
        console.error(`[Proxy] Error: ${response.status} ${response.statusText}`)

        // Handle market closed (425) by returning fallback data
        if (response.status === 425 && path.includes('latest')) {
          console.log('[Proxy] Market closed, providing fallback historical data')
          const fallbackResponse = await fetch(
            `${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=1`,
            {
              signal: AbortSignal.timeout(15000),
              cache: 'no-store',
            }
          )

          if (fallbackResponse.ok) {
            const fallbackData = await fallbackResponse.json()
            if (fallbackData.data && fallbackData.data.length > 0) {
              const latestCandle = fallbackData.data[fallbackData.data.length - 1]
              const fallbackResult = {
                symbol: 'USDCOP',
                price: latestCandle.close,
                timestamp: new Date(latestCandle.time).toISOString(),
                volume: latestCandle.volume,
                bid: latestCandle.close - 0.5,
                ask: latestCandle.close + 0.5,
                source: 'database_historical_real'
              }
              return { response: fallbackResponse, data: fallbackResult }
            }
          }
        }

        throw new Error(`Trading API error: ${response.status}`)
      }

      const data = await response.json()
      return { response, data }
    })

    // Return data directly from Trading API without additional wrapping
    // This maintains compatibility with MarketDataFetcher expectations
    // Add cache headers for candlestick data (can be cached briefly)
    const headers: Record<string, string> = {}
    if (path.includes('candlesticks')) {
      headers['Cache-Control'] = 'private, max-age=5'  // Cache for 5 seconds
    }

    return NextResponse.json(data, {
      status: response.status,
      headers,
    })

  } catch (error) {
    console.error('[Proxy] Error:', error)

    // For health endpoint, return fallback response instead of error
    // This allows the frontend to gracefully handle backend unavailability
    if (path === 'health') {
      console.log('[Proxy] Health endpoint failed, returning fallback response')
      return NextResponse.json({
        ...HEALTH_FALLBACK_RESPONSE,
        timestamp: new Date().toISOString(),
        error: String(error)
      }, { status: 200 })
    }

    // Find matching mock response to use as fallback
    for (const [key, value] of Object.entries(MOCK_RESPONSES)) {
      if (path.includes(key)) {
        console.log(`[Proxy] Returning mock data for ${path} due to error`);
        return NextResponse.json({
          ...(typeof value === 'object' ? value : { data: value }),
          timestamp: new Date().toISOString(),
          source: 'mock-data-fallback',
        });
      }
    }

    if (error instanceof CircuitOpenError) {
      // Return default mock response instead of error
      return NextResponse.json({
        data: [],
        timestamp: new Date().toISOString(),
        source: 'mock-data-circuit-breaker',
        message: 'Service temporarily unavailable - using fallback data'
      })
    }

    // Return empty data instead of error for unknown paths
    console.log(`[Proxy] No mock data available for ${path}, returning empty response`);
    return NextResponse.json({
      data: [],
      timestamp: new Date().toISOString(),
      source: 'mock-data-fallback',
      path
    })
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  // Resolve params early (once) to avoid multiple awaits
  const resolvedParams = await params;
  const path = resolvedParams.path.join('/');

  // DEV MODE: Return mock success response without backend
  if (process.env.NEXT_PUBLIC_SKIP_BACKEND === 'true') {
    return NextResponse.json({
      success: true,
      message: 'Mock operation completed',
      path,
      source: 'mock-data'
    });
  }

  // Protect route with authentication
  const auth = await protectApiRoute(request);
  if (!auth.authenticated) {
    return NextResponse.json(
      { error: auth.error, timestamp: new Date().toISOString() },
      { status: auth.status || 401 }
    );
  }

  try {
    const body = await request.text()

    const targetUrl = `${TRADING_API_URL}/api/${path}`

    console.log(`[Proxy] POST ${targetUrl}`)

    const { response, data } = await circuitBreaker.execute(async () => {
      const response = await fetch(targetUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Connection': 'keep-alive',
        },
        body,
        signal: AbortSignal.timeout(15000),  // Increased from 10s
        cache: 'no-store',
      })

      if (!response.ok) {
        console.error(`[Proxy] Error: ${response.status} ${response.statusText}`)
        throw new Error(`Trading API error: ${response.status}`)
      }

      const data = await response.json()
      return { response, data }
    })

    // Return data directly from Trading API without additional wrapping
    return NextResponse.json(data, { status: response.status })

  } catch (error) {
    console.error('[Proxy] Error:', error)

    if (error instanceof CircuitOpenError) {
      return NextResponse.json(
        {
          error: 'Trading proxy circuit breaker is open',
          message: 'Service temporarily unavailable due to repeated failures'
        },
        { status: 503 }
      )
    }

    // Check if it's a timeout error and provide clearer message
    const errorMessage = String(error)
    const isTimeout = errorMessage.includes('timeout') || errorMessage.includes('abort')

    return NextResponse.json(
      {
        error: isTimeout ? 'Request timeout' : 'Proxy connection failed',
        message: errorMessage,
        hint: isTimeout ? 'The backend took too long to respond. Please try again.' : undefined
      },
      { status: isTimeout ? 504 : 500 }
    )
  }
}