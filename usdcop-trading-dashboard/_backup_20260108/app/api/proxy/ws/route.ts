/**
 * WebSocket Proxy for Real-time Data
 * ==================================
 *
 * Proxies WebSocket connections to the internal WebSocket service
 * Converts HTTP requests to WebSocket connections for real-time data
 *
 * Returns data directly in the format expected by MarketDataFetcher:
 * { symbol, price, timestamp, volume, bid, ask, source }
 *
 * Features:
 * - Retry logic for transient failures (ECONNRESET, ECONNREFUSED)
 * - Graceful degradation to cached/fallback data
 * - Proper error categorization (connection vs server errors)
 */

import { NextRequest, NextResponse } from 'next/server'
import { withAuth } from '@/lib/auth/api-auth'

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8006'

// Retry configuration
const MAX_RETRIES = 3
const INITIAL_RETRY_DELAY_MS = 100
const REQUEST_TIMEOUT_MS = 5000

// Cache for fallback data when connection fails
let lastSuccessfulData: {
  price: number
  timestamp: string
  volume: number
  bid: number
  ask: number
} | null = null

/**
 * Check if an error is a connection reset or similar transient error
 */
function isTransientError(error: unknown): boolean {
  if (error instanceof Error) {
    const errorCode = (error as NodeJS.ErrnoException).code
    const transientCodes = ['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN']
    if (errorCode && transientCodes.includes(errorCode)) {
      return true
    }
    // Also check error message for common connection issues
    const message = error.message.toLowerCase()
    if (message.includes('fetch failed') ||
      message.includes('network') ||
      message.includes('connection') ||
      message.includes('socket')) {
      return true
    }
  }
  return false
}

/**
 * Fetch with timeout to prevent hanging connections
 */
async function fetchWithTimeout(url: string, timeoutMs: number = REQUEST_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(url, { signal: controller.signal })
    return response
  } finally {
    clearTimeout(timeoutId)
  }
}

/**
 * Fetch with retry logic for transient errors
 */
async function fetchWithRetry(
  url: string,
  maxRetries: number = MAX_RETRIES,
  initialDelay: number = INITIAL_RETRY_DELAY_MS
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetchWithTimeout(url)
      return response
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))

      // Only retry on transient errors
      if (!isTransientError(error) || attempt === maxRetries) {
        throw lastError
      }

      // Exponential backoff
      const delay = initialDelay * Math.pow(2, attempt)
      console.log(`[WebSocket Proxy] Retry ${attempt + 1}/${maxRetries} after ${delay}ms - ${lastError.message}`)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  throw lastError || new Error('Fetch failed after retries')
}

/**
 * Get fallback response when all data sources fail
 */
function getFallbackResponse(reason: string): NextResponse {
  // If we have cached data, return it with a stale indicator
  if (lastSuccessfulData) {
    console.log('[WebSocket Proxy] Using cached fallback data')
    return NextResponse.json({
      type: 'price_update',
      symbol: 'USDCOP',
      price: lastSuccessfulData.price,
      timestamp: lastSuccessfulData.timestamp,
      volume: lastSuccessfulData.volume,
      bid: lastSuccessfulData.bid,
      ask: lastSuccessfulData.ask,
      source: 'proxy_cached_fallback',
      stale: true,
      cacheAge: Date.now() - new Date(lastSuccessfulData.timestamp).getTime()
    })
  }

  // No cached data available - return a clear error (not 503 to avoid triggering error pages)
  return NextResponse.json({
    type: 'error',
    error: 'connection_unavailable',
    message: reason,
    retryable: true,
    retryAfterMs: 5000,
    symbol: 'USDCOP',
    timestamp: new Date().toISOString()
  }, {
    status: 200, // Return 200 with error payload to prevent 503 page renders
    headers: {
      'X-Data-Status': 'unavailable',
      'X-Retry-After': '5'
    }
  })
}

export const GET = withAuth(async (request, { user }) => {
  try {
    // First try the real-time endpoint with retry logic
    let response: Response
    try {
      response = await fetchWithRetry(`${TRADING_API_URL}/api/latest/USDCOP`)
    } catch (fetchError) {
      console.error('[WebSocket Proxy] Real-time fetch failed:', fetchError)

      // Try historical endpoint as fallback
      try {
        response = await fetchWithRetry(`${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=1`)

        if (!response.ok) {
          return getFallbackResponse('Historical data endpoint returned error')
        }

        const candleData = await response.json()

        if (candleData.data && candleData.data.length > 0) {
          const latestCandle = candleData.data[candleData.data.length - 1]

          // Cache this data for future fallback
          lastSuccessfulData = {
            price: latestCandle.close,
            timestamp: new Date(latestCandle.time).toISOString(),
            volume: latestCandle.volume,
            bid: latestCandle.close - 0.5,
            ask: latestCandle.close + 0.5
          }

          return NextResponse.json({
            type: 'price_update',
            symbol: 'USDCOP',
            price: latestCandle.close,
            timestamp: new Date(latestCandle.time).toISOString(),
            volume: latestCandle.volume,
            bid: latestCandle.close - 0.5,
            ask: latestCandle.close + 0.5,
            source: 'proxy_historical_fallback'
          })
        }

        return getFallbackResponse('No historical data available')
      } catch (historicalError) {
        console.error('[WebSocket Proxy] Historical fallback also failed:', historicalError)
        return getFallbackResponse('All data sources unavailable - connection reset')
      }
    }

    // If market is closed (425 status), fall back to historical data
    if (response.status === 425 || !response.ok) {
      console.log('[WebSocket Proxy] Market closed, using historical fallback')

      try {
        response = await fetchWithRetry(`${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=1`)
      } catch (fetchError) {
        console.error('[WebSocket Proxy] Historical fetch failed:', fetchError)
        return getFallbackResponse('Historical data fetch failed after retries')
      }

      if (!response.ok) {
        return getFallbackResponse('Historical endpoint returned error status')
      }

      const candleData = await response.json()

      if (candleData.data && candleData.data.length > 0) {
        const latestCandle = candleData.data[candleData.data.length - 1]

        // Cache this data for future fallback
        lastSuccessfulData = {
          price: latestCandle.close,
          timestamp: new Date(latestCandle.time).toISOString(),
          volume: latestCandle.volume,
          bid: latestCandle.close - 0.5,
          ask: latestCandle.close + 0.5
        }

        // Return data directly in the format expected by MarketDataFetcher
        return NextResponse.json({
          type: 'price_update',
          symbol: 'USDCOP',
          price: latestCandle.close,
          timestamp: new Date(latestCandle.time).toISOString(),
          volume: latestCandle.volume,
          bid: latestCandle.close - 0.5,
          ask: latestCandle.close + 0.5,
          source: 'proxy_historical_real'
        })
      }
    } else {
      // Market is open, use real-time data
      const data = await response.json()

      // Cache this data for future fallback
      lastSuccessfulData = {
        price: data.price,
        timestamp: data.timestamp,
        volume: data.volume,
        bid: data.bid,
        ask: data.ask
      }

      // Return data directly in the format expected by MarketDataFetcher
      return NextResponse.json({
        type: 'price_update',
        symbol: data.symbol,
        price: data.price,
        timestamp: data.timestamp,
        volume: data.volume,
        bid: data.bid,
        ask: data.ask,
        source: 'proxy_realtime'
      })
    }

    return getFallbackResponse('No data available from any source')

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    const errorCode = (error as NodeJS.ErrnoException).code || 'UNKNOWN'

    console.error('[WebSocket Proxy] Unhandled error:', { code: errorCode, message: errorMessage })

    // Check if it's a transient error and return appropriate response
    if (isTransientError(error)) {
      return getFallbackResponse(`Connection error (${errorCode}): ${errorMessage}`)
    }

    // For non-transient errors, still avoid 503 but indicate it's not retryable
    return NextResponse.json({
      type: 'error',
      error: 'server_error',
      message: errorMessage,
      code: errorCode,
      retryable: false,
      symbol: 'USDCOP',
      timestamp: new Date().toISOString()
    }, {
      status: 200, // Return 200 with error payload to prevent 503 page renders
      headers: {
        'X-Data-Status': 'error'
      }
    })
  }
})