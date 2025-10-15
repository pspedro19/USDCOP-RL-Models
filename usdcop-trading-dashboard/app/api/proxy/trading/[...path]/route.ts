/**
 * Trading API Proxy
 * =================
 *
 * Proxy requests to the Trading API to overcome CORS and firewall issues
 * Routes all /api/proxy/trading/* requests to the internal Trading API
 */

import { NextRequest, NextResponse } from 'next/server'

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://usdcop-trading-api:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/')
    const url = new URL(request.url)
    const queryString = url.search

    const targetUrl = `${TRADING_API_URL}/api/${path}${queryString}`

    console.log(`[Proxy] GET ${targetUrl}`)

    const response = await fetch(targetUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      console.error(`[Proxy] Error: ${response.status} ${response.statusText}`)

      // Handle market closed (425) by returning fallback data
      if (response.status === 425 && path.includes('latest')) {
        console.log('[Proxy] Market closed, providing fallback historical data')
        const fallbackResponse = await fetch(`${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=1`)

        if (fallbackResponse.ok) {
          const fallbackData = await fallbackResponse.json()
          if (fallbackData.data && fallbackData.data.length > 0) {
            const latestCandle = fallbackData.data[fallbackData.data.length - 1]
            return NextResponse.json({
              symbol: 'USDCOP',
              price: latestCandle.close,
              timestamp: new Date(latestCandle.time).toISOString(),
              volume: latestCandle.volume,
              bid: latestCandle.close - 0.5,
              ask: latestCandle.close + 0.5,
              source: 'database_historical_real'
            })
          }
        }
      }

      return NextResponse.json(
        { error: `Trading API error: ${response.status}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('[Proxy] Error:', error)
    return NextResponse.json(
      { error: 'Proxy connection failed' },
      { status: 500 }
    )
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const resolvedParams = await params;
    const path = resolvedParams.path.join('/')
    const body = await request.text()

    const targetUrl = `${TRADING_API_URL}/api/${path}`

    console.log(`[Proxy] POST ${targetUrl}`)

    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body,
    })

    if (!response.ok) {
      console.error(`[Proxy] Error: ${response.status} ${response.statusText}`)
      return NextResponse.json(
        { error: `Trading API error: ${response.status}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('[Proxy] Error:', error)
    return NextResponse.json(
      { error: 'Proxy connection failed' },
      { status: 500 }
    )
  }
}