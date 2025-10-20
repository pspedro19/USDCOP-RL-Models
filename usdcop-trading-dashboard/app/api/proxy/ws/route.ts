/**
 * WebSocket Proxy for Real-time Data
 * ==================================
 *
 * Proxies WebSocket connections to the internal WebSocket service
 * Converts HTTP requests to WebSocket connections for real-time data
 */

import { NextRequest, NextResponse } from 'next/server'

const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    // First try the real-time endpoint
    let response = await fetch(`${TRADING_API_URL}/api/latest/USDCOP`)

    // If market is closed (425 status), fall back to historical data
    if (response.status === 425 || !response.ok) {
      console.log('[WebSocket Proxy] Market closed, using historical fallback')
      response = await fetch(`${TRADING_API_URL}/api/candlesticks/USDCOP?timeframe=5m&limit=1`)

      if (!response.ok) {
        return NextResponse.json(
          { error: 'Failed to fetch data - both real-time and historical failed' },
          { status: 500 }
        )
      }

      const candleData = await response.json()

      if (candleData.data && candleData.data.length > 0) {
        const latestCandle = candleData.data[candleData.data.length - 1]

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

    return NextResponse.json(
      { error: 'No data available' },
      { status: 404 }
    )

  } catch (error) {
    console.error('[WebSocket Proxy] Error:', error)
    return NextResponse.json(
      { error: 'WebSocket proxy connection failed' },
      { status: 500 }
    )
  }
}