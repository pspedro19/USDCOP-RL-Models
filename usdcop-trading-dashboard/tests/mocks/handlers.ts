import { http, HttpResponse, delay } from 'msw'

// Mock market data
const generateCandleData = (count: number, startTime: number = Date.now() - (count * 5 * 60 * 1000)) => {
  return Array.from({ length: count }, (_, i) => {
    const time = startTime + (i * 5 * 60 * 1000) // 5-minute intervals
    const basePrice = 4000 + Math.sin(i / 10) * 100 // Oscillating around 4000
    const volatility = 20

    const open = basePrice + (Math.random() - 0.5) * volatility
    const high = open + Math.random() * volatility
    const low = open - Math.random() * volatility
    const close = open + (Math.random() - 0.5) * volatility
    const volume = 1000000 + Math.random() * 500000

    return {
      datetime: new Date(time).toISOString(),
      timestamp: Math.floor(time / 1000),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.floor(volume)
    }
  })
}

// Mock technical indicators data
const generateTechnicalIndicators = (data: any[]) => {
  return data.map((candle, index) => ({
    ...candle,
    ema20: index >= 19 ? candle.close + (Math.random() - 0.5) * 5 : null,
    ema50: index >= 49 ? candle.close + (Math.random() - 0.5) * 8 : null,
    ema200: index >= 199 ? candle.close + (Math.random() - 0.5) * 15 : null,
    bb_upper: candle.close + 20 + Math.random() * 10,
    bb_middle: candle.close + (Math.random() - 0.5) * 5,
    bb_lower: candle.close - 20 - Math.random() * 10,
    rsi: Math.random() * 100,
    macd: (Math.random() - 0.5) * 10,
    macd_signal: (Math.random() - 0.5) * 8,
    macd_histogram: (Math.random() - 0.5) * 5,
    volume_sma: candle.volume + (Math.random() - 0.5) * 100000
  }))
}

// Mock WebSocket messages
export const webSocketMessages = {
  priceUpdate: (price: number = 4000) => ({
    type: 'price_update',
    data: {
      symbol: 'USDCOP',
      price: Number((price + (Math.random() - 0.5) * 10).toFixed(2)),
      change: Number(((Math.random() - 0.5) * 20).toFixed(2)),
      changePercent: Number(((Math.random() - 0.5) * 0.5).toFixed(3)),
      volume: Math.floor(Math.random() * 1000000),
      timestamp: Date.now()
    }
  }),

  newCandle: () => {
    const now = Date.now()
    const basePrice = 4000 + Math.sin(now / 100000) * 100
    const volatility = 20

    return {
      type: 'new_candle',
      data: {
        datetime: new Date(now).toISOString(),
        timestamp: Math.floor(now / 1000),
        open: Number((basePrice + (Math.random() - 0.5) * volatility).toFixed(2)),
        high: Number((basePrice + Math.random() * volatility).toFixed(2)),
        low: Number((basePrice - Math.random() * volatility).toFixed(2)),
        close: Number((basePrice + (Math.random() - 0.5) * volatility).toFixed(2)),
        volume: Math.floor(1000000 + Math.random() * 500000)
      }
    }
  },

  marketStatus: (status: 'open' | 'closed' | 'pre_market' | 'after_hours' = 'open') => ({
    type: 'market_status',
    data: {
      status,
      nextOpen: status === 'closed' ? Date.now() + 8 * 60 * 60 * 1000 : null,
      nextClose: status === 'open' ? Date.now() + 6 * 60 * 60 * 1000 : null,
      timezone: 'America/Bogota'
    }
  }),

  error: (message: string = 'Connection error') => ({
    type: 'error',
    data: {
      message,
      code: 'WS_ERROR',
      timestamp: Date.now()
    }
  })
}

export const handlers = [
  // Market data endpoints
  http.get('/api/market/data', async ({ request }) => {
    const url = new URL(request.url)
    const symbol = url.searchParams.get('symbol') || 'USDCOP'
    const timeframe = url.searchParams.get('timeframe') || '5m'
    const limit = parseInt(url.searchParams.get('limit') || '1000')
    const from = url.searchParams.get('from')
    const to = url.searchParams.get('to')

    // Simulate network delay
    await delay(100)

    const startTime = from ? new Date(from).getTime() : Date.now() - (limit * 5 * 60 * 1000)
    const data = generateCandleData(limit, startTime)

    return HttpResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        candles: data,
        meta: {
          count: data.length,
          from: new Date(startTime).toISOString(),
          to: new Date(startTime + (limit * 5 * 60 * 1000)).toISOString()
        }
      }
    })
  }),

  // Real-time price endpoint
  http.get('/api/market/realtime', async () => {
    await delay(50)

    const price = 4000 + Math.sin(Date.now() / 100000) * 100

    return HttpResponse.json({
      success: true,
      data: {
        symbol: 'USDCOP',
        price: Number((price + (Math.random() - 0.5) * 10).toFixed(2)),
        change: Number(((Math.random() - 0.5) * 20).toFixed(2)),
        changePercent: Number(((Math.random() - 0.5) * 0.5).toFixed(3)),
        volume: Math.floor(Math.random() * 1000000),
        bid: Number((price - 0.5).toFixed(2)),
        ask: Number((price + 0.5).toFixed(2)),
        spread: 1.0,
        timestamp: Date.now(),
        marketStatus: 'open'
      }
    })
  }),

  // Technical indicators endpoint
  http.get('/api/market/indicators', async ({ request }) => {
    const url = new URL(request.url)
    const symbol = url.searchParams.get('symbol') || 'USDCOP'
    const timeframe = url.searchParams.get('timeframe') || '5m'
    const indicators = url.searchParams.get('indicators')?.split(',') || ['ema20', 'ema50', 'rsi', 'macd']

    await delay(150)

    const baseData = generateCandleData(200)
    const dataWithIndicators = generateTechnicalIndicators(baseData)

    return HttpResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        indicators: indicators,
        data: dataWithIndicators
      }
    })
  }),

  // Volume profile endpoint
  http.get('/api/market/volume-profile', async ({ request }) => {
    const url = new URL(request.url)
    const symbol = url.searchParams.get('symbol') || 'USDCOP'
    const from = url.searchParams.get('from')
    const to = url.searchParams.get('to')
    const levels = parseInt(url.searchParams.get('levels') || '50')

    await delay(200)

    // Generate volume profile data
    const minPrice = 3900
    const maxPrice = 4100
    const priceStep = (maxPrice - minPrice) / levels

    const profileLevels = Array.from({ length: levels }, (_, i) => {
      const price = minPrice + (i * priceStep)
      const volume = Math.floor(Math.random() * 2000000) + 500000

      return {
        price: Number(price.toFixed(2)),
        volume,
        percentOfTotal: Number((Math.random() * 5 + 1).toFixed(2))
      }
    })

    // Sort by volume to find POC
    const sortedByVolume = [...profileLevels].sort((a, b) => b.volume - a.volume)
    const poc = sortedByVolume[0].price

    return HttpResponse.json({
      success: true,
      data: {
        symbol,
        levels: profileLevels,
        poc,
        valueAreaHigh: poc + 30,
        valueAreaLow: poc - 30,
        totalVolume: profileLevels.reduce((sum, level) => sum + level.volume, 0),
        valueAreaVolume: profileLevels.reduce((sum, level) => sum + level.volume, 0) * 0.7
      }
    })
  }),

  // ML predictions endpoint
  http.get('/api/ml/predictions', async ({ request }) => {
    const url = new URL(request.url)
    const symbol = url.searchParams.get('symbol') || 'USDCOP'
    const horizon = parseInt(url.searchParams.get('horizon') || '24')

    await delay(300)

    const currentPrice = 4000 + Math.sin(Date.now() / 100000) * 100
    const predictions = Array.from({ length: horizon }, (_, i) => {
      const time = Date.now() + ((i + 1) * 5 * 60 * 1000)
      const trend = Math.sin(i / 5) * 0.1
      const noise = (Math.random() - 0.5) * 0.05
      const predictedPrice = currentPrice * (1 + trend + noise)
      const confidence = Math.max(0.5, 0.95 - (i * 0.02))
      const variance = 20 * (1 - confidence)

      return {
        time: Math.floor(time / 1000),
        predicted_price: Number(predictedPrice.toFixed(2)),
        confidence: Number(confidence.toFixed(3)),
        upper_bound: Number((predictedPrice + variance).toFixed(2)),
        lower_bound: Number((predictedPrice - variance).toFixed(2)),
        trend: trend > 0.02 ? 'bullish' : trend < -0.02 ? 'bearish' : 'neutral'
      }
    })

    return HttpResponse.json({
      success: true,
      data: {
        symbol,
        horizon,
        model: 'LSTM-GRU-Ensemble',
        predictions,
        meta: {
          accuracy: 0.847,
          lastTrained: Date.now() - 2 * 60 * 60 * 1000,
          confidence: 0.923
        }
      }
    })
  }),

  // Market status endpoint
  http.get('/api/market/status', async () => {
    await delay(50)

    const now = new Date()
    const hour = now.getHours()
    let status = 'open'

    if (hour < 8 || hour >= 17) {
      status = 'closed'
    } else if (hour < 9) {
      status = 'pre_market'
    } else if (hour >= 16) {
      status = 'after_hours'
    }

    return HttpResponse.json({
      success: true,
      data: {
        status,
        marketOpen: '09:00',
        marketClose: '16:00',
        timezone: 'America/Bogota',
        nextOpen: status === 'closed' ? '09:00' : null,
        nextClose: status === 'open' ? '16:00' : null,
        isHoliday: false,
        tradingHours: {
          regular: { open: '09:00', close: '16:00' },
          extended: { open: '08:00', close: '17:00' }
        }
      }
    })
  }),

  // Trading operations endpoints
  http.post('/api/trading/order', async ({ request }) => {
    const body = await request.json() as any
    await delay(200)

    // Validate order
    if (!body.symbol || !body.quantity || !body.side) {
      return HttpResponse.json({
        success: false,
        error: {
          code: 'INVALID_ORDER',
          message: 'Missing required order fields'
        }
      }, { status: 400 })
    }

    // Simulate order execution
    const orderId = `ORDER_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const currentPrice = 4000 + Math.sin(Date.now() / 100000) * 100
    const executionPrice = body.type === 'market'
      ? currentPrice + (Math.random() - 0.5) * 2
      : body.price

    return HttpResponse.json({
      success: true,
      data: {
        orderId,
        symbol: body.symbol,
        side: body.side,
        quantity: body.quantity,
        type: body.type,
        status: 'filled',
        executionPrice: Number(executionPrice.toFixed(2)),
        executedQuantity: body.quantity,
        timestamp: Date.now(),
        commission: Number((body.quantity * executionPrice * 0.001).toFixed(2))
      }
    })
  }),

  // Portfolio endpoints
  http.get('/api/portfolio/positions', async () => {
    await delay(100)

    const positions = [
      {
        symbol: 'USDCOP',
        quantity: 10000,
        averagePrice: 3950.25,
        currentPrice: 4025.80,
        unrealizedPnL: 755.0,
        unrealizedPnLPercent: 1.91,
        marketValue: 40258.00,
        side: 'long'
      },
      {
        symbol: 'EURUSD',
        quantity: -5000,
        averagePrice: 1.0850,
        currentPrice: 1.0820,
        unrealizedPnL: 150.0,
        unrealizedPnLPercent: 0.28,
        marketValue: -5410.00,
        side: 'short'
      }
    ]

    return HttpResponse.json({
      success: true,
      data: {
        positions,
        totalValue: positions.reduce((sum, pos) => sum + pos.marketValue, 0),
        totalPnL: positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0),
        totalPnLPercent: 1.85
      }
    })
  }),

  // Error simulation endpoints
  http.get('/api/test/error/:type', async ({ params }) => {
    const { type } = params
    await delay(100)

    switch (type) {
      case 'timeout':
        await delay(30000) // Simulate timeout
        return HttpResponse.json({ success: true })

      case 'server-error':
        return HttpResponse.json({
          success: false,
          error: {
            code: 'INTERNAL_ERROR',
            message: 'Internal server error'
          }
        }, { status: 500 })

      case 'unauthorized':
        return HttpResponse.json({
          success: false,
          error: {
            code: 'UNAUTHORIZED',
            message: 'Authentication required'
          }
        }, { status: 401 })

      case 'rate-limit':
        return HttpResponse.json({
          success: false,
          error: {
            code: 'RATE_LIMIT_EXCEEDED',
            message: 'Too many requests'
          }
        }, { status: 429 })

      default:
        return HttpResponse.json({
          success: false,
          error: {
            code: 'UNKNOWN_ERROR',
            message: 'Unknown error type'
          }
        }, { status: 400 })
    }
  }),

  // Performance test endpoint
  http.get('/api/test/performance/:size', async ({ params }) => {
    const { size } = params
    const dataSize = parseInt(size as string) || 1000

    const startTime = Date.now()
    const data = generateCandleData(dataSize)
    const processingTime = Date.now() - startTime

    return HttpResponse.json({
      success: true,
      data: {
        candles: data,
        meta: {
          count: dataSize,
          processingTime,
          generatedAt: Date.now()
        }
      }
    })
  })
]

// Default export for handlers
export default handlers