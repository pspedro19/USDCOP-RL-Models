/**
 * Market Data Service - Professional Trading Data
 * ==============================================
 *
 * Servicio optimizado para obtener datos reales de la base de datos PostgreSQL
 * Conecta directamente con la API de trading para TradingView charts
 */

export interface MarketDataPoint {
  symbol: string
  price: number
  timestamp: number
  volume: number
  bid?: number
  ask?: number
  source?: string
}

export interface CandlestickData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface TechnicalIndicators {
  ema_20?: number
  ema_50?: number
  ema_200?: number
  bb_upper?: number
  bb_middle?: number
  bb_lower?: number
  rsi?: number
}

export interface CandlestickResponse {
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  count: number
  data: (CandlestickData & { indicators?: TechnicalIndicators })[]
}

export class MarketDataService {
  private static API_BASE_URL = process.env.NEXT_PUBLIC_TRADING_API_URL || (typeof window === 'undefined' ? 'http://usdcop-trading-api:8000/api' : '/api/proxy/trading')
  private static WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8082'
  private static websocket: WebSocket | null = null
  private static subscribers: Array<(data: MarketDataPoint) => void> = []

  /**
   * Connect to WebSocket for real-time updates
   */
  static connectWebSocket(): WebSocket | null {
    if (typeof window === 'undefined') return null

    try {
      console.log(`🔌 Connecting to WebSocket: ${this.WS_URL}`)
      this.websocket = new WebSocket(`${this.WS_URL}/ws`)

      this.websocket.onopen = () => {
        console.log('✅ WebSocket connected to enhanced websocket service on port 8082')

        // Send subscription message for USDCOP
        if (this.websocket?.readyState === WebSocket.OPEN) {
          this.websocket.send(JSON.stringify({
            type: 'subscribe',
            symbol: 'USDCOP'
          }))
        }
      }

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('📥 Received WebSocket data:', data)

          // Handle different message types from enhanced websocket service
          if (data.type === 'price_update' || data.type === 'market_data') {
            const marketData: MarketDataPoint = {
              symbol: data.symbol || 'USDCOP',
              price: data.price || data.close,
              timestamp: new Date(data.timestamp).getTime(),
              volume: data.volume || 0,
              bid: data.bid,
              ask: data.ask,
              source: data.source || 'websocket'
            }

            // Notify all subscribers
            this.subscribers.forEach(callback => callback(marketData))
          } else if (data.type === 'status') {
            console.log('📊 WebSocket service status:', data)
          }
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error)
        }
      }

      this.websocket.onclose = (event) => {
        console.log(`🔌 WebSocket disconnected (code: ${event.code}), attempting to reconnect in 5s...`)
        setTimeout(() => this.connectWebSocket(), 5000)
      }

      this.websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error)
      }

      return this.websocket
    } catch (error) {
      console.error('❌ Failed to connect WebSocket:', error)
      return null
    }
  }

  /**
   * Subscribe to real-time price updates via polling (WebSocket proxy alternative)
   */
  static subscribeToRealTimeUpdates(callback: (data: MarketDataPoint) => void): () => void {
    this.subscribers.push(callback)

    // Use polling instead of WebSocket due to firewall constraints
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch('/api/proxy/ws')
        if (response.ok) {
          const data = await response.json()
          console.log('📊 Received poll data:', data)

          const marketData: MarketDataPoint = {
            symbol: data.symbol || 'USDCOP',
            price: data.price,
            timestamp: new Date(data.timestamp).getTime(),
            volume: data.volume || 0,
            bid: data.bid,
            ask: data.ask,
            source: data.source || 'polling'
          }

          // Notify all subscribers
          this.subscribers.forEach(sub => sub(marketData))
        } else {
          console.error('❌ Polling response not ok:', response.status, response.statusText)
        }
      } catch (error) {
        console.error('❌ Polling error:', error)
      }
    }, 2000) // Poll every 2 seconds

    // Return unsubscribe function
    return () => {
      clearInterval(pollInterval)
      const index = this.subscribers.indexOf(callback)
      if (index > -1) {
        this.subscribers.splice(index, 1)
      }
    }
  }

  /**
   * Get real-time latest price data via REST API
   */
  static async getRealTimeData(): Promise<MarketDataPoint[]> {
    try {
      const response = await fetch(`${this.API_BASE_URL}/latest/USDCOP`)

      if (!response.ok) {
        // Check if market is closed (status 425 - Too Early)
        if (response.status === 425) {
          console.log('Market is closed, using historical data fallback')
          return await this.getHistoricalFallback()
        }
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      return [{
        symbol: data.symbol,
        price: data.price,
        timestamp: new Date(data.timestamp).getTime(),
        volume: data.volume,
        bid: data.bid,
        ask: data.ask,
        source: data.source
      }]
    } catch (error) {
      console.error('Error fetching real-time data:', error)

      // Fallback to historical data when market is closed
      return await this.getHistoricalFallback()
    }
  }

  /**
   * Get latest real historical data when market is closed
   */
  static async getHistoricalFallback(): Promise<MarketDataPoint[]> {
    console.log('🔄 Market closed - fetching latest real historical data from database')

    try {
      // Fetch the most recent real data from database via candlesticks endpoint
      const response = await fetch(`${this.API_BASE_URL}/candlesticks/USDCOP?timeframe=5m&limit=1`)

      if (!response.ok) {
        throw new Error(`Failed to fetch historical data: ${response.status}`)
      }

      const data = await response.json()

      if (data.data && data.data.length > 0) {
        const latestCandle = data.data[data.data.length - 1]

        return [{
          symbol: 'USDCOP',
          price: latestCandle.close,
          timestamp: latestCandle.time,
          volume: latestCandle.volume,
          bid: latestCandle.close - 0.5, // Conservative spread
          ask: latestCandle.close + 0.5,
          source: 'database_historical_real'
        }]
      }

      throw new Error('No historical data available')
    } catch (error) {
      console.error('❌ Failed to fetch real historical data:', error)
      throw new Error('Real data unavailable - no fallback allowed')
    }
  }

  /**
   * Get candlestick data for charts
   */
  static async getCandlestickData(
    symbol: string = 'USDCOP',
    timeframe: string = '5m',
    startDate?: string,
    endDate?: string,
    limit: number = 1000,
    includeIndicators: boolean = true
  ): Promise<CandlestickResponse> {
    try {
      const params = new URLSearchParams({
        timeframe,
        limit: limit.toString(),
        include_indicators: includeIndicators.toString(),
      })

      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)

      const response = await fetch(`${this.API_BASE_URL}/candlesticks/${symbol}?${params}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error fetching candlestick data:', error)
      throw error
    }
  }

  /**
   * Get symbol statistics (with fallback to candlestick data if stats endpoint not available)
   */
  static async getSymbolStats(symbol: string = 'USDCOP') {
    try {
      const response = await fetch(`${this.API_BASE_URL}/stats/${symbol}`)

      if (!response.ok) {
        // If stats endpoint doesn't exist, calculate stats from recent candlestick data
        console.log(`[MarketDataService] Stats endpoint not available (${response.status}), using candlestick fallback`)
        return await this.getStatsFromCandlesticks(symbol)
      }

      return await response.json()
    } catch (error) {
      console.error('Error fetching symbol stats, using candlestick fallback:', error)
      return await this.getStatsFromCandlesticks(symbol)
    }
  }

  /**
   * Calculate basic statistics from recent candlestick data
   */
  static async getStatsFromCandlesticks(symbol: string = 'USDCOP') {
    try {
      // Get last 24 hours of 5-minute data (288 candles)
      const response = await this.getCandlestickData(symbol, '5m', undefined, undefined, 288, false)

      if (response.data && response.data.length > 0) {
        const candles = response.data
        const prices = candles.map(c => c.close)
        const volumes = candles.map(c => c.volume)
        const highs = candles.map(c => c.high)
        const lows = candles.map(c => c.low)

        const currentPrice = prices[prices.length - 1]
        const openPrice = candles[0].open
        const high24h = Math.max(...highs)
        const low24h = Math.min(...lows)
        const volume24h = volumes.reduce((sum, vol) => sum + vol, 0)
        const priceChange = currentPrice - openPrice
        const priceChangePercent = (priceChange / openPrice) * 100

        return {
          symbol,
          price: currentPrice,
          open_24h: openPrice,
          high_24h: high24h,
          low_24h: low24h,
          volume_24h: volume24h,
          change_24h: priceChange,
          change_percent_24h: priceChangePercent,
          spread: high24h - low24h,
          timestamp: new Date().toISOString(),
          source: 'calculated_from_candlesticks'
        }
      }

      // Return default stats if no data available
      return {
        symbol,
        price: 0,
        open_24h: 0,
        high_24h: 0,
        low_24h: 0,
        volume_24h: 0,
        change_24h: 0,
        change_percent_24h: 0,
        spread: 0,
        timestamp: new Date().toISOString(),
        source: 'default_fallback'
      }
    } catch (error) {
      console.error('Error calculating stats from candlesticks:', error)
      // Return default stats as final fallback
      return {
        symbol,
        price: 0,
        open_24h: 0,
        high_24h: 0,
        low_24h: 0,
        volume_24h: 0,
        change_24h: 0,
        change_percent_24h: 0,
        spread: 0,
        timestamp: new Date().toISOString(),
        source: 'error_fallback'
      }
    }
  }

  /**
   * Check API health
   */
  static async checkAPIHealth() {
    try {
      const response = await fetch(`${this.API_BASE_URL}/health`)
      const healthData = await response.json()

      console.log('🏥 Trading API Health:', {
        status: healthData.status,
        database: healthData.database,
        records: healthData.total_records,
        market_status: healthData.market_status
      })

      return healthData
    } catch (error) {
      console.error('❌ Error checking API health:', error)
      return { status: 'error', message: 'API not available' }
    }
  }

  /**
   * Check if market is currently open (8:00-12:55 COT, Mon-Fri)
   */
  static async isMarketOpen(): Promise<boolean> {
    try {
      const health = await this.checkAPIHealth()
      return health.market_status?.is_open || false
    } catch (error) {
      console.error('❌ Error checking market status:', error)
      return false
    }
  }

  /**
   * Format price for display
   */
  static formatPrice(price: number, decimals: number = 4): string {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(price)
  }

  /**
   * Calculate price change
   */
  static calculatePriceChange(current: number, previous: number) {
    const change = current - previous
    const changePercent = (change / previous) * 100

    return {
      change,
      changePercent,
      isPositive: change >= 0
    }
  }
}
