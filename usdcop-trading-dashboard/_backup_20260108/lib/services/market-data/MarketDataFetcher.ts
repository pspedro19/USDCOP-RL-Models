/**
 * Market Data Fetcher - REST API Data Retrieval
 * =============================================
 *
 * Single Responsibility: Fetch market data from REST API endpoints
 * - Real-time price data
 * - Historical candlestick data
 * - Symbol statistics
 * - API health checks
 */

import { createLogger } from '@/lib/utils/logger'
import type {
  MarketDataPoint,
  CandlestickResponse,
  SymbolStats,
  APIHealthResponse,
} from './types'

const logger = createLogger('MarketDataFetcher')

export interface FetcherConfig {
  apiBaseUrl: string
}

export class MarketDataFetcher {
  private config: FetcherConfig

  constructor(config: FetcherConfig) {
    this.config = config
  }

  /**
   * Get real-time latest price data for a symbol
   */
  async getRealTimeData(symbol: string): Promise<MarketDataPoint[]> {
    try {
      const url = `${this.config.apiBaseUrl}/latest/${symbol}`
      logger.debug(`Fetching real-time data for ${symbol} from ${url}`)

      const response = await fetch(url, {
        credentials: 'include',  // Include session cookies for authentication
        signal: AbortSignal.timeout(10000), // Timeout after 10 seconds for realtime data
      })

      if (!response.ok) {
        // Market closed (status 425 - Too Early)
        if (response.status === 425) {
          logger.info('Market is closed, using historical data fallback')
          return await this.getHistoricalFallback(symbol)
        }
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      return [
        {
          symbol: data.symbol,
          price: data.price,
          timestamp: new Date(data.timestamp).getTime(),
          volume: data.volume,
          bid: data.bid,
          ask: data.ask,
          source: data.source,
        },
      ]
    } catch (error) {
      logger.error('Error fetching real-time data:', error)
      // Fallback to historical data when market is closed or error occurs
      return await this.getHistoricalFallback(symbol)
    }
  }

  /**
   * Get latest historical data when market is closed
   */
  async getHistoricalFallback(symbol: string): Promise<MarketDataPoint[]> {
    logger.info('Market closed - fetching latest historical data from database')

    try {
      // Fetch the most recent real data from database via candlesticks endpoint
      const response = await this.getCandlestickData(symbol, '5m', undefined, undefined, 1, false)

      if (response.data && response.data.length > 0) {
        const latestCandle = response.data[response.data.length - 1]

        return [
          {
            symbol: symbol,
            price: latestCandle.close,
            timestamp: latestCandle.time,
            volume: latestCandle.volume,
            bid: latestCandle.close - 0.5, // Conservative spread
            ask: latestCandle.close + 0.5,
            source: 'database_historical_real',
          },
        ]
      }

      throw new Error('No historical data available')
    } catch (error) {
      logger.error('Failed to fetch historical data:', error)
      throw new Error('Real data unavailable - no fallback allowed')
    }
  }

  /**
   * Get candlestick data for charts
   */
  async getCandlestickData(
    symbol: string,
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

      const url = `${this.config.apiBaseUrl}/candlesticks/${symbol}?${params}`
      logger.debug(`Fetching candlestick data from ${url}`)

      const response = await fetch(url, {
        credentials: 'include',  // Include session cookies for authentication
        signal: AbortSignal.timeout(65000), // 65s timeout - slightly longer than proxy's 60s to let proxy timeout first
        cache: 'no-store',  // Prevent stale cached responses
      })

      if (!response.ok) {
        logger.error(`Candlestick API error: ${response.status} ${response.statusText}`)
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      // Provide more informative error messages for timeouts
      if (error instanceof Error) {
        if (error.name === 'TimeoutError' || error.message.includes('signal timed out')) {
          logger.error(`Candlestick data request timed out for ${symbol}`)
          throw new Error(`Request timed out. The chart will retry automatically.`)
        }
        if (error.name === 'AbortError') {
          logger.warn('Candlestick request was aborted')
          throw error
        }
      }
      logger.error('Error fetching candlestick data:', error)
      throw error
    }
  }

  /**
   * Get symbol statistics
   */
  async getSymbolStats(symbol: string): Promise<SymbolStats> {
    try {
      const url = `${this.config.apiBaseUrl}/stats/${symbol}`
      logger.debug(`Fetching stats for ${symbol} from ${url}`)

      const response = await fetch(url, {
        credentials: 'include',  // Include session cookies for authentication
        signal: AbortSignal.timeout(30000), // Timeout after 30 seconds for stats (backend may be slow)
      })

      if (!response.ok) {
        // If stats endpoint doesn't exist, calculate from candlesticks
        logger.info(`Stats endpoint not available (${response.status}), using candlestick fallback`)
        return await this.getStatsFromCandlesticks(symbol)
      }

      return await response.json()
    } catch (error) {
      logger.error('Error fetching symbol stats, using candlestick fallback:', error)
      try {
        return await this.getStatsFromCandlesticks(symbol)
      } catch (fallbackError) {
        logger.error('Candlestick fallback also failed, returning default stats:', fallbackError)
        return this.getDefaultStats(symbol, 'error_fallback')
      }
    }
  }

  /**
   * Calculate statistics from recent candlestick data
   */
  private async getStatsFromCandlesticks(symbol: string): Promise<SymbolStats> {
    try {
      // Get last 24 hours of 5-minute data (288 candles)
      const response = await this.getCandlestickData(symbol, '5m', undefined, undefined, 288, false)

      if (response.data && response.data.length > 0) {
        const candles = response.data
        const prices = candles.map((c) => c.close)
        const volumes = candles.map((c) => c.volume)
        const highs = candles.map((c) => c.high)
        const lows = candles.map((c) => c.low)

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
          source: 'calculated_from_candlesticks',
        }
      }

      // Return default stats if no data available
      return this.getDefaultStats(symbol, 'default_fallback')
    } catch (error) {
      logger.error('Error calculating stats from candlesticks:', error)
      return this.getDefaultStats(symbol, 'error_fallback')
    }
  }

  /**
   * Get default stats structure
   */
  private getDefaultStats(symbol: string, source: string): SymbolStats {
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
      source,
    }
  }

  /**
   * Check API health
   */
  async checkAPIHealth(): Promise<APIHealthResponse> {
    try {
      const url = `${this.config.apiBaseUrl}/health`
      logger.debug(`Checking API health at ${url}`)

      const response = await fetch(url, {
        credentials: 'include',  // Include session cookies for authentication
        signal: AbortSignal.timeout(10000), // Timeout after 10 seconds for health check
      })
      const healthData = await response.json()

      logger.info('Trading API Health:', {
        status: healthData.status,
        database: healthData.database,
        records: healthData.total_records,
        market_status: healthData.market_status,
      })

      return healthData
    } catch (error) {
      logger.error('Error checking API health:', error)
      return { status: 'error', message: 'API not available' }
    }
  }

  /**
   * Check if market is currently open
   */
  async isMarketOpen(): Promise<boolean> {
    try {
      const health = await this.checkAPIHealth()
      return health.market_status?.is_open || false
    } catch (error) {
      logger.error('Error checking market status:', error)
      return false
    }
  }

  /**
   * Subscribe to real-time updates via polling (WebSocket alternative)
   */
  subscribeToPolling(
    callback: (data: MarketDataPoint) => void,
    interval: number = 2000
  ): () => void {
    logger.info(`Starting polling with ${interval}ms interval`)
    let consecutiveErrors = 0
    const MAX_CONSECUTIVE_ERRORS = 5

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch('/api/proxy/ws', {
          credentials: 'include',  // Include session cookies for authentication
          signal: AbortSignal.timeout(10000), // 10 second timeout for polling
        })
        if (response.ok) {
          const data = await response.json()

          // Check if response contains error payload (ws proxy returns 200 with error)
          if (data.type === 'error') {
            if (consecutiveErrors === 0) {
              logger.warn('Poll returned error payload:', data.message)
            }
            consecutiveErrors++
            return
          }

          logger.debug('Received poll data:', data)
          consecutiveErrors = 0 // Reset on success

          const marketData: MarketDataPoint = {
            symbol: data.symbol,
            price: data.price,
            timestamp: new Date(data.timestamp).getTime(),
            volume: data.volume || 0,
            bid: data.bid,
            ask: data.ask,
            source: data.source || 'polling',
          }

          callback(marketData)
        } else {
          consecutiveErrors++
          if (consecutiveErrors <= MAX_CONSECUTIVE_ERRORS) {
            logger.warn(`Polling response not ok (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, response.status)
          }
        }
      } catch (error) {
        consecutiveErrors++
        if (consecutiveErrors <= MAX_CONSECUTIVE_ERRORS) {
          logger.warn(`Polling error (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, error)
        }
      }
    }, interval)

    // Return unsubscribe function
    return () => {
      logger.info('Stopping polling')
      clearInterval(pollInterval)
    }
  }
}
