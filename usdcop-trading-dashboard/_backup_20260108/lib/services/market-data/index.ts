/**
 * Market Data Services - Unified Facade
 * =====================================
 *
 * Provides a backwards-compatible interface to the refactored market data services.
 * Follows the Facade pattern to maintain compatibility while using modular components.
 *
 * Module Structure:
 * - WebSocketConnector: WebSocket lifecycle management
 * - MarketDataFetcher: REST API data retrieval
 * - DataTransformer: Data formatting and transformation
 * - StatisticsCalculator: Market statistics calculations
 */

// Re-export all types
export * from './types'

// Re-export all modules for direct access
// Note: WebSocketConnector is deprecated in favor of unified-websocket-manager
export { WebSocketConnector } from './WebSocketConnector'
export type { WebSocketConfig } from './WebSocketConnector'

export { MarketDataFetcher } from './MarketDataFetcher'
export type { FetcherConfig } from './MarketDataFetcher'

export { DataTransformer } from './DataTransformer'
export type { FormatOptions } from './DataTransformer'

export { StatisticsCalculator } from './StatisticsCalculator'
export type { PriceChange, PriceStatistics, VolumeStatistics } from './StatisticsCalculator'

// Re-export unified WebSocket manager as the preferred option
export { getUnifiedWebSocketManager, UnifiedWebSocketManager } from '../unified-websocket-manager'

import { getUnifiedWebSocketManager } from '../unified-websocket-manager'
import { MarketDataFetcher } from './MarketDataFetcher'
import { DataTransformer } from './DataTransformer'
import { StatisticsCalculator } from './StatisticsCalculator'
import type { MarketDataPoint, CandlestickResponse, SymbolStats } from './types'

/**
 * Backwards-compatible MarketDataService class
 * Delegates to specialized modules while maintaining the original API
 */
export class MarketDataService {
  private static API_BASE_URL =
    typeof window === 'undefined' ? 'http://localhost:8000/api' : '/api/proxy/trading'
  private static fetcher: MarketDataFetcher | null = null
  private static subscribers: Array<(data: MarketDataPoint) => void> = []
  private static wsUnsubscribe: (() => void) | null = null

  /**
   * Get fetcher instance (lazy initialization)
   */
  private static getFetcher(): MarketDataFetcher {
    if (!this.fetcher) {
      this.fetcher = new MarketDataFetcher({
        apiBaseUrl: this.API_BASE_URL,
      })
    }
    return this.fetcher
  }

  /**
   * Connect to WebSocket for real-time updates
   * Uses the unified WebSocket manager with authentication and validation
   * @deprecated Use getUnifiedWebSocketManager() directly for more control
   */
  static connectWebSocket(symbol: string = 'USDCOP'): void {
    const wsManager = getUnifiedWebSocketManager()

    // Set up subscriber forwarding
    this.wsUnsubscribe = wsManager.on<MarketDataPoint>('market_data', (data) => {
      this.subscribers.forEach((callback) => callback(data))
    })

    wsManager.connect()
    wsManager.subscribe('market_data')
  }

  /**
   * Subscribe to real-time price updates via polling
   * @deprecated Use MarketDataFetcher.subscribeToPolling directly
   */
  static subscribeToRealTimeUpdates(callback: (data: MarketDataPoint) => void): () => void {
    this.subscribers.push(callback)

    const fetcher = this.getFetcher()
    const unsubscribePolling = fetcher.subscribeToPolling(callback, 2000)

    // Return combined unsubscribe function
    return () => {
      unsubscribePolling()
      const index = this.subscribers.indexOf(callback)
      if (index > -1) {
        this.subscribers.splice(index, 1)
      }
    }
  }

  /**
   * Get real-time latest price data
   */
  static async getRealTimeData(symbol: string = 'USDCOP'): Promise<MarketDataPoint[]> {
    return this.getFetcher().getRealTimeData(symbol)
  }

  /**
   * Get latest historical data when market is closed
   */
  static async getHistoricalFallback(symbol: string = 'USDCOP'): Promise<MarketDataPoint[]> {
    return this.getFetcher().getHistoricalFallback(symbol)
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
    return this.getFetcher().getCandlestickData(
      symbol,
      timeframe,
      startDate,
      endDate,
      limit,
      includeIndicators
    )
  }

  /**
   * Get symbol statistics
   */
  static async getSymbolStats(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    return this.getFetcher().getSymbolStats(symbol)
  }

  /**
   * Calculate statistics from candlesticks (internal method)
   * @deprecated This is now handled internally by MarketDataFetcher
   */
  static async getStatsFromCandlesticks(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    // Delegate to fetcher's internal method via getSymbolStats
    return this.getFetcher().getSymbolStats(symbol)
  }

  /**
   * Check API health
   */
  static async checkAPIHealth() {
    return this.getFetcher().checkAPIHealth()
  }

  /**
   * Check if market is currently open
   */
  static async isMarketOpen(): Promise<boolean> {
    return this.getFetcher().isMarketOpen()
  }

  /**
   * Format price for display
   */
  static formatPrice(price: number, decimals: number = 4): string {
    return DataTransformer.formatPrice(price, decimals)
  }

  /**
   * Calculate price change
   */
  static calculatePriceChange(current: number, previous: number) {
    return StatisticsCalculator.calculatePriceChange(current, previous)
  }

  // Additional utility methods exposed for convenience

  /**
   * Format volume for display
   */
  static formatVolume(volume: number): string {
    return DataTransformer.formatVolume(volume)
  }

  /**
   * Format timestamp to readable date
   */
  static formatTimestamp(timestamp: number): string {
    return DataTransformer.formatTimestamp(timestamp)
  }

  /**
   * Calculate statistics from candlestick data
   */
  static calculatePriceStats(candles: Parameters<typeof StatisticsCalculator.calculatePriceStats>[0]) {
    return StatisticsCalculator.calculatePriceStats(candles)
  }

  /**
   * Calculate 24h statistics
   */
  static calculate24hStats(candles: Parameters<typeof StatisticsCalculator.calculate24hStats>[0]) {
    return StatisticsCalculator.calculate24hStats(candles)
  }
}
