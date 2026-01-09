/**
 * IDataProvider Interface
 * =======================
 *
 * Interface for data providers that fetch market data, historical data, and symbol statistics.
 */

import {
  CandlestickExtended,
  SymbolStats,
  MarketDataPoint,
  BacktestResult,
  PerformanceMetrics
} from '@/types/trading';
import { ApiResponse } from '@/types/common';

/**
 * Data provider interface for market data operations
 */
export interface IDataProvider {
  /**
   * Get real-time market data for a symbol
   */
  getRealTimeData(symbol: string): Promise<MarketDataPoint>;

  /**
   * Get historical candlestick data
   */
  getHistoricalData(
    symbol: string,
    timeframe: string,
    startDate?: string,
    endDate?: string,
    limit?: number
  ): Promise<CandlestickExtended[]>;

  /**
   * Get symbol statistics (24h data)
   */
  getSymbolStats(symbol: string): Promise<SymbolStats>;

  /**
   * Get backtest results for a strategy
   */
  getBacktestResults(
    strategyCode: string,
    startDate?: string,
    endDate?: string
  ): Promise<BacktestResult>;

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(
    strategyCode?: string,
    timeRange?: { start: string; end: string }
  ): Promise<PerformanceMetrics>;

  /**
   * Health check for the data provider
   */
  healthCheck(): Promise<boolean>;
}

/**
 * Extended data provider with additional capabilities
 */
export interface IExtendedDataProvider extends IDataProvider {
  /**
   * Get multiple symbols data in batch
   */
  getBatchData(symbols: string[]): Promise<Record<string, MarketDataPoint>>;

  /**
   * Stream data updates
   */
  streamData(
    symbol: string,
    callback: (data: MarketDataPoint) => void
  ): () => void; // Returns unsubscribe function

  /**
   * Cache management
   */
  clearCache(): void;
  invalidateCache(symbol: string): void;
  getCacheStats(): { hits: number; misses: number; size: number };
}

/**
 * Configuration for data providers
 */
export interface DataProviderConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  cacheEnabled?: boolean;
  cacheTTL?: number;
}
