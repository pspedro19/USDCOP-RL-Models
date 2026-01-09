/**
 * API Data Provider
 * ==================
 *
 * Provides real market data from the Trading API
 * Implements IDataProvider interface
 */

import type {
  IDataProvider,
  MarketDataPoint,
  CandlestickResponse,
  SymbolStats,
} from '@/lib/core/interfaces';

export interface ApiDataProviderConfig {
  apiBaseUrl?: string;
  timeout?: number;
}

export class ApiDataProvider implements IDataProvider {
  private apiBaseUrl: string;
  private timeout: number;
  private subscribers: Array<(data: MarketDataPoint) => void> = [];
  private pollInterval: NodeJS.Timeout | null = null;

  constructor(config?: ApiDataProviderConfig) {
    this.apiBaseUrl =
      config?.apiBaseUrl ||
      (typeof window === 'undefined' ? 'http://localhost:8000/api' : '/api/proxy/trading');
    this.timeout = config?.timeout || 10000;
  }

  /**
   * Get real-time market data
   */
  async getRealTimeData(): Promise<MarketDataPoint[]> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/latest/USDCOP`, {
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        // Check if market is closed (status 425 - Too Early)
        if (response.status === 425) {
          console.log('Market is closed, using historical data fallback');
          return await this.getHistoricalFallback();
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

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
      ];
    } catch (error) {
      console.error('Error fetching real-time data:', error);
      // Fallback to historical data when market is closed
      return await this.getHistoricalFallback();
    }
  }

  /**
   * Get candlestick data for charts
   */
  async getCandlestickData(
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
      });

      if (startDate) params.append('start_date', startDate);
      if (endDate) params.append('end_date', endDate);

      const response = await fetch(`${this.apiBaseUrl}/candlesticks/${symbol}?${params}`, {
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching candlestick data:', error);
      throw error;
    }
  }

  /**
   * Get symbol statistics
   */
  async getSymbolStats(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/stats/${symbol}`, {
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        // If stats endpoint doesn't exist, calculate stats from recent candlestick data
        console.log(
          `[ApiDataProvider] Stats endpoint not available (${response.status}), using candlestick fallback`
        );
        return await this.getStatsFromCandlesticks(symbol);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching symbol stats, using candlestick fallback:', error);
      return await this.getStatsFromCandlesticks(symbol);
    }
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<{ status: string; message?: string }> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/health`, {
        signal: AbortSignal.timeout(this.timeout),
      });

      if (!response.ok) {
        return {
          status: 'unhealthy',
          message: `API returned status ${response.status}`,
        };
      }

      const healthData = await response.json();

      return {
        status: healthData.status || 'healthy',
        message: JSON.stringify(healthData),
      };
    } catch (error) {
      return {
        status: 'error',
        message: error instanceof Error ? error.message : 'API not available',
      };
    }
  }

  /**
   * Subscribe to real-time updates via polling
   */
  subscribeToRealTimeUpdates(callback: (data: MarketDataPoint) => void): () => void {
    this.subscribers.push(callback);

    // Start polling if this is the first subscriber
    if (this.subscribers.length === 1) {
      this.startPolling();
    }

    // Return unsubscribe function
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }

      // Stop polling if no more subscribers
      if (this.subscribers.length === 0) {
        this.stopPolling();
      }
    };
  }

  /**
   * Get latest historical data when market is closed
   */
  private async getHistoricalFallback(): Promise<MarketDataPoint[]> {
    console.log('[ApiDataProvider] Market closed - fetching latest historical data from database');

    try {
      const response = await this.getCandlestickData('USDCOP', '5m', undefined, undefined, 1);

      if (response.data && response.data.length > 0) {
        const latestCandle = response.data[response.data.length - 1];

        return [
          {
            symbol: 'USDCOP',
            price: latestCandle.close,
            timestamp: latestCandle.time,
            volume: latestCandle.volume,
            bid: latestCandle.close - 0.5,
            ask: latestCandle.close + 0.5,
            source: 'database_historical',
          },
        ];
      }

      throw new Error('No historical data available');
    } catch (error) {
      console.error('[ApiDataProvider] Failed to fetch historical data:', error);
      throw new Error('Real data unavailable - no fallback allowed');
    }
  }

  /**
   * Calculate statistics from recent candlestick data
   */
  private async getStatsFromCandlesticks(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    try {
      // Get last 24 hours of 5-minute data (288 candles)
      const response = await this.getCandlestickData(symbol, '5m', undefined, undefined, 288, false);

      if (response.data && response.data.length > 0) {
        const candles = response.data;
        const prices = candles.map((c) => c.close);
        const volumes = candles.map((c) => c.volume);
        const highs = candles.map((c) => c.high);
        const lows = candles.map((c) => c.low);

        const currentPrice = prices[prices.length - 1];
        const openPrice = candles[0].open;
        const high24h = Math.max(...highs);
        const low24h = Math.min(...lows);
        const volume24h = volumes.reduce((sum, vol) => sum + vol, 0);
        const priceChange = currentPrice - openPrice;
        const priceChangePercent = (priceChange / openPrice) * 100;

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
        };
      }

      throw new Error('No candlestick data available');
    } catch (error) {
      console.error('Error calculating stats from candlesticks:', error);
      throw error;
    }
  }

  /**
   * Start polling for updates
   */
  private startPolling(): void {
    if (this.pollInterval) return;

    this.pollInterval = setInterval(async () => {
      try {
        const data = await this.getRealTimeData();
        if (data.length > 0) {
          this.subscribers.forEach((callback) => callback(data[0]));
        }
      } catch (error) {
        console.error('[ApiDataProvider] Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds
  }

  /**
   * Stop polling for updates
   */
  private stopPolling(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }
}
