/**
 * Mock Data Provider
 * ==================
 *
 * Provides mock market data for testing and development
 * Implements IDataProvider interface for seamless testing
 *
 * WARNING: This provider should NEVER be used in production environments
 */

// Production environment guard
if (process.env.NODE_ENV === 'production') {
  throw new Error('MockDataProvider cannot be used in production');
}

import type {
  IDataProvider,
  MarketDataPoint,
  CandlestickData,
  CandlestickResponse,
  SymbolStats,
} from '@/lib/core/interfaces';

export class MockDataProvider implements IDataProvider {
  private subscribers: Array<(data: MarketDataPoint) => void> = [];
  private updateInterval: NodeJS.Timeout | null = null;
  private basePrice = 4200;
  private priceVolatility = 0.001; // 0.1% volatility

  /**
   * Get mock real-time data
   */
  async getRealTimeData(): Promise<MarketDataPoint[]> {
    const price = this.generatePrice();
    const timestamp = Date.now();

    return [
      {
        symbol: 'USDCOP',
        price,
        timestamp,
        volume: Math.floor(Math.random() * 1000000),
        bid: price - 0.5,
        ask: price + 0.5,
        source: 'mock',
      },
    ];
  }

  /**
   * Get mock candlestick data
   */
  async getCandlestickData(
    symbol: string = 'USDCOP',
    timeframe: string = '5m',
    startDate?: string,
    endDate?: string,
    limit: number = 1000,
    includeIndicators: boolean = true
  ): Promise<CandlestickResponse> {
    const now = Date.now();
    const fiveMinutes = 5 * 60 * 1000;
    const data: (CandlestickData & { indicators?: any })[] = [];

    // Generate mock candles
    for (let i = limit - 1; i >= 0; i--) {
      const time = now - i * fiveMinutes;
      const open = this.generatePrice();
      const close = this.generatePrice();
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;
      const volume = Math.floor(Math.random() * 1000000);

      const candle: CandlestickData & { indicators?: any } = {
        time,
        open,
        high,
        low,
        close,
        volume,
      };

      if (includeIndicators) {
        candle.indicators = {
          ema_20: close + (Math.random() - 0.5) * 5,
          ema_50: close + (Math.random() - 0.5) * 10,
          ema_200: close + (Math.random() - 0.5) * 20,
          bb_upper: high + Math.random() * 3,
          bb_middle: close,
          bb_lower: low - Math.random() * 3,
          rsi: Math.random() * 100,
        };
      }

      data.push(candle);
    }

    return {
      symbol,
      timeframe,
      start_date: startDate || new Date(now - limit * fiveMinutes).toISOString(),
      end_date: endDate || new Date(now).toISOString(),
      count: data.length,
      data,
    };
  }

  /**
   * Get mock symbol statistics
   */
  async getSymbolStats(symbol: string = 'USDCOP'): Promise<SymbolStats> {
    const currentPrice = this.generatePrice();
    const openPrice = this.basePrice;
    const change = currentPrice - openPrice;
    const changePercent = (change / openPrice) * 100;

    return {
      symbol,
      price: currentPrice,
      open_24h: openPrice,
      high_24h: this.basePrice + 50,
      low_24h: this.basePrice - 50,
      volume_24h: Math.floor(Math.random() * 100000000),
      change_24h: change,
      change_percent_24h: changePercent,
      spread: 1.0,
      timestamp: new Date().toISOString(),
      source: 'mock',
    };
  }

  /**
   * Check mock provider health
   */
  async checkHealth(): Promise<{ status: string; message?: string }> {
    return {
      status: 'healthy',
      message: 'Mock data provider is operational',
    };
  }

  /**
   * Subscribe to mock real-time updates
   */
  subscribeToRealTimeUpdates(callback: (data: MarketDataPoint) => void): () => void {
    this.subscribers.push(callback);

    // Start sending updates if this is the first subscriber
    if (this.subscribers.length === 1) {
      this.startUpdates();
    }

    // Return unsubscribe function
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }

      // Stop updates if no more subscribers
      if (this.subscribers.length === 0) {
        this.stopUpdates();
      }
    };
  }

  /**
   * Generate a realistic price with volatility
   */
  private generatePrice(): number {
    const change = (Math.random() - 0.5) * 2 * this.priceVolatility * this.basePrice;
    this.basePrice += change;
    return Number(this.basePrice.toFixed(4));
  }

  /**
   * Start sending periodic updates to subscribers
   */
  private startUpdates(): void {
    if (this.updateInterval) return;

    this.updateInterval = setInterval(() => {
      const price = this.generatePrice();
      const marketData: MarketDataPoint = {
        symbol: 'USDCOP',
        price,
        timestamp: Date.now(),
        volume: Math.floor(Math.random() * 1000000),
        bid: price - 0.5,
        ask: price + 0.5,
        source: 'mock',
      };

      this.subscribers.forEach((callback) => callback(marketData));
    }, 2000); // Update every 2 seconds
  }

  /**
   * Stop sending updates
   */
  private stopUpdates(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}
