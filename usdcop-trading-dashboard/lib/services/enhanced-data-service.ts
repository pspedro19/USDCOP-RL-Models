export interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  timestamp: number
  volume?: number
  high?: number
  low?: number
  open?: number
}

export interface EnhancedCandle {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source?: string;
}

export class EnhancedDataService {
  private static instance: EnhancedDataService
  private wsConnection: WebSocket | null = null
  private subscribers: ((data: MarketData[]) => void)[] = []

  static getInstance(): EnhancedDataService {
    if (!this.instance) {
      this.instance = new EnhancedDataService()
    }
    return this.instance
  }

  subscribe(callback: (data: MarketData[]) => void): () => void {
    this.subscribers.push(callback)
    return () => {
      const index = this.subscribers.indexOf(callback)
      if (index > -1) {
        this.subscribers.splice(index, 1)
      }
    }
  }

  async connect(): Promise<void> {
    // Mock connection for now
    return Promise.resolve()
  }

  disconnect(): void {
    if (this.wsConnection) {
      this.wsConnection.close()
      this.wsConnection = null
    }
  }

  async getHistoricalData(
    symbol: string,
    startTime: number,
    endTime: number
  ): Promise<MarketData[]> {
    try {
      // Try to fetch from the main API through proxy
      const response = await fetch(`/api/proxy/trading/api/market/historical?symbol=${symbol}&start=${startTime}&end=${endTime}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Main API not available, using fallback data');
    }

    // Generate realistic mock data for USD/COP
    const data: MarketData[] = [];
    const basePrice = 4200; // USD/COP base price
    let currentPrice = basePrice;

    for (let i = 0; i < 100; i++) {
      const change = (Math.random() - 0.5) * 20;
      currentPrice += change;

      data.push({
        symbol,
        price: Math.round(currentPrice * 100) / 100,
        change: Math.round(change * 100) / 100,
        changePercent: Math.round((change / currentPrice) * 10000) / 100,
        timestamp: startTime + (i * 60000), // 1 minute intervals
        volume: Math.floor(Math.random() * 10000000),
        high: Math.round((currentPrice + Math.random() * 10) * 100) / 100,
        low: Math.round((currentPrice - Math.random() * 10) * 100) / 100,
        open: Math.round((currentPrice - change) * 100) / 100
      });
    }

    return data;
  }

  async loadCompleteHistory(symbol: string = 'USDCOP'): Promise<EnhancedCandle[]> {
    try {
      // Try to fetch from the main API through proxy
      const response = await fetch(`/api/proxy/trading/api/market/complete-history?symbol=${symbol}`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn('Complete history API not available, generating mock data');
    }

    // Generate realistic OHLC data for charts
    const data: EnhancedCandle[] = [];
    const basePrice = 4200;
    let currentPrice = basePrice;
    const now = new Date();

    // Generate 1000 5-minute candles (about 3.5 days of data)
    for (let i = 999; i >= 0; i--) {
      const candleTime = new Date(now.getTime() - (i * 5 * 60 * 1000));

      const open = currentPrice;
      const volatility = Math.random() * 30;
      const high = open + Math.random() * volatility;
      const low = open - Math.random() * volatility;
      const close = low + Math.random() * (high - low);

      currentPrice = close; // Next candle starts where this one ends

      data.push({
        datetime: candleTime.toISOString(),
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume: Math.floor(Math.random() * 5000000),
        source: 'mock'
      });
    }

    return data;
  }
}

export const dataService = EnhancedDataService.getInstance()
export const enhancedDataService = dataService
