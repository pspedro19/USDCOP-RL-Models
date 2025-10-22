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

    // NO MOCK DATA - Return empty array if API fails
    // UI should display "No historical data available - API unavailable"
    console.error('[EnhancedDataService] Historical data unavailable - API must be running');
    console.error('[EnhancedDataService] Start Trading API on port 8000');
    return [];
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

    // NO MOCK DATA - Return empty array if API fails
    // UI should display "No historical data available - API unavailable"
    console.error('[EnhancedDataService] Complete history unavailable - API must be running');
    console.error('[EnhancedDataService] Start Trading API on port 8000');
    return [];
  }
}

export const dataService = EnhancedDataService.getInstance()
export const enhancedDataService = dataService
