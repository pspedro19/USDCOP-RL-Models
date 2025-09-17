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
    // Mock historical data
    return []
  }
}

export const dataService = EnhancedDataService.getInstance()
