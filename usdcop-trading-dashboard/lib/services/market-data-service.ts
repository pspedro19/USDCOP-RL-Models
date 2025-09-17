export interface MarketDataPoint {
  symbol: string
  price: number
  timestamp: number
  volume: number
}

export class MarketDataService {
  static async getRealTimeData(): Promise<MarketDataPoint[]> {
    return [
      { symbol: 'USDCOP', price: 4200, timestamp: Date.now(), volume: 1000 }
    ]
  }
}
